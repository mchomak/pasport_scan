"""
Тестовый скрипт распознавания паспорта РФ через Tesseract OCR.

Установка:
  1. Скачать Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
     (добавить путь к tesseract.exe в PATH)
  2. Скачать русский язык:
     https://github.com/tessdata/fast/blob/master/rus.traineddata
     Скопировать rus.traineddata в папку tessdata (рядом с tesseract.exe)
  3. pip install pytesseract opencv-python numpy Pillow
"""

import os
import re
import cv2
import numpy as np
import json
from datetime import datetime

try:
    import pytesseract
except ImportError:
    print("❌ pytesseract не установлен: pip install pytesseract")
    exit(1)


# ---------------------------------------------------------------------------
# Конфигурации Tesseract (PSM + whitelist)
# ---------------------------------------------------------------------------
# PSM 3  — auto page segmentation (русский текст: несколько блоков на паспорте)
# PSM 6  — однородный блок текста (MRZ: два аккуратных ряда)
# PSM 7  — одна текстовая строка (серия/номер, MRZ fallback)
# PSM 11 — текст без порядка (разрозненные цифры)

# Серия + номер: только цифры
CFG_DIGITS_LINE = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
CFG_DIGITS_BLOCK = "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789"

# MRZ: латиница + цифры + < (два варианта PSM — пробуем оба)
CFG_MRZ_BLOCK = "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
CFG_MRZ_LINE  = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

# Русский текст — PSM 3 (auto page segmentation): паспорт содержит несколько блоков текста
CFG_RUSSIAN = "--oem 3 --psm 3"


# ---------------------------------------------------------------------------
# Транслитерация (ГОСТ 7.79-2000)
# ---------------------------------------------------------------------------
_TRANSLIT_MAP = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'YO',
    'Ж': 'ZH', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'H', 'Ц': 'TS', 'Ч': 'CH', 'Ш': 'SH', 'Щ': 'SHCH',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'YU', 'Я': 'YA',
}


def transliterate(text):
    if not text:
        return None
    return ''.join(_TRANSLIT_MAP.get(ch, ch) for ch in text.upper())


# ---------------------------------------------------------------------------
# Парсинг дат
# ---------------------------------------------------------------------------
def parse_date_dmy(date_str):
    if not date_str:
        return None
    for fmt in ('%d.%m.%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def parse_mrz_date(six_digits):
    if not six_digits or len(six_digits) != 6 or not six_digits.isdigit():
        return None
    try:
        yy, mm, dd = int(six_digits[0:2]), int(six_digits[2:4]), int(six_digits[4:6])
        year = yy + (1900 if yy > 50 else 2000)
        return datetime(year, mm, dd).date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Document detection & cropping
# ---------------------------------------------------------------------------
def order_points(pts):
    """Упорядочиваем 4 точки: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left (наименьшая сумма)
    rect[2] = pts[np.argmax(s)]   # bottom-right (наибольшая сумма)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def detect_and_crop_document(image_path, output_dir='preprocessed'):
    """
    Детектирует документ в кадре, применяет perspective transform.
    Сохраняет выровненный crop и возвращает путь к нему.
    Если документ не найден, сохраняет оригинал.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur для снижения шума на Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edged = cv2.Canny(blurred, 50, 200)

    # Морфология: закрываем разрывы на контуре документа
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # Находим контуры
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ищем самый большой прямоугольный контур
    doc_contour = None
    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        # Документ должен быть минимум 30% от площади кадра
        if area < (w * h * 0.3):
            continue

        # Аппроксимируем контур до многоугольника
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Если получили четырёхугольник — кандидат
        if len(approx) == 4 and area > max_area:
            max_area = area
            doc_contour = approx

    # Если не нашли прямоугольный контур — сохраняем оригинал
    if doc_contour is None:
        print("  ⚠️  Документ не детектирован, используем полное изображение")
        cropped_path = f"{output_dir}/{base}_cropped.jpg"
        cv2.imwrite(cropped_path, img)
        return cropped_path

    # Применяем perspective transform
    pts = doc_contour.reshape(4, 2)
    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    # Вычисляем новую ширину/высоту
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Целевые координаты для transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # Паспорт РФ — формат А5 (портрейт: высота > ширина).
    # Если после perspective transform получилась ландшафтная ориентация —
    # контур был определён в повёрнутом состоянии. Поворачиваем в портрейт.
    wh, ww = warped.shape[:2]
    if ww > wh:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        print(f"  ↻ Портрейт: {ww}x{wh} → {warped.shape[1]}x{warped.shape[0]}")

    print(f"  ✓ Документ детектирован и выровнен: {w}x{h} → {warped.shape[1]}x{warped.shape[0]}")

    cropped_path = f"{output_dir}/{base}_cropped.jpg"
    cv2.imwrite(cropped_path, warped)
    return cropped_path


# ---------------------------------------------------------------------------
# Препроцессинг (три зоны)
# ---------------------------------------------------------------------------
def _to_grayscale(img_bgr):
    """
    Чистое серое изображение — без blur, denoise, CLAHE.
    Tesseract 4+ LSTM использует полутоны напрямую для распознавания.
    Любая дополнительная обработка (denoise, CLAHE) размывает границы букв
    и снижает качество распознавания.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def preprocess_full(image_path, output_dir='preprocessed'):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить: {image_path}")

    h, w = img.shape[:2]
    # Tesseract LSTM нуждается в минимум 300 DPI эквивалентом.
    # Для паспорта (~A5) это примерно 1240x1748px.
    # Берём 3x от оригинала или минимум 3000px ширины — whichever больше.
    target_w = max(w * 3, 3000)
    if target_w > w:
        scale = target_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        print(f"  ↗ upscale {w}x{h} → {img.shape[1]}x{img.shape[0]}")

    gray = _to_grayscale(img)
    path = f"{output_dir}/{base}_full.jpg"
    cv2.imwrite(path, gray)
    print(f"  ✓ Полное изображение обработано")
    return path


def preprocess_edge_strips(image_path, output_dir='preprocessed'):
    """
    Серия/номер может оказаться на любом из 4 краёв после perspective transform.
    Вырезаем полоску (20%) с каждого края и для каждой делаем 2 ротации.
    Итого 8 вариантов — OCR возьмёт тот, где больше цифр.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]

    # 4 края — 20% от соответствующей размерности
    edges = [
        ('right',  img[:, int(w * 0.80):, :]),
        ('left',   img[:, :int(w * 0.20), :]),
        ('top',    img[:int(h * 0.20), :, :]),
        ('bottom', img[int(h * 0.80):, :, :]),
    ]

    paths = []
    for edge_name, strip in edges:
        for rot_name, rotated in [
            ('ccw', cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ('cw',  cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE)),
        ]:
            sh, sw = rotated.shape[:2]
            rotated = cv2.resize(rotated, (sw * 5, sh * 5), interpolation=cv2.INTER_CUBIC)
            gray = _to_grayscale(rotated)
            path = f"{output_dir}/{base}_strip_{edge_name}_{rot_name}.jpg"
            cv2.imwrite(path, gray)
            paths.append(path)

    print(f"  ✓ Полосы 4 края: {len(paths)} вариантов")
    return paths


def preprocess_mrz(image_path, output_dir='preprocessed'):
    """
    MRZ может быть внизу (норма) или вверху (если документ перевёрнут 180°).
    Кроп верхних 35% поворачиваем на 180° для корректного чтения.
    Возвращает список путей — OCR проверит все кандидаты и выберет лучший.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]

    candidates = [
        ('bottom', img[int(h * 0.65):, :]),                                      # нижние 35% (основной)
        ('top',    cv2.rotate(img[:int(h * 0.35), :, :], cv2.ROTATE_180)),  # верхние 35°, повёрнутые 180°
    ]

    paths = []
    for name, crop in candidates:
        ch, cw = crop.shape[:2]
        crop = cv2.resize(crop, (cw * 4, ch * 4), interpolation=cv2.INTER_CUBIC)
        gray = _to_grayscale(crop)
        path = f"{output_dir}/{base}_mrz_{name}.jpg"
        cv2.imwrite(path, gray)
        paths.append(path)

    print(f"  ✓ MRZ: {len(paths)} кандидатов (bottom + top)")
    return paths


# ---------------------------------------------------------------------------
# OCR через Tesseract
# ---------------------------------------------------------------------------
def ocr_russian(image_path):
    """Полное изображение — русский текст. Возвращает список строк."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    raw = pytesseract.image_to_string(img, lang='rus', config=CFG_RUSSIAN)
    return [line.strip() for line in raw.split('\n') if line.strip()]


def ocr_digits(strip_paths):
    """
    Правая полоса — только цифры.
    Пробуем оба варианта поворота и оба PSM (7 и 11).
    Возвращаем строку с максимумом цифр.
    """
    best = ''
    for path in strip_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        for cfg in (CFG_DIGITS_LINE, CFG_DIGITS_BLOCK):
            raw = pytesseract.image_to_string(img, lang='eng', config=cfg)
            digits = re.sub(r'[^\d]', '', raw)
            if len(digits) > len(best):
                best = digits
    return best


def ocr_mrz(mrz_paths):
    """
    MRZ через английский с whitelist.
    Перебираем все кандидатные crop (bottom/top) × PSM (6/7).
    Берём вариант с лучшим скором: count('<') + наличие PNRUS.
    """
    if not mrz_paths:
        return []

    best_lines = []
    best_score = -1

    for mrz_path in mrz_paths:
        img = cv2.imread(mrz_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for cfg in (CFG_MRZ_BLOCK, CFG_MRZ_LINE):
            raw = pytesseract.image_to_string(img, lang='eng', config=cfg)
            lines = [l.strip() for l in raw.split('\n') if l.strip()]
            text = ' '.join(lines)

            score = text.count('<')
            if 'PNRUS' in text or 'P<RUS' in text:
                score += 100

            print(f"  📝 MRZ {os.path.basename(mrz_path)} PSM={cfg.split()[1][-1]}: score={score}, lines={lines}")
            if score > best_score:
                best_score = score
                best_lines = lines

    return best_lines


# ---------------------------------------------------------------------------
# Извлечение полей
# ---------------------------------------------------------------------------
def _fix_mrz_fio_chars(s):
    """Исправляем OCR-замены в буквенной части MRZ: 4->S, 3->Z, 1->I, 0->O, QQ-><<."""
    s = s.replace('QQ', '<<')
    result = []
    for ch in s:
        if ch == '<':
            result.append('<')
        elif ch == '4':
            result.append('S')
        elif ch == '3':
            result.append('Z')
        elif ch == '1':
            result.append('I')
        elif ch == '0':
            result.append('O')
        else:
            result.append(ch)
    return ''.join(result)


def extract_series_number(digits_str, russian_lines):
    """Серия (4) + номер (6) из строки цифр. Fallback на русский текст."""
    print(f"  📝 Цифры из полосы: {digits_str}")

    if len(digits_str) < 10:
        print(f"  ⚠️  Мало цифр в полосе, пробуем fallback...")
        fallback = ''
        for line in russian_lines:
            stripped = line.strip()
            if stripped.isdigit() and 1 <= len(stripped) <= 4:
                fallback += stripped
        print(f"  📝 Fallback цифры: {fallback}")
        if len(fallback) > len(digits_str):
            digits_str = fallback

    digits_str = digits_str[:10]
    series = digits_str[:4] if len(digits_str) >= 4 else None
    number = digits_str[4:10] if len(digits_str) >= 10 else None
    return series, number


def extract_fio_from_mrz(mrz_lines):
    """ФИО из MRZ строк (латиница). Ищем строку с PNRUS."""
    for line in mrz_lines:
        t = line.upper().replace(' ', '')
        t = t.replace('{', '<').replace('[', '<').replace('(', '<')

        match = re.search(r'P[N<]RUS', t)
        if not match:
            continue

        fio_part = t[match.end():]
        print(f"  📝 MRZ FIO (raw):  ...{fio_part[:50]}")
        fio_part = _fix_mrz_fio_chars(fio_part)
        print(f"  📝 MRZ FIO (fixed): ...{fio_part[:50]}")

        parts = re.split(r'<{2,}', fio_part)
        parts = [p.replace('<', '').strip() for p in parts if p.replace('<', '').strip()]

        surname = parts[0] if len(parts) >= 1 else None
        name = middle_name = None
        if len(parts) >= 2:
            nm = [x.strip() for x in parts[1].split('<') if x.strip()]
            name = nm[0] if len(nm) >= 1 else None
            middle_name = nm[1] if len(nm) >= 2 else None

        if surname:
            return surname, name, middle_name

    return None, None, None


def extract_meta_from_mrz(mrz_lines):
    """Дата рождения, пол, гражданство из MRZ (строка 2)."""
    result = {}
    for line in mrz_lines:
        t = line.upper().replace(' ', '')
        t = t.replace('{', '<').replace('[', '<').replace('(', '<')
        t = t.replace('О', '0').replace('I', '1').replace('l', '1')

        # RUS + дата(6) + контрольная(1) + пол
        m = re.search(r'RUS(\d{6})\d([MF<])', t)
        if m:
            bd = parse_mrz_date(m.group(1))
            if bd:
                result['birth_date'] = bd
                print(f"  📝 MRZ дата рождения: {bd}")

            result['citizenship'] = 'RU'
            g = m.group(2)
            if g in ('M', 'F'):
                result['gender'] = g
                print(f"  📝 MRZ пол: {g}")
            break
    return result


def extract_from_russian(russian_lines):
    """Даты (DD.MM.YYYY) и пол (МУЖ/ЖЕН) из русского текста."""
    dates = []
    gender = None

    for line in russian_lines:
        # Даты
        for m in re.finditer(r'(\d{2}[.\-/]\d{2}[.\-/]\d{4})', line):
            d = parse_date_dmy(m.group(1).replace('-', '.').replace('/', '.'))
            if d:
                dates.append(d)

        # Пол — точное и нечёткое
        if gender is None:
            low = line.lower().strip().rstrip('.')
            if re.search(r'м\s*у\s*ж', low):
                gender = 'M'
            elif re.search(r'ж\s*е\s*н', low):
                gender = 'F'
            elif re.match(r'^[а-яё]{3}$', low):
                muж = sum(1 for a, b in zip(low, 'муж') if a == b)
                жен = sum(1 for a, b in zip(low, 'жен') if a == b)
                if muж >= 2:
                    gender = 'M'
                elif жен >= 2:
                    gender = 'F'

    dates.sort()
    birth_date = dates[0] if len(dates) >= 1 else None
    issue_date = dates[1] if len(dates) >= 2 else None

    if birth_date and not issue_date and birth_date.year >= 2010:
        issue_date, birth_date = birth_date, None

    return birth_date, issue_date, gender


def extract_fio_russian(russian_lines):
    """ФИО из русского текста -> транслитерация."""
    excluded = {
        'РФ', 'МУЖ', 'ЖЕН', 'НУХ', 'НУЖ', 'ПОЛ', 'РОССИЙСКАЯ', 'ФЕДЕРАЦИЯ',
        'ПАСПОРТ', 'ВЫДАН', 'ОБЛАСТИ', 'ОБЛАСТЬ', 'МОСКВА', 'ГОР', 'ПОЛЯ',
    }
    names = []
    for line in russian_lines:
        t = line.strip().upper().rstrip('.')
        if not re.match(r'^[А-ЯЁ]+(\-[А-ЯЁ]+)?$', t):
            continue
        if len(t) < 3 or len(t) > 15 or t in excluded:
            continue
        names.append(t)

    print(f"  📝 Слова-кандидаты: {names}")
    su, na, mi = (names[i] if len(names) > i else None for i in range(3))
    return (transliterate(su), transliterate(na), transliterate(mi), su, na, mi)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" TESSERACT OCR — Распознавание паспорта РФ")
    print(" Источники: вертикальные цифры | MRZ | русский текст")
    print("=" * 70)
    print()

    # Проверяем что tesseract установлен
    try:
        ver = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract v{ver}")
    except Exception:
        print("❌ Tesseract не найден. Установите tesseract-ocr и добавьте в PATH")
        return

    while True:
        print("\n📁 Путь к изображению паспорта (или 'exit'):")
        image_path = input("> ").strip().strip('"').strip("'")

        if image_path.lower() == 'exit':
            print("\n👋 Выход")
            break
        if not image_path or not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}\n")
            continue

        print(f"\n🔍 Обработка: {os.path.basename(image_path)}")
        print("=" * 70)

        try:
            # ----------------------------------------------------------
            # ЭТАП 0: Document detection & cropping
            # ----------------------------------------------------------
            print("\n[0/4] Детектирование документа...")
            cropped_doc_path = detect_and_crop_document(image_path)
            if cropped_doc_path is None:
                print("❌ Не удалось загрузить изображение")
                continue

            # ----------------------------------------------------------
            # ЭТАП 1: Препроцессинг (работаем с cropped документом)
            # ----------------------------------------------------------
            print("\n[1/4] Препроцессинг...")
            full_path = preprocess_full(cropped_doc_path)
            strip_paths = preprocess_edge_strips(cropped_doc_path)
            mrz_paths = preprocess_mrz(cropped_doc_path)

            # ----------------------------------------------------------
            # ЭТАП 2: Tesseract OCR
            # ----------------------------------------------------------
            print("\n[2/4] Распознавание (Tesseract)...")
            russian_lines = ocr_russian(full_path)
            print(f"  ✓ Русский текст: {len(russian_lines)} строк")

            digits_str = ocr_digits(strip_paths)
            print(f"  ✓ Полоса цифр: '{digits_str}'")

            mrz_lines = ocr_mrz(mrz_paths)
            print(f"  ✓ MRZ: {len(mrz_lines)} строк")

            # ----------------------------------------------------------
            # ЭТАП 3: Извлечение
            # ----------------------------------------------------------
            print("\n[3/4] Извлечение данных...")

            print("  >> Серия и номер:")
            series, number = extract_series_number(digits_str, russian_lines)
            print(f"     Серия: {series}, Номер: {number}")

            print("  >> ФИО (MRZ):")
            surname_lat, name_lat, middle_name_lat = extract_fio_from_mrz(mrz_lines)
            print(f"     {surname_lat} {name_lat} {middle_name_lat}")

            print("  >> Метаданные (MRZ):")
            mrz_meta = extract_meta_from_mrz(mrz_lines)
            birth_date_mrz = mrz_meta.get('birth_date')
            citizenship = mrz_meta.get('citizenship')
            gender_mrz = mrz_meta.get('gender')

            print("  >> Даты и пол (русский текст):")
            birth_date_ru, issue_date_ru, gender_ru = extract_from_russian(russian_lines)
            print(f"     Даты: {birth_date_ru} / {issue_date_ru},  Пол: {gender_ru}")

            print("  >> ФИО (русский текст + транслитерация):")
            (su_fb, na_fb, mi_fb, su_ru, na_ru, mi_ru) = extract_fio_russian(russian_lines)
            print(f"     RU: {su_ru} {na_ru} {mi_ru}")
            print(f"     Translit: {su_fb} {na_fb} {mi_fb}")

            # ----------------------------------------------------------
            # ЭТАП 4: Сборка
            # ----------------------------------------------------------
            print("\n[4/4] Сборка результата...")

            fio_latin = (surname_lat or su_fb, name_lat or na_fb, middle_name_lat or mi_fb)
            fio_russian = (su_ru, na_ru, mi_ru)

            birth_date = birth_date_ru or birth_date_mrz
            issue_date = issue_date_ru
            if birth_date and issue_date and birth_date > issue_date:
                birth_date, issue_date = issue_date, birth_date

            gender = gender_ru or gender_mrz
            if not citizenship:
                citizenship = 'RU'

            result = {
                'required_data': {
                    'fio_latin': {
                        'surname': fio_latin[0],
                        'name': fio_latin[1],
                        'middle_name': fio_latin[2],
                    },
                    'birth_date': birth_date.isoformat() if birth_date else None,
                    'issue_date': issue_date.isoformat() if issue_date else None,
                    'citizenship': citizenship,
                    'gender': gender,
                    'series': series,
                    'number': number,
                },
                'all_detected_data': {
                    'fio_russian': {
                        'surname': fio_russian[0],
                        'name': fio_russian[1],
                        'middle_name': fio_russian[2],
                    },
                    'birth_date': birth_date.isoformat() if birth_date else None,
                    'issue_date': issue_date.isoformat() if issue_date else None,
                    'citizenship': citizenship,
                    'gender': gender,
                    'series': series,
                    'number': number,
                    'raw_russian_lines': russian_lines,
                    'raw_mrz_lines': mrz_lines,
                    'raw_digits': digits_str,
                },
            }

            # Вывод
            req = result['required_data']
            print("\n" + "=" * 70)
            print("📋 ИТОГОВЫЕ ДАННЫЕ (required_data)")
            print("=" * 70)
            print(f"  👤 Фамилия:     {req['fio_latin']['surname']}")
            print(f"  👤 Имя:         {req['fio_latin']['name']}")
            print(f"  👤 Отчество:    {req['fio_latin']['middle_name']}")
            print(f"  🎂 Д. рожд.:    {req['birth_date']}")
            print(f"  📅 Д. выдачи:   {req['issue_date']}")
            print(f"  🌍 Гражданство: {req['citizenship']}")
            print(f"  ⚥  Пол:         {req['gender']}")
            print(f"  📄 Серия:       {req['series']}")
            print(f"  📄 Номер:       {req['number']}")

            out_path = os.path.splitext(image_path)[0] + '_tesseract_result.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 JSON: {out_path}")
            print(f"💾 Изображения: preprocessed/")

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()