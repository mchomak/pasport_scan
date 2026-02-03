"""
Улучшенный скрипт распознавания паспорта РФ через EasyOCR.

Источники данных:
  - Правая вертикальная полоса -> серия и номер
  - MRZ (нижние строки с <<<) -> ФИО латиницей, дата рождения, пол, гражданство
  - Русский текст -> даты, пол (fallback)
  - Транслитерация русского ФИО -> ФИО латиницей (fallback если MRZ не распознана)
"""

import os
import re
import cv2
import numpy as np
import easyocr
from datetime import datetime
import json


# ---------------------------------------------------------------------------
# Транслитерация (ГОСТ 7.79-2000, система Б) — fallback если MRZ не найдена
# ---------------------------------------------------------------------------
_TRANSLIT_MAP = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'YO',
    'Ж': 'ZH', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'H', 'Ц': 'TS', 'Ч': 'CH', 'Ш': 'SH', 'Щ': 'SHCH',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'YU', 'Я': 'YA',
}


def transliterate(text):
    """Транслитерация кириллицы в латиницу."""
    result = []
    for ch in text.upper():
        result.append(_TRANSLIT_MAP.get(ch, ch))
    return ''.join(result)


# ---------------------------------------------------------------------------
# Парсинг дат
# ---------------------------------------------------------------------------
def parse_date_dmy(date_str):
    """DD.MM.YYYY -> date"""
    if not date_str:
        return None
    for fmt in ('%d.%m.%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def parse_mrz_date(six_digits):
    """YYMMDD -> date"""
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
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_and_crop_document(image_path, output_dir='preprocessed'):
    """
    Детектирует документ в кадре, применяет perspective transform.
    Сохраняет выровненный crop и возвращает путь к нему.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    doc_contour = None
    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < (w * h * 0.3):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            doc_contour = approx

    if doc_contour is None:
        print("  ⚠️  Документ не детектирован, используем полное изображение")
        cropped_path = f"{output_dir}/{base}_cropped.jpg"
        cv2.imwrite(cropped_path, img)
        return cropped_path

    pts = doc_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

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
# Препроцессинг общего изображения
# ---------------------------------------------------------------------------
def preprocess_full(image_path, output_dir='preprocessed'):
    """
    Полный препроцессинг: upscale -> grayscale -> denoise -> CLAHE -> adaptive threshold.
    Возвращает путь к обработанному изображению.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить: {image_path}")

    h, w = img.shape[:2]
    # 3x от оригинала или минимум 3000px для высокого качества OCR
    target_w = max(w * 3, 3000)
    if target_w > w:
        scale = target_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        print(f"  ↗ upscale {w}x{h} → {img.shape[1]}x{img.shape[0]}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    path = f"{output_dir}/{base}_full.jpg"
    cv2.imwrite(path, contrast)
    print(f"  ✓ Полное изображение обработано")
    return path


# ---------------------------------------------------------------------------
# Вырезка и обработка правой вертикальной полосы (серия + номер)
# ---------------------------------------------------------------------------
def _process_strip(strip_bgr, output_dir, name):
    """Общий препроцессинг полосы: resize 5x -> gray -> denoise -> CLAHE -> Otsu."""
    sh, sw = strip_bgr.shape[:2]
    strip_bgr = cv2.resize(strip_bgr, (sw * 5, sh * 5), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    path = f"{output_dir}/{name}.jpg"
    cv2.imwrite(path, binary)
    return path


def preprocess_edge_strips(image_path, output_dir='preprocessed'):
    """
    Серия/номер может оказаться на любом из 4 краёв после perspective transform.
    Вырезаем полоску (20%) с каждого края × 2 ротации = 8 вариантов.
    OCR выберет тот, где больше цифр.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]

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
            path = _process_strip(rotated, output_dir, f"{base}_strip_{edge_name}_{rot_name}")
            paths.append(path)

    print(f"  ✓ Полосы 4 края: {len(paths)} вариантов")
    return paths


# ---------------------------------------------------------------------------
# Вырезка и обработка MRZ зоны (нижние ~25 % изображения)
# ---------------------------------------------------------------------------
def _process_mrz(crop_bgr, output_dir, name):
    """Общий препроцессинг MRZ crop: 4x resize -> gray -> denoise -> CLAHE -> Otsu."""
    ch, cw = crop_bgr.shape[:2]
    crop_bgr = cv2.resize(crop_bgr, (cw * 4, ch * 4), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    path = f"{output_dir}/{name}.jpg"
    cv2.imwrite(path, binary)
    return path


def preprocess_mrz(image_path, output_dir='preprocessed'):
    """
    MRZ кандидаты: нижние 35% + верхние 35% (повёрнутые 180°).
    Возвращает список путей — OCR проверит оба кандидата.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]

    candidates = [
        ('bottom', img[int(h * 0.65):, :]),                                      # нижние 35% (основной)
        ('top',    cv2.rotate(img[:int(h * 0.35), :, :], cv2.ROTATE_180)),  # верхние 35%, повёрнутые 180°
    ]

    paths = []
    for name, crop in candidates:
        path = _process_mrz(crop, output_dir, f"{base}_mrz_{name}")
        paths.append(path)

    print(f"  ✓ MRZ: {len(paths)} кандидатов (bottom + top)")
    return paths


# ---------------------------------------------------------------------------
# Извлечение серии и номера из распознанного текста правой полосы
# ---------------------------------------------------------------------------
def extract_series_number(strip_ocr_variants, full_ocr_lines):
    """
    Серия (4 цифры) + номер (6 цифр) из правой полосы.

    strip_ocr_variants — список результатов OCR для каждого варианта поворота.
    Берём вариант с максимумом цифр. Если в обоих меньше 10 цифр —
    пробуем найти серию/номер как fallback из полного текста изображения.
    """
    best_digits = ''

    for variant_lines in strip_ocr_variants:
        digits = ''
        for _, text, _ in variant_lines:
            digits += re.sub(r'[^\d]', '', text)
        if len(digits) > len(best_digits):
            best_digits = digits

    print(f"  📝 Лучший вариант полосы ({len(best_digits)} цифр): {best_digits}")

    # Fallback: если полоса не дала 10 цифр, ищем в полном тексте
    # Вертикальные цифры иногда распознаются как отдельные однозначные строки
    # подряд идущие по Y-координатам. Собираем все однозначные/двузначные числа.
    if len(best_digits) < 10:
        print(f"  ⚠️  Полоса дала мало цифр, пробуем fallback из полного текста...")
        fallback_digits = ''
        for bbox, text, conf in full_ocr_lines:
            stripped = text.strip()
            # Берём только чисто-цифровые строки длиной 1-4 символа
            if stripped.isdigit() and 1 <= len(stripped) <= 4:
                fallback_digits += stripped
        print(f"  📝 Fallback цифры: {fallback_digits}")
        if len(fallback_digits) > len(best_digits):
            best_digits = fallback_digits

    # Берём первые 10 цифр
    best_digits = best_digits[:10]

    series = best_digits[:4] if len(best_digits) >= 4 else None
    number = best_digits[4:10] if len(best_digits) >= 10 else None

    return series, number


# ---------------------------------------------------------------------------
# Извлечение ФИО из MRZ строк
# ---------------------------------------------------------------------------
def _fix_mrz_fio_chars(s):
    """
    Исправляем типичные замены OCR в буквенной части MRZ.
    В MRZ после PNRUS идут ТОЛЬКО латинские буквы и '<'.
    OCR часто путает: S<->4, Z<->3, I<->1, O<->0, Q<-><.
    """
    # QQ подряд -> << (OCR часто читает << как QQ)
    s = s.replace('QQ', '<<')
    # Теперь заменяем цифры на похожие буквы, но только вне '<'
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


def extract_fio_from_mrz(ocr_lines):
    """
    Ищем строку начинающуюся с PNRUS (или похожую).
    Формат: PNRUS + ФАМИЛИЯ<<ИМЯ<ОТЧЕСТВО<<<...
    """
    for _, text, conf in ocr_lines:
        # Нормализация: заменяем символы похожие на '<'
        t = text.upper().replace(' ', '')
        t = t.replace('{', '<').replace('[', '<').replace('(', '<')
        t = t.replace('С', 'C').replace('Р', 'P').replace('Е', 'E')  # кириллица -> латиница

        # Ищем начало: PNRUS
        match = re.search(r'P[N<]RUS', t)
        if not match:
            continue

        fio_part = t[match.end():]
        print(f"  📝 MRZ FIO (raw):  ...{fio_part[:50]}")

        # Исправляем OCR-ошибки в буквенной части
        fio_part = _fix_mrz_fio_chars(fio_part)
        print(f"  📝 MRZ FIO (fixed): ...{fio_part[:50]}")

        # Разделяем по двойным <<
        parts = re.split(r'<{2,}', fio_part)
        parts = [p.replace('<', '').strip() for p in parts if p.replace('<', '').strip()]

        surname = parts[0] if len(parts) >= 1 else None

        # Имя и отчество через одиночный <
        name = None
        middle_name = None
        if len(parts) >= 2:
            name_middle = parts[1].split('<')
            name_middle = [x.strip() for x in name_middle if x.strip()]
            if len(name_middle) >= 1:
                name = name_middle[0]
            if len(name_middle) >= 2:
                middle_name = name_middle[1]

        if surname:
            return surname, name, middle_name

    return None, None, None


# ---------------------------------------------------------------------------
# Извлечение даты рождения, пола и гражданства из MRZ
# ---------------------------------------------------------------------------
def extract_meta_from_mrz(ocr_lines):
    """
    Ищем строку с паттерном: ...RUS YYMMDD [M|F] ...
    """
    result = {}
    for _, text, _ in ocr_lines:
        t = text.upper().replace(' ', '')
        t = t.replace('{', '<').replace('[', '<').replace('(', '<')
        # Заменяем кириллические О, S и т.д. которые OCR может путать
        t = t.replace('О', '0').replace('I', '1').replace('l', '1')

        # Формат: RUS + дата(6) + контрольная_цифра(1) + пол(M/F)
        m = re.search(r'RUS(\d{6})\d([MF<])', t)
        if m:
            bd = parse_mrz_date(m.group(1))
            if bd:
                result['birth_date'] = bd
                print(f"  📝 MRZ дата рождения: {bd}")

            result['citizenship'] = 'RU'

            g = m.group(2)
            if g == 'M':
                result['gender'] = 'M'
            elif g == 'F':
                result['gender'] = 'F'
            print(f"  📝 MRZ пол: {result.get('gender', '?')}")
            break

    return result


# ---------------------------------------------------------------------------
# Извлечение дат и пола из русского текста (fallback / дополнение)
# ---------------------------------------------------------------------------
def extract_from_russian_text(ocr_lines):
    """Ищем даты (DD.MM.YYYY) и пол (МУЖ/ЖЕН) в кириллическом тексте."""
    dates = []
    gender = None

    for _, text, _ in ocr_lines:
        # Даты
        for m in re.finditer(r'(\d{2}[.\-/]\d{2}[.\-/]\d{4})', text):
            d = parse_date_dmy(m.group(1).replace('-', '.').replace('/', '.'))
            if d:
                dates.append(d)

        # Пол — точное и нечёткое сравнение
        # OCR часто путает: МУЖ -> НУХ, НУЖ, МУЭ и т.п.
        if gender is None:
            low = text.lower().strip().rstrip('.')
            # Точное
            if re.search(r'м\s*у\s*ж', low):
                gender = 'M'
            elif re.search(r'ж\s*е\s*н', low):
                gender = 'F'
            # Нечёткое: строка из 3 букв, похожая на МУЖ/ЖЕН
            elif re.match(r'^[а-яё]{3}$', low):
                # Сравниваем по количеству совпадающих букв с МУЖ
                muж_score = sum(1 for a, b in zip(low, 'муж') if a == b)
                жен_score = sum(1 for a, b in zip(low, 'жен') if a == b)
                if muж_score >= 2:
                    gender = 'M'
                elif жен_score >= 2:
                    gender = 'F'

    # Сортируем даты: меньшая = дата рождения, большая = дата выдачи
    dates.sort()
    birth_date = dates[0] if len(dates) >= 1 else None
    issue_date = dates[1] if len(dates) >= 2 else None

    # Если только одна дата — определяем по году
    if birth_date and not issue_date:
        if birth_date.year >= 2010:
            # Слишком молодо для даты рождения, значит дата выдачи
            issue_date = birth_date
            birth_date = None

    return birth_date, issue_date, gender


# ---------------------------------------------------------------------------
# Извлечение ФИО из русского текста + транслитерация (fallback)
# ---------------------------------------------------------------------------
def extract_fio_russian(ocr_lines):
    """
    Ищем строки с кириллическим ФИО.
    На паспорте РФ порядок: Фамилия, Имя, Отчество.

    Имя / фамилия — это всегда ОДНО слово (или через дефис).
    Отчество может заканчиваться на -ВИЧ / -ВНА.
    Длина: от 3 до 15 букв.
    """
    excluded = {
        'РФ', 'МУЖ', 'ЖЕН', 'НУХ', 'НУЖ', 'ПОЛ', 'РОССИЙСКАЯ', 'ФЕДЕРАЦИЯ',
        'ПАСПОРТ', 'ВЫДАН', 'ОБЛАСТИ', 'ОБЛАСТЬ', 'МОСКВА', 'ГОР', 'ПОЛЯ',
    }

    names = []
    for _, text, _ in ocr_lines:
        t = text.strip().upper()

        # Убираем точки в конце (МУЖ. -> МУЖ)
        t = t.rstrip('.')

        # Строго одно слово кириллицы (допускаем дефис для составных имён)
        if not re.match(r'^[А-ЯЁ]+(\-[А-ЯЁ]+)?$', t):
            continue

        if len(t) < 3 or len(t) > 15:
            continue

        if t in excluded:
            continue

        names.append(t)

    print(f"  📝 Найдено слов-кандидатов: {names}")

    surname_ru = names[0] if len(names) >= 1 else None
    name_ru = names[1] if len(names) >= 2 else None
    middle_name_ru = names[2] if len(names) >= 3 else None

    # Транслитерация
    surname_lat = transliterate(surname_ru) if surname_ru else None
    name_lat = transliterate(name_ru) if name_ru else None
    middle_name_lat = transliterate(middle_name_ru) if middle_name_ru else None

    return (surname_lat, name_lat, middle_name_lat,
            surname_ru, name_ru, middle_name_ru)


# ---------------------------------------------------------------------------
# Основная сборка результатов
# ---------------------------------------------------------------------------
def build_result(series, number,
                 fio_latin, fio_russian,
                 birth_date, issue_date, citizenship, gender,
                 all_ocr_texts):
    """Собираем итоговый словарь с двумя уровнями."""

    required = {
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
    }

    all_detected = {
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
        'raw_text_lines': all_ocr_texts,
    }

    return {'required_data': required, 'all_detected_data': all_detected}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" УЛУЧШЕННЫЙ OCR — Распознавание паспорта РФ")
    print(" Источники: вертикальные цифры | MRZ | русский текст")
    print("=" * 70)
    print()

    print("⏳ Инициализация EasyOCR (ru + en)...")
    try:
        reader = easyocr.Reader(['ru', 'en'], gpu=False)
        print("✅ EasyOCR готов\n")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return

    while True:
        print("📁 Путь к изображению паспорта (или 'exit'):")
        image_path = input("> ").strip().strip('"').strip("'")

        if image_path.lower() == 'exit':
            print("\n👋 Выход")
            break

        if not image_path or not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}\n")
            continue

        print()
        print(f"🔍 Обработка: {os.path.basename(image_path)}")
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
            # ЭТАП 1: Препроцессинг трёх зон (работаем с cropped документом)
            # ----------------------------------------------------------
            print("\n[1/4] Препроцессинг...")
            full_path = preprocess_full(cropped_doc_path)
            strip_paths = preprocess_edge_strips(cropped_doc_path)
            mrz_paths = preprocess_mrz(cropped_doc_path)

            # ----------------------------------------------------------
            # ЭТАП 2: OCR зон
            # ----------------------------------------------------------
            print("\n[2/4] Распознавание текста...")

            lines_full = reader.readtext(full_path)
            print(f"  ✓ Основное изображение: {len(lines_full)} строк")

            # OCR каждого варианта полосы (4 края × 2 ротации)
            strip_ocr_variants = []
            for i, sp in enumerate(strip_paths):
                lines = reader.readtext(sp)
                strip_ocr_variants.append(lines)
                print(f"  ✓ Полоса вариант {i+1}: {len(lines)} строк")

            # MRZ: все кандидаты (bottom + top)
            lines_mrz = []
            for mp in mrz_paths:
                lines_mrz.extend(reader.readtext(mp))
            print(f"  ✓ MRZ зона: {len(lines_mrz)} строк")

            # Объединяем MRZ из двух источников
            all_mrz_lines = lines_mrz + lines_full

            # ----------------------------------------------------------
            # ЭТАП 3: Извлечение данных
            # ----------------------------------------------------------
            print("\n[3/4] Извлечение данных...")

            # Серия и номер: пробуем полосу, если не хватит цифр — fallback на полный текст
            print("  >> Серия и номер:")
            series, number = extract_series_number(strip_ocr_variants, lines_full)
            print(f"     Серия: {series}, Номер: {number}")

            # ФИО из MRZ
            print("  >> ФИО (MRZ):")
            surname_lat, name_lat, middle_name_lat = extract_fio_from_mrz(all_mrz_lines)
            print(f"     {surname_lat} {name_lat} {middle_name_lat}")

            # Дата рождения, пол, гражданство из MRZ
            print("  >> Метаданные (MRZ):")
            mrz_meta = extract_meta_from_mrz(all_mrz_lines)
            birth_date_mrz = mrz_meta.get('birth_date')
            citizenship = mrz_meta.get('citizenship')
            gender_mrz = mrz_meta.get('gender')

            # Даты и пол из русского текста
            print("  >> Даты и пол (русский текст):")
            birth_date_ru, issue_date_ru, gender_ru = extract_from_russian_text(lines_full)
            print(f"     Даты: {birth_date_ru} / {issue_date_ru},  Пол: {gender_ru}")

            # ФИО из русского текста + транслитерация (fallback)
            print("  >> ФИО (русский текст + транслитерация):")
            (surname_lat_fb, name_lat_fb, middle_name_lat_fb,
             surname_ru, name_ru, middle_name_ru) = extract_fio_russian(lines_full)
            print(f"     RU: {surname_ru} {name_ru} {middle_name_ru}")
            print(f"     Translit: {surname_lat_fb} {name_lat_fb} {middle_name_lat_fb}")

            # ----------------------------------------------------------
            # ЭТАП 4: Сборка итогов
            # ----------------------------------------------------------
            print("\n[4/4] Сборка результата...")

            # ФИО: MRZ приоритет, иначе транслитерация
            fio_latin = (
                surname_lat or surname_lat_fb,
                name_lat or name_lat_fb,
                middle_name_lat or middle_name_lat_fb,
            )
            fio_russian = (surname_ru, name_ru, middle_name_ru)

            # Даты: из русского текста (надёжнее DD.MM.YYYY)
            birth_date = birth_date_ru or birth_date_mrz
            issue_date = issue_date_ru

            # Валидация дат
            if birth_date and issue_date and birth_date > issue_date:
                birth_date, issue_date = issue_date, birth_date

            # Пол: русский текст приоритет (МУЖ/ЖЕН надёжнее чем M из MRZ)
            gender = gender_ru or gender_mrz

            # Гражданство: всегда из MRZ (или RU по умолчанию если паспорт РФ)
            if not citizenship:
                citizenship = 'RU'

            all_strip_lines = [line for variant in strip_ocr_variants for line in variant]
            all_texts = [t for _, t, _ in (lines_full + all_strip_lines + lines_mrz)]

            result = build_result(
                series, number,
                fio_latin, fio_russian,
                birth_date, issue_date, citizenship, gender,
                all_texts
            )

            # ----------------------------------------------------------
            # ВЫВОД
            # ----------------------------------------------------------
            req = result['required_data']

            print("\n" + "=" * 70)
            print("📋 ИТОГОВЫЕ ДАННЫЕ (required_data)")
            print("=" * 70)
            print(f"  👤 Фамилия:   {req['fio_latin']['surname']}")
            print(f"  👤 Имя:       {req['fio_latin']['name']}")
            print(f"  👤 Отчество:  {req['fio_latin']['middle_name']}")
            print(f"  🎂 Д. рожд.:  {req['birth_date']}")
            print(f"  📅 Д. выдачи: {req['issue_date']}")
            print(f"  🌍 Гражданство: {req['citizenship']}")
            print(f"  ⚥  Пол:       {req['gender']}")
            print(f"  📄 Серия:     {req['series']}")
            print(f"  📄 Номер:     {req['number']}")

            # Сохранение
            out_path = os.path.splitext(image_path)[0] + '_result.json'
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 JSON сохранён: {out_path}")
            print(f"💾 Промежуточные изображения: preprocessed/")

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