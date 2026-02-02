"""
Гибридный скрипт распознавания паспорта РФ: EasyOCR + Tesseract.

Обе библиотеки запускаются на одинаковых препроцессированных зонах.
Для каждого поля выбирается лучший результат; в JSON указан источник (source).

Зоны:
  full   — полное изображение (русский текст + fallback MRZ)
  strip  — правая вертикальная полоса (серия + номер, 2 поворота)
  mrz    — нижние 25% (MRZ зона)

Логика выбора (merge):
  series/number — движок с полными 10 цифрами; tie → EasyOCR
  fio_latin     — движок с больше заполненными полями; tie → EasyOCR
  birth_date    — оба нашли и совпали → "agreed"; иначе EasyOCR > Tesseract
  issue_date    — аналогично
  gender        — аналогично
  citizenship   — всегда RU (из MRZ или по умолчанию)

Установка:
  pip install -r requirements_hybrid.txt
  + Tesseract binary в PATH + rus.traineddata в tessdata/
"""

import os
import re
import cv2
import numpy as np
import json
from datetime import datetime

try:
    import easyocr
except ImportError:
    print("❌ easyocr не установлен: pip install easyocr")
    exit(1)

try:
    import pytesseract
except ImportError:
    print("❌ pytesseract не установлен: pip install pytesseract")
    exit(1)


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
# Препроцессинг (binary для EasyOCR, grayscale для Tesseract)
# ---------------------------------------------------------------------------
def _to_binary(img_bgr):
    """BGR -> gray -> denoise -> CLAHE -> Otsu (для EasyOCR)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _to_grayscale(img_bgr):
    """BGR -> gray -> denoise -> CLAHE. Без бинаризации (для Tesseract)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def _preprocess(img_bgr, binary=True):
    """binary=True -> Otsu (EasyOCR), False -> grayscale (Tesseract)."""
    return _to_binary(img_bgr) if binary else _to_grayscale(img_bgr)


def preprocess_full(image_path, output_dir='preprocessed', binary=True):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить: {image_path}")

    h, w = img.shape[:2]
    if w < 2000:
        scale = 2000 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        print(f"  ↗ upscale {w}x{h} → {img.shape[1]}x{img.shape[0]}")

    suffix = 'bin' if binary else 'gray'
    processed = _preprocess(img, binary)
    path = f"{output_dir}/{base}_full_{suffix}.jpg"
    cv2.imwrite(path, processed)
    print(f"  ✓ Полное изображение ({suffix})")
    return path


def preprocess_right_strip(image_path, output_dir='preprocessed', binary=True):
    """Правая полоса (~15%), два поворота (90°/270°)."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    strip = img[:, int(w * 0.85):, :]

    suffix = 'bin' if binary else 'gray'
    paths = []
    for name, rotated in [
        ('ccw', cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ('cw',  cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE)),
    ]:
        sh, sw = rotated.shape[:2]
        rotated = cv2.resize(rotated, (sw * 3, sh * 3), interpolation=cv2.INTER_CUBIC)
        processed = _preprocess(rotated, binary)
        path = f"{output_dir}/{base}_strip_{name}_{suffix}.jpg"
        cv2.imwrite(path, processed)
        paths.append(path)

    print(f"  ✓ Правая полоса: 2 варианта поворота ({suffix})")
    return paths


def preprocess_mrz(image_path, output_dir='preprocessed', binary=True):
    """Нижние ~25% изображения (MRZ)."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    crop = img[int(h * 0.75):, :]
    ch, cw = crop.shape[:2]
    crop = cv2.resize(crop, (cw * 3, ch * 3), interpolation=cv2.INTER_CUBIC)

    suffix = 'bin' if binary else 'gray'
    processed = _preprocess(crop, binary)
    path = f"{output_dir}/{base}_mrz_{suffix}.jpg"
    cv2.imwrite(path, processed)
    print(f"  ✓ MRZ зона ({suffix})")
    return path


# ---------------------------------------------------------------------------
# Tesseract конфигурации
# ---------------------------------------------------------------------------
CFG_DIGITS_LINE  = "--psm 7 -c tessedit_char_whitelist=0123456789"
CFG_DIGITS_BLOCK = "--psm 11 -c tessedit_char_whitelist=0123456789"
CFG_MRZ          = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
CFG_RUSSIAN      = "--psm 6"


# ---------------------------------------------------------------------------
# OCR runners
# Нормализуют результаты обоих движков к общему формату:
#   russian_lines : [str, ...]  — текст из полного изображения
#   digits_str    : str         — все цифры из правой полосы
#   mrz_lines     : [str, ...]  — строки MRZ (+ fallback из full)
# ---------------------------------------------------------------------------
def run_easyocr(reader, full_path, strip_paths, mrz_path):
    """EasyOCR на все три зоны."""
    lines_full = reader.readtext(full_path)
    russian_lines = [text for _, text, _ in lines_full]

    # Полоса — берём вариант с максимумом цифр
    digits_str = ''
    for sp in strip_paths:
        lines = reader.readtext(sp)
        variant_digits = ''
        for _, text, _ in lines:
            variant_digits += re.sub(r'[^\d]', '', text)
        if len(variant_digits) > len(digits_str):
            digits_str = variant_digits

    # MRZ + fallback из full
    lines_mrz = reader.readtext(mrz_path) if mrz_path else []
    mrz_lines = [text for _, text, _ in lines_mrz] + russian_lines

    return russian_lines, digits_str, mrz_lines


def run_tesseract(full_path, strip_paths, mrz_path):
    """Tesseract на все три зоны."""
    # Полное изображение — русский
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    raw = pytesseract.image_to_string(img, lang='rus', config=CFG_RUSSIAN)
    russian_lines = [l.strip() for l in raw.split('\n') if l.strip()]

    # Полоса — цифры (2 PSM × 2 поворота, берём лучший)
    digits_str = ''
    for sp in strip_paths:
        img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        for cfg in (CFG_DIGITS_LINE, CFG_DIGITS_BLOCK):
            raw = pytesseract.image_to_string(img, lang='eng', config=cfg)
            digits = re.sub(r'[^\d]', '', raw)
            if len(digits) > len(digits_str):
                digits_str = digits

    # MRZ + fallback из full (english run)
    mrz_lines = []
    if mrz_path:
        img = cv2.imread(mrz_path, cv2.IMREAD_GRAYSCALE)
        raw = pytesseract.image_to_string(img, lang='eng', config=CFG_MRZ)
        mrz_lines += [l.strip() for l in raw.split('\n') if l.strip()]

    img_full = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    raw_full = pytesseract.image_to_string(img_full, lang='eng', config=CFG_MRZ)
    mrz_lines += [l.strip() for l in raw_full.split('\n') if l.strip()]

    return russian_lines, digits_str, mrz_lines


# ---------------------------------------------------------------------------
# Извлечение полей (общие функции — работают на plain text lists)
# ---------------------------------------------------------------------------
def _fix_mrz_fio_chars(s):
    """OCR-замены в буквенной части MRZ: 4->S, 3->Z, 1->I, 0->O, QQ-><<."""
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
    """Серия(4)+номер(6). Fallback на русский текст если полоса дала мало цифр."""
    if len(digits_str) < 10:
        fallback = ''
        for line in russian_lines:
            stripped = line.strip()
            if stripped.isdigit() and 1 <= len(stripped) <= 4:
                fallback += stripped
        if len(fallback) > len(digits_str):
            digits_str = fallback

    digits_str = digits_str[:10]
    series = digits_str[:4] if len(digits_str) >= 4 else None
    number = digits_str[4:10] if len(digits_str) >= 10 else None
    return series, number


def extract_fio_from_mrz(mrz_lines):
    """ФИО из MRZ (латиница). Ищем строку с PNRUS."""
    for text in mrz_lines:
        t = text.upper().replace(' ', '')
        t = t.replace('{', '<').replace('[', '<').replace('(', '<')

        match = re.search(r'P[N<]RUS', t)
        if not match:
            continue

        fio_part = t[match.end():]
        fio_part = _fix_mrz_fio_chars(fio_part)

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
    for text in mrz_lines:
        t = text.upper().replace(' ', '')
        t = t.replace('{', '<').replace('[', '<').replace('(', '<')
        t = t.replace('О', '0').replace('I', '1').replace('l', '1')

        # RUS + дата(6) + контрольная(1) + пол
        m = re.search(r'RUS(\d{6})\d([MF<])', t)
        if m:
            bd = parse_mrz_date(m.group(1))
            if bd:
                result['birth_date'] = bd
            result['citizenship'] = 'RU'
            g = m.group(2)
            if g in ('M', 'F'):
                result['gender'] = g
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
    """ФИО из русского текста + транслитерация (fallback)."""
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

    su, na, mi = (names[i] if len(names) > i else None for i in range(3))
    return (transliterate(su), transliterate(na), transliterate(mi), su, na, mi)


def extract_all(russian_lines, digits_str, mrz_lines, label=''):
    """
    Полный цикл извлечения на нормализованных данных одного движка.
    Возвращает словарь со всеми полями.
    """
    series, number = extract_series_number(digits_str, russian_lines)
    surname_lat, name_lat, middle_name_lat = extract_fio_from_mrz(mrz_lines)

    mrz_meta = extract_meta_from_mrz(mrz_lines)
    birth_date_mrz = mrz_meta.get('birth_date')
    citizenship = mrz_meta.get('citizenship')
    gender_mrz = mrz_meta.get('gender')

    birth_date_ru, issue_date_ru, gender_ru = extract_from_russian(russian_lines)
    (su_fb, na_fb, mi_fb, su_ru, na_ru, mi_ru) = extract_fio_russian(russian_lines)

    # Приоритеты (MRZ > translit для FIO, русский текст > MRZ для дат/пола)
    fio_latin = (surname_lat or su_fb, name_lat or na_fb, middle_name_lat or mi_fb)
    fio_russian = (su_ru, na_ru, mi_ru)

    birth_date = birth_date_ru or birth_date_mrz
    issue_date = issue_date_ru
    if birth_date and issue_date and birth_date > issue_date:
        birth_date, issue_date = issue_date, birth_date

    gender = gender_ru or gender_mrz
    if not citizenship:
        citizenship = 'RU'

    if label:
        print(f"  [{label:<10}] series={series} number={number} "
              f"fio={fio_latin} bd={birth_date} id={issue_date} gender={gender}")

    return {
        'fio_latin':     {'surname': fio_latin[0], 'name': fio_latin[1], 'middle_name': fio_latin[2]},
        'fio_russian':   {'surname': fio_russian[0], 'name': fio_russian[1], 'middle_name': fio_russian[2]},
        'birth_date':    birth_date.isoformat() if birth_date else None,
        'issue_date':    issue_date.isoformat() if issue_date else None,
        'citizenship':   citizenship,
        'gender':        gender,
        'series':        series,
        'number':        number,
        'raw_russian_lines': russian_lines,
        'raw_mrz_lines':     mrz_lines,
        'raw_digits':        digits_str,
    }


# ---------------------------------------------------------------------------
# Merge — выбор лучшего результата per field
# ---------------------------------------------------------------------------
def _fio_score(fio_dict):
    """Количество заполненных полей в FIO."""
    return sum(1 for v in fio_dict.values() if v is not None)


def _pick_date(easy_val, tess_val):
    """Выбор даты + аннотация источника."""
    if easy_val and tess_val:
        src = 'easyocr (совпадение)' if easy_val == tess_val else 'easyocr'
        return easy_val, src
    if easy_val:
        return easy_val, 'easyocr'
    if tess_val:
        return tess_val, 'tesseract'
    return None, None


def merge(easy, tess):
    """
    Объединяет результаты двух движков.
    Для каждого поля — лучший результат с аннотацией источника.
    """
    merged = {}

    # --- series + number ---
    easy_full = easy['series'] is not None and easy['number'] is not None
    tess_full = tess['series'] is not None and tess['number'] is not None

    if easy_full and tess_full:
        merged['series'], merged['number'] = easy['series'], easy['number']
        merged['series_number_source'] = 'easyocr (оба нашли)'
    elif easy_full:
        merged['series'], merged['number'] = easy['series'], easy['number']
        merged['series_number_source'] = 'easyocr'
    elif tess_full:
        merged['series'], merged['number'] = tess['series'], tess['number']
        merged['series_number_source'] = 'tesseract'
    else:
        # Ни у кого нет полных 10 цифр — берём больше
        easy_len = len((easy['series'] or '') + (easy['number'] or ''))
        tess_len = len((tess['series'] or '') + (tess['number'] or ''))
        if easy_len >= tess_len:
            merged['series'], merged['number'] = easy['series'], easy['number']
            merged['series_number_source'] = 'easyocr (частично)'
        else:
            merged['series'], merged['number'] = tess['series'], tess['number']
            merged['series_number_source'] = 'tesseract (частично)'

    # --- fio_latin ---
    e_score = _fio_score(easy['fio_latin'])
    t_score = _fio_score(tess['fio_latin'])
    if e_score >= t_score:
        merged['fio_latin'] = easy['fio_latin']
        merged['fio_latin_source'] = 'easyocr'
    else:
        merged['fio_latin'] = tess['fio_latin']
        merged['fio_latin_source'] = 'tesseract'

    # --- dates ---
    merged['birth_date'], merged['birth_date_source'] = _pick_date(
        easy['birth_date'], tess['birth_date'])
    merged['issue_date'], merged['issue_date_source'] = _pick_date(
        easy['issue_date'], tess['issue_date'])

    # --- gender ---
    if easy['gender'] and tess['gender']:
        merged['gender'] = easy['gender']
        merged['gender_source'] = 'easyocr (совпадение)' if easy['gender'] == tess['gender'] else 'easyocr'
    elif easy['gender']:
        merged['gender'] = easy['gender']
        merged['gender_source'] = 'easyocr'
    elif tess['gender']:
        merged['gender'] = tess['gender']
        merged['gender_source'] = 'tesseract'
    else:
        merged['gender'] = None
        merged['gender_source'] = None

    # --- citizenship ---
    merged['citizenship'] = easy['citizenship'] or tess['citizenship'] or 'RU'
    merged['citizenship_source'] = 'mrz/default'

    return merged


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _v(x):
    """Форматирование значения для вывода."""
    return str(x) if x is not None else 'не найдено'


def main():
    print("=" * 70)
    print(" HYBRID OCR — EasyOCR + Tesseract")
    print(" Распознавание паспорта РФ с голосованием")
    print("=" * 70)
    print()

    # Проверка Tesseract
    try:
        ver = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract v{ver}")
    except Exception:
        print("❌ Tesseract не найден. Установите tesseract-ocr и добавьте в PATH")
        return

    # Инициализация EasyOCR
    print("⏳ Инициализация EasyOCR (ru + en)...")
    try:
        reader = easyocr.Reader(['ru', 'en'], gpu=False)
        print("✅ EasyOCR готов\n")
    except Exception as e:
        print(f"❌ Ошибка EasyOCR: {e}")
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
            # ЭТАП 1: Препроцессинг (binary для EasyOCR, grayscale для Tesseract)
            # ----------------------------------------------------------
            print("\n[1/5] Препроцессинг...")
            easy_full_path   = preprocess_full(image_path, binary=True)
            easy_strip_paths = preprocess_right_strip(image_path, binary=True)
            easy_mrz_path    = preprocess_mrz(image_path, binary=True)
            tess_full_path   = preprocess_full(image_path, binary=False)
            tess_strip_paths = preprocess_right_strip(image_path, binary=False)
            tess_mrz_path    = preprocess_mrz(image_path, binary=False)

            # ----------------------------------------------------------
            # ЭТАП 2: EasyOCR (на бинаризованных изображениях)
            # ----------------------------------------------------------
            print("\n[2/5] EasyOCR...")
            easy_russian, easy_digits, easy_mrz = run_easyocr(
                reader, easy_full_path, easy_strip_paths, easy_mrz_path)
            print(f"  ✓ Русский текст: {len(easy_russian)} строк")
            print(f"  ✓ Цифры полосы:  '{easy_digits}'")
            print(f"  ✓ MRZ:           {len(easy_mrz)} строк")

            # ----------------------------------------------------------
            # ЭТАП 3: Tesseract (на серых изображениях)
            # ----------------------------------------------------------
            print("\n[3/5] Tesseract...")
            tess_russian, tess_digits, tess_mrz = run_tesseract(
                tess_full_path, tess_strip_paths, tess_mrz_path)
            print(f"  ✓ Русский текст: {len(tess_russian)} строк")
            print(f"  ✓ Цифры полосы:  '{tess_digits}'")
            print(f"  ✓ MRZ:           {len(tess_mrz)} строк")

            # ----------------------------------------------------------
            # ЭТАП 4: Извлечение (по одному разу на движок)
            # ----------------------------------------------------------
            print("\n[4/5] Извлечение данных...")
            easy_result = extract_all(easy_russian, easy_digits, easy_mrz, label='EasyOCR')
            tess_result = extract_all(tess_russian, tess_digits, tess_mrz, label='Tesseract')

            # ----------------------------------------------------------
            # ЭТАП 5: Merge + вывод
            # ----------------------------------------------------------
            print("\n[5/5] Объединение результатов (голосование)...")
            merged = merge(easy_result, tess_result)

            # Итоговый результат
            result = {
                'required_data': {
                    'fio_latin':    merged['fio_latin'],
                    'birth_date':   merged['birth_date'],
                    'issue_date':   merged['issue_date'],
                    'citizenship':  merged['citizenship'],
                    'gender':       merged['gender'],
                    'series':       merged['series'],
                    'number':       merged['number'],
                },
                'sources': {
                    'fio_latin':     merged['fio_latin_source'],
                    'birth_date':    merged['birth_date_source'],
                    'issue_date':    merged['issue_date_source'],
                    'citizenship':   merged['citizenship_source'],
                    'gender':        merged['gender_source'],
                    'series_number': merged['series_number_source'],
                },
                'all_detected_data': {
                    'easyocr':   {k: v for k, v in easy_result.items()  if not k.startswith('raw_')},
                    'tesseract': {k: v for k, v in tess_result.items() if not k.startswith('raw_')},
                },
                'raw_data': {
                    'easyocr': {
                        'russian_lines': easy_result['raw_russian_lines'],
                        'mrz_lines':     easy_result['raw_mrz_lines'],
                        'digits':        easy_result['raw_digits'],
                    },
                    'tesseract': {
                        'russian_lines': tess_result['raw_russian_lines'],
                        'mrz_lines':     tess_result['raw_mrz_lines'],
                        'digits':        tess_result['raw_digits'],
                    },
                },
            }

            # Вывод
            req = result['required_data']
            src = result['sources']

            print("\n" + "=" * 70)
            print("📋 ИТОГОВЫЕ ДАННЫЕ (merged / required_data)")
            print("=" * 70)
            print(f"  👤 Фамилия:     {_v(req['fio_latin']['surname']):<20} ← {src['fio_latin']}")
            print(f"  👤 Имя:         {_v(req['fio_latin']['name']):<20} ← {src['fio_latin']}")
            print(f"  👤 Отчество:    {_v(req['fio_latin']['middle_name']):<20} ← {src['fio_latin']}")
            print(f"  🎂 Д. рожд.:    {_v(req['birth_date']):<20} ← {src['birth_date'] or '—'}")
            print(f"  📅 Д. выдачи:   {_v(req['issue_date']):<20} ← {src['issue_date'] or '—'}")
            print(f"  🌍 Гражданство: {_v(req['citizenship']):<20} ← {src['citizenship']}")
            print(f"  ⚥  Пол:         {_v(req['gender']):<20} ← {src['gender'] or '—'}")
            print(f"  📄 Серия:       {_v(req['series']):<20} ← {src['series_number']}")
            print(f"  📄 Номер:       {_v(req['number']):<20} ← {src['series_number']}")

            # Сохранение
            out_path = os.path.splitext(image_path)[0] + '_hybrid_result.json'
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