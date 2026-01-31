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
    if w < 2000:
        scale = 2000 / w
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
    """Общий препроцессинг полосы: resize 3x -> gray -> denoise -> CLAHE -> Otsu."""
    sh, sw = strip_bgr.shape[:2]
    strip_bgr = cv2.resize(strip_bgr, (sw * 3, sh * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    path = f"{output_dir}/{name}.jpg"
    cv2.imwrite(path, binary)
    return path


def preprocess_right_strip(image_path, output_dir='preprocessed'):
    """
    Вырезаем правую полосу (~15 % ширины) и генерируем два варианта поворота
    (90° и 270°). Серия и номер на РФ паспорте печатаются вертикально справа.
    Возвращаем список путей — OCR будет запущен на каждый.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]

    # Вырезка: правые ~15 % ширины (шире чтобы точно захватить цифры)
    strip_x = int(w * 0.85)
    strip = img[:, strip_x:, :]

    # Два варианта поворота — берём тот, где больше цифр
    ccw = cv2.rotate(strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cw = cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE)

    path_ccw = _process_strip(ccw, output_dir, f"{base}_strip_ccw")
    path_cw = _process_strip(cw, output_dir, f"{base}_strip_cw")

    print(f"  ✓ Правая полоса: два варианта поворота сохранены")
    return [path_ccw, path_cw]


# ---------------------------------------------------------------------------
# Вырезка и обработка MRZ зоны (нижние ~25 % изображения)
# ---------------------------------------------------------------------------
def preprocess_mrz(image_path, output_dir='preprocessed'):
    """Специальная обработка нижней части с MRZ."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    crop = img[int(h * 0.75):, :]

    # x3 upscale
    ch, cw = crop.shape[:2]
    crop = cv2.resize(crop, (cw * 3, ch * 3), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    path = f"{output_dir}/{base}_mrz.jpg"
    cv2.imwrite(path, binary)
    print(f"  ✓ MRZ зона обработана")
    return path


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
            # ЭТАП 1: Препроцессинг трёх зон
            # ----------------------------------------------------------
            print("\n[1/4] Препроцессинг...")
            full_path = preprocess_full(image_path)
            strip_paths = preprocess_right_strip(image_path)  # список путей (2 варианта поворота)
            mrz_path = preprocess_mrz(image_path)

            # ----------------------------------------------------------
            # ЭТАП 2: OCR трёх зон
            # ----------------------------------------------------------
            print("\n[2/4] Распознавание текста...")

            lines_full = reader.readtext(full_path)
            print(f"  ✓ Основное изображение: {len(lines_full)} строк")

            # OCR каждого варианта полосы
            strip_ocr_variants = []
            for i, sp in enumerate(strip_paths):
                lines = reader.readtext(sp)
                strip_ocr_variants.append(lines)
                print(f"  ✓ Полоса вариант {i+1}: {len(lines)} строк")

            lines_mrz = reader.readtext(mrz_path) if mrz_path else []
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