"""
Тестовый скрипт для распознавания паспорта через EasyOCR
Документация: https://github.com/JaidedAI/EasyOCR
"""

import os
import re
import easyocr
from datetime import datetime
import json


def parse_date(date_str):
    """Парсинг даты из различных форматов."""
    if not date_str:
        return None

    date_str = date_str.strip()
    formats = ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def parse_mrz_date(mrz_date):
    """Парсинг даты из MRZ формата YYMMDD."""
    if not mrz_date or len(mrz_date) != 6:
        return None

    try:
        year = int(mrz_date[0:2])
        month = int(mrz_date[2:4])
        day = int(mrz_date[4:6])

        # Определяем век (если год > 50, то 19XX, иначе 20XX)
        if year > 50:
            year += 1900
        else:
            year += 2000

        return datetime(year, month, day).date()
    except (ValueError, IndexError):
        return None


def parse_mrz(all_text):
    """
    Парсинг машиночитаемой зоны (MRZ) российского паспорта.

    Формат MRZ:
    Строка 1: PNRUS + Фамилия<<Имя<<Отчество<<< (всего 44 символа)
    Строка 2: Номер(10) + RUS + ДатаРожд(6) + Пол(1) + <<< + ДопДанные
    """
    mrz_data = {}

    # Ищем строки с MRZ (содержат много символов '<' и латиницу)
    mrz_lines = []
    for item in all_text:
        text = item['text'].upper()
        # MRZ содержит много '<' и только латинские буквы и цифры
        if text.count('<') >= 3 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<' for c in text):
            mrz_lines.append(text)

    if len(mrz_lines) >= 2:
        # Сортируем по длине (более длинные строки скорее всего правильные)
        mrz_lines.sort(key=len, reverse=True)

        # Первая строка: ФИО
        line1 = mrz_lines[0].replace(' ', '')
        if line1.startswith('PN') or line1.startswith('P<'):
            # Убираем префикс PNRUS или P<RUS
            fio_part = line1[5:] if line1.startswith('PNRUS') else line1[5:]

            # Разделяем по двойным <<
            parts = fio_part.split('<<')
            if len(parts) >= 2:
                mrz_data['surname_latin'] = parts[0].replace('<', '').strip()
                name_parts = parts[1].replace('<', ' ').strip().split()
                if name_parts:
                    mrz_data['name_latin'] = name_parts[0]
                if len(name_parts) > 1:
                    mrz_data['middle_name_latin'] = name_parts[1]

        # Вторая строка: номер паспорта, дата рождения, пол
        line2 = mrz_lines[1].replace(' ', '')

        # Номер паспорта (первые 10 символов)
        if len(line2) >= 10 and line2[:10].replace('<', '').isdigit():
            passport_num = line2[:10].replace('<', '')
            if passport_num.isdigit() and len(passport_num) == 10:
                mrz_data['passport_number'] = passport_num

        # Дата рождения (после RUS, 6 цифр) и пол (M/F)
        # Формат: 4614000500RUS0506166M
        match = re.search(r'RUS(\d{6})([MF<])', line2)
        if match:
            birth_date = parse_mrz_date(match.group(1))
            if birth_date:
                mrz_data['birth_date'] = birth_date

            gender_code = match.group(2)
            if gender_code == 'M':
                mrz_data['gender'] = 'муж'
            elif gender_code == 'F':
                mrz_data['gender'] = 'жен'

        # Дополнительные данные (дата выдачи + код подразделения)
        # Формат: 9190627500065 (YYMMDD + код без дефиса)
        match_additional = re.search(r'<+(\d{13})', line2)
        if match_additional:
            additional = match_additional.group(1)
            # Первые 7 символов: 91 + дата YYMMDD
            if len(additional) >= 7 and additional[0] == '9' and additional[1] == '1':
                issue_date = parse_mrz_date(additional[1:7])
                if issue_date:
                    mrz_data['issue_date'] = issue_date

            # Последние 6 цифр - код подразделения
            if len(additional) >= 13:
                code = additional[7:13]
                if len(code) == 6 and code.isdigit():
                    mrz_data['subdivision_code'] = f"{code[:3]}-{code[3:]}"

    return mrz_data


def is_cyrillic(text):
    """Проверка, содержит ли текст кириллицу."""
    return bool(re.search('[а-яА-ЯёЁ]', text))


def is_likely_name(text):
    """Проверка, похож ли текст на имя/фамилию."""
    # Только кириллица и пробелы, длина от 2 до 30 символов
    if not text or len(text) < 2 or len(text) > 30:
        return False
    # Содержит только буквы, пробелы и дефисы
    return bool(re.match(r'^[А-ЯЁа-яё\s\-]+$', text)) and is_cyrillic(text)


def extract_passport_fields(ocr_results):
    """
    Извлекает поля паспорта из результатов OCR.

    EasyOCR возвращает массив: [bbox, text, confidence]
    """
    all_text = []

    # Собираем весь распознанный текст
    for item in ocr_results:
        bbox, text, confidence = item
        all_text.append({
            'text': text,
            'confidence': confidence,
            'bbox': bbox
        })

    # Сортируем по позиции (сверху вниз, слева направо)
    all_text.sort(key=lambda x: (x['bbox'][0][1], x['bbox'][0][0]))

    # Инициализация данных паспорта
    passport_data = {
        'surname': None,
        'name': None,
        'middle_name': None,
        'gender': None,
        'birth_date': None,
        'birth_place': None,
        'passport_number': None,
        'issue_date': None,
        'issued_by': None,
        'subdivision_code': None
    }

    # ШАГ 1: Парсинг MRZ (приоритетный источник данных)
    mrz_data = parse_mrz(all_text)
    if mrz_data:
        print(f"🔍 Найдена MRZ зона: {mrz_data}")
        # Используем данные из MRZ как основные
        passport_data.update({k: v for k, v in mrz_data.items() if k in passport_data})

    # ШАГ 2: Извлечение данных из русского текста
    # Фильтруем только кириллические строки (исключаем MRZ)
    russian_text = [item for item in all_text if is_cyrillic(item['text']) and '<<' not in item['text']]

    dates_found = []
    names_found = []

    for i, item in enumerate(russian_text):
        text = item['text'].strip()
        text_upper = text.upper()
        text_lower = text.lower()

        # Поиск серии и номера паспорта (10 цифр)
        if not passport_data['passport_number']:
            # Удаляем пробелы и дефисы
            digits_only = text.replace(' ', '').replace('-', '').replace('№', '')
            if len(digits_only) == 10 and digits_only.isdigit():
                passport_data['passport_number'] = digits_only

        # Поиск кода подразделения (формат: 123-456 или 123 456)
        if not passport_data['subdivision_code']:
            # Убираем лишние символы
            clean = text.replace(' ', '').replace('-', '')
            if len(clean) == 6 and clean.isdigit():
                # Проверяем, есть ли дефис в оригинале
                if '-' in text:
                    passport_data['subdivision_code'] = text.strip()
                else:
                    passport_data['subdivision_code'] = f"{clean[:3]}-{clean[3:]}"

        # Поиск дат (формат DD.MM.YYYY)
        if '.' in text and any(c.isdigit() for c in text):
            date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', text)
            if date_match:
                parsed_date = parse_date(date_match.group(1))
                if parsed_date:
                    dates_found.append(parsed_date)

        # Поиск пола
        if not passport_data['gender']:
            if 'муж' in text_lower:
                passport_data['gender'] = 'муж'
            elif 'жен' in text_lower:
                passport_data['gender'] = 'жен'

        # Поиск места рождения (содержит "ГОР." или "ОБЛ." или другие географические маркеры)
        if not passport_data['birth_place']:
            if any(marker in text_upper for marker in ['ГОР.', 'Г.', 'ОБЛ.', 'РЕСП.', 'КРАЙ']):
                passport_data['birth_place'] = text

        # Поиск кем выдан (содержит "МВД" или "УФМС" и т.д.)
        if not passport_data['issued_by']:
            if any(marker in text_upper for marker in ['МВД', 'УФМС', 'ГУ МВД', 'ОТДЕЛОМ']):
                passport_data['issued_by'] = text

        # Поиск ФИО (только большие кириллические буквы, длина 2-30 символов)
        if is_likely_name(text_upper) and len(text_upper) >= 2:
            # Исключаем короткие слова и не-имена
            if text_upper not in ['ПОЛ', 'МУЖ', 'ЖЕН', 'РФ', 'РОССИЙСКАЯ', 'ФЕДЕРАЦИЯ']:
                names_found.append(text_upper)

    # ШАГ 3: Определение дат с учетом логики (дата рождения < дата выдачи)
    if dates_found:
        # Сортируем даты по возрастанию
        dates_found.sort()

        if len(dates_found) >= 2:
            # Первая (самая ранняя) - дата рождения
            if not passport_data['birth_date']:
                passport_data['birth_date'] = dates_found[0]
            # Вторая (более поздняя) - дата выдачи
            if not passport_data['issue_date']:
                passport_data['issue_date'] = dates_found[1]
        elif len(dates_found) == 1:
            # Если только одна дата, нужно понять какая это
            # Если дата < 2000 года, скорее всего дата рождения
            if dates_found[0].year < 2000:
                if not passport_data['birth_date']:
                    passport_data['birth_date'] = dates_found[0]
            else:
                # Иначе предполагаем дата выдачи
                if not passport_data['issue_date']:
                    passport_data['issue_date'] = dates_found[0]

    # Валидация дат
    if passport_data['birth_date'] and passport_data['issue_date']:
        if passport_data['birth_date'] > passport_data['issue_date']:
            # Меняем местами если перепутали
            passport_data['birth_date'], passport_data['issue_date'] = \
                passport_data['issue_date'], passport_data['birth_date']

    # ШАГ 4: Определение ФИО из найденных имен
    # В российском паспорте обычно идут: Фамилия, Имя, Отчество (в порядке появления на странице)
    if names_found and len(names_found) >= 3:
        if not passport_data['surname']:
            passport_data['surname'] = names_found[0]
        if not passport_data['name']:
            passport_data['name'] = names_found[1]
        if not passport_data['middle_name']:
            passport_data['middle_name'] = names_found[2]

    # Преобразуем даты обратно в строки ISO
    if passport_data['birth_date']:
        passport_data['birth_date'] = passport_data['birth_date'].isoformat()
    if passport_data['issue_date']:
        passport_data['issue_date'] = passport_data['issue_date'].isoformat()

    # Собираем весь распознанный текст для отображения
    passport_data['all_text'] = [item['text'] for item in all_text]
    passport_data['confidence_avg'] = sum(item['confidence'] for item in all_text) / len(all_text) if all_text else 0

    # Добавляем данные MRZ для отладки
    passport_data['mrz_data'] = mrz_data

    return passport_data


def main():
    print("=" * 70)
    print("EasyOCR - Тест распознавания паспорта РФ")
    print("=" * 70)
    print()

    # Инициализация EasyOCR
    print("⏳ Инициализация EasyOCR (первый запуск может занять время)...")
    try:
        # Инициализация с русским и английским языками
        reader = easyocr.Reader(['ru', 'en'], gpu=False)
        print("✅ EasyOCR инициализирован успешно")
        print()
    except Exception as e:
        print(f"❌ Ошибка инициализации EasyOCR: {e}")
        print("\nПроверьте установку:")
        print("  pip install easyocr")
        return

    # Запрос пути к изображению
    while True:
        print("📁 Введите путь к изображению паспорта (или 'exit' для выхода):")
        image_path = input("> ").strip().strip('"').strip("'")

        if image_path.lower() == 'exit':
            print("\n👋 Выход из программы")
            break

        if not image_path:
            print("⚠️  Путь не может быть пустым\n")
            continue

        if not os.path.exists(image_path):
            print(f"❌ Файл не найден: {image_path}\n")
            continue

        print()
        print(f"🔍 Обработка: {os.path.basename(image_path)}")
        print("-" * 70)

        try:
            # Запуск распознавания
            print("⏳ Распознавание...")
            result = reader.readtext(image_path)

            if not result:
                print("⚠️  Текст не распознан")
                print()
                continue

            # Извлечение полей паспорта
            passport_data = extract_passport_fields(result)

            print("\n" + "=" * 70)
            print("📋 РАСПОЗНАННЫЕ ДАННЫЕ ПАСПОРТА")
            print("=" * 70)

            # Вывод структурированных данных
            print(f"\n📄 Серия и номер: {passport_data['passport_number'] or '❌ не найдено'}")
            print(f"👤 Фамилия: {passport_data['surname'] or '❌ не найдено'}")
            print(f"👤 Имя: {passport_data['name'] or '❌ не найдено'}")
            print(f"👤 Отчество: {passport_data['middle_name'] or '❌ не найдено'}")
            print(f"⚥  Пол: {passport_data['gender'] or '❌ не найдено'}")
            print(f"🎂 Дата рождения: {passport_data['birth_date'] or '❌ не найдено'}")
            print(f"🏙️  Место рождения: {passport_data['birth_place'] or '❌ не найдено'}")
            print(f"📅 Дата выдачи: {passport_data['issue_date'] or '❌ не найдено'}")
            print(f"🏛️  Кем выдан: {passport_data['issued_by'] or '❌ не найдено'}")
            print(f"🔢 Код подразделения: {passport_data['subdivision_code'] or '❌ не найдено'}")

            # Показываем данные из MRZ если найдены
            if passport_data.get('mrz_data'):
                print("\n" + "-" * 70)
                print("🔐 Данные из машиночитаемой зоны (MRZ):")
                mrz = passport_data['mrz_data']
                if mrz.get('surname_latin'):
                    print(f"  Фамилия (латиница): {mrz['surname_latin']}")
                if mrz.get('name_latin'):
                    print(f"  Имя (латиница): {mrz['name_latin']}")
                if mrz.get('middle_name_latin'):
                    print(f"  Отчество (латиница): {mrz['middle_name_latin']}")

            print(f"\n📊 Средняя уверенность: {passport_data['confidence_avg']:.2%}")

            # Вывод всего распознанного текста
            print("\n" + "=" * 70)
            print("📝 ВЕСЬ РАСПОЗНАННЫЙ ТЕКСТ")
            print("=" * 70)
            for idx, text in enumerate(passport_data['all_text'], 1):
                print(f"{idx:2d}. {text}")

            # Вывод сырых данных (для отладки)
            print("\n" + "=" * 70)
            print("🔧 СЫРЫЕ ДАННЫЕ (для отладки)")
            print("=" * 70)

            for i, item in enumerate(result):
                bbox, text, confidence = item
                print(f"{i+1:2d}. [{confidence:.2%}] {text}")
                print(f"    Координаты: {bbox[0]} -> {bbox[2]}")

            # Сохранение результата в JSON
            output_file = image_path.rsplit('.', 1)[0] + '_easyocr_result.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'passport_data': passport_data,
                    'raw_ocr_result': [[bbox, text, float(conf)] for bbox, text, conf in result]
                }, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n💾 Результаты сохранены в: {output_file}")

        except Exception as e:
            print(f"❌ Ошибка при обработке: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()