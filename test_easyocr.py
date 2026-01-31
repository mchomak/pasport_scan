"""
Тестовый скрипт для распознавания паспорта через EasyOCR
Документация: https://github.com/JaidedAI/EasyOCR
"""

import os
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
            return datetime.strptime(date_str, fmt).date().isoformat()
        except ValueError:
            continue
    return date_str


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

    # Пытаемся найти ключевые поля
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

    # Простой поиск по ключевым словам
    for i, item in enumerate(all_text):
        text_lower = item['text'].lower()

        # Поиск серии и номера (например: 4619 709685 или 4619709685)
        if len(item['text'].replace(' ', '').replace('-', '')) == 10 and item['text'][0].isdigit():
            passport_data['passport_number'] = item['text'].replace(' ', '').replace('-', '')

        # Поиск кода подразделения (формат: 123-456)
        if '-' in item['text'] and len(item['text'].replace('-', '')) == 6:
            if item['text'].replace('-', '').isdigit():
                passport_data['subdivision_code'] = item['text']

        # Поиск даты рождения и даты выдачи
        if any(char.isdigit() for char in item['text']) and '.' in item['text']:
            parsed = parse_date(item['text'])
            if parsed:
                # Первая найденная дата - вероятно дата рождения
                if not passport_data['birth_date']:
                    passport_data['birth_date'] = parsed
                elif not passport_data['issue_date']:
                    passport_data['issue_date'] = parsed

        # Поиск пола
        if 'муж' in text_lower or 'male' in text_lower:
            passport_data['gender'] = 'муж'
        elif 'жен' in text_lower or 'female' in text_lower:
            passport_data['gender'] = 'жен'

    # Собираем весь распознанный текст для отображения
    passport_data['all_text'] = [item['text'] for item in all_text]
    passport_data['confidence_avg'] = sum(item['confidence'] for item in all_text) / len(all_text) if all_text else 0

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
