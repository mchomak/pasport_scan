"""
Распознавание паспорта через Yandex Vision OCR API
Документация: https://yandex.cloud/ru/docs/vision/concepts/ocr/template-recognition
"""

import base64
import requests
import json

# ============ НАСТРОЙКИ ============
IMAGE_PATH = "./data/1.png"  # Укажи путь к фото паспорта
IAM_TOKEN = ""          # Твой IAM токен
FOLDER_ID = "" 

# Endpoint для OCR API
OCR_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"


def encode_image_to_base64(image_path: str) -> str:
    """Кодирует изображение в base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(image_path: str) -> str:
    """Определяет MIME тип по расширению файла"""
    ext = image_path.lower().split(".")[-1]
    mime_types = {
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "png": "PNG",
        "pdf": "PDF"
    }
    return mime_types.get(ext, "JPEG")


def recognize_passport(image_path: str, iam_token: str, folder_id: str) -> dict:
    """
    Отправляет изображение паспорта на распознавание
    
    Args:
        image_path: путь к изображению
        iam_token: IAM токен для авторизации
        folder_id: ID каталога в Yandex Cloud
    
    Returns:
        dict с результатами распознавания
    """
    # Готовим тело запроса
    body = {
        "mimeType": get_mime_type(image_path),
        "languageCodes": ["*"],  # Автоопределение языка
        "model": "passport",     # Модель для паспортов
        "content": encode_image_to_base64(image_path)
    }
    
    # Заголовки с авторизацией и folder_id
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}",
        "x-folder-id": folder_id
    }
    
    # Отправляем запрос
    response = requests.post(OCR_URL, headers=headers, json=body)
    
    if response.status_code != 200:
        raise Exception(f"Ошибка API: {response.status_code}\n{response.text}")
    
    return response.json()


def parse_passport_data(response: dict) -> dict:
    """
    Извлекает данные паспорта из ответа API
    
    Args:
        response: ответ от API
    
    Returns:
        dict с полями паспорта
    """
    result = {}
    
    # Ищем entities в ответе
    if "result" in response:
        text_annotation = response["result"].get("textAnnotation", {})
        entities = text_annotation.get("entities", [])
    else:
        # Прямой ответ
        text_annotation = response.get("textAnnotation", {})
        entities = text_annotation.get("entities", [])
    
    # Названия полей на русском
    field_names = {
        "name": "Имя",
        "middle_name": "Отчество",
        "surname": "Фамилия",
        "gender": "Пол",
        "citizenship": "Гражданство",
        "birth_date": "Дата рождения",
        "birth_place": "Место рождения",
        "number": "Номер паспорта",
        "issued_by": "Кем выдан",
        "issue_date": "Дата выдачи",
        "subdivision": "Код подразделения",
        "expiration_date": "Срок действия"
    }
    
    for entity in entities:
        name = entity.get("name", "")
        text = entity.get("text", "")
        ru_name = field_names.get(name, name)
        result[ru_name] = text
    
    return result


def main():
    print("=" * 50)
    print("Yandex Vision OCR - Распознавание паспорта")
    print("=" * 50)
    
    try:
        print(f"\nЗагружаю изображение: {IMAGE_PATH}")
        
        # Распознаём паспорт
        response = recognize_passport(IMAGE_PATH, IAM_TOKEN, FOLDER_ID)
        
        # Парсим данные
        passport_data = parse_passport_data(response)
        
        if passport_data:
            print("\n✅ Распознанные данные паспорта:\n")
            for field, value in passport_data.items():
                print(f"  {field}: {value}")
        else:
            print("\n⚠️ Данные паспорта не найдены в ответе")
            print("\nПолный ответ API:")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            
    except FileNotFoundError:
        print(f"\n❌ Файл не найден: {IMAGE_PATH}")
        print("Укажите правильный путь к изображению паспорта")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()