
import os
import base64
import json
from pathlib import Path
from typing import Any, Dict
from openai import OpenAI, BadRequestError

OPENROUTER_API_KEY = "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"  # https://openrouter.ai/keys
if not OPENROUTER_API_KEY:
    raise RuntimeError("Set OPENROUTER_API_KEY env var")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Можно заменить на deepseek/deepseek-v3.2 (см. комментарий ниже),
# но для OCR/документов обычно лучше брать специализированную VL-модель.
MODEL_ID = "qwen/qwen2.5-vl-72b-instruct"


def encode_image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    ext = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{media_type};base64,{b64}"


def _extract_text_content(content: Any) -> str:
    """
    Унифицируем ответ: иногда content может прийти строкой,
    иногда списком частей (в зависимости от провайдера/обвязки).
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # OpenAI-compatible providers обычно используют {"type":"text","text":"..."}
                if item.get("type") == "text" and "text" in item:
                    parts.append(item["text"])
                # Иногда бывает {"text": "..."} без type
                elif "text" in item:
                    parts.append(item["text"])
        return "\n".join(parts).strip()

    return str(content).strip()


def _safe_json_loads(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()

    # Защита от ```json ... ```
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


def extract_passport_data(image_path: str) -> Dict[str, Any]:
    data_url = encode_image_to_data_url(image_path)

    system_prompt = (
        "Ты — система распознавания документов. "
        "Верни ТОЛЬКО валидный JSON без markdown и пояснений. "
        "Если поле не читается — ставь null."
    )

    user_prompt = """
Извлеки данные с паспорта РФ. Верни строго JSON-объект со схемой:

{
  "surname": "фамилия",
  "name": "имя",
  "patronymic": "отчество (если есть)",
  "birth_date": "ДД.ММ.ГГГГ",
  "gender": "М или Ж",
  "birth_place": "место рождения",
  "passport_series": "серия",
  "passport_number": "номер",
  "issue_date": "ДД.ММ.ГГГГ",
  "issued_by": "кем выдан",
  "department_code": "код подразделения",
  "mrz_line1": "первая строка MRZ (если видна)",
  "mrz_line2": "вторая строка MRZ (если видна)"
}

Правила:
- Не выдумывай значения.
- Если не видно/нечитаемо — null.
- Даты строго в формате ДД.ММ.ГГГГ.
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        # По рекомендации OpenRouter — сначала текст, потом картинка
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=1200,
            # Просим JSON-объект (если модель/провайдер поддерживает)
            response_format={"type": "json_object"},
        )
    except BadRequestError as e:
        # Очень полезно для дебага model-id / unsupported params
        raise RuntimeError(f"OpenRouter BadRequest: {e}") from e

    raw = _extract_text_content(response.choices[0].message.content)
    return _safe_json_loads(raw)


if __name__ == "__main__":
    image_path = r"C:\Users\McHomak\Projects\pasport_scan\data\1.png"
    print(f"Обрабатываем: {image_path}\n")

    result = extract_passport_data(image_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))