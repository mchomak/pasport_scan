# Passport OCR Bot

Telegram бот для распознавания паспортов с использованием гибридного OCR пайплайна (OpenRouter LLM + Yandex OCR + Tesseract MRZ) и сохранением результатов в PostgreSQL.

## Возможности

- Гибридное распознавание паспортов из фотографий (JPEG, PNG) и PDF
- Три OCR модуля с настраиваемым приоритетом: OpenRouter LLM, Yandex Cloud OCR, Tesseract MRZ
- Автоматическое слияние результатов — каждый следующий модуль дополняет пропущенные поля
- Определение пола по отчеству/фамилии
- Определение страны по месту рождения
- Настраиваемые форматы вывода через шаблоны
- Экспорт данных в CSV и Excel (для администраторов)
- Запуск одной командой через Docker Compose

## Структура проекта

```
pasport_scan/
├── bot/                  # Telegram bot handlers, keyboards
├── db/                   # SQLAlchemy models, repository, database init
├── ocr/                  # OCR модули (hybrid, openrouter, yandex, provider)
├── services/             # Обработка изображений, PDF, экспорт
├── utils/                # Логгер, форматирование, транслитерация, MRZ
├── alembic/              # Миграции базы данных
├── config.py             # Конфигурация (pydantic-settings)
├── main.py               # Точка входа
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Быстрый старт (Docker)

### 1. Настройка

```bash
cp .env.example .env
```

Заполните обязательные параметры в `.env`:

```env
BOT_TOKEN=your_telegram_bot_token
ADMIN_IDS=123456789

YC_FOLDER_ID=your_yandex_cloud_folder_id
YC_IAM_TOKEN=your_yandex_iam_token

# Рекомендуется для лучшего качества распознавания:
OPENROUTER_API_KEY=sk-or-your-key-here
```

### 2. Запуск

Проект доступен в двух вариантах:

**Light** (OpenRouter + Yandex OCR, маленький образ, меньше ОЗУ):

```bash
docker compose --profile light up -d
```

**Full** (+ Tesseract MRZ, образ больше на ~65MB):

```bash
docker compose --profile full up -d
```

```bash
# Логи
docker compose logs -f bot-light   # или bot-full

# Остановка
docker compose --profile light down   # или --profile full

# Пересборка после изменений
docker compose --profile light up -d --build
```

### 3. Локальная разработка (без Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Для full-варианта с Tesseract:
# pip install -r requirements-full.txt
# + установите tesseract-ocr: apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus

# Запустите PostgreSQL и задайте DATABASE_URL в .env
alembic upgrade head
python main.py
```

## Использование

1. Откройте бота в Telegram и отправьте `/start`
2. Отправьте фото паспорта или PDF документ
3. Бот вернет закодированные строки и развернутые детали по каждому модулю

Администраторы (`ADMIN_IDS`) могут выгрузить все записи командой `/export`.

## Конфигурация

### OCR модули

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `OCR_MODULE_PRIORITY` | Приоритет модулей (через запятую) | `openrouter,yandex_ocr` |
| `OPENROUTER_API_KEY` | Ключ OpenRouter API | — |
| `OPENROUTER_MODEL` | Модель для OpenRouter | `google/gemini-flash-1.5` |
| `YC_FOLDER_ID` | Folder ID Yandex Cloud | — |
| `YC_IAM_TOKEN` | IAM токен Yandex Cloud | — |

### Форматы вывода

| Параметр | По умолчанию |
|----------|--------------|
| `FORMAT_TYPE1` | `{country}/{number}/{country}/{birth_date_long}/{gender}/{expiry_long}/{surname}/{name}` |
| `FORMAT_TYPE2` | `-{surname} {name} {birth_date_short}+{gender}/{country}/{doc_type} {number}/{expiry_short}` |

Доступные переменные: `country`, `number`, `surname`, `name`, `gender`, `doc_type`, `birth_date_long`, `birth_date_short`, `expiry_long`, `expiry_short`.

### Остальные параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `DATABASE_URL` | Строка подключения к PostgreSQL | — |
| `OCR_MAX_FILE_MB` | Макс. размер файла (MB) | `10` |
| `PDF_RENDER_DPI` | DPI для рендера PDF | `200` |
| `LOG_LEVEL` | Уровень логирования | `INFO` |

## Алгоритм распознавания

1. **Нормализация** — EXIF, RGB, ресайз
2. **Гибридный пайплайн** — модули запускаются по приоритету; если все ключевые поля заполнены первым модулем, остальные пропускаются
3. **Слияние** — результаты объединяются; для имен выбирается вариант с лучшим quality score
4. **Постобработка** — транслитерация, определение пола, очистка MRZ-артефактов
5. **Сохранение** — запись в PostgreSQL с raw payload для аудита

## Миграции

```bash
alembic upgrade head          # применить
alembic downgrade -1          # откатить
alembic revision --autogenerate -m "description"  # создать
```

## Лицензия

MIT