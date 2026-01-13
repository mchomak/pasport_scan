# Passport OCR Bot

Telegram бот для распознавания паспортов РФ с использованием Yandex Cloud OCR и сохранением результатов в PostgreSQL.

## Возможности

- Распознавание паспортов РФ из фотографий (JPEG, PNG)
- Обработка многостраничных PDF документов
- Автоматическое определение ориентации паспорта (поворот на 90°, 270°)
- Сохранение всех распознанных данных в PostgreSQL
- Экспорт данных в CSV и Excel (только для администраторов)
- Асинхронная обработка с rate limiting для OCR API

## Архитектура

Проект построен на модульной архитектуре с возможностью замены OCR провайдера:

```
passport_ocr_bot/
├── app/
│   ├── bot/                 # Telegram bot handlers
│   ├── db/                  # Database models & repository
│   ├── ocr/                 # OCR providers (extensible)
│   ├── services/            # Business logic services
│   ├── utils/               # Utilities (logging, rate limiter)
│   ├── config.py            # Configuration management
│   └── main.py              # Application entry point
├── alembic/                 # Database migrations
├── docker-compose.yml       # Docker setup
├── Dockerfile               # Container definition
└── requirements.txt         # Python dependencies
```

## Технологический стек

- **Python 3.11+**
- **aiogram 3** - асинхронный Telegram bot framework
- **PostgreSQL** - база данных
- **SQLAlchemy 2.0** - async ORM
- **Alembic** - миграции базы данных
- **Yandex Cloud OCR** - распознавание текста
- **PyMuPDF** - обработка PDF
- **Pillow + OpenCV** - обработка изображений
- **Pandas + OpenPyXL** - экспорт данных
- **Pydantic Settings** - управление конфигурацией

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd passport_ocr_bot
```

### 2. Настройка окружения

Скопируйте `.env.example` в `.env` и заполните настройки:

```bash
cp .env.example .env
```

Обязательные параметры в `.env`:

```env
# Telegram Bot
BOT_TOKEN=your_telegram_bot_token
ADMIN_IDS=123456789,987654321

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/passports

# Yandex Cloud OCR
YC_FOLDER_ID=your_yandex_cloud_folder_id
YC_AUTH_MODE=api_key
YC_API_KEY=your_yandex_api_key
```

### 3. Получение Telegram Bot Token

1. Откройте [@BotFather](https://t.me/BotFather) в Telegram
2. Создайте нового бота командой `/newbot`
3. Скопируйте полученный токен в `BOT_TOKEN`

### 4. Настройка Yandex Cloud OCR

1. Создайте аккаунт в [Yandex Cloud](https://cloud.yandex.ru/)
2. Создайте folder и скопируйте его ID в `YC_FOLDER_ID`
3. Создайте API ключ:
   - Перейдите в IAM
   - Создайте сервисный аккаунт
   - Выдайте роль `ocr.user`
   - Создайте API ключ
   - Скопируйте ключ в `YC_API_KEY`

### 5. Запуск с Docker

```bash
# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f bot

# Остановка
docker-compose down
```

### 6. Локальная разработка (без Docker)

Установите зависимости:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

Запустите PostgreSQL локально или измените `DATABASE_URL` в `.env`.

Примените миграции:

```bash
alembic upgrade head
```

Запустите бота:

```bash
python -m app.main
```

## Использование

### Для пользователей

1. Откройте бота в Telegram
2. Отправьте команду `/start`
3. Отправьте фото паспорта или PDF документ
4. Получите распознанные данные

### Для администраторов

Администраторы (пользователи из `ADMIN_IDS`) имеют доступ к команде `/export`:

1. Отправьте `/export`
2. Выберите формат: CSV или Excel
3. Получите файл с выгрузкой всех данных

## Конфигурация

### Основные параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `OCR_PROVIDER_MODEL` | OCR провайдер | `yandex` |
| `OCR_MAX_FILE_MB` | Максимальный размер файла (MB) | `10` |
| `OCR_MAX_MEGAPIXELS` | Максимальный размер изображения (MP) | `20` |
| `OCR_RATE_LIMIT_RPS` | Ограничение запросов в секунду | `1` |
| `PDF_RENDER_DPI` | DPI для рендера PDF | `200` |
| `LOG_LEVEL` | Уровень логирования | `INFO` |

### Аутентификация Yandex Cloud

Поддерживается два режима:

**1. API Key (рекомендуется)**
```env
YC_AUTH_MODE=api_key
YC_API_KEY=your_api_key
```

**2. IAM Token**
```env
YC_AUTH_MODE=iam_token
YC_IAM_TOKEN=your_iam_token
YC_FOLDER_ID=your_folder_id
```

## База данных

### Структура таблицы `passport_records`

| Поле | Тип | Описание |
|------|-----|----------|
| `id` | UUID | Уникальный идентификатор |
| `created_at` | TIMESTAMP | Время создания |
| `tg_user_id` | BIGINT | Telegram user ID |
| `tg_username` | VARCHAR | Telegram username |
| `source_type` | VARCHAR | Тип источника (photo/pdf_page/image_document) |
| `passport_number` | VARCHAR | Серия и номер паспорта |
| `issued_by` | TEXT | Кем выдан |
| `issue_date` | DATE | Дата выдачи |
| `subdivision_code` | VARCHAR | Код подразделения |
| `surname` | VARCHAR | Фамилия |
| `name` | VARCHAR | Имя |
| `middle_name` | VARCHAR | Отчество |
| `gender` | VARCHAR | Пол |
| `birth_date` | DATE | Дата рождения |
| `birth_place` | TEXT | Место рождения |
| `raw_payload` | JSONB | Сырые данные OCR |
| `quality_score` | INTEGER | Количество заполненных полей |

### Миграции

Создание новой миграции:
```bash
alembic revision --autogenerate -m "description"
```

Применение миграций:
```bash
alembic upgrade head
```

Откат миграции:
```bash
alembic downgrade -1
```

## Алгоритм распознавания

1. **Нормализация изображения**
   - Исправление EXIF ориентации
   - Конвертация в RGB
   - Уменьшение до максимального размера

2. **OCR с эвристикой поворота**
   - Первая попытка: исходная ориентация
   - Если серия/номер не найдены: попытка с поворотом 90° и 270°
   - Выбор результата с максимальным количеством заполненных полей

3. **Извлечение данных**
   - Парсинг полей из OCR ответа
   - Конвертация типов данных
   - Подсчет quality score

4. **Сохранение в БД**
   - Сохранение всех полей
   - Сохранение raw payload для аудита

## Обработка ошибок

- Timeout для HTTP запросов: 30 секунд
- Автоматические ретраи: 2 попытки с экспоненциальным backoff
- Rate limiting с учетом лимитов Yandex OCR
- Graceful degradation: сохранение записи даже при частичном распознавании

## Разработка

### Добавление нового OCR провайдера

1. Создайте файл `app/ocr/new_provider.py`
2. Реализуйте класс, наследующий `OcrProvider`
3. Добавьте логику в `get_ocr_provider()` в `app/ocr/provider.py`

Пример:
```python
from app.ocr.provider import OcrProvider
from app.ocr.models import OcrResult

class NewOcrProvider(OcrProvider):
    async def recognize_passport(self, image_bytes: bytes, mime_type: str) -> OcrResult:
        # Ваша реализация
        pass
```

### Тестирование

```bash
# Установите dev зависимости
pip install pytest pytest-asyncio pytest-cov

# Запустите тесты
pytest
```

## Мониторинг и логи

Все логи структурированы и выводятся в stdout с использованием `structlog`:

```bash
# Docker logs
docker-compose logs -f bot

# Фильтрация по уровню
docker-compose logs -f bot | grep ERROR
```

## Безопасность

- API ключи хранятся в переменных окружения
- База данных изолирована в Docker network
- Rate limiting для защиты от злоупотребления
- Валидация всех входных данных

## Лицензия

MIT

## Поддержка

Если у вас возникли вопросы или проблемы, создайте issue в репозитории.
