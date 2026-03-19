# Passport OCR Bot

Telegram-бот для распознавания паспортов с гибридным OCR-пайплайном (OpenRouter LLM + Yandex OCR + Tesseract MRZ) и сохранением результатов в PostgreSQL.

## Возможности

- Гибридное распознавание паспортов из фотографий (JPEG, PNG) и PDF
- Три OCR-модуля с настраиваемым приоритетом: OpenRouter LLM, Yandex Cloud OCR, Tesseract MRZ
- Автоматическое слияние результатов — каждый следующий модуль дополняет пропущенные поля
- Определение пола по отчеству/фамилии
- Определение страны по месту рождения
- Настраиваемые форматы вывода через шаблоны
- Rate limiting для OpenRouter API (RPM)
- Экспорт данных в CSV и Excel (для администраторов)
- Два профиля Docker: light (без Tesseract) и full
- Исходники монтируются как volume — изменения применяются без пересборки образа

## Структура проекта

```
pasport_scan/
├── bot/                  # Telegram bot handlers, keyboards
├── db/                   # SQLAlchemy models, repository, database init
├── ocr/                  # OCR модули (hybrid, openrouter, yandex, provider)
├── services/             # Обработка изображений, PDF, экспорт
├── utils/                # Логгер, форматирование, транслитерация, MRZ, rate limiter
├── alembic/              # Миграции базы данных
├── config.py             # Конфигурация (pydantic-settings)
├── main.py               # Точка входа
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Быстрый старт

### 1. PostgreSQL

Бот использует PostgreSQL. Создайте базу данных:

```bash
createdb pasport_scans
```

### 2. Настройка

```bash
cp .env.example .env
```

Заполните параметры в `.env` (подробное описание — ниже в разделе «Параметры конфигурации»).

### 3. Запуск (Docker)

**Light** (OpenRouter + Yandex OCR, маленький образ):

```bash
docker compose --profile light up -d
```

**Full** (+ Tesseract MRZ, образ больше на ~65 MB):

```bash
docker compose --profile full up -d
```

```bash
# Логи
docker compose --profile light logs -f

# Остановка
docker compose --profile light down

# Пересборка (нужна только при изменении зависимостей или Dockerfile)
docker compose --profile light up -d --build
```

> Исходный код монтируется через volume, поэтому при изменении `.py`-файлов достаточно перезапуска контейнера без `--build`.

### 4. Запуск без Docker

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Для full-варианта с Tesseract:
# pip install -r requirements-full.txt
# + apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus

alembic upgrade head
python main.py
```

---

## Параметры конфигурации (.env)

### Telegram Bot

| Параметр | Обязательный | По умолчанию | Описание | Где получить |
|----------|:---:|---|---|---|
| `BOT_TOKEN` | да | — | Токен Telegram-бота | Создать бота через [@BotFather](https://t.me/BotFather), команда `/newbot` |
| `ADMIN_IDS` | да | — | ID администраторов через запятую | Узнать свой ID: отправить любое сообщение боту [@userinfobot](https://t.me/userinfobot) |

### База данных

| Параметр | Обязательный | По умолчанию | Описание | Где получить |
|----------|:---:|---|---|---|
| `DATABASE_URL` | да | — | Строка подключения PostgreSQL | Формат: `postgresql+asyncpg://user:password@host:5432/dbname`. Для Docker используйте `host.docker.internal` вместо `localhost` |

### Yandex Cloud OCR

Для работы Yandex OCR нужен `YC_FOLDER_ID` и **один** из трёх вариантов авторизации.

| Параметр | Обязательный | По умолчанию | Описание | Где получить |
|----------|:---:|---|---|---|
| `YC_FOLDER_ID` | да* | `""` | ID каталога в Yandex Cloud | [Консоль YC](https://console.yandex.cloud/) → выберите каталог → ID в верхней части страницы |
| `YC_API_KEY` | — | `""` | API-ключ сервисного аккаунта (**рекомендуется**, не истекает) | `yc iam api-key create --service-account-name <имя>` или [Консоль YC](https://console.yandex.cloud/) → Сервисные аккаунты → Создать API-ключ |
| `YC_OAUTH_TOKEN` | — | `""` | OAuth-токен (IAM обновляется автоматически каждые 12 ч) | Перейти по [ссылке](https://oauth.yandex.ru/authorize?response_type=token&client_id=1a6990aa636648e9b2ef855fa7bec2fb), скопировать токен |
| `YC_IAM_TOKEN` | — | `""` | IAM-токен вручную (истекает через 12 ч) | `yc iam create-token` |
| `YC_OCR_ENDPOINT` | нет | `https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText` | Endpoint Yandex OCR API | Менять не нужно, если не используется приватный endpoint |
| `OCR_DOCUMENT_MODEL` | нет | `passport` | Модель распознавания документа | Оставить `passport` |
| `OCR_LANGUAGE_CODES` | нет | `*` | Языки распознавания | `*` — автоопределение. Можно указать `ru,en` |

> \* Обязателен, если Yandex OCR включён в `OCR_MODULE_PRIORITY`.

### OpenRouter (Vision LLM)

| Параметр | Обязательный | По умолчанию | Описание | Где получить |
|----------|:---:|---|---|---|
| `OPENROUTER_API_KEY` | да* | `""` | API-ключ OpenRouter | Зарегистрироваться на [openrouter.ai](https://openrouter.ai/), раздел Keys → Create Key |
| `OPENROUTER_MODEL` | нет | `google/gemini-flash-1.5` | Vision-модель для распознавания | [Список моделей OpenRouter](https://openrouter.ai/models). Рекомендуются: `google/gemini-flash-1.5`, `anthropic/claude-sonnet-4` |
| `OPENROUTER_RPM` | нет | `0` | Лимит запросов в минуту (0 = без лимита) | Зависит от вашего тарифа на OpenRouter. Бесплатный план — обычно 3-5 RPM |

> \* Обязателен, если `openrouter` включён в `OCR_MODULE_PRIORITY`.

### Приоритет OCR-модулей

| Параметр | Обязательный | По умолчанию | Описание |
|----------|:---:|---|---|
| `OCR_MODULE_PRIORITY` | нет | `openrouter,yandex_ocr` | Порядок вызова OCR-модулей через запятую. Первый модуль — наивысший приоритет. Если первый модуль распознал все поля, остальные пропускаются |

Доступные модули:
- `openrouter` — Vision LLM через OpenRouter (лучшее качество, требует `OPENROUTER_API_KEY`)
- `yandex_ocr` — Yandex Cloud Vision OCR (требует `YC_FOLDER_ID` + авторизацию)
- `rupasportread` — локальный Tesseract MRZ (только в `full`-профиле, работает без API)

### Форматы вывода

| Параметр | По умолчанию |
|----------|---|
| `FORMAT_TYPE1` | `{country}/{number}/{country}/{birth_date_long}/{gender}/{expiry_long}/{surname}/{name}` |
| `FORMAT_TYPE2` | `-{surname} {name} {birth_date_short}+{gender}/{country}/{doc_type} {number}/{expiry_short}` |

Доступные переменные в шаблонах:

| Переменная | Пример | Описание |
|---|---|---|
| `{country}` | `RUS` | Код страны (3 буквы) |
| `{number}` | `123456789` | Номер документа |
| `{surname}` | `IVANOV` | Фамилия (латиница) |
| `{name}` | `IVAN` | Имя (латиница) |
| `{gender}` | `M` / `F` | Пол |
| `{doc_type}` | `P` | Тип документа |
| `{birth_date_long}` | `01.01.1990` | Дата рождения (ДД.ММ.ГГГГ) |
| `{birth_date_short}` | `010190` | Дата рождения (ДДММГГ) |
| `{expiry_long}` | `01.01.2030` | Срок действия (ДД.ММ.ГГГГ) |
| `{expiry_short}` | `010130` | Срок действия (ДДММГГ) |

### Лимиты и обработка файлов

| Параметр | По умолчанию | Описание |
|----------|---|---|
| `OCR_MAX_FILE_MB` | `10` | Максимальный размер загружаемого файла (МБ) |
| `OCR_MAX_MEGAPIXELS` | `20` | Максимальное разрешение изображения (мегапиксели) |
| `OCR_RATE_LIMIT_RPS` | `1` | Общий rate limit OCR-запросов (запросов/сек) |
| `PDF_RENDER_DPI` | `200` | DPI при рендере страниц PDF в изображения |

### Хранение и логирование

| Параметр | По умолчанию | Описание |
|----------|---|---|
| `TMP_DIR` | `./tmp` | Директория для временных файлов |
| `STORE_SOURCE_FILES` | `false` | Сохранять исходные файлы после обработки |
| `LOG_LEVEL` | `INFO` | Уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Использование

1. Откройте бота в Telegram и отправьте `/start`
2. Отправьте фото паспорта или PDF-документ
3. Бот вернёт распознанные данные в настроенном формате

Администраторы (`ADMIN_IDS`) могут выгрузить все записи командой `/export`.

## Алгоритм распознавания

1. **Нормализация** — EXIF-ротация, конвертация в RGB, ресайз
2. **Гибридный пайплайн** — модули запускаются по приоритету; если все ключевые поля заполнены первым модулем, остальные пропускаются
3. **Слияние** — результаты объединяются; для имён выбирается вариант с лучшим quality score
4. **Постобработка** — транслитерация, определение пола, очистка MRZ-артефактов
5. **Сохранение** — запись в PostgreSQL с raw payload для аудита

## Миграции

```bash
alembic upgrade head                              # применить все
alembic downgrade -1                              # откатить последнюю
alembic revision --autogenerate -m "description"  # создать новую
```

## Лицензия

MIT
