# Построенные контракты

## OCR-конфигурация

- Оба Compose-сервиса получают `OCR_MODULE_PRIORITY` и OpenRouter-переменные из общего окружения.
- `BOT_OCR_MODULE_PRIORITY` задаёт общий приоритет; если переменная отсутствует, совместимый fallback читает `MAX_OCR_MODULE_PRIORITY`; последний fallback — `openrouter,rupasportread`.
- Приоритет локальной настройки остаётся доступен через `OCR_MODULE_PRIORITY` для прямого запуска.

## Логирование

- Entry points вызывают `setup_logger(log_level, messenger_source=...)`; structlog добавляет `messenger=telegram` или `messenger=max` к сообщениям процесса.
- Инициализационные записи показывают разрешённые имена OCR-модулей и boolean-статус OpenRouter. Значение ключа и содержимое документов не логируются.

## Диагностика OCR

- `HybridResult.modules_used` содержит модули, вернувшие данные для результата; `modules_attempted` отдельно хранит все вызванные модули.
- Для новой записи `services/passport_processing.py` передаёт в `raw_payload` только `modules_used: list[str]` и `field_providers: dict[str, str]`.
- `field_providers` связывает имя поля с модулем-источником либо `inferred`; сырой OCR-текст и ответы провайдера не сохраняются в диагностических метаданных.
- Используется существующая JSONB-колонка; миграция схемы не нужна.

## Runtime-границы

- Telegram запускается через `python main.py`, MAX — через `python -m bot.max_main`.
- Общая обработка использует `core/messaging.py`, `services/passport_processing.py` и `services/runtime.py`; shared layer не импортирует SDK или event types мессенджеров.
- Статические проверки этой задачи: `python -m py_compile` изменённых модулей и `docker compose config --quiet`; тесты не запускались.
