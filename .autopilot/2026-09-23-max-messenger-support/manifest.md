# Манифест требований

Coverage note: после G2 независимой проверки все R01–R53 покрыты `spec.md`; точные разделы сгруппированы в разделе `## Покрытие требований` спецификации, а дополнительные process invariants — в `## G2 coverage resolution`.

Ticket trace (authoritative for the `Где` column below): R04–R10, R18–R24, R33, R36–R38, R41–R43, R49 → ticket 01; R01, R07, R17, R32–R33, R39 → ticket 02; R02–R03, R07, R11–R17, R34–R38, R43–R45, R52 → ticket 03; R25–R31, R46–R48, R53 → ticket 04; R40–R45, R50–R51 → ticket 05. This trace supersedes the provisional `spec pending` text in individual rows.

| D01 | Blind acceptance proved that the MAX SDK session carries Authorization into arbitrary image URLs; forwarded events can expose the current user through `event.from_user` while `message.sender` is absent; OCR-empty output must be treated as failure | discovered | required by security and functional acceptance | tickets 06–08 |

Источник: `2026-09-23-brief.md`. Статусы будут уточнены в спецификации и при приёмке.

## Самобрифинг full mode

- `ASSUMPTION` — MAX запускается через long polling `maxapi`, потому что текущий Telegram уже работает так, а публичный webhook URL и TLS-конфигурация в репозитории не заданы.
- `ASSUMPTION` — для MAX используется тот же образ и код, а web-интерфейс остаётся включён только в Telegram-процессе, чтобы сохранить текущий порт и не создавать конфликт слушателей.
- `ASSUMPTION` — существующие Telegram-колонки сохраняются для обратной совместимости; новые внешние идентификаторы хранятся строками с полем `source`.
- `ASSUMPTION` — MAX поддерживает фотографии из обычного сообщения и из `link.message` пересланного сообщения; документы/PDF для MAX не расширяются сверх явно требуемого сценария, а получают понятный ответ о неподдерживаемом вложении.
- `ASSUMPTION` — Telegram сохраняет текущий HTML-ответ; MAX получает тот же результат в безопасном plain-text представлении, так как MAX-specific форматирование не должно менять данные результата.
- `ASSUMPTION` — HTTP-сессия MAX SDK переиспользуется для скачивания URL, с проверкой статуса, таймаутом и лимитом `OCR_MAX_FILE_MB`; изображение живёт только в памяти.
- `ASSUMPTION` — экспорт `/export` остаётся Telegram-only административной функцией, так как пользователь отдельно потребовал только основной OCR-сценарий MAX.

| ID | Из брифа (дословно) | Статус | Основание | Где |
|---|---|---|---|---|
| R01 | «Telegram-бот продолжил работать без изменений для пользователей» | in-spec | подтверждено self-briefing | spec pending |
| R02 | «MAX-бот работал параллельно и использовал ту же бизнес-логику» | in-spec | подтверждено self-briefing | spec pending |
| R03 | «Для MAX использовать Python-библиотеку `maxapi`» | in-spec | подтверждено self-briefing | spec pending |
| R04 | «НЕ создавать отдельную копию существующего проекта и НЕ дублировать бизнес-логику» | in-spec | подтверждено self-briefing | spec pending |
| R05 | «Сначала внимательно изучи весь репозиторий» | in-spec | выполнено до проектирования | spec pending |
| R06 | «После анализа самостоятельно выбери минимально инвазивную архитектуру» | in-spec | принято: общий service seam поверх текущих модулей | spec pending |
| R07 | «Telegram и MAX должны отвечать только за messenger-specific вещи» | in-spec | принято | spec pending |
| R08 | «Вся логика ... должна быть общей» | in-spec | принято | spec pending |
| R09 | «Не привязывай OCR-core непосредственно к объектам Telegram или MAX SDK» | in-spec | принято | spec pending |
| R10 | «привести входящие изображения к собственной общей структуре/интерфейсу» | in-spec | принято: IncomingImage | spec pending |
| R11 | «Пользователь отправляет фотографию паспорта» | in-spec | принято | spec pending |
| R12 | «пользователь может переслать в бота сообщение с фотографией паспорта из другого MAX-чата» | in-spec | принято: link.message | spec pending |
| R13 | «из обычного сообщения с image attachment» | in-spec | подтверждено контрактом maxapi | spec pending |
| R14 | «из пересланного сообщения с изображением» | in-spec | подтверждено контрактом maxapi | spec pending |
| R15 | «Изображение передаётся в существующую OCR/business logic» | in-spec | принято | spec pending |
| R16 | «Результат сохраняется в общую PostgreSQL» | in-spec | принято | spec pending |
| R17 | «Бот отправляет пользователю такой же или максимально эквивалентный результат» | in-spec | принято с platform presentation | spec pending |
| R18 | «Добавь понятие источника сообщения» | in-spec | принято: MessengerSource | spec pending |
| R19 | «В ... добавь поле: source / messenger / platform» | in-spec | выбрано `source` | spec pending |
| R20 | «Для существующих записей из Telegram должно быть безопасное значение: telegram» | in-spec | миграция server_default `telegram` | spec pending |
| R21 | «Если используется Alembic — создай нормальную миграцию» | in-spec | принято | spec pending |
| R22 | «Не удаляй существующие данные» | in-spec | принято | spec pending |
| R23 | «не считай их глобально уникальными между Telegram и MAX» | in-spec | принято, без новых лишних constraints | spec pending |
| R24 | «Если Telegram IDs сейчас INTEGER/BIGINT, проверь, подходит ли тип для MAX» | in-spec | MAX message ID string; external IDs будут String | spec pending |
| R25 | «Добавь как минимум: MAX_BOT_TOKEN=» | in-spec | принято | spec pending |
| R26 | «Обнови существующий Config / Settings / Pydantic Settings-класс» | in-spec | принято | spec pending |
| R27 | «Но не hardcode секреты» | in-spec | принято | spec pending |
| R28 | «Добавь `maxapi` в существующий механизм зависимостей» | in-spec | принято в requirements.txt | spec pending |
| R29 | «Оба сервиса: используют один Docker image; используют один код; получают `.env`; подключаются к одной PostgreSQL» | in-spec | принято через compose anchor/shared image | spec pending |
| R30 | «Один контейнер = один основной процесс» | in-spec | принято | spec pending |
| R31 | «Dockerfile должен остаться универсальным для обоих сервисов» | in-spec | принято | spec pending |
| R32 | «Предпочтительно иметь две небольшие точки входа» | in-spec | принято в `main.py` и `bot/max_main.py` | spec pending |
| R33 | «отдели данные результата от platform-specific presentation» | in-spec | принято через shared result/presentations | spec pending |
| R34 | «Обработай как минимум: сообщение без фотографии; неподдерживаемое вложение; несколько фотографий; ошибка скачивания изображения; timeout; слишком большой файл; ошибка OCR; OCR ничего не распознал; ошибка PostgreSQL; исключение API MAX» | in-spec | принято | spec pending |
| R35 | «Не допускай падения bot process из-за одного плохого сообщения» | in-spec | принято: adapter-level exception boundary | spec pending |
| R36 | «не логируй содержимое изображения; не логируй Base64; не логируй секретные токены; ... не добавляй паспортные данные в debug-логи» | in-spec | принято, включая исправление существующих чувствительных логов | spec pending |
| R37 | «установи разумный timeout; проверяй HTTP status; ограничивай размер скачиваемого файла; корректно закрывай клиент при shutdown» | in-spec | принято | spec pending |
| R38 | «Реализуй корректный startup/shutdown» | in-spec | принято через shared runtime | spec pending |
| R39 | «после изменений существующий Telegram-бот должен продолжать работать точно так же» | in-spec | принято как regression gate | spec pending |
| R40 | «Добавь минимальные полезные тесты» | in-spec | принято | spec pending |
| R41 | «общий OCR service не зависит от Telegram/MAX SDK» | in-spec | принято как тестируемый seam | spec pending |
| R42 | «Telegram source сохраняется как `telegram`» | in-spec | принято | spec pending |
| R43 | «MAX source сохраняется как `max`» | in-spec | принято | spec pending |
| R44 | «MAX normal image attachment корректно преобразуется в общий input» | in-spec | принято | spec pending |
| R45 | «forwarded MAX image корректно преобразуется в общий input» | in-spec | принято | spec pending |
| R46 | «Обнови README или существующую документацию запуска» | in-spec | принято | spec pending |
| R47 | «docker compose up -d --build» | in-spec | принято с новым default compose | spec pending |
| R48 | «docker compose logs -f telegram_bot» и «docker compose logs -f max_bot» | in-spec | принято с такими именами сервисов | spec pending |
| R49 | «один OCR/backend, несколько messenger adapters» | in-spec | принято | spec pending |
| R50 | «Запусти доступные tests/lint/type-check» | in-spec | принято | spec pending |
| R51 | «Проверь, что imports работают внутри Docker» | in-spec | принято | spec pending |
| R52 | «Проверь, что Telegram и MAX могут запускаться как два независимых процесса» | in-spec | принято | spec pending |
| R53 | «Проверь, что оба используют одну и ту же PostgreSQL» | in-spec | принято | spec pending |
