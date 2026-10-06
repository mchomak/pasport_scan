# 01 — Общая конфигурация OCR

**Требования:** R01, R02
**Blocked by:** нет
**Зона:** `docker-compose.yml`, `.env.example`, `ops/`
**Волна:** 1
**Status:** ready

## Что должно заработать

Telegram и MAX получают одинаковые OCR-приоритет и общий ключ OpenRouter. Новая переменная `BOT_OCR_MODULE_PRIORITY` управляет обоими; если она не задана, используется прежняя `MAX_OCR_MODULE_PRIORITY`, затем `openrouter,rupasportread`. Удаляется Telegram override, который отключал OpenRouter.

## Из брифа, дословно

> «сделай так чтобы и для telegram и для max были одинаковые настройки, то есть чтобы в telegram тоже работал openrouter»

## Разделы спецификации

«Решение», решения 1–2; истории R01, R01.1 и R02.

## Критерии приёмки

- [ ] Compose задаёт одинаковый итоговый `OCR_MODULE_PRIORITY` обоим сервисам с приоритетом OpenRouter над локальным OCR.
- [ ] Общий ключ из `.env` доступен обоим ботам; Compose больше не подменяет его пустым значением для Telegram.
- [ ] Существующий `MAX_OCR_MODULE_PRIORITY` продолжает работать как fallback; `.env.example` описывает новую общую настройку и совместимость.
- [ ] `ops/start_max_bot.sh` проверяет эффективную общую настройку и не принимает несовместимую конфигурацию незаметно.
- [ ] Синтаксис Compose проверяется через `docker compose config --quiet`, не выводя resolved secrets.

## Не менять

Код entrypoints, OCR merge/persistence, миграции и рабочий `.env`.
