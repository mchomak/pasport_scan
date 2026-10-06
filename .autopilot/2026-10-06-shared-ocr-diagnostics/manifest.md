# Manifest требований

Источник: `2026-10-06-brief.md`. Требование удаляется только по прямой просьбе пользователя.

| ID | Из brief (дословно) | Статус | Основание | Где |
|---|---|---|---|---|
| R01 | «сделай так чтобы и для telegram и для max были одинаковые настройки» | done (01) | ASSUMPTION — one shared openrouter,rupasportread priority preserves the current MAX mode and aligns Telegram. | Solution 1-2 / story 1 |
| R02 | «то есть чтобы в telegram тоже работал openrouter» | done (01) | ASSUMPTION — both containers use the OpenRouter key already supplied through the shared .env. | Solution 1-2 / story 3 |
| R03 | «Добавить безопасные логи инициализации в обоих ботах с подписью через какой мессенджер бот работает (откуда логи)» | done (03, 04) | — | Solution 6 / story 4 |
| R04 | «Сохранять в диагностических метаданных записи `modules_used` и `field_providers`» | done (02, 05) | Blind G4 found and task 05 corrected the source mismatch: persisted `modules_used` now comes from the pipeline's result list. | Solution 3-4 / story 5 |
| R05 | «скорректируй код, коммит и пуш в main затем задеплой на сервер» | done (04, VPS) | Code commits through `b6592d3` are on `origin/main`; both bot containers run that revision with the shared OpenRouter priority, PostgreSQL is healthy, and pre-existing server-side files were preserved. | Solution 5 / story 6 |
