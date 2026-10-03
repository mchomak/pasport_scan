# Autopilot workspace

- `dashboard.html` is the current run dashboard snapshot.
- Dated run folders preserve each brief, manifest, specification, and interface notes.
- A folder ending in `--wip` is an active run; completed runs use the bare name.

| Started | Run | Status | Outcome |
|---|---|---|---|
| 2026-09-23 | `2026-09-23-max-messenger-support` | complete | Shared OCR/backend and PostgreSQL support Telegram and MAX adapters; forwarded and ordinary MAX photos, source-aware migration, Compose services, and tests were added. |
| 2026-09-25 | `2026-09-25-numpy-opencv-ocr` | сдано | Закреплён NumPy `<2` для OpenCV 4.9; добавлена проверка OCR-импортов при сборке; перезапущен только бот MAX. |
| 2026-09-25 | `2026-09-25-passport-ocr-recovery` | сдано с открытой runtime-проверкой | Код закоммичен и развернут; бот и суточные метрики ожидают ручного заполнения секретов пользователем. |
