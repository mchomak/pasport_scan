window.STATE =
{
    "slug":  "passport-ocr-recovery",
    "dir":  "2026-09-25-passport-ocr-recovery",
    "title":  "Восстановление распознавания паспорта",
    "mode":  "full",
    "depth":  "normal",
    "polish":  null,
    "tier":  "T0",
    "briefFile":  "2026-09-25-brief.md",
    "memoryFile":  "AGENTS.md",
    "skillDir":  "C:/Users/McHomak/.agents/skills/autopilot",
    "startedAt":  "2026-09-25T23:03:21+03:00",
    "updatedAt":  "2026-10-03T13:56:35+03:00",
    "finishedAt":  "2026-10-03T13:56:35+03:00",
    "stages":  [
                   {
                       "id":  "preflight",
                       "status":  "done",
                       "startedAt":  "2026-09-25T23:03:21+03:00",
                       "finishedAt":  "2026-09-25T23:03:21+03:00"
                   },
                   {
                       "id":  "manifest",
                       "status":  "done",
                       "startedAt":  "2026-09-25T23:03:21+03:00",
                       "finishedAt":  "2026-09-25T23:42:55+03:00"
                   },
                   {
                       "id":  "briefing",
                       "status":  "done",
                       "startedAt":  "2026-09-25T23:42:55+03:00",
                       "finishedAt":  "2026-09-26T00:14:43+03:00",
                       "note":  "Пользователь выбрал OpenRouter/Qwen и разрешил тест с паспортными образцами"
                   },
                   {
                       "id":  "spec",
                       "status":  "done",
                       "startedAt":  "2026-09-26T00:14:43+03:00",
                       "finishedAt":  "2026-09-26T11:00:21+03:00"
                   },
                   {
                       "id":  "plan",
                       "status":  "skipped",
                       "note":  "ярус T0 — без разбивки на таски"
                   },
                   {
                       "id":  "build",
                       "status":  "done",
                       "startedAt":  "2026-10-03T12:51:07+03:00",
                       "note":  "Код и операционная подготовка развернуты; live запуск отложен до ручного заполнения секретов. Результат точности G03 остаётся проваленным: 27/30 применимых полей, 5/8 критических.",
                       "finishedAt":  "2026-10-03T13:56:35+03:00"
                   },
                   {
                       "id":  "review",
                       "status":  "done",
                       "startedAt":  "2026-09-26T11:02:42+03:00",
                       "finishedAt":  "2026-09-26T11:04:22+03:00",
                       "note":  "Compose-изоляция и regression suite проверены"
                   },
                   {
                       "id":  "final",
                       "status":  "done",
                       "startedAt":  "2026-10-03T13:56:35+03:00",
                       "finishedAt":  "2026-10-03T13:56:35+03:00",
                       "note":  "Закрытие с открытыми пунктами: ручной ввод секретов, live-запуск MAX/VPN-проверка и суточные метрики."
                   }
               ],
    "requirements":  {
                         "total":  13,
                         "done":  6,
                         "inTicket":  0,
                         "inSpec":  0,
                         "placeholder":  1,
                         "deferred":  0,
                         "dropped":  0,
                         "partial":  5,
                         "failed":  1
                     },
    "tickets":  [

                ],
    "singlePass":  null,
    "tests":  {
                  "passed":  26,
                  "failed":  0
              },
    "debt":  {
                 "placeholders":  [

                                  ],
                 "assumptions":  [
                                     "A01 — тест не сохраняется в PostgreSQL",
                                     "A02 — критерий точности перед перезапуском",
                                     "A03 — перезапуск только max_bot",
                                     "A04 — продолжить диагностику при остаточном сбое",
                                     "A05 — обезличенные логи/отчёт",
                                     "A06 — ключ и OCR-приоритет только MAX",
                                     "A07 — общий OCR/backend без копии"
                                 ],
                 "emptyEnv":  [
                                  "MAX_BOT_TOKEN",
                                  "OPENROUTER_API_KEY",
                                  "ADMIN_IDS",
                                  "POSTGRES_PASSWORD",
                                  "DATABASE_URL"
                              ]
             },
    "additions":  [
                      "A01 → R01: тест без записи в БД",
                      "A02 → G03: порог приемлемой точности",
                      "A03 → R03: перезапуск только MAX",
                      "A04 → R02: продолжить диагностику",
                      "A05 → G03: обезличенный отчёт",
                      "A06 → G02: провайдер изолирован для MAX",
                      "A07 → R02: использовать общий OCR/backend"
                  ],
    "coverage":  {
                     "found":  16,
                     "fixed":  15,
                     "deferred":  1,
                     "findings":  [
                                      "Qwen manual evaluation: 27/30 applicable fields; critical fields 5/8; G03 remains failed",
                                      "Local rupasportread returned no fields on 3/4 samples and surname only on 1/4",
                                      "MAX-only redeploy explicitly authorized by user and verified running; PostgreSQL unchanged; no OCR data was written",
                                      "Independent G2 check found four follow-on omissions: compare OpenRouter-only and hybrid capacity, attribute Tesseract CPU/RSS, define VPN/port criteria including client test, and give exact environment/start commands; spec and operational setup now cover them",
                                      "G4 по brief 2026-09-25 и 2026-10-03: независимая проверка кода сочла OCR sample-run и live runtime частично/не подтверждёнными; checker не открывал приватные образцы, .env или VPS.",
                                      "Предстартовая проверка VPS: 2 vCPU, около 1.9 GiB RAM; VPN-контейнеры и слушающие сокеты сохранились; внешний TCP 443 доступен. Сквозной VPN-клиентский тест не выполнялся.",
                                      "Серверный collector прошёл 3-секундный smoke-run; непрерывный 24-часовой сбор не запускался, Tesseract-процесс в smoke-run отсутствовал.",
                                      "На VPS развернут runtime-коммит e53c4a5; startup guard подтвердил, что пустые обязательные поля блокируют запуск до старта контейнеров."
                                  ]
                 },
    "concerns":  [

                 ],
    "reviewers":  {
                      "manifestSpec":  "01a0da99-59bd-7172-967f-b06e780f3bfc",
                      "craft":  null
                  },
    "blind":  {
                  "status":  "partial",
                  "summary":  "Независимый G4 охватил оба брифа. Кодовые/операционные механизмы видны, но checker не открывал паспортные образцы и .env и не подключался к VPS; поэтому запуск, точность, порты после запуска и 24-часовая нагрузка остались не подтверждены.",
                  "requirements":  {
                                       "realized":  0,
                                       "partial":  9,
                                       "missing":  3
                                   },
                  "findings":  [
                                   "R01 частично: обработчик изображений существует, но checker не запускал OCR на приватных образцах; отдельный прежний прогон записан в спецификации.",
                                   "R02 частично: MAX использует приоритет OpenRouter → rupasportread, однако фактический отказ на изображении проверкой не воспроизведён.",
                                   "R03 нет в слепой проверке: процесс бота checker не запускал.",
                                   "G01/G02/G04 частично: настройки и guard видны в коде; содержимое .env не проверялось.",
                                   "G03 нет в слепой проверке: тестовые документы не открывались, точность не проверялась.",
                                   "Задачи 2026-10-03 о VPS, VPN, запуске и метриках проверяемы только частично по коду; текущие внешние операции отдельно проверил оркестратор."
                               ]
              }
}
