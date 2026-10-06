window.STATE =
{
  "slug": "shared-ocr-diagnostics",
  "dir": "2026-10-06-shared-ocr-diagnostics",
  "title": "Одинаковый OpenRouter в Telegram и MAX",
  "mode": "full",
  "depth": "normal",
  "polish": null,
  "tier": "T2",
  "briefFile": "2026-10-06-brief.md",
  "memoryFile": "AGENTS.md",
  "skillDir": "C:/Users/McHomak/.agents/skills/autopilot",
  "startedAt": "2026-10-06T02:16:32+03:00",
  "updatedAt": "2026-10-06T11:09:30+03:00",
  "finishedAt": "2026-10-06T11:09:30+03:00",
  "stages": [
    {
      "id": "preflight",
      "status": "done",
      "startedAt": "2026-10-06T02:16:32+03:00",
      "finishedAt": "2026-10-06T02:18:36+03:00"
    },
    {
      "id": "manifest",
      "status": "done",
      "startedAt": "2026-10-06T02:18:36+03:00",
      "finishedAt": "2026-10-06T02:34:00+03:00"
    },
    {
      "id": "briefing",
      "status": "skipped",
      "note": "Режим full: требования уточнены по исходному brief."
    },
    {
      "id": "spec",
      "status": "done",
      "startedAt": "2026-10-06T02:34:00+03:00",
      "finishedAt": "2026-10-06T02:36:00+03:00"
    },
    {
      "id": "plan",
      "status": "done",
      "startedAt": "2026-10-06T02:36:00+03:00",
      "finishedAt": "2026-10-06T02:40:00+03:00"
    },
    {
      "id": "build",
      "status": "done",
      "startedAt": "2026-10-06T02:40:00+03:00",
      "finishedAt": "2026-10-06T09:53:06+03:00"
    },
    {
      "id": "review",
      "status": "done",
      "startedAt": "2026-10-06T02:57:50+03:00",
      "finishedAt": "2026-10-06T09:53:06+03:00"
    },
    {
      "id": "final",
      "status": "done",
      "startedAt": "2026-10-06T03:46:48+03:00",
      "finishedAt": "2026-10-06T11:09:30+03:00",
      "note": "Код отправлен в main и развёрнут; оба бота и общий OpenRouter проверены на VPS."
    }
  ],
  "requirements": {
    "total": 5,
    "done": 5,
    "inTicket": 0,
    "inSpec": 0,
    "placeholder": 0,
    "deferred": 0,
    "dropped": 0
  },
  "tickets": [
    {
      "id": "01",
      "title": "Общая OCR-конфигурация",
      "requirements": [
        "R01",
        "R02"
      ],
      "blockedBy": [],
      "wave": 1,
      "zone": [
        "docker-compose.yml",
        ".env.example",
        "ops/"
      ],
      "status": "done",
      "retries": 0,
      "repairs": 0,
      "handoffs": 0,
      "startedAt": "2026-10-06T02:51:04+03:00",
      "finishedAt": "2026-10-06T03:14:54+03:00",
      "commit": "af8df94",
      "tests": "Not run by instruction; docker compose config --quiet passed."
    },
    {
      "id": "02",
      "title": "Диагностические метаданные",
      "requirements": [
        "R04"
      ],
      "blockedBy": [],
      "wave": 1,
      "zone": [
        "ocr/",
        "services/",
        "db/"
      ],
      "status": "done",
      "retries": 0,
      "repairs": 0,
      "handoffs": 0,
      "startedAt": "2026-10-06T02:51:04+03:00",
      "finishedAt": "2026-10-06T03:21:12+03:00",
      "commit": "d782444",
      "tests": "Not run by instruction; python -m py_compile passed."
    },
    {
      "id": "03",
      "title": "Логи запуска Telegram",
      "requirements": [
        "R03"
      ],
      "blockedBy": [],
      "wave": 1,
      "zone": [
        "utils/logger.py",
        "main.py"
      ],
      "status": "done",
      "retries": 0,
      "repairs": 0,
      "handoffs": 0,
      "startedAt": "2026-10-06T02:51:04+03:00",
      "finishedAt": "2026-10-06T03:08:37+03:00",
      "commit": "4d0943f",
      "tests": "Not run by instruction; python -m py_compile passed."
    },
    {
      "id": "04",
      "title": "Логи запуска MAX",
      "requirements": [
        "R03",
        "R05"
      ],
      "blockedBy": [
        "03"
      ],
      "wave": 2,
      "zone": [
        "bot/max_main.py",
        "release/"
      ],
      "status": "done",
      "retries": 0,
      "repairs": 1,
      "handoffs": 0,
      "startedAt": "2026-10-06T03:08:37+03:00",
      "finishedAt": "2026-10-06T03:46:48+03:00",
      "tests": "Not run by instruction; python -m py_compile passed.",
      "commit": "3ee2cf8"
    },
    {
      "id": "05",
      "title": "Источник modules_used",
      "requirements": [
        "R04"
      ],
      "blockedBy": [
        "02"
      ],
      "wave": 3,
      "zone": [
        "services/passport_processing.py"
      ],
      "status": "done",
      "retries": 0,
      "repairs": 0,
      "handoffs": 0,
      "startedAt": "2026-10-06T04:00:00+03:00",
      "tests": "Not run by instruction; python -m py_compile and git diff --check passed.",
      "finishedAt": "2026-10-06T09:53:06+03:00",
      "commit": "b6592d3"
    }
  ],
  "singlePass": null,
  "tests": null,
  "debt": {
    "placeholders": [],
    "assumptions": [
      "Общий OCR-приоритет — openrouter,rupasportread; старый приоритет MAX принимается как fallback.",
      "Диагностические метаданные сохраняются в существующем JSONB raw_payload; сырые OCR/API-ответы не пишутся."
    ],
    "emptyEnv": []
  },
  "additions": [],
  "coverage": {
    "found": 1,
    "fixed": 1,
    "deferred": 0,
    "summary": "G2 пройден. G4 нашёл несовпадение источника modules_used; оно исправлено и проверено."
  },
  "concerns": [],
  "reviewers": {
    "manifestSpec": "/root/review_manifest_spec2",
    "craft": "/root/review_craft",
    "task05": "/root/ticket05_review"
  },
  "blind": {
    "status": "partial",
    "summary": "Независимая проверка подтвердила кодовые требования и push. Развёртывание не проверялось blind-checker по условиям задачи; его отдельно подтвердил оркестратор по VPS.",
    "requirements": {
      "realized": 4,
      "partial": 1,
      "missing": 0
    },
    "findings": [
      "Общая OpenRouter-first конфигурация, безопасные messenger-логи и обе диагностические метки реализованы.",
      "Blind-checker не подключался к VPS. Отдельная проверка оркестратора подтвердила HEAD b6592d3, работающие контейнеры Telegram и MAX, одинаковый OCR-приоритет, настроенный ключ OpenRouter, безопасные стартовые логи и healthy PostgreSQL."
    ]
  }
}
