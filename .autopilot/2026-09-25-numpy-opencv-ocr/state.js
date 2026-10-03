window.STATE =
{
  "slug": "numpy-opencv-ocr",
  "dir": "2026-09-25-numpy-opencv-ocr",
  "title": "Совместимость NumPy/OpenCV в OCR",
  "mode": "full",
  "depth": "normal",
  "tier": "T0",
  "polish": null,
  "briefFile": "2026-09-25-brief.md",
  "memoryFile": "AGENTS.md",
  "skillDir": "C:/Users/McHomak/.agents/skills/autopilot",
  "startedAt": "2026-09-25T21:18:07+03:00",
  "updatedAt": "2026-09-25T22:17:20+03:00",
  "finishedAt": "2026-09-25T22:17:20+03:00",
  "stages": [
    { "id": "preflight", "status": "done", "startedAt": "2026-09-25T21:18:07+03:00", "finishedAt": "2026-09-25T21:26:04+03:00" },
    { "id": "manifest", "status": "done", "startedAt": "2026-09-25T21:26:04+03:00", "finishedAt": "2026-09-25T21:28:10+03:00" },
    { "id": "briefing", "status": "skipped", "note": "Полный автомат — само-брифинг" },
    { "id": "spec", "status": "done", "startedAt": "2026-09-25T21:28:10+03:00", "finishedAt": "2026-09-25T21:42:47+03:00" },
    { "id": "plan", "status": "skipped", "note": "Ярус T0 — без разбиения на задачи" },
    { "id": "build", "status": "done", "startedAt": "2026-09-25T21:43:42+03:00", "finishedAt": "2026-09-25T21:49:20+03:00" },
    { "id": "review", "status": "done", "startedAt": "2026-09-25T21:49:20+03:00", "finishedAt": "2026-09-25T22:11:53+03:00", "note": "T0 inline code review; blind acceptance confirmed the requested fix and MAX runtime" },
    { "id": "final", "status": "done", "startedAt": "2026-09-25T22:11:53+03:00", "finishedAt": "2026-09-25T22:17:20+03:00" }
  ],
  "requirements": {
    "total": 3, "done": 3, "inTicket": 0, "inSpec": 0,
    "placeholder": 0, "deferred": 0, "dropped": 0
  },
  "tickets": [],
  "singlePass": {
    "startedAt": "2026-09-25T21:43:42+03:00",
    "finishedAt": "2026-09-25T21:49:20+03:00",
    "files": ["Dockerfile", "requirements.txt", "AGENTS.md"],
    "tests": { "passed": 23, "failed": 0 },
    "commit": "included in the final single-pass commit"
  },
  "tests": { "passed": 23, "failed": 0 },
  "debt": { "placeholders": [], "assumptions": [], "emptyEnv": [] },
  "additions": [],
  "coverage": { "found": 3, "fixed": 3, "deferred": 0 },
  "concerns": [],
  "reviewers": { "manifestSpec": "01a0d9df-23c7-79c1-95b6-88943856471b", "craft": null },
  "blind": {
    "status": "pass",
    "summary": "Independent checker found no unmet user outcome; NumPy/OpenCV fix and MAX-only current runtime verified. Checker did not run tests; orchestrator independently ran all 23 successfully.",
    "requirements": { "realized": 3, "partial": 0, "missing": 0 }
  }
}
