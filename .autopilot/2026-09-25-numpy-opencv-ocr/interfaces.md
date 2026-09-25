# Build interfaces and repository rules

## Resolved boundaries

| Unit | Owns | Exposes | Hides |
|---|---|---|---|
| `requirements.txt` | Shared Python dependency constraints for every image variant | NumPy's supported version range | The exact package version selected by pip inside that range |
| Full-image build in `Dockerfile` | OpenCV/Tesseract installation and import-time validation of the OCR stack | A full image that builds only if NumPy, OpenCV, and `utils.rupasportread` import successfully | OS and pip installation details |

There is no cross-module public API change. The regression seam is the built `full` image: `python -c "import numpy, cv2, utils.rupasportread"`.
The Dockerfile supplies non-secret dummy `ADMIN_IDS` and `DATABASE_URL` only to this build-time import because importing `utils` eagerly validates settings; no database connection is opened.

## Project rules the implementer cannot infer

- Python 3.11, Docker Compose, and `requirements.txt` are the existing stack and dependency source of truth.
- `.env` selects `BOT_VARIANT=full`; never read out, print, modify, or commit `.env` values.
- `opencv-python-headless==4.9.0.80` and `utils/rupasportread.py` remain unchanged; fix resolver compatibility at the NumPy constraint.
- Preserve the pending Dockerfile change from the Debian Trixie repair (`libgl1`, `libglib2.0-0t64`). Do not restore the old package names or rewrite unrelated Docker setup.
- Only the MAX service is in scope for deployment. Do not recreate or start `telegram_bot`, PostgreSQL, or `migration`. No schema change is part of this task.
- Keep OCR/passport data and secrets out of logs and artifacts. Do not use a real passport photo for a regression test.
- If a required test/runtime dependency is absent in the local environment, report it as `BLOCKED`; do not install it to the host.

## Run and verify

```text
python -m unittest discover -s tests -v
docker compose config --quiet
docker compose build max_bot
docker compose up -d --no-deps max_bot
docker compose ps
docker compose logs --tail=100 max_bot
```

The deployment command is intentionally `--no-deps`: the shared PostgreSQL and successful migration already exist, and this change only adjusts image dependencies. Before redeploy, a successful full-image build and the import smoke check are required.
