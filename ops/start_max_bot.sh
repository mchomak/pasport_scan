#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  echo "Файл .env не найден в /opt/passport-scan" >&2
  exit 1
fi

python3 - .env <<'PY'
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

values = {}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[7:].lstrip()
    key, sep, value = line.partition("=")
    if not sep:
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    values[key.strip()] = value

required = ("MAX_BOT_TOKEN", "OPENROUTER_API_KEY", "ADMIN_IDS", "POSTGRES_PASSWORD", "DATABASE_URL")
missing = [key for key in required if not values.get(key) or values[key].strip().lower() in {"change-me", "changeme", "your-token", "your-api-key"}]
if missing:
    print("Заполните в .env обязательные поля: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
if values["ADMIN_IDS"] == "123456789,987654321":
    print("Замените пример ADMIN_IDS на свои Telegram/ MAX ID.", file=sys.stderr)
    raise SystemExit(1)
if values.get("BOT_VARIANT") != "full":
    print("Для локального OCR требуется BOT_VARIANT=full.", file=sys.stderr)
    raise SystemExit(1)
ocr_priority = (
    values.get("BOT_OCR_MODULE_PRIORITY")
    or values.get("MAX_OCR_MODULE_PRIORITY")
    or "openrouter,rupasportread"
)
if ocr_priority != "openrouter,rupasportread":
    print(
        "Первый запуск измеряет гибридный режим: BOT_OCR_MODULE_PRIORITY=openrouter,rupasportread "
        "(или MAX_OCR_MODULE_PRIORITY при unset общей переменной).",
        file=sys.stderr,
    )
    raise SystemExit(1)
if values.get("MAX_BOT_ENABLED", "true").lower() != "true" or values.get("TELEGRAM_BOT_ENABLED", "false").lower() != "false":
    print("Ожидаются MAX_BOT_ENABLED=true и TELEGRAM_BOT_ENABLED=false.", file=sys.stderr)
    raise SystemExit(1)
if values.get("OPENROUTER_MODEL", "qwen/qwen2.5-vl-72b-instruct") != "qwen/qwen2.5-vl-72b-instruct":
    print("Проверьте OPENROUTER_MODEL в .env.", file=sys.stderr)
    raise SystemExit(1)

try:
    parsed = urlsplit(values["DATABASE_URL"])
    password_matches = unquote(parsed.password or "") == values["POSTGRES_PASSWORD"]
except ValueError:
    parsed = urlsplit("")
    password_matches = False
if (
    parsed.scheme != "postgresql+asyncpg"
    or parsed.hostname != "postgres"
    or not password_matches
    or unquote(parsed.username or "") != values.get("POSTGRES_USER", "postgres")
    or parsed.path.lstrip("/") != values.get("POSTGRES_DB", "passports")
):
    print("DATABASE_URL должен совпадать с POSTGRES_USER, POSTGRES_PASSWORD и POSTGRES_DB.", file=sys.stderr)
    raise SystemExit(1)
PY

docker compose version >/dev/null
docker compose config --quiet

METRICS_ROOT="$ROOT_DIR/metrics"
RUN_DIR="$METRICS_ROOT/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"

for name in amnezia-xray amnezia-awg2; do
  if ! docker inspect "$name" >/dev/null 2>&1 || [[ "$(docker inspect -f '{{.State.Running}}' "$name")" != "true" ]]; then
    echo "VPN-контейнер $name не активен; бот не запускаю." >&2
    exit 1
  fi
done

vpn_snapshot() {
  for name in amnezia-xray amnezia-awg2; do
    docker inspect -f '{{.Name}} {{.State.Running}} {{.RestartCount}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$name"
  done | sort -u
}

ss -H -lntu | awk '{print $1, $5}' | sort -u > "$RUN_DIR/listeners-before.txt"
vpn_snapshot > "$RUN_DIR/vpn-before.txt"

python3 - "$RUN_DIR/listeners-before.txt" <<'PY'
import sys
from pathlib import Path

listeners = [line.split() for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
present = {(proto, address.rsplit(":", 1)[-1]) for proto, address in listeners}
expected = {("tcp", "443"), ("udp", "443"), ("udp", "585")}
missing = sorted(expected - present)
if missing:
    print("До запуска не найдены VPN-порты: " + ", ".join(f"{proto}/{port}" for proto, port in missing), file=sys.stderr)
    raise SystemExit(1)
PY

echo "Собираю и запускаю только MAX-бота с его зависимостями..."
docker compose up -d --build max_bot

MAX_ID="$(docker compose ps -q max_bot)"
POSTGRES_ID="$(docker compose ps -q postgres)"
MIGRATION_ID="$(docker compose ps -a -q migration)"
if [[ -z "$MAX_ID" || -z "$POSTGRES_ID" || -z "$MIGRATION_ID" ]]; then
  echo "Не найдены ожидаемые контейнеры MAX/PostgreSQL/migration." >&2
  docker compose ps -a
  exit 1
fi

sleep 30
MAX_STATE="$(docker inspect -f '{{.State.Status}}' "$MAX_ID")"
MAX_RESTARTS="$(docker inspect -f '{{.RestartCount}}' "$MAX_ID")"
POSTGRES_STATE="$(docker inspect -f '{{.State.Status}}' "$POSTGRES_ID")"
POSTGRES_HEALTH="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$POSTGRES_ID")"
MIGRATION_RESULT="$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$MIGRATION_ID")"

if [[ "$MAX_STATE" != "running" || "$MAX_RESTARTS" != "0" || "$POSTGRES_STATE" != "running" || "$POSTGRES_HEALTH" != "healthy" || "$MIGRATION_RESULT" != "exited 0" ]]; then
  echo "Проверка запуска не пройдена: max=$MAX_STATE/restarts=$MAX_RESTARTS postgres=$POSTGRES_STATE/$POSTGRES_HEALTH migration=$MIGRATION_RESULT" >&2
  docker compose ps -a
  exit 1
fi

for name in amnezia-xray amnezia-awg2; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$name")" != "true" ]]; then
    echo "VPN-контейнер $name остановился после запуска бота." >&2
    exit 1
  fi
done

ss -H -lntu | awk '{print $1, $5}' | sort -u > "$RUN_DIR/listeners-after.txt"
if ! cmp -s "$RUN_DIR/listeners-before.txt" "$RUN_DIR/listeners-after.txt"; then
  echo "Список слушающих портов изменился; проверьте diff ниже." >&2
  diff -u "$RUN_DIR/listeners-before.txt" "$RUN_DIR/listeners-after.txt" || true
  exit 1
fi
if [[ -n "$(docker port "$MAX_ID")" ]]; then
  echo "У MAX-бота неожиданно опубликован host-порт." >&2
  docker port "$MAX_ID"
  exit 1
fi

vpn_snapshot > "$RUN_DIR/vpn-after.txt"
if ! cmp -s "$RUN_DIR/vpn-before.txt" "$RUN_DIR/vpn-after.txt"; then
  echo "Статусы VPN-контейнеров изменились; проверьте diff ниже." >&2
  diff -u "$RUN_DIR/vpn-before.txt" "$RUN_DIR/vpn-after.txt" || true
  exit 1
fi

chmod 700 ops/collect_server_metrics.py
nohup python3 -u ops/collect_server_metrics.py \
  --duration-seconds 86400 \
  --interval-seconds 10 \
  --mode openrouter,rupasportread \
  --output-dir "$RUN_DIR" \
  >> "$RUN_DIR/collector.log" 2>&1 < /dev/null &
COLLECTOR_PID=$!
echo "$COLLECTOR_PID" > "$RUN_DIR/collector.pid"
sleep 1
if ! kill -0 "$COLLECTOR_PID" 2>/dev/null; then
  echo "Не удалось запустить сборщик метрик; бот запущен, логи: $RUN_DIR" >&2
  exit 1
fi

echo "MAX-бот работает; PostgreSQL healthy; миграция завершена; VPN-порты совпадают с исходным снимком."
echo "Сбор метрик на 24 часа запущен: $RUN_DIR"
echo "Контейнеры: docker compose ps"
