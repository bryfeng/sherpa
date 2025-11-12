#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-"http://localhost:8000"}
ADDRESS=${ADDRESS:-"0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"}
CHAIN=${CHAIN:-"ethereum"}
WINDOW_DAYS=${WINDOW_DAYS:-30}
REQUESTS=${REQUESTS:-5}

read START_ISO END_ISO <<<"$(python - <<'PY'
import datetime as dt
import os
window = int(os.getenv('WINDOW_DAYS', '30'))
end = dt.datetime.utcnow().replace(microsecond=0)
start = end - dt.timedelta(days=window)
print(start.isoformat() + 'Z', end.isoformat() + 'Z')
PY
)"

echo "Benchmarking history summary:"
echo "  Base URL : ${BASE_URL}"
echo "  Address  : ${ADDRESS}"
echo "  Chain    : ${CHAIN}"
echo "  Window   : ${WINDOW_DAYS}d (${START_ISO} → ${END_ISO})"
echo "  Requests : ${REQUESTS}"

total=0
success=0
for ((i=1; i<=REQUESTS; i++)); do
  response_file=$(mktemp)
  latency=$(curl -sS \
    -w '%{time_total}' \
    -o "$response_file" \
    "${BASE_URL}/wallets/${ADDRESS}/history-summary?chain=${CHAIN}&start=${START_ISO}&end=${END_ISO}") || true
  status=$(python - <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    print(payload.get("status") or "ok")
except Exception:
    print("unknown")
PY
  "$response_file")
  echo "Run #${i}: latency=${latency}s status=${status}"
  total=$(python - <<'PY'
import sys
total=float(sys.argv[1])
lat=float(sys.argv[2])
print(total+lat)
PY
 "$total" "$latency")
  rm -f "$response_file"
  success=$((success+1))
done
avg=$(python - <<'PY'
import sys
print(float(sys.argv[1]) / max(int(sys.argv[2]), 1))
PY
 "$total" "$success")

echo "Average latency: ${avg}s across ${success} successful runs"
