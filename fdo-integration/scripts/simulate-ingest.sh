#!/usr/bin/env bash
# Reference simulator for demo rehearsal: POST one score reading to the
# Flask backend as a given machine, using that machine's onboarded FDO GUID.
#
# threshold/passed are optional, matching what a real evaluator
# (Dynamic_Trust_Evaluation/test_model.py's 門檻/分數/是否通過 output) would
# send. Omit them to let the backend fall back to its own internal
# simulation (see demo_web/README.md).
#
# Usage: simulate-ingest.sh <machine_id> <score> [threshold] [passed(true|false)]
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./lib.sh

MACHINE_ID=${1:?usage: simulate-ingest.sh <machine_id> <score> [threshold] [passed(true|false)]}
SCORE=${2:?usage: simulate-ingest.sh <machine_id> <score> [threshold] [passed(true|false)]}
THRESHOLD=${3:-}
PASSED=${4:-}
BACKEND_URL=http://localhost:5181

GUID="$(python - "${GUID_MAP_FILE}" "${MACHINE_ID}" <<'PY'
import json
import sys
from pathlib import Path

path, machine_id = sys.argv[1], int(sys.argv[2])
data = json.loads(Path(path).read_text()) if Path(path).exists() else {}
for guid, info in data.items():
    if info.get("machineId") == machine_id:
        print(guid)
        break
PY
)"

if [ -z "${GUID}" ]; then
  log_err "no onboarded GUID found for machine ${MACHINE_ID} in ${GUID_MAP_FILE} — run 04-onboard-machine.sh ${MACHINE_ID} first"
  exit 1
fi

BODY="$(python - "${SCORE}" "${THRESHOLD}" "${PASSED}" <<'PY'
import json
import sys

score, threshold, passed = sys.argv[1:4]
body = {"score": float(score)}
if threshold:
    body["threshold"] = float(threshold)
if passed:
    body["passed"] = passed.lower() == "true"
print(json.dumps(body))
PY
)"

curl -sS -X POST "${BACKEND_URL}/api/ingest" \
  -H "Content-Type: application/json" \
  -H "X-Device-Guid: ${GUID}" \
  -d "${BODY}"
echo
