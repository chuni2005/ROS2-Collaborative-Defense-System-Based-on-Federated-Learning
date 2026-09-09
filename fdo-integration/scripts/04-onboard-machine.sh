#!/usr/bin/env bash
# Onboard one simulated "machine" through the real FDO flow:
#   DI -> extract GUID -> voucher handoff (Manufacturing -> Owner, triggers
#   TO0) -> onboard (TO1+TO2) -> resolve the final (possibly rotated) GUID ->
#   record it in demo_web/backend/guid_machine_map.json
#
# Usage: 04-onboard-machine.sh <machine_id 1-5> [device-info-label]
# Safe to re-run for the same machine id (simulates re-onboarding; the GUID
# mapped to that machine id will be replaced).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./lib.sh

MACHINE_ID=${1:?usage: 04-onboard-machine.sh <machine_id> [device-info-label]}
LABEL=${2:-machine-${MACHINE_ID}}

wait_for_health "${MANUFACTURER_URL}" manufacturer
wait_for_health "${RENDEZVOUS_URL}" rendezvous
wait_for_health "${OWNER_URL}" owner

MACHINE_DIR="${CREDS_DIR}/machine-${MACHINE_ID}"
mkdir -p "${MACHINE_DIR}" "${WORKDIR}/tmp"

BLOB_HOST="${MACHINE_DIR}/cred.bin"
BLOB_CONTAINER="/workdir/device-credentials/machine-${MACHINE_ID}/cred.bin"
rm -f "${BLOB_HOST}"

log_info "machine ${MACHINE_ID}: running Device Initialization (DI)"
run_client device-init "http://manufacturer:8038" \
  --device-info "${LABEL}" --key ec256 --blob "${BLOB_CONTAINER}"

log_info "machine ${MACHINE_ID}: extracting GUID from credential blob"
DI_GUID="$(run_client print --blob "${BLOB_CONTAINER}" | grep -oE '[0-9a-fA-F]{32}' | head -n1)"
if [ -z "${DI_GUID}" ]; then
  log_err "machine ${MACHINE_ID}: could not extract a GUID from 'print' output — aborting, not writing anything"
  exit 1
fi
log_info "machine ${MACHINE_ID}: DI GUID = ${DI_GUID}"

log_info "machine ${MACHINE_ID}: fetching ownership voucher from Manufacturing"
VOUCHER_FILE="${MACHINE_DIR}/voucher.pem"
get_ov_from_manufacturer "${DI_GUID}" "${VOUCHER_FILE}"

log_info "machine ${MACHINE_ID}: uploading voucher to Owner (triggers TO0)"
send_ov_to_owner "${VOUCHER_FILE}"

log_info "machine ${MACHINE_ID}: waiting for TO0 to settle"
sleep 10

log_info "machine ${MACHINE_ID}: running onboarding (TO1 + TO2)"
ONBOARD_LOG="${MACHINE_DIR}/onboard.log"
run_client onboard --key ec256 --kex ECDH256 --blob "${BLOB_CONTAINER}" | tee "${ONBOARD_LOG}"
if ! grep -qF 'FIDO Device Onboard Complete' "${ONBOARD_LOG}"; then
  log_err "machine ${MACHINE_ID}: onboarding did not report completion — aborting, not writing anything"
  exit 1
fi

log_info "machine ${MACHINE_ID}: resolving final (post-TO2) GUID"
FINAL_GUID="$(owner_device_by_old_guid "${DI_GUID}" | json_field "data[0]['guid'] if data else ''")"
if [ -z "${FINAL_GUID}" ]; then
  log_warn "machine ${MACHINE_ID}: no rotated GUID found, keeping DI-time GUID"
  FINAL_GUID="${DI_GUID}"
fi
log_info "machine ${MACHINE_ID}: final GUID = ${FINAL_GUID}"

python - "${GUID_MAP_FILE}" "${FINAL_GUID}" "${MACHINE_ID}" "${LABEL}" <<'PY'
import json
import sys
from pathlib import Path

path, guid, machine_id, label = sys.argv[1:5]
p = Path(path)
data = json.loads(p.read_text()) if p.exists() else {}

# drop any stale entries previously mapped to this machine id
data = {g: v for g, v in data.items() if v.get("machineId") != int(machine_id)}
data[guid] = {"machineId": int(machine_id), "deviceInfo": label}

p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {path}")
PY

log_info "machine ${MACHINE_ID} onboarded, guid=${FINAL_GUID}"
