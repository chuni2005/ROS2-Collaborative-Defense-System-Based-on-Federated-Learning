#!/usr/bin/env bash
# Shared helpers for fdo-integration scripts.
#
# The curl calls here mirror the tested logic in
# FDO/go-fdo-server/scripts/{cert-utils,fdo-utils}.sh and
# FDO/go-fdo-server/test/{ci,container}/utils.sh (the repo's own CI test
# harness for exactly this onboarding flow), adapted to:
#   - run go-fdo-client via Docker Compose instead of a bare-metal binary
#   - use published localhost ports for host-side curl calls instead of
#     editing /etc/hosts (avoids needing sudo)
#   - use the V1 API (/api/v1/...), same as the tested harness, not V2
set -euo pipefail

FDO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
INTEGRATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
WORKDIR="${INTEGRATION_DIR}/workdir"
CERTS_DIR="${WORKDIR}/certs"
CREDS_DIR="${WORKDIR}/device-credentials"

SERVER_COMPOSE="${FDO_ROOT}/FDO/go-fdo-server/deployments/compose/server/fdo-onboarding-servers.yaml"
CLIENT_COMPOSE="${INTEGRATION_DIR}/docker-compose.client.yaml"

MANUFACTURER_URL="http://localhost:8038"
RENDEZVOUS_URL="http://localhost:8041"
OWNER_URL="http://localhost:8043"

GUID_MAP_FILE="${FDO_ROOT}/demo_web/backend/guid_machine_map.json"

log_info() { echo "[fdo] $*"; }
log_warn() { echo "[fdo][warn] $*" >&2; }
log_err() { echo "[fdo][error] $*" >&2; }

wait_for_health() {
  local url=$1 name=$2 retries=0 max_retries=30
  log_info "waiting for ${name} health at ${url}/health"
  until curl -fsS -o /dev/null "${url}/health" 2>/dev/null; do
    retries=$((retries + 1))
    if [ "${retries}" -ge "${max_retries}" ]; then
      log_err "${name} did not become healthy in time"
      return 1
    fi
    sleep 2
  done
  log_info "${name} healthy"
}

# --- RVInfo (Manufacturing) / device-ca (Rendezvous) / owner redirect (Owner) ---
# All V1 API, matching FDO/go-fdo-server/scripts/fdo-utils.sh exactly.

get_rendezvous_info() {
  curl -fsS --insecure -H 'Content-Type: text/plain' "${MANUFACTURER_URL}/api/v1/rvinfo"
}
set_rendezvous_info() {
  curl -fsS --insecure -X POST -H 'Content-Type: application/json' --data-raw "$1" "${MANUFACTURER_URL}/api/v1/rvinfo"
}
update_rendezvous_info() {
  curl -fsS --insecure -X PUT -H 'Content-Type: application/json' --data-raw "$1" "${MANUFACTURER_URL}/api/v1/rvinfo"
}

add_device_ca_cert() {
  curl -fsS --insecure -X POST -H 'Content-Type: application/x-pem-file' \
    --data-binary "@${CERTS_DIR}/device_ca.crt" "${RENDEZVOUS_URL}/api/v1/device-ca"
}

get_owner_redirect_info() {
  curl -fsS --insecure -H 'Content-Type: text/plain' "${OWNER_URL}/api/v1/owner/redirect"
}
set_owner_redirect_info() {
  local ip=$1 dns=$2 port=$3 protocol=$4
  local payload="[{\"ip\": \"${ip}\", \"dns\": \"${dns}\", \"port\": \"${port}\", \"protocol\": \"${protocol}\"}]"
  curl -fsS --insecure -X POST -H 'Content-Type: text/plain' --data-raw "${payload}" "${OWNER_URL}/api/v1/owner/redirect"
}
update_owner_redirect_info() {
  local ip=$1 dns=$2 port=$3 protocol=$4
  local payload="[{\"ip\": \"${ip}\", \"dns\": \"${dns}\", \"port\": \"${port}\", \"protocol\": \"${protocol}\"}]"
  curl -fsS --insecure -X PUT -H 'Content-Type: text/plain' --data-raw "${payload}" "${OWNER_URL}/api/v1/owner/redirect"
}

get_real_ip() {
  docker inspect --format='{{.NetworkSettings.Networks.fdo.IPAddress}}' "$1"
}

# --- go-fdo-client, run via Docker Compose (no local Go toolchain available) ---

run_client() {
  docker compose --file "${CLIENT_COMPOSE}" run --rm go-fdo-client "$@"
}

# --- voucher handoff (V1 API) ---

get_ov_from_manufacturer() {
  local guid=$1 output=$2
  curl -fsS --insecure -H 'Accept: application/x-pem-file' "${MANUFACTURER_URL}/api/v1/vouchers/${guid}" -o "${output}"
}
send_ov_to_owner() {
  local output=$1
  curl -fsS --insecure -X POST --data-binary "@${output}" "${OWNER_URL}/api/v1/owner/vouchers"
}

# --- owner device status ---

owner_device_by_old_guid() {
  curl -fsS --insecure "${OWNER_URL}/api/v1/owner/devices?old_guid=$1"
}

# field <json> <python-expr-on-data> — tiny JSON helper since jq isn't installed here.
json_field() {
  python -c "import sys, json; data = json.loads(sys.stdin.read()); print($1)"
}
