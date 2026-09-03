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

# NOTE on Git Bash path conversion: MSYS rewrites bare Unix-style CLI
# arguments (e.g. "/certs/foo") into Windows paths before they reach a native
# binary like docker.exe. That's exactly what we want for *host* paths (e.g.
# the compose --file path below, curl -o targets), but it's wrong for
# arguments that name a path *inside a container* (e.g. "/certs" as the
# right-hand side of a -v mount, or "/workdir/..." passed to go-fdo-client).
# Disabling it globally breaks the host-path case instead (notably curl's
# "-o /dev/null" — Windows curl.exe can't open a literal "/dev/null" once
# MSYS stops translating it), so MSYS_NO_PATHCONV is set locally, only around
# the specific docker invocations that need it (see run_client() below and
# 01-gen-certs.sh), not exported for the whole script.

FDO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
INTEGRATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
WORKDIR="${INTEGRATION_DIR}/workdir"
CERTS_DIR="${WORKDIR}/certs"
CREDS_DIR="${WORKDIR}/device-credentials"

SERVER_COMPOSE="${FDO_ROOT}/FDO/go-fdo-server/deployments/compose/server/fdo-onboarding-servers.yaml"
CLIENT_COMPOSE="${INTEGRATION_DIR}/docker-compose.client.yaml"
# Our own override of FDO/go-fdo-client/Dockerfile (not a submodule file —
# pins a working golang builder tag; see docker/go-fdo-client.Dockerfile for
# why). docker-compose.client.yaml picks it up via ${client_dockerfile}.
CLIENT_DOCKERFILE="${INTEGRATION_DIR}/docker/go-fdo-client.Dockerfile"

# `docker compose --file <path>` shells out to a native Windows binary that
# needs a real Windows path, not the MSYS "/c/Users/..." form these were just
# built in. With MSYS_NO_PATHCONV=1 (above) disabling bash's usual automatic
# conversion, we have to do it ourselves for the handful of paths that name a
# file on the *host* rather than inside a container.
if command -v cygpath &>/dev/null; then
  SERVER_COMPOSE="$(cygpath -w "${SERVER_COMPOSE}")"
  CLIENT_COMPOSE="$(cygpath -w "${CLIENT_COMPOSE}")"
  CLIENT_DOCKERFILE="$(cygpath -w "${CLIENT_DOCKERFILE}")"
fi
export client_dockerfile="${CLIENT_DOCKERFILE}"

# 127.0.0.1, not "localhost" — on this Windows/Docker Desktop/WSL2 setup,
# resolving "localhost" sometimes prefers IPv6 (::1) and the connection stalls
# for a long time before failing over to IPv4, which made wait_for_health
# below flaky/slow even though the service was already up and healthy.
MANUFACTURER_URL="http://127.0.0.1:8038"
RENDEZVOUS_URL="http://127.0.0.1:8041"
OWNER_URL="http://127.0.0.1:8043"

GUID_MAP_FILE="${FDO_ROOT}/demo_web/backend/guid_machine_map.json"

log_info() { echo "[fdo] $*"; }
log_warn() { echo "[fdo][warn] $*" >&2; }
log_err() { echo "[fdo][error] $*" >&2; }

wait_for_health() {
  local url=$1 name=$2 retries=0 max_retries=30
  log_info "waiting for ${name} health at ${url}/health"
  until curl -fsS --max-time 3 "${url}/health" >/dev/null 2>&1; do
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
  # go-fdo-client's own CLI args (--blob /workdir/...) are container-internal
  # paths and must not be MSYS-translated; --file "${CLIENT_COMPOSE}" is a
  # host path and was already resolved to native Windows form above.
  MSYS_NO_PATHCONV=1 docker compose --file "${CLIENT_COMPOSE}" run --rm go-fdo-client "$@"
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
