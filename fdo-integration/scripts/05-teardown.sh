#!/usr/bin/env bash
# Stop the FDO server containers. Pass --purge to also wipe workdir/ (certs,
# sqlite DBs, per-machine credentials) so the next run starts from scratch.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./lib.sh

log_info "stopping FDO server containers"
docker compose --file "${SERVER_COMPOSE}" down

if [ "${1:-}" = "--purge" ]; then
  log_info "purging ${WORKDIR}"
  rm -rf "${WORKDIR}"
  log_info "purging ${GUID_MAP_FILE}"
  rm -f "${GUID_MAP_FILE}"
fi
