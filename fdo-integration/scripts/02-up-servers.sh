#!/usr/bin/env bash
# Bring up the three FDO server roles (manufacturer/rendezvous/owner) via the
# compose file already vendored in the go-fdo-server submodule (it builds the
# image from local source, ../../.. from that file's location, so no network
# fetch is needed at demo time beyond the base images).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./lib.sh

if [ ! -f "${CERTS_DIR}/manufacturer.crt" ]; then
  log_err "certs not found — run ./01-gen-certs.sh first"
  exit 1
fi

mkdir -p "${WORKDIR}"

export base_dir="${WORKDIR}"
export container_user="65532:65532"
export container_working_dir="/workdir"

log_info "starting manufacturer / rendezvous / owner containers"
# MSYS_NO_PATHCONV=1: on Git Bash, "container_working_dir=/workdir" (a
# container-internal path baked into the compose file's volume spec) can get
# silently rewritten into a Windows path before Docker sees it, corrupting
# the "host:container" volume string and causing a "too many colons" mount
# error. Scoped to just this command.
MSYS_NO_PATHCONV=1 docker compose --file "${SERVER_COMPOSE}" up -d --build

wait_for_health "${MANUFACTURER_URL}" manufacturer
wait_for_health "${RENDEZVOUS_URL}" rendezvous
wait_for_health "${OWNER_URL}" owner

log_info "all three FDO server roles are up"
