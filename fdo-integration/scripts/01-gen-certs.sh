#!/usr/bin/env bash
# Generate the manufacturer / device-CA / owner keys+certs the FDO servers need.
#
# Filenames/formats match exactly what FDO/go-fdo-server's Docker image and
# compose file expect (see FDO/go-fdo-server/docs/user-guide/dockerfile-usage.md
# and deployments/compose/server/fdo-onboarding-servers.yaml):
#   certs/manufacturer.key (DER) + certs/manufacturer.crt (PEM)
#   certs/device_ca.key    (DER) + certs/device_ca.crt    (PEM)
#   certs/owner.key        (DER) + certs/owner.crt        (PEM)
#
# Generated inside a disposable container running as UID 65532 (the UID the
# go-fdo-server distroless:nonroot image runs as) instead of on the Windows
# host + chown, since bind-mount ownership from a Windows host into Docker
# Desktop's Linux VM doesn't reliably support host-side chown.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./lib.sh

FORCE=${1:-}

if [ -f "${CERTS_DIR}/manufacturer.crt" ] && [ "${FORCE}" != "--force" ]; then
  log_info "certs already exist in ${CERTS_DIR}, skipping (use --force to regenerate)"
  exit 0
fi

mkdir -p "${CERTS_DIR}"

gen_one() {
  local name=$1 subj=$2
  log_info "generating ${name} key+cert"
  # MSYS_NO_PATHCONV=1 scoped to just these two calls: "/certs/..." here names
  # a path inside the container, not on the host, and must not be rewritten
  # into a Windows path by Git Bash.
  MSYS_NO_PATHCONV=1 docker run --rm -u 65532:65532 -v "${CERTS_DIR}:/certs" alpine/openssl:latest \
    ecparam -name prime256v1 -genkey -outform der -out "/certs/${name}.key"
  MSYS_NO_PATHCONV=1 docker run --rm -u 65532:65532 -v "${CERTS_DIR}:/certs" alpine/openssl:latest \
    req -x509 -key "/certs/${name}.key" -keyform der -subj "${subj}" -days 365 -out "/certs/${name}.crt"
}

gen_one manufacturer "/C=US/O=FDO/CN=Manufacturer"
gen_one device_ca "/C=US/O=FDO/CN=Device CA"
gen_one owner "/C=US/O=FDO/CN=Owner"

log_info "certs ready in ${CERTS_DIR}"
ls -l "${CERTS_DIR}"
