#!/usr/bin/env bash
# One-time server configuration required before any device can onboard.
# Mirrors the exact sequence FDO/go-fdo-server's own CI test harness uses
# (test/ci/test-onboarding.sh + scripts/fdo-utils.sh), just called directly
# instead of sourcing their whole test framework:
#   1. tell Manufacturing where Rendezvous is (RVInfo)
#   2. tell Rendezvous to trust our device CA (otherwise TO0/TO1 reject the
#      device's voucher — this step is easy to miss, it's not mentioned in
#      the quick-start doc's V2 walkthrough but IS required by this codebase)
#   3. tell Owner its own reachable address, so Rendezvous can redirect
#      devices to it during TO1 (RVTO2Addr)
# All idempotent: safe to re-run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./lib.sh

wait_for_health "${MANUFACTURER_URL}" manufacturer
wait_for_health "${RENDEZVOUS_URL}" rendezvous
wait_for_health "${OWNER_URL}" owner

log_info "configuring RVInfo on Manufacturing"
RV_INFO='[{"dns": "rendezvous", "device_port": "8041", "protocol": "http", "ip": "127.0.0.1", "owner_port": "8041"}]'
if [ -z "$(get_rendezvous_info)" ]; then
  set_rendezvous_info "${RV_INFO}"
else
  update_rendezvous_info "${RV_INFO}"
fi
log_info "RVInfo now: $(get_rendezvous_info)"

log_info "trusting our device CA on Rendezvous"
existing_ca="$(curl -fsS --insecure "${RENDEZVOUS_URL}/api/v1/device-ca")"
if [ "${existing_ca}" = "[]" ] || [ -z "${existing_ca}" ]; then
  add_device_ca_cert
else
  log_info "device CA already registered on Rendezvous, skipping"
fi

log_info "configuring RVTO2Addr on Owner"
OWNER_IP="$(get_real_ip owner)"
if [ -z "${OWNER_IP}" ]; then
  log_err "could not resolve owner container's IP on the 'fdo' network"
  exit 1
fi
if [ -z "$(get_owner_redirect_info)" ]; then
  set_owner_redirect_info "${OWNER_IP}" "owner" "8043" "http"
else
  update_owner_redirect_info "${OWNER_IP}" "owner" "8043" "http"
fi
log_info "RVTO2Addr now: $(get_owner_redirect_info)"

log_info "server configuration complete"
