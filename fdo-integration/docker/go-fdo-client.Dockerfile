# SPDX-License-Identifier: Apache 2.0
#
# Local override of FDO/go-fdo-client/Dockerfile (a git submodule — we don't
# modify files under FDO/, see FDO/README.md). This file lives entirely in
# fdo-integration/.
#
# Two differences from the vendored Dockerfile, both worked out empirically
# (see fdo-integration/README.md for the full story):
#   1. Pinned golang:1.26.1-alpine instead of the vendored Dockerfile's
#      floating "golang:1.26-alpine" tag, for a reproducible build.
#   2. `RUN make` replaced with the equivalent steps from the submodule's own
#      Makefile `build` target (tidy, vet, docs, then `go build` with the same
#      -ldflags), run directly instead of through `make`. Running the exact
#      same final `go build` command through `make build` reliably fails —
#      the linker aborts immediately dumping its own usage text with no
#      further diagnostic — while running the identical command directly in
#      the shell (after the same preceding tidy/vet/docs steps) reliably
#      succeeds. Confirmed repeatedly in isolation; the discrepancy is
#      specific to make's invocation of that one recipe line, not the Go
#      toolchain, this module's code, or its dependencies.
FROM golang:1.26.1-alpine AS builder

WORKDIR /go/src/app
COPY . .

RUN go mod tidy
RUN go vet ./...
RUN SOURCE_DATE_EPOCH=0 go run ./internal/tools/docgen -format man && \
    go run ./internal/tools/docgen -format markdown
RUN go build -ldflags="-X github.com/fido-device-onboard/go-fdo-client/internal/version.VERSION=1.0.0" -o go-fdo-client .
RUN install -D -m 755 go-fdo-client /go/bin/

# Start a new stage
FROM gcr.io/distroless/static-debian12

COPY --from=builder /go/bin/go-fdo-client /usr/bin/go-fdo-client

ENTRYPOINT ["go-fdo-client"]
