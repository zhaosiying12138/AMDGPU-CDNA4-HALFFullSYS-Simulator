#!/usr/bin/env bash
# Build and stage the AgentENV WSL kernel without changing the active WSL
# configuration. Activation is intentionally a separate, manual maintenance
# step because wsl --shutdown terminates every running WSL workload.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PIN=14794180686c2fb6307fbe359c359bec765249f3
KERNEL_VERSION=6.18.40.1
OUT=${AGENTENV_KERNEL_OUT:-"$ROOT/build/agentenv-kernel"}
SRC="$OUT/src"
CFG="$SRC/Microsoft/config-wsl-agentenv"
JOBS=${AGENTENV_KERNEL_JOBS:-4}
SUBMODULE="$ROOT/projects/WSL2-Linux-Kernel"
WIN_STAGE=${AGENTENV_KERNEL_WIN_STAGE:-/mnt/c/Users/Admin1/wsl-kernels/agentenv-6.18.40.1}

die() { printf 'agentenv-kernel: %s\n' "$*" >&2; exit 1; }
require_pin() {
  test -d "$SUBMODULE" || die "kernel gitlink is not checked out: $SUBMODULE"
  actual=$(git -C "$SUBMODULE" rev-parse HEAD 2>/dev/null) || \
    die "kernel gitlink is not a usable checkout: $SUBMODULE"
  test "$actual" = "$PIN" || die "kernel pin mismatch: $actual != $PIN"
}

prepare() {
  require_pin
  mkdir -p "$OUT"
  if [[ ! -e "$SRC/Makefile" ]]; then
    rm -rf "$SRC"
    mkdir -p "$SRC"
    git -C "$SUBMODULE" archive --format=tar "$PIN" | tar -xf - -C "$SRC"
  fi
  test -f "$SRC/Makefile" || die "kernel source archive is incomplete"
  cp -f "$SRC/Microsoft/config-wsl" "$CFG"
  "$SRC/scripts/config" --file "$CFG" --module BLK_DEV_UBLK
  "$SRC/scripts/config" --file "$CFG" --set-str LOCALVERSION "-aenv-ublk-6.18.40.1"
  make -C "$SRC" KCONFIG_CONFIG="$CFG" olddefconfig
  grep -q '^CONFIG_BLK_DEV_UBLK=m$' "$CFG" || die "UBLK was not enabled as a module"
  grep -q '^CONFIG_KVM=m$' "$CFG" || die "KVM baseline unexpectedly changed"
  make -s -C "$SRC" KCONFIG_CONFIG="$CFG" kernelrelease > "$OUT/kernelrelease"
  sha256sum "$CFG" > "$OUT/config.sha256"
  printf 'prepared %s\n' "$(cat "$OUT/kernelrelease")"
}

build() {
  prepare
  make -C "$SRC" -j"$JOBS" KCONFIG_CONFIG="$CFG"
  rm -rf "$OUT/modules" "$OUT/headers" "$OUT/perf"
  make -C "$SRC" KCONFIG_CONFIG="$CFG" INSTALL_MOD_PATH="$OUT/modules" modules_install
  make -C "$SRC" headers_install INSTALL_HDR_PATH="$OUT/headers"
  make -C "$SRC/tools/perf" -j"$JOBS" \
    NO_JEVENTS=1 NO_JVMTI=1 NO_LIBTRACEEVENT=1 \
    install DESTDIR="$OUT/perf" prefix=/
  test -s "$SRC/arch/x86/boot/bzImage" || die "bzImage was not produced"
  test -d "$OUT/modules/lib/modules/$(cat "$OUT/kernelrelease")" || die "modules tree missing"
  test -d "$OUT/headers" || die "headers tree missing"
  test -d "$OUT/perf" || die "perf tree missing"
  rm -f "$OUT/modules.vhdx"
  "$SRC/Microsoft/scripts/gen_artifacts_vhdx.sh" \
    "$OUT/modules" "$OUT/headers" "$OUT/perf" \
    "$(cat "$OUT/kernelrelease")" "$OUT/modules.vhdx"
  sha256sum "$SRC/arch/x86/boot/bzImage" "$OUT/modules.vhdx" > "$OUT/artifacts.sha256"
  printf 'built artifacts in %s\n' "$OUT"
}

stage() {
  test -s "$SRC/arch/x86/boot/bzImage" || die "run build first"
  test -s "$OUT/modules.vhdx" || die "run build first"
  mkdir -p "$WIN_STAGE"
  cp -f "$SRC/arch/x86/boot/bzImage" "$WIN_STAGE/bzImage"
  cp -f "$OUT/modules.vhdx" "$WIN_STAGE/modules.vhdx"
  candidate="$OUT/wslconfig.candidate"
  python3 "$ROOT/tools/agentenv_wslconfig.py" \
    --active /mnt/c/Users/Admin1/.wslconfig \
    --output "$candidate" \
    --stage "$WIN_STAGE"
  sha256sum "$WIN_STAGE/bzImage" "$WIN_STAGE/modules.vhdx" "$OUT/wslconfig.candidate"
  printf 'staged artifacts and candidate config; active .wslconfig was not modified\n'
}

status() {
  printf 'running kernel: '; uname -r
  printf 'expected pin: %s (%s)\n' "$PIN" "$KERNEL_VERSION"
  printf 'ublk device: '; test -e /dev/ublk-control && echo present || echo absent
  printf 'candidate: '; test -f "$OUT/wslconfig.candidate" && echo "$OUT/wslconfig.candidate" || echo absent
  printf 'active config sha: '; sha256sum /mnt/c/Users/Admin1/.wslconfig 2>/dev/null || echo unavailable
}

case "${1:-status}" in
  prepare) prepare ;;
  build) build ;;
  stage) stage ;;
  status) status ;;
  *) die "usage: $0 {prepare|build|stage|status}" ;;
esac
