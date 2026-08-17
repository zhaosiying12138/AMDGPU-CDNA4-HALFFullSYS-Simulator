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
# Use a new artifact directory so stage never overwrites the VHD attached by
# the current WSL VM. The previous directory remains an immediate rollback.
WIN_STAGE=${AGENTENV_KERNEL_WIN_STAGE:-/mnt/c/Users/Admin1/wsl-kernels/agentenv-6.18.40.1-dual-v1}

die() { printf 'agentenv-kernel: %s\n' "$*" >&2; exit 1; }
require_pin() {
  test -d "$SUBMODULE" || die "kernel gitlink is not checked out: $SUBMODULE"
  actual=$(git -C "$SUBMODULE" rev-parse HEAD 2>/dev/null) || \
    die "kernel gitlink is not a usable checkout: $SUBMODULE"
  test "$actual" = "$PIN" || die "kernel pin mismatch: $actual != $PIN"
}

kernel_release() {
  local release
  test -s "$OUT/kernelrelease" || die "kernel release is missing: $OUT/kernelrelease"
  IFS= read -r release < "$OUT/kernelrelease"
  case "$release" in
    ""|.|..|*/*) die "invalid kernel release in $OUT/kernelrelease: $release" ;;
  esac
  printf '%s\n' "$release"
}

# WSL 2.7.10 mounts the root of kernelModules directly over
# /usr/lib/modules/$(uname -r), while the pinned 6.18 source generator targets a
# newer nested artifact contract. Materialize both views without duplicating
# modules: canonical nested artifacts plus relative root compatibility links.
pack_modules() (
  local release modules_root tmp_dir staging_dir artifacts_dir nested_modules
  local image_size image_blocks inode_count output_tmp entry name
  release=$(kernel_release)
  modules_root="$OUT/modules/lib/modules/$release"
  test -d "$modules_root/kernel" || die "module kernel tree missing: $modules_root/kernel"
  test -f "$modules_root/modules.dep" || die "module dependency index missing: $modules_root/modules.dep"
  test -d "$OUT/headers" || die "installed headers tree missing: $OUT/headers"
  test -d "$OUT/perf" || die "installed perf tree missing: $OUT/perf"
  command -v mke2fs >/dev/null || die "mke2fs is required to pack modules.vhdx"
  command -v qemu-img >/dev/null || die "qemu-img is required to pack modules.vhdx"

  tmp_dir=$(mktemp -d "$OUT/.modules-vhd.XXXXXX")
  output_tmp="$OUT/.modules.vhdx.$$"
  trap 'rm -rf "$tmp_dir"; rm -f "$output_tmp"' EXIT
  staging_dir="$tmp_dir/root"
  artifacts_dir="$staging_dir/$release"
  nested_modules="$artifacts_dir/modules"
  mkdir -p "$nested_modules" "$artifacts_dir/linux-headers" "$artifacts_dir/perf"
  cp -a "$modules_root/." "$nested_modules/"
  cp -a "$OUT/headers/." "$artifacts_dir/linux-headers/"
  cp -a "$OUT/perf/." "$artifacts_dir/perf/"

  # These links name the build host and are invalid in a WSL distribution.
  rm -f "$nested_modules/build" "$nested_modules/source"
  test -d "$nested_modules/kernel" || die "staged module kernel tree is missing"
  test -f "$nested_modules/modules.dep" || die "staged module dependency index is missing"

  while IFS= read -r -d '' entry; do
    name=${entry##*/}
    case "$name" in
      build|source) continue ;;
      "$release") die "module entry collides with kernel release: $name" ;;
    esac
    ln -s "$release/modules/$name" "$staging_dir/$name"
  done < <(find "$nested_modules" -mindepth 1 -maxdepth 1 -print0)
  test -L "$staging_dir/kernel" || die "flat compatibility kernel link is missing"
  test -L "$staging_dir/modules.dep" || die "flat compatibility dependency link is missing"

  image_size=$(du -bs "$staging_dir" | awk '{print $1}')
  image_size=$((image_size + (256 * (1 << 20))))
  image_blocks=$(((image_size + 4095) / 4096))
  inode_count=$(find "$staging_dir" -xdev | wc -l)
  inode_count=$((inode_count + 4096))

  mke2fs -q -L '' -d "$staging_dir" -N "$inode_count" -b 4096 \
    -t ext4 "$tmp_dir/modules.img" "$image_blocks"
  qemu-img convert -O vhdx "$tmp_dir/modules.img" "$output_tmp"
  mv -f "$output_tmp" "$OUT/modules.vhdx"
  printf 'packed dual-layout WSL module VHD for %s\n' "$release"
)

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
  pack_modules
  sha256sum "$SRC/arch/x86/boot/bzImage" "$OUT/modules.vhdx" > "$OUT/artifacts.sha256"
  printf 'built artifacts in %s\n' "$OUT"
}

stage() {
  test -s "$SRC/arch/x86/boot/bzImage" || die "run build first"
  test -s "$OUT/modules.vhdx" || die "run build first"
  test ! -e "$WIN_STAGE" || \
    die "immutable stage target already exists; choose a new AGENTENV_KERNEL_WIN_STAGE: $WIN_STAGE"
  mkdir -p "$(dirname "$WIN_STAGE")"
  mkdir "$WIN_STAGE"
  cp "$SRC/arch/x86/boot/bzImage" "$WIN_STAGE/bzImage"
  cp "$OUT/modules.vhdx" "$WIN_STAGE/modules.vhdx"
  candidate="$OUT/wslconfig.candidate"
  python3 "$ROOT/tools/agentenv_wslconfig.py" \
    --active /mnt/c/Users/Admin1/.wslconfig \
    --output "$candidate" \
    --stage "$WIN_STAGE"
  sha256sum "$WIN_STAGE/bzImage" "$WIN_STAGE/modules.vhdx" "$OUT/wslconfig.candidate"
  printf 'staged artifacts and candidate config; active .wslconfig was not modified\n'
}

status() {
  local running module_dir
  running=$(uname -r)
  module_dir="/lib/modules/$running"
  printf 'running kernel: '; uname -r
  printf 'expected pin: %s (%s)\n' "$PIN" "$KERNEL_VERSION"
  printf 'ublk device: '; test -e /dev/ublk-control && echo present || echo absent
  printf 'module layout: '
  if [[ -d "$module_dir/$running/modules" && -L "$module_dir/kernel" && -L "$module_dir/modules.dep" ]]; then
    echo 'dual-compatible'
  elif [[ -d "$module_dir/$running/modules" ]]; then
    echo 'nested-release (repack and reactivate kernelModules)'
  elif [[ -f "$module_dir/modules.dep" && -d "$module_dir/kernel" ]]; then
    echo 'module-root'
  else
    echo 'missing-or-unknown'
  fi
  printf 'candidate: '; test -f "$OUT/wslconfig.candidate" && echo "$OUT/wslconfig.candidate" || echo absent
  printf 'active config sha: '; sha256sum /mnt/c/Users/Admin1/.wslconfig 2>/dev/null || echo unavailable
}

case "${1:-status}" in
  prepare) prepare ;;
  build) build ;;
  pack-modules) pack_modules ;;
  stage) stage ;;
  status) status ;;
  *) die "usage: $0 {prepare|build|pack-modules|stage|status}" ;;
esac
