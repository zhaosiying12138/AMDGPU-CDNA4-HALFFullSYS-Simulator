#!/usr/bin/env bash
# One-key publish: push the full simulation stack into a running AgentENV
# sandbox and assemble a runnable host-mirrored tree at /home/zhaosiying.
#
# Layout (CP-0148 bring-up decision): the guest mirrors the host lane tree at
# /home/zhaosiying/zcode-lane/... exactly, so every baked-in absolute path --
# the runtime's compiled-in managed-session defaults (which point into
# /home/zhaosiying/amdgpu-sim), aiter's ROCm probing, sysroot .so closures,
# the conda interpreter's RPATHs -- resolves identically inside the microVM,
# and vm_run_sglang.sh is a byte-level mirror of the accepted host demo lane.
#
# Streams:
#   base     : gem5 + configs + hello fixture + ROCr stage + self-runtime +
#              topology + product + arch shim
#   python   : conda interpreter (+headers) + site-packages incl. torch,
#              triton, trimmed aiter (with the 3 modules the lane loads),
#              torchvision, pandas + sglang source + overlay
#   caches   : warm triton cache + tvm-ffi cache
#   sysroot  : the phase-A library closure (maps capture of a live TP2 run):
#              rocm-sysroot subset + system-runtime + soname/dev symlink
#              chains + .info/version + share/hip + rocm bin
#   hostlibs : the host Ubuntu 26.04 glibc 2.43 loader set + libpython3.14 +
#              python3.14 stdlib.  gem5.opt is built on the host distro and
#              needs GLIBC_2.43/libpython3.14; the sandbox is Ubuntu 24.04,
#              so gem5 launches through this private loader (see the gem5
#              launcher scripts installed by the guest assembly).
#   model    : Qwen3.5-0.8B weights (dereferenced from the checkout symlink)
#
# Usage: ./publish_sim_stack.sh [--full] <sandbox-id>
#   --full publishes every stream; default publishes base only.
set -euo pipefail

ROOT="/home/zhaosiying/zcode-lane"
CONDA="${ROOT}/env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d"
PRODUCT="env/rocm/product-v1-4d9d40454031c7345f25da81b6781995b09a3b10e4dd66026e019306fc7ee39b"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="/tmp/amdgpu-sim-stack-publish"
LANE_REL="home/zhaosiying/zcode-lane"   # archive paths are relative to /
HASH=rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d

FULL=0
for a in "$@"; do
  [[ "$a" == --full ]] && FULL=1
done
SBX="${@: -1}"
if [[ -z "$SBX" || "$SBX" == --* ]]; then
  echo "usage: $0 [--full] <sandbox-id>" >&2; exit 2
fi

# SKIP_STAGE=1 reuses the tars already in $STAGE (use when only the guest
# needs rebuilding); default re-stages from the host tree.
echo "[publish] staging for sandbox ${SBX} (full=${FULL})..."
if [[ "${SKIP_STAGE:-0}" != 1 ]]; then
rm -rf "$STAGE"; mkdir -p "$STAGE"

# --- base stream -------------------------------------------------------------
tar czf "${STAGE}/base.tar.gz" \
  --exclude='__pycache__' \
  -C / \
  "${LANE_REL}/build/rocr-stage-zcode" \
  "${LANE_REL}/build/rocr_logging_preload.so" \
  "${LANE_REL}/projects/self-amdgpu-runtime/build/cp28-runtime-clang" \
  "home/zhaosiying/amdgpu-sim/artifacts/topology" \
  "${LANE_REL}/projects/gem5/build/VEGA_X86/gem5.opt" \
  "${LANE_REL}/projects/gem5/configs" \
  "${LANE_REL}/projects/gem5/tests/test-progs/hello" \
  "${LANE_REL}/tools/sim_amdgpu_arch.sh" \
  "${LANE_REL}/${PRODUCT}"

if (( FULL )); then
  # --- python stream ---------------------------------------------------------
  # Conda runtime: interpreter + stdlib + non-site-packages libs + headers
  # (triton's hip_utils JIT compile needs include/python3.12).  activate.d is
  # dropped on purpose; vm_run_sglang.sh assembles the environment itself.
  tar czf "${STAGE}/conda.tar.gz" \
    --exclude='__pycache__' \
    --exclude='*/site-packages' \
    -C / \
    "${LANE_REL}/env/conda/${HASH}/bin" \
    "${LANE_REL}/env/conda/${HASH}/lib" \
    "${LANE_REL}/env/conda/${HASH}/include" \
    "${LANE_REL}/env/conda/${HASH}/etc" \
    "${LANE_REL}/env/conda/${HASH}/compiler_compat" \
    "${LANE_REL}/env/conda/${HASH}/ssl" \
    "${LANE_REL}/env/conda/${HASH}/x86_64-conda-linux-gnu"

  # Site-packages wholesale minus the heavy packages the sglang lane never
  # imports.  torchvision and pandas stay: sglang.srt.utils.common imports
  # torchvision unguarded, and aiter's import chain reaches pandas.
  tar czf "${STAGE}/sitepkgs.tar.gz" \
    --exclude='__pycache__' \
    -C "${CONDA}/lib/python3.12/site-packages" \
    --exclude='/vllm*' \
    --exclude='/torchaudio*' \
    --exclude='/flash_attn*' \
    --exclude='/scipy*' \
    --exclude='/onnx*' \
    --exclude='/pyarrow*' \
    --exclude='/plotly*' \
    --exclude='/cv2*' \
    --exclude='/opencv*' \
    --exclude='/runai_model_streamer*' \
    --exclude='/numba*' \
    --exclude='/llvmlite*' \
    --exclude='/tilelang*' \
    --exclude='/flydsl*' \
    --exclude='/pip' \
    --exclude='/aiter' \
    --exclude='/torch' --exclude='torch-*.dist-info' \
    --exclude='/triton' --exclude='triton-*.dist-info' \
    .

  tar czf "${STAGE}/torch.tar.gz" \
    --exclude='__pycache__' \
    -C "${CONDA}/lib/python3.12/site-packages" \
    torch "torch-2.12.0+git6bbd260.dist-info"

  tar czf "${STAGE}/triton.tar.gz" \
    --exclude='__pycache__' \
    -C "${CONDA}/lib/python3.12/site-packages" \
    triton "triton-3.7.1+git0263a6a6.dist-info"

  # aiter: python + configs + the three compiled modules the lane loads at
  # import (phase-A maps evidence).  The 2.2 GB flydsl_cache and the rest of
  # the per-op jit .so blobs never load on the triton-attention path.
  tar cf "${STAGE}/aiter.tar" \
    --exclude='__pycache__' \
    --exclude='aiter/jit/*.so' \
    --exclude='aiter/jit/flydsl_cache' \
    -C "${CONDA}/lib/python3.12/site-packages" \
    aiter
  tar rf "${STAGE}/aiter.tar" \
    -C "${CONDA}/lib/python3.12/site-packages" \
    aiter/jit/module_aiter_core.so \
    aiter/jit/module_gemm_common.so \
    aiter/jit/module_sample.so
  gzip "${STAGE}/aiter.tar"

  tar czf "${STAGE}/sglang-stack.tar.gz" \
    --exclude='__pycache__' \
    --exclude='*/.git' \
    -C / \
    "${LANE_REL}/projects/sglang-0.5.17" \
    "home/zhaosiying/amdgpu-sim/env/sglang-overlay-cp312"

  # --- caches stream ---------------------------------------------------------
  tar czf "${STAGE}/caches.tar.gz" -C / \
    "${LANE_REL}/artifacts/zcode-cache/triton" \
    "home/zhaosiying/.cache/tvm-ffi"

  # --- sysroot stream --------------------------------------------------------
  # The phase-A library closure (gen_phaseA_libs.sh + phaseA_sysroot.list):
  # exactly what a live TP2 run maps, plus symlink chains and probe files.
  if [[ -f "${HERE}/phaseA_libs.list" ]]; then
    grep -v -E "^${LANE_REL}/(env/conda|projects/sglang|env/sglang-overlay)" \
      "${HERE}/phaseA_libs.list" | \
      grep -E "^/(home|opt|usr/local)" | sort -u > "${STAGE}/syslibs.copy"
    tar czf "${STAGE}/syslibs.tar.gz" -C / -T \
      <(sed 's|^/||' "${STAGE}/syslibs.copy")
  fi
  if [[ -f "${HERE}/phaseA_sysroot.list" ]]; then
    tar czf "${STAGE}/sysroot.tar.gz" -C / -T \
      <(sed 's|^/||' "${HERE}/phaseA_sysroot.list")
  fi

  # --- hostlibs stream -------------------------------------------------------
  rm -rf "${STAGE}/hostlibs"; mkdir -p "${STAGE}/hostlibs/lib" "${STAGE}/hostlibs/usr/lib"
  for so in ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libstdc++.so.6 \
            libgcc_s.so.1 libz.so.1 libprotobuf.so.32 libexpat.so.1 \
            libpython3.14.so.1.0; do
    cp -L "/usr/lib/x86_64-linux-gnu/${so}" "${STAGE}/hostlibs/lib/" \
      || cp -L "/lib/x86_64-linux-gnu/${so}" "${STAGE}/hostlibs/lib/" \
      || echo "[publish] WARN: host lib missing: ${so}"
  done
  cp -a /usr/lib/python3.14 "${STAGE}/hostlibs/usr/lib/"
  rm -rf "${STAGE}/hostlibs/usr/lib/python3.14/test" \
         "${STAGE}"/hostlibs/usr/lib/python3.14/*/test
  tar czf "${STAGE}/hostlibs.tar.gz" -C "${STAGE}" hostlibs

  # --- model stream ----------------------------------------------------------
  tar czfh "${STAGE}/model.tar.gz" -C / "${LANE_REL}/models/Qwen3.5-0.8B"
fi
fi  # SKIP_STAGE guard

echo "[publish] uploading..."
aenv upload "$SBX" "${STAGE}/base.tar.gz" /tmp/base.tar.gz
if (( FULL )); then
  for f in conda sitepkgs torch triton aiter sglang-stack caches syslibs sysroot hostlibs model; do
    if [[ -f "${STAGE}/${f}.tar.gz" ]]; then
      aenv upload "$SBX" "${STAGE}/${f}.tar.gz" "/tmp/${f}.tar.gz"
    fi
  done
fi

echo "[publish] extracting and assembling on the guest..."
read -r -d '' GUEST_ASSEMBLE <<'GASSEMBLE' || true
set -e
Z=/home/zhaosiying/zcode-lane
C=$Z/env/conda/HASHHASH
mkdir -p /home/zhaosiying /tmp/amdgpu-sim-demo-gen
cd /
for t in base conda sitepkgs torch triton aiter sglang-stack caches syslibs sysroot hostlibs model; do
  [[ -f "/tmp/$t.tar.gz" ]] || continue
  case "$t" in
    sitepkgs|torch|triton|aiter)
      mkdir -p "$C/lib/python3.12/site-packages" &&
        tar xzf "/tmp/$t.tar.gz" -C "$C/lib/python3.12/site-packages" ;;
    hostlibs) tar xzf /tmp/hostlibs.tar.gz -C /home/zhaosiying ;;
    *) tar xzf "/tmp/$t.tar.gz" -C / ;;
  esac
  rm -f "/tmp/$t.tar.gz"
  echo "[guest] $t extracted"
done

# --- host-tree mirrors -------------------------------------------------------
# zcode-lane symlinks exactly as on the host: overlay + topology point into
# the main checkout; the sysroot closure extracts at amdgpu-sim realpaths
# (phase-A maps realpaths), so the conda prefix needs these two links.
mkdir -p "$Z/artifacts" "$Z/env"
ln -sfn /home/zhaosiying/amdgpu-sim/artifacts/topology "$Z/artifacts/topology"
ln -sfn /home/zhaosiying/amdgpu-sim/env/sglang-overlay-cp312 "$Z/env/sglang-overlay-cp312"
rm -rf "$C/system-runtime" "$C/rocm-sysroot"
ln -sfn /home/zhaosiying/amdgpu-sim/env/conda/HASHHASH/system-runtime "$C/system-runtime"
ln -sfn /home/zhaosiying/amdgpu-sim/env/conda/HASHHASH/rocm-sysroot "$C/rocm-sysroot"

# The runtime .so bakes managed-session defaults pointing into
# /home/zhaosiying/amdgpu-sim; mirror the gem5 tree there.  gem5.opt itself
# becomes a launcher through the host glibc-2.43 loader set (see hostlibs).
G5Z=$Z/projects/gem5/build/VEGA_X86
G5A=/home/zhaosiying/amdgpu-sim/projects/gem5
mkdir -p "$G5A/build/VEGA_X86" "$G5A/configs/example" "$G5A/tests"
mv -f "$G5Z/gem5.opt" "$G5Z/gem5.opt.raw" 2>/dev/null || true
chmod +x "$G5Z/gem5.opt.raw"
cat > "$G5Z/gem5.opt.fastwrap" <<'FW'
#!/bin/sh
# gem5.opt.raw was built on Ubuntu 26.04 (glibc 2.43, libpython3.14); this
# sandbox is Ubuntu 24.04.  Launch through the shipped host loader set.
export PYTHONHOME=/home/zhaosiying/hostlibs/usr
exec /home/zhaosiying/hostlibs/lib/ld-linux-x86-64.so.2 \
  --library-path /home/zhaosiying/hostlibs/lib \
  G5ZRAW DOLLARAT --functional-fast
FW
sed -i -e "s|G5ZRAW|$G5Z/gem5.opt.raw|" -e "s|DOLLARAT|\"\$@\"|" "$G5Z/gem5.opt.fastwrap"
chmod +x "$G5Z/gem5.opt.fastwrap"
cat > "$G5A/build/VEGA_X86/gem5.opt" <<'FW'
#!/bin/sh
# Same launcher at the runtime's baked-default path; the raw binary lives
# only at the lane path.
export PYTHONHOME=/home/zhaosiying/hostlibs/usr
exec /home/zhaosiying/hostlibs/lib/ld-linux-x86-64.so.2 \
  --library-path /home/zhaosiying/hostlibs/lib \
  G5ZRAW DOLLARAT
FW
sed -i -e "s|G5ZRAW|$G5Z/gem5.opt.raw|" -e "s|DOLLARAT|\"\$@\"|" "$G5A/build/VEGA_X86/gem5.opt"
chmod +x "$G5A/build/VEGA_X86/gem5.opt"
ln -sfn "$Z/projects/gem5/configs/example/gemsim" "$G5A/configs/example/gemsim"
ln -sfn "$Z/projects/gem5/tests/test-progs" "$G5A/tests/test-progs"
"$G5A/build/VEGA_X86/gem5.opt" --version >/dev/null 2>&1 \
  || echo "[guest] WARN: gem5 launcher check failed"

# --- guest system packages ---------------------------------------------------
# Host system libs the lane maps (NCCL NET/hwloc probe, protobuf, image libs)
# and the toolchain triton needs to JIT its hip_utils launcher module.
echo "nameserver 8.8.8.8" > /etc/resolv.conf
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq --no-install-recommends \
  libibverbs1 libmlx5-1 libnl-3-200 libnl-route-3-200 libhwloc15 \
  libudev1 libzstd1 libprotobuf32t64 libjpeg-turbo8 libfribidi0 \
  libexpat1 libbz2-1.0 liblzma5 gcc g++ make strace file >/dev/null 2>&1 \
  || echo "[guest] WARN: apt system package install incomplete"

# RCcl's rocm_smi_init treats /dev/dxg presence as "WSL2 environment" and
# skips the rsmi/ARSMI path entirely (there is no real RSMI to query under
# the simulator; the WSL host took this same branch, which is why the host
# lane worked).  A zero-byte marker is all access(F_OK) checks for.
[[ -e /dev/dxg ]] || touch /dev/dxg

echo "[publish] DONE"
df -h / | tail -1
GASSEMBLE
GUEST_ASSEMBLE="${GUEST_ASSEMBLE//HASHHASH/${HASH}}"
aenv exec "$SBX" /bin/bash -c "$GUEST_ASSEMBLE"
