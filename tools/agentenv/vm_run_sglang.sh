#!/usr/bin/env bash
# Guest-side: assemble the full simulation environment and run SGLang TP2
# to produce one token, verifying the AgentENV-hosted stack end to end.
#
# Precondition: publish_sim_stack.sh --full has delivered the host-mirrored
# tree at /home/zhaosiying/zcode-lane (interpreter, site-packages, sglang,
# product, gem5, topology, caches, syslibs, model).
#
# The environment below is a byte-level mirror of the accepted host demo lane
# (tools/demo_gen.py build_worker_env + fastcopy fast mode): every path that
# exists on the host exists at the same absolute location inside the microVM.
# Two host-only adaptations remain: no NVML masking (the microVM has no NVIDIA
# card to hide), and the tool shim is created here rather than by demo_gen.
#
# Usage (inside the sandbox):
#   /home/zhaosiying/zcode-lane/tools/agentenv/vm_run_sglang.sh
set -euo pipefail

ROOT=/home/zhaosiying/zcode-lane
CONDA_PREFIX="$ROOT/env/conda/rocm-pytorch-v3-fa8414cce688f934f538163621423376c2542acff3e4d3e403df4340d90fcd6d"
RUNTIME_BUILD="$ROOT/projects/self-amdgpu-runtime/build/cp28-runtime-clang"
ROCR_LIB="$ROOT/build/rocr-stage-zcode/lib"
PRODUCT="$ROOT/env/rocm/product-v1-4d9d40454031c7345f25da81b6781995b09a3b10e4dd66026e019306fc7ee39b"
STATE_DIR=/tmp/amdgpu-sim-demo-gen

# The host lane runs with unlimited locked memory; the guest default (8 MB)
# makes hsa_init fail with 4104 (HSA_STATUS_ERROR_OUT_OF_RESOURCES) because
# the topology's GPU memory regions are backed by pinned host pages.  We run
# as root inside the sandbox, so raising both limits is allowed.
ulimit -l unlimited
ulimit -n 1048576

# RCcl skips its rsmi/ARSMI probe when /dev/dxg exists (the WSL host takes
# this branch natively); a zero-byte marker is all access(F_OK) needs.
[[ -e /dev/dxg ]] || touch /dev/dxg

# --- tool shim (mirrors demo_gen: aiter resolves gfx via this rocminfo) -----
mkdir -p "$STATE_DIR/tool-shim" "$STATE_DIR/aiter-config" /tmp/sagr-lane-zcode-demo-tp2
ln -sfn "$RUNTIME_BUILD/sagr-rocminfo" "$STATE_DIR/tool-shim/rocminfo"
ln -sfn "$ROOT/tools/sim_amdgpu_arch.sh" "$STATE_DIR/tool-shim/amdgpu-arch"
AITER_CSV="$CONDA_PREFIX/lib/python3.12/site-packages/aiter/configs/bf16_tuned_gemm.csv"
[[ -f "$AITER_CSV" ]] && cp -f "$AITER_CSV" "$STATE_DIR/aiter-config/bf16_tuned_gemm.csv"

# --- environment assembly (mirror of the host lane) --------------------------
export HOME=/home/zhaosiying
export TERM=dumb
export PATH="$STATE_DIR/tool-shim:$CONDA_PREFIX/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH="$ROOT/projects/sglang-0.5.17:$ROOT/env/sglang-overlay-cp312"
export LD_LIBRARY_PATH="$CONDA_PREFIX/system-runtime/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/rocm-sysroot/opt/rocm-7.2.3/lib:$CONDA_PREFIX/rocm-sysroot/opt/rocm-7.2.3/lib/rocm_sysdeps/lib:$CONDA_PREFIX/lib:$RUNTIME_BUILD:$ROCR_LIB"
export LD_PRELOAD="$ROOT/build/rocr_logging_preload.so:$ROCR_LIB/libhsa-runtime64.so.1"
export ROCM_SIM_ROOT="$PRODUCT"
export HSA_PATH="$PRODUCT"
export HSA_MODEL_LIB="$RUNTIME_BUILD/libself_amdgpu_hsakmt_model.so.1"
export HSA_MODEL_TOPOLOGY="$ROOT/artifacts/topology/gpu-2"
export HSA_ENABLE_DXG_DETECTION=0
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_ENABLE_INTERRUPT=0
export HIP_PLATFORM=amd
export ROCM_PATH="$CONDA_PREFIX/rocm-sysroot/opt/rocm-7.2.3"
export HIP_PATH="$CONDA_PREFIX/rocm-sysroot/opt/rocm-7.2.3"
export HIP_CLANG_PATH="$CONDA_PREFIX/rocm-sysroot/opt/rocm-7.2.3/lib/llvm/bin"
export PYTORCH_ROCM_ARCH=gfx950
export GPU_ARCHS=gfx950
export SAGR_SIM_ROCMINFO="$STATE_DIR/tool-shim/rocminfo"
export SAGR_MANAGED_RUN_ROOT=/tmp/sagr-lane-zcode-demo-tp2
export SAGR_ROCR_LIBRARY_DIR="$ROCR_LIB"
export SAGR_MANAGED_GEM5="$ROOT/projects/gem5/build/VEGA_X86/gem5.opt.fastwrap"
export SAGR_MANAGED_GEM5_CONFIG="$ROOT/projects/gem5/configs/example/gemsim/host_dispatch.py"
export SAGR_MANAGED_REPO_ROOT="$ROOT"
export TRITON_CACHE_AUTOTUNING=1
export SGLANG_USE_AITER=1
export FLA_CACHE_RESULTS=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export AITER_CONFIG_GEMM_BF16="$STATE_DIR/aiter-config/bf16_tuned_gemm.csv"
export TRITON_CACHE_DIR="$ROOT/artifacts/zcode-cache/triton"
export XDG_CACHE_HOME="$STATE_DIR/xdg-cache"
# fastcopy fast mode (scripts/fastcopy_mode.sh fast): DTIF weight injection
export HSA_ENABLE_DTIF_FAST_COPY=1
export SAGR_HSAKMT_MODEL_FAST_COPY=1
mkdir -p "$XDG_CACHE_HOME"
: > "$STATE_DIR/progress.jsonl"

for f in "$SAGR_MANAGED_GEM5" "$SAGR_MANAGED_GEM5_CONFIG" "$HSA_MODEL_TOPOLOGY" "$CONDA_PREFIX/bin/python"; do
  if [[ ! -e "$f" ]]; then
    echo "[vm-run] ERROR: missing $f — was publish_sim_stack.sh --full run?" >&2
    exit 2
  fi
done

echo "[vm-run] environment assembled; launching SGLang TP2 for 1 token..."
# A real file, not a heredoc: with tp>1 sglang spawns schedulers through
# multiprocessing spawn, and spawn re-imports the parent __main__ from its
# recorded path — a stdin-fed program records "<stdin>" and every scheduler
# dies with FileNotFoundError (same trap scripts/run_engine_lane.sh notes).
cat > "$STATE_DIR/vm_worker.py" <<'PYEOF'
def _main():
    import json, os, time

    progress = os.environ.get("DEMO_PROGRESS_FILE", "/tmp/amdgpu-sim-demo-gen/progress.jsonl")

    def emit(rec):
        with open(progress, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    from sglang.srt.entrypoints.engine import Engine

    t0 = time.time()
    emit({"event": "engine_building"})
    engine = Engine(
        model_path="/home/zhaosiying/zcode-lane/models/Qwen3.5-0.8B",
        tp_size=2,
        dtype="bfloat16",
        attention_backend="triton",
        disable_cuda_graph=True,
        disable_custom_all_reduce=True,
        max_total_tokens=256,
        # The topology declares MI350X-scale (288 GiB) VRAM, and the model
        # backs it with committed host pages on touch.  Left uncapped, each
        # rank derives its pools from 0.906 of phantom VRAM and the guest
        # OOM-kills the schedulers; a small explicit fraction bounds the
        # touched pages to what a 0.8B model with 256 token slots needs.
        mem_fraction_static=0.03,
        max_running_requests=1,
        max_mamba_cache_size=5,
        random_seed=0,
        watchdog_timeout=86400,
        dist_timeout=86400,
        context_length=512,
        chunked_prefill_size=-1,
        skip_tokenizer_init=True,
        log_level="info",
    )
    emit({"event": "engine_ready", "load_s": round(time.time() - t0, 1)})

    PROMPT_IDS = [248044, 266]  # frozen golden prompt (tools/qwen35_token_gate.py)
    out = engine.generate(
        input_ids=PROMPT_IDS,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
    )
    ids = out["output_ids"]
    emit({"event": "token", "ids": ids})
    print(f"[vm-run] generated token ids: {ids}")
    print(f"[vm-run] expected: [27841]")
    print(f"[vm-run] {'PASS' if ids == [27841] else 'FAIL'}")
    engine.shutdown()
    import sys
    sys.exit(0 if ids == [27841] else 1)


if __name__ == "__main__":
    _main()

PYEOF
"$CONDA_PREFIX/bin/python" "$STATE_DIR/vm_worker.py"
