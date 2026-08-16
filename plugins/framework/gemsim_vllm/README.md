# GemSim vLLM plugin

This package is the formal out-of-tree PyTorch/vLLM integration boundary for
`amdgpu-sim`. It is installed only in the repository's private AMD environment.
It does not modify or monkey patch PyTorch or vLLM. Tensor collectives that do
not have an accepted project communicator implementation remain outside the
accepted execution scope and are rejected at the OOT communicator boundary.

The production TP design reuses vLLM's standard model, parameter loader,
parallel-state initialization, named `GroupCoordinator`, and layer contracts.
The plugin does not write private vLLM TP globals or implement a parallel
sharding framework. It supplies only the official OOT platform, layer/operator,
attention, and `DeviceCommunicator` responsibilities; self-runtime, CCL, and
gem5 details remain below those interfaces. The normative layering is
`docs/framework-runtime-layering.md` at the repository root.

The package boundary provides:

- `vllm.platform_plugins` discovery for the `gemsim_amd` platform;
- `vllm.general_plugins` registration of project-owned `torch.library` ops;
- CPU-staging implementations backed by normal Triton `gemsim_amd` launches;
- FakeTensor implementations for graph and shape propagation;
- formal vLLM OOT replacements for dense linear, RoPE/MRoPE, Gemma RMSNorm,
  SiluAndMul, RMSNormGated, and Qwen Gated DeltaNet decode;
- a bounded GemSim full-attention backend with NHD KV-cache update and GQA
  decode; and
- a separately registered text-only `GemsimQwen3_5ForCausalLM` architecture
  that reuses the pinned upstream model and weight loader.

The registered text-only model now completes a checkpoint-backed,
manually-scheduled single-token forward across all 24 decoder layers. It loads
the official text weights, executes 278 generic Triton dispatches, updates all
18 GDN and 6 NHD attention-cache states, and exactly matches the direct
runner's final BF16 hidden tensor with zero fallback. The OOT model also
preserves the checkpoint's FP32 GDN output-norm parameters rather than silently
narrowing them under the surrounding BF16 model-construction dtype.

The same formal model also executes a bounded one-request, two-token
empty-cache prefill through layers 0..3. It uses sequence GDN state kernels and
causal NHD attention in 141 generic dispatches; both returned BF16 rows, all
three executed GDN states, and the first full-attention KV cache exactly match
serial decode with zero fallback.

The two-token request also executes all 24 layers in exactly 278 dispatches.
Every GDN/NHD state is finite and mutated, all dispatches retire durably, and
the session cleans up. This is not yet a full numerical acceptance claim: the
frozen independent-NVIDIA pointwise gate reports 318 final-hidden outliers,
despite relative-L2 `0.10932212645964957` and cosine `0.9940072673597194`.
Teacher-forced layer 16 GDN and layer 23 full-attention reruns both pass their
strict output and state gates with zero mismatch, identifying accumulated BF16
cross-architecture drift rather than an intrinsic failure in those layers.

The package's standalone communicator is now formally accepted at
N=2/BF16/1024. The live entry is the pinned vLLM
`GroupCoordinator.all_reduce`, which selects the exact out-of-tree
`GemsimDeviceCommunicator` and reusable CCL engine. Both ranks match an
independently reconstructed ring oracle bitwise, execute one normal Triton SUM
each, preserve input storage, return fresh outputs, and report zero Gloo tensor
API calls, host reduction, fallback, FD leak, or orphan process. The accepted
bundle is
`artifacts/evidence/vllm-ccl-live-n2-bf16-1024-v1-accepted`.

This does not yet claim a complete vLLM worker/scheduler path, a sharded
RowParallel layer, full prefix numerical acceptance, batched or
cache-preserving multi-token decode, or tensor-parallel model execution. The
next TP gate uses the real Qwen hidden size 1024 to validate TP2
`RowParallelLinear` weight ranges, local dense output, out-of-place allreduce,
and single-device equivalence before enabling other TP adapters. The first
attention backend slice remains fail-closed to one request, BF16, at most 16
new tokens, empty-cache prefill or single-token decode, and context length at
most 128.

Build or verify the repository-owned content-addressed conda product, then use
the plugin through ordinary vLLM imports after activation. Project packages are
installed as noneditable wheels and no workspace `PYTHONPATH` is required:

```bash
./scripts/setup_conda_env.sh --install
./scripts/setup_conda_env.sh --verify
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(./scripts/setup_conda_env.sh --print-prefix)"
python examples/quickstart/vllm_silu.py
```

The product reuses the pinned vLLM/PyTorch/Triton installation, installs this
plugin through entry-point metadata, and binds the exact native runtime and
gem5 artifacts. It does not write the pinned upstream checkouts and does not
compile vLLM, gem5, self-runtime, or a Triton kernel. Operator execution never
calls the setup command.

Both entry points are inert unless `TRITON_DEFAULT_BACKEND=gemsim_amd` and
`ROCM_SIM_ROOT` are present. The registered Torch operators are
`dense_linear`, `embedding`, `rotary_embedding`, `gemma_rms_norm`,
`fused_add_gemma_rms_norm`, `silu_and_mul`, `sigmoid_output_gate`,
`rms_norm_gated`, `gdn_conv_decode`, and `gdn_recurrent_decode`. Their vLLM
adapters use `CustomOp.register_oot`, `PluggableLayer.register_oot`, the
attention-backend selector, and `ModelRegistry`; they do not patch torch or
vLLM source. Adding or running an adapter JIT-compiles only a missing Triton
kernel into the persistent cache. It never builds gem5 or self-runtime.
