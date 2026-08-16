#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
vllm_dir="${root_dir}/projects/vllm"
plugin_dir="${root_dir}/plugins/framework/gemsim_vllm"
ccl_plugin_dir="${root_dir}/plugins/collectives/gemsim_ccl"
expected_vllm_head="8d9b52f7c2514490bdadfd5eb0c931e58625df2e"
expected_common_sha="d0f997a49ee1c8f2f952378818e345d7b054824a6abb7a65a47952804b345614"
vllm_version="0.0.dev0+g${expected_vllm_head:0:10}"
mode=install

if (( $# > 1 )); then
  printf 'usage: %s [--verify-only]\n' "${0##*/}" >&2
  exit 2
fi
if (( $# == 1 )); then
  if [[ "$1" != --verify-only ]]; then
    printf 'usage: %s [--verify-only]\n' "${0##*/}" >&2
    exit 2
  fi
  mode=verify
fi

actual_vllm_head=$(git -C "${vllm_dir}" rev-parse HEAD)
if [[ "${actual_vllm_head}" != "${expected_vllm_head}" ]]; then
  printf 'pinned vLLM HEAD mismatch: expected %s, got %s\n' \
    "${expected_vllm_head}" "${actual_vllm_head}" >&2
  exit 1
fi
if ! git -C "${vllm_dir}" diff --quiet HEAD --; then
  printf 'pinned vLLM tracked source is dirty\n' >&2
  exit 1
fi
actual_common_sha=$(sha256sum "${vllm_dir}/requirements/common.txt" | cut -d' ' -f1)
if [[ "${actual_common_sha}" != "${expected_common_sha}" ]]; then
  printf 'vLLM common requirements digest mismatch\n' >&2
  exit 1
fi

if [[ "${mode}" == install ]]; then
  prefix=$("${root_dir}/scripts/setup_rocm_env.sh" --print-base-prefix)
else
  prefix=$("${root_dir}/scripts/setup_rocm_env.sh" --print-prefix)
fi
prefix=$(realpath -e "${prefix}")
python="${prefix}/venv/bin/python"
state_root="${prefix}"
product_launcher=""
product_schema=$(
  /usr/bin/python3 - "${prefix}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

prefix = Path(sys.argv[1])
manifest_path = prefix / "manifest.json"
payload = manifest_path.read_bytes()
document = json.loads(payload.decode("ascii"))
if document.get("schema") != "amdgpu-sim.product-prefix.v1":
    print("legacy")
    raise SystemExit(0)
canonical = (json.dumps(
    document, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
) + "\n").encode("ascii")
metadata = manifest_path.lstat()
if (
    payload != canonical
    or not stat.S_ISREG(metadata.st_mode)
    or metadata.st_uid != os.getuid()
    or metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
    or document.get("prefix") != str(prefix)
):
    raise SystemExit("active product manifest is not canonical/private")
base = document.get("base")
plugins = document.get("plugins")
python_record = base.get("python") if isinstance(base, dict) else None
python = Path(python_record.get("path", "")) if isinstance(python_record, dict) else Path()
base_prefix = Path(base.get("prefix", "")) if isinstance(base, dict) else Path()
state_root = Path(document.get("runtime_state_root", ""))
launcher = Path(plugins.get("bootstrap", "")) if isinstance(plugins, dict) else Path()
for path in (base_prefix, python, state_root, launcher):
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise SystemExit("active product execution path is invalid")
try:
    python.relative_to(base_prefix)
    launcher.relative_to(prefix)
except ValueError as error:
    raise SystemExit("active product execution path escapes its owner") from error
observed = python.read_bytes()
if (
    len(observed) != python_record.get("bytes")
    or hashlib.sha256(observed).hexdigest() != python_record.get("sha256")
    or not os.access(python, os.X_OK)
    or not launcher.is_file()
):
    raise SystemExit("active product Python/bootstrap identity is invalid")
print("product")
print(python)
print(state_root)
print(launcher)
PY
)
if [[ "${product_schema%%$'\n'*}" == product ]]; then
  if [[ "${mode}" != verify ]]; then
    printf 'install mode must target the writable schema-8 base prefix\n' >&2
    exit 1
  fi
  mapfile -t product_fields <<< "${product_schema}"
  if (( ${#product_fields[@]} != 4 )); then
    printf 'active product execution record is incomplete\n' >&2
    exit 1
  fi
  python="${product_fields[1]}"
  state_root="${product_fields[2]}"
  product_launcher="${product_fields[3]}"
fi
if [[ ! -x "${python}" ]]; then
  printf 'private Python is unavailable: %s\n' "${python}" >&2
  exit 1
fi

for relative in home tmp cache config data cache/pip; do
  mkdir -p "${state_root}/${relative}"
done

private_env=(
  /usr/bin/env -i
  "HOME=${state_root}/home"
  "TMPDIR=${state_root}/tmp"
  "XDG_CACHE_HOME=${state_root}/cache"
  "XDG_CONFIG_HOME=${state_root}/config"
  "XDG_DATA_HOME=${state_root}/data"
  "PIP_CACHE_DIR=${state_root}/cache/pip"
  "PATH=/usr/bin:/bin"
  "LC_ALL=C"
  "PYTHONNOUSERSITE=1"
  "PYTHONDONTWRITEBYTECODE=1"
  "PIP_DISABLE_PIP_VERSION_CHECK=1"
  "PIP_NO_INPUT=1"
)

export_dir=""
verification_script=""
cleanup() {
  if [[ -n "${verification_script}" ]]; then
    rm -f -- "${verification_script}"
  fi
  if [[ -n "${export_dir}" ]]; then
    rm -rf -- "${export_dir}"
  fi
}
trap cleanup EXIT

if [[ "${mode}" == install ]]; then
  export_dir=$(mktemp -d "${state_root}/tmp/vllm-python-export.XXXXXX")

  git -C "${vllm_dir}" archive --format=tar "${expected_vllm_head}" \
    | tar -xf - -C "${export_dir}"

  "${private_env[@]}" "${python}" -I -m pip install \
    --only-binary=:all: \
    'packaging>=24.2' \
    'setuptools>=77.0.3,<81.0.0' \
    'setuptools-scm>=8.0' \
    'setuptools-rust>=1.9.0' \
    wheel jinja2
  "${private_env[@]}" "${python}" -I -m pip install \
    --only-binary=:all: \
    --requirement "${vllm_dir}/requirements/common.txt"
  "${private_env[@]}" "${python}" -I -m pip install \
    --only-binary=:all: --no-deps \
    --index-url https://download.pytorch.org/whl/cpu \
    'torchvision==0.28.0+cpu'
  "${private_env[@]}" \
    "VLLM_TARGET_DEVICE=empty" \
    "VLLM_VERSION_OVERRIDE=${vllm_version}" \
    "${python}" -I -m pip install \
    --no-deps --no-build-isolation "${export_dir}"
  "${private_env[@]}" "${python}" -I -m pip install \
    --no-deps --no-build-isolation -e "${ccl_plugin_dir}"
  "${private_env[@]}" "${python}" -I -m pip install \
    --no-deps --no-build-isolation -e "${plugin_dir}"
fi

"${private_env[@]}" "${python}" -I -m pip check
verification_script=$(mktemp "${state_root}/tmp/framework-verify.XXXXXX.py")
chmod 0600 "${verification_script}"
cat > "${verification_script}" <<'PY'
import importlib.metadata as m
import os

assert m.version("vllm") == os.environ["EXPECTED_VLLM_VERSION"]
assert m.version("gemsim-ccl") == "0.1.0"
eps = {(e.group, e.name, e.value) for e in m.entry_points()}
assert (
    "vllm.platform_plugins", "gemsim_amd", "gemsim_vllm:platform_plugin"
) in eps
assert (
    "vllm.general_plugins", "gemsim_qwen35_ops", "gemsim_vllm:register_ops"
) in eps
import torch
import torchvision
import vllm.model_executor.models.qwen3_5 as q
assert torch.__version__.startswith("2.13.")
assert hasattr(q, "Qwen3_5ForCausalLM")
PY

verification_command=("${python}" -I "${verification_script}")
if [[ -n "${product_launcher}" ]]; then
  verification_command=("${python}" -I "${product_launcher}" "${verification_script}")
fi
"${private_env[@]}" \
  "EXPECTED_VLLM_VERSION=${vllm_version}" \
  "ROCM_SIM_ROOT=${prefix}" \
  "TRITON_DEFAULT_BACKEND=gemsim_amd" \
  "TRITON_CACHE_DIR=${state_root}/cache/triton" \
  "${verification_command[@]}"

if ! git -C "${vllm_dir}" diff --quiet HEAD --; then
  printf 'framework setup modified pinned vLLM tracked source\n' >&2
  exit 1
fi

printf 'framework environment verified\n'
printf 'prefix=%s\n' "${prefix}"
printf 'vllm_head=%s\n' "${actual_vllm_head}"
printf 'vllm_version=%s\n' "${vllm_version}"
printf 'plugin=%s\n' "${plugin_dir}"
printf 'ccl_plugin=%s\n' "${ccl_plugin_dir}"
