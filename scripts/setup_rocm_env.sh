#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

requested_prefix=${ROCM_PREFIX-}
requested_jobs=${ROCM_JOBS-}
unset BASH_ENV ENV CDPATH GLOBIGNORE PYTHONHOME PYTHONPATH PYTHONOPTIMIZE
unset -f git find grep awk install chmod mkdir dirname basename 2>/dev/null || true
export PATH=/usr/bin:/bin
IFS=$' \t\n'
umask 022

root_dir=$(cd "$(/usr/bin/dirname "$0")/.." && pwd -P)
llvm_source="$root_dir/projects/llvm-project"
device_source="$llvm_source/amd/device-libs"
rocm_source="$root_dir/projects/rocm-systems"
runtime_source="$root_dir/projects/self-amdgpu-runtime"
gem5_source="$root_dir/projects/gem5"
gem5_binary="$gem5_source/build/VEGA_X86/gem5.opt"
gem5_config="$gem5_source/configs/example/gemsim/host_dispatch.py"

setup_schema=5
llvm_head_expected=73f2a21fe16b34e35fd0e149564b8664e59da392
llvm_tree_expected=d589480097e8a30fd1df38435ccc9a9fca71f489
rocm_head_expected=92115a2941982a384de161be3f78cf9bff547027
rocm_tree_expected=28bf42b65f7aad25167180543dda69b5fc6caf58
gem5_head_expected=ed808f6a57b843c040ba864b9e9aad188d0eab36
gem5_tree_expected=27fdc466df4eeebda19e8c6ef18234b07c3ec79f
runtime_head_expected=749717e1a5c7c337d73ab26eae5b0827a34a795a
runtime_tree_expected=fb6dec26311ff4aebfe31c4364cfa9a2575bc0fb
locked_hsaco_sha=7b6a4d2bb7f9c4e7466bcf69f3110ecbfab54d07abd4c70b6bd96b6a6fb9de56
locked_hsaco="$rocm_source/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/gpuReadWrite_kernels.hsaco"
kernel_source="$rocm_source/projects/rdc/rdc_libs/rdc_modules/kernels/gpuReadWrite_kernels.cl"

# Keep the profile path neutral: the legacy dependency audit scans ldd output
# for simulator library names, including path components.
profile_name="gfx950-v${setup_schema}-llvm-${llvm_head_expected:0:12}-rocm-${rocm_head_expected:0:12}-sim-${gem5_head_expected:0:12}-runtime-${runtime_head_expected:0:12}"
rocm_root=$(/usr/bin/realpath -m -- "$root_dir/env/rocm")
prefix="$rocm_root/$profile_name"
[[ -n "$requested_prefix" ]] && prefix="$requested_prefix"
jobs=${requested_jobs:-$(/usr/bin/getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4')}
mode=verify

usage() {
    cat >&2 <<'EOF'
usage: scripts/setup_rocm_env.sh [--verify-only|--all|--compiler|--runtime|--print-prefix]
       [--prefix PATH] [--jobs N]

The default is --verify-only. --all builds the pinned LLVM/Clang/lld,
device-libs, gpuReadWrite code object, shared self-amdgpu-runtime, local
libOpenCL, and direct opencl-vecadd executable under a repository-local
env/rocm prefix. It never installs into /opt/rocm.
EOF
}

while (($#)); do
    case "$1" in
        --verify-only) mode=verify; shift ;;
        --all) mode=all; shift ;;
        --compiler) mode=compiler; shift ;;
        --runtime) mode=runtime; shift ;;
        --print-prefix) mode=print; shift ;;
        --prefix)
            (($# >= 2)) || { usage; exit 2; }
            prefix=$2
            shift 2
            ;;
        --jobs)
            (($# >= 2)) || { usage; exit 2; }
            jobs=$2
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; usage; exit 2 ;;
    esac
done

case "$jobs" in
    ''|*[!0-9]*|0) printf '%s\n' '--jobs must be a positive integer' >&2; exit 2 ;;
esac

if [[ "$prefix" != /* ]]; then
    prefix="$root_dir/$prefix"
fi
prefix=$(/usr/bin/realpath -m -- "$prefix")
[[ "$prefix" != "$rocm_root" && "$(/usr/bin/dirname "$prefix")" == "$rocm_root" ]] || {
    printf 'ROCM_PREFIX must be a direct versioned child of %s (got %s)\n' \
        "$rocm_root" "$prefix" >&2
    exit 1
}
case "$(/usr/bin/basename "$prefix")" in
    ''|*[!A-Za-z0-9._-]*)
        printf 'ROCM_PREFIX profile name must match [A-Za-z0-9._-]+ (got %s)\n' \
            "$(/usr/bin/basename "$prefix")" >&2
        exit 1
        ;;
esac

if [[ "$mode" == print ]]; then
    printf '%s\n' "$prefix"
    exit 0
fi

for required in "$llvm_source" "$device_source" "$rocm_source" \
    "$runtime_source" "$gem5_source" "$gem5_binary" "$gem5_config" \
    "$kernel_source" "$locked_hsaco"; do
    [[ -e "$required" ]] || {
        printf 'missing required source: %s\n' "$required" >&2
        exit 1
    }
done

git_head() {
    git -C "$1" rev-parse HEAD
}

git_tree() {
    git -C "$1" rev-parse 'HEAD^{tree}'
}

assert_repo_lock() {
    local label=$1 repo=$2 expected_head=$3 expected_tree=$4
    local actual_head actual_tree dirty
    actual_head=$(git_head "$repo")
    actual_tree=$(git_tree "$repo")
    [[ "$actual_head" == "$expected_head" ]] || {
        printf '%s lock mismatch: head=%s expected=%s\n' \
            "$label" "$actual_head" "$expected_head" >&2
        return 1
    }
    [[ "$actual_tree" == "$expected_tree" ]] || {
        printf '%s tree mismatch: tree=%s expected=%s\n' \
            "$label" "$actual_tree" "$expected_tree" >&2
        return 1
    }
    dirty=$(git -C "$repo" status --porcelain=v1 --untracked-files=all)
    [[ -z "$dirty" ]] || {
        printf '%s source tree is dirty:\n%s\n' "$label" "$dirty" >&2
        return 1
    }
}

assert_source_locks() {
    assert_repo_lock llvm-project "$llvm_source" \
        "$llvm_head_expected" "$llvm_tree_expected"
    assert_repo_lock rocm-systems "$rocm_source" \
        "$rocm_head_expected" "$rocm_tree_expected"
    assert_repo_lock gem5 "$gem5_source" \
        "$gem5_head_expected" "$gem5_tree_expected"
    assert_repo_lock self-amdgpu-runtime "$runtime_source" \
        "$runtime_head_expected" "$runtime_tree_expected"
}

assert_host_tools() {
    local tool
    for tool in /usr/bin/cmake /usr/bin/ninja /usr/bin/clang /usr/bin/clang++ \
        /usr/bin/python3 /usr/bin/sha256sum /usr/bin/readelf /usr/bin/realpath \
        /usr/bin/flock /usr/bin/file; do
        [[ -x "$tool" ]] || {
            printf 'required bootstrap tool missing: %s\n' "$tool" >&2
            return 1
        }
    done
}

assert_symlinks_local() {
    local link target
    while IFS= read -r -d '' link; do
        target=$(/usr/bin/readlink -f -- "$link" 2>/dev/null || true)
        [[ -n "$target" && "$target" == "$prefix/"* ]] || {
            printf 'prefix symlink escapes local profile: %s -> %s\n' \
                "$link" "${target:-unresolved}" >&2
            return 1
        }
    done < <(find "$prefix" -type l -print0 2>/dev/null)
}

assert_elf_dependencies() {
    local candidate dynamic
    local -a roots=()
    [[ -d "$prefix/bin" ]] && roots+=("$prefix/bin")
    [[ -d "$prefix/lib" ]] && roots+=("$prefix/lib")
    [[ -d "$prefix/lib64" ]] && roots+=("$prefix/lib64")
    [[ -d "$prefix/libexec" ]] && roots+=("$prefix/libexec")
    ((${#roots[@]})) || return 0
    while IFS= read -r -d '' candidate; do
        /usr/bin/readelf -h "$candidate" >/dev/null 2>&1 || continue
        dynamic=$(/usr/bin/readelf -d "$candidate" 2>/dev/null || true)
        if grep -Eiq \
            'libhsa-runtime64|libhsakmt|libamdhip64|libcuda|libcudart|/opt/rocm|/usr/local/cuda|\.triton/manual-llvm|/miniforge3' \
            <<<"$dynamic"; then
            printf 'forbidden ELF dependency or runpath in %s:\n%s\n' \
                "$candidate" "$dynamic" >&2
            return 1
        fi
    done < <(find "${roots[@]}" \( -type f -o -type l \) -print0 2>/dev/null)
}

assert_clean_prefix() {
    local forbidden production
    forbidden=$(grep -R -n -I -E \
        '/opt/rocm|/usr/local/cuda|\.triton/manual-llvm|/tmp/cp17|/miniforge3' \
        "$prefix" --include='CMakeCache.txt' --include='*.cmake' \
        --include='*.ninja' --include='*.json' --include='*.txt' 2>/dev/null || true)
    if [[ -n "$forbidden" ]]; then
        printf 'prefix contains a forbidden host/toolchain path:\n%s\n' "$forbidden" >&2
        return 1
    fi
    production=$(find "$prefix" \( -type f -o -type l \) \
        \( -name 'libhsa-runtime64.so*' -o -name 'libhsakmt.so*' \
        -o -name 'libamdhip64.so*' \) -print -quit 2>/dev/null || true)
    [[ -z "$production" ]] || {
        printf 'production ROCr/HIP library is not allowed: %s\n' "$production" >&2
        return 1
    }
    assert_symlinks_local
    assert_elf_dependencies
}

run_clean() {
    env -i \
        HOME="$prefix/home" TMPDIR="$prefix/tmp" XDG_CACHE_HOME="$prefix/cache" \
        XDG_CONFIG_HOME="$prefix/config" XDG_DATA_HOME="$prefix/data" \
        SOURCE_DATE_EPOCH=0 PATH="/usr/bin:/bin:$prefix/bin" \
        CC=/usr/bin/cc CXX=/usr/bin/c++ AR=/usr/bin/ar RANLIB=/usr/bin/ranlib \
        "$@"
}

llvm_build="$prefix/build/llvm"
device_build="$prefix/build/device-libs"
runtime_build="$prefix/build/self-amdgpu-runtime"
runtime_endpoint="$prefix/libexec/amdgpu-sim/generic-dispatch-v2-endpoint-test"
opencl_library="$prefix/lib/libOpenCL.so.1.2.0"
opencl_executable="$prefix/bin/opencl-vecadd"
opencl_source="$prefix/share/self-amdgpu-runtime/opencl/vecadd.cl"
device_lib_dir="$prefix/amdgcn/bitcode"
device_lib_names=(
    opencl.bc
    ocml.bc
    ockl.bc
    oclc_isa_version_950.bc
    oclc_abi_version_400.bc
    oclc_wavefrontsize64_on.bc
    oclc_wavefrontsize64_off.bc
    oclc_unsafe_math_on.bc
    oclc_unsafe_math_off.bc
    oclc_finite_only_on.bc
    oclc_finite_only_off.bc
)
compiled_hsaco="$prefix/kernels/gfx950/compiled/gpuReadWrite_kernels.hsaco"
installed_locked_hsaco="$prefix/kernels/gfx950/locked/gpuReadWrite_kernels.hsaco"
manifest="$prefix/manifest.json"
setup_stamp="$prefix/setup-script.sha256"
setup_sha=$(/usr/bin/sha256sum "$0" | awk '{print $1}')

write_activation() {
    local activation="$prefix/activate"
    local temporary="$prefix/.activate.tmp.$$"
    {
        printf '%s\n' '# Source in a disposable shell; no system files are modified.'
        printf 'export ROCM_SIM_ROOT=%q\n' "$prefix"
        printf '%s\n' 'export ROCM_PATH="$ROCM_SIM_ROOT"'
        printf '%s\n' 'export HIP_PATH="$ROCM_SIM_ROOT"'
        printf '%s\n' 'export HSA_PATH="$ROCM_SIM_ROOT"'
        printf '%s\n' 'export PATH="$ROCM_SIM_ROOT/bin:/usr/bin:/bin"'
        printf '%s\n' 'export LD_LIBRARY_PATH="$ROCM_SIM_ROOT/lib:$ROCM_SIM_ROOT/lib64"'
        printf '%s\n' 'export CMAKE_PREFIX_PATH="$ROCM_SIM_ROOT"'
        printf '%s\n' 'export PKG_CONFIG_PATH="$ROCM_SIM_ROOT/lib/pkgconfig:$ROCM_SIM_ROOT/lib64/pkgconfig"'
        printf '%s\n' 'unset CUDA_HOME CUDA_PATH CUDACXX LLVM_SYSPATH TRITON_BUILD_WITH_CCACHE'
    } > "$temporary"
    chmod 0555 "$temporary"
    mv -f "$temporary" "$activation"
}

build_llvm() {
    mkdir -p "$llvm_build"
    run_clean /usr/bin/cmake -S "$llvm_source/llvm" -B "$llvm_build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DLLVM_ENABLE_PROJECTS='clang;lld' \
        -DLLVM_TARGETS_TO_BUILD='X86;AMDGPU' \
        -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_BUILD_TESTS=OFF \
        -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_CURL=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_BUILD_LLVM_DYLIB=OFF \
        -DLLVM_LINK_LLVM_DYLIB=OFF \
        -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_35 \
        -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON
    assert_clean_prefix
    run_clean /usr/bin/cmake --build "$llvm_build" --parallel "$jobs" --target \
        clang lld llvm-link llvm-objdump opt FileCheck
    mkdir -p "$prefix/bin"
    local tool
    for tool in clang clang++ lld ld.lld llvm-link llvm-objdump opt FileCheck; do
        [[ -x "$llvm_build/bin/$tool" ]] || {
            printf 'required LLVM tool was not built: %s\n' "$llvm_build/bin/$tool" >&2
            return 1
        }
        ln -sfn "$llvm_build/bin/$tool" "$prefix/bin/$tool"
    done
    assert_clean_prefix
}

build_device_libs() {
    mkdir -p "$device_build"
    run_clean /usr/bin/cmake -S "$device_source" -B "$device_build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DCMAKE_PREFIX_PATH="$prefix" \
        -DLLVM_DIR="$llvm_build/lib/cmake/llvm" \
        -DClang_DIR="$llvm_build/lib/cmake/clang" \
        -DCLANG_OPTIONS_APPEND=-mcpu=gfx950
    assert_clean_prefix
    run_clean /usr/bin/cmake --build "$device_build" --parallel "$jobs" \
        --target rocm-device-libs
    run_clean /usr/bin/cmake --install "$device_build"
    assert_clean_prefix
}

compile_kernel() {
    local resource_dir header sha
    resource_dir=$(run_clean "$prefix/bin/clang" -print-resource-dir)
    [[ "$resource_dir" == "$prefix/"* ]] || {
        printf 'Clang resource directory escapes local prefix: %s\n' "$resource_dir" >&2
        return 1
    }
    header="$resource_dir/include/opencl-c.h"
    [[ -f "$header" ]] || {
        printf 'Clang OpenCL header not found: %s\n' "$header" >&2
        return 1
    }
    mkdir -p "$(/usr/bin/dirname "$compiled_hsaco")" \
        "$(/usr/bin/dirname "$installed_locked_hsaco")"
    run_clean "$prefix/bin/clang" \
        -D ROCRTST_GPU=0x950 -x cl -target amdgcn-amd-amdhsa \
        -include "$header" -mcpu=gfx950 -cl-std=CL2.0 -mcode-object-version=4 \
        --rocm-path="$prefix" "$kernel_source" -o "$compiled_hsaco"
    install -m 0444 "$locked_hsaco" "$installed_locked_hsaco"
    /usr/bin/readelf -h "$compiled_hsaco" >/dev/null
    sha=$(/usr/bin/sha256sum "$compiled_hsaco" | awk '{print $1}')
    printf 'freshly compiled HSACO: %s\n' "$compiled_hsaco"
    printf 'freshly compiled SHA-256: %s\n' "$sha"
    printf 'currently executable locked HSACO: %s\n' "$installed_locked_hsaco"
    if [[ "$sha" != "$locked_hsaco_sha" ]]; then
        printf '%s\n' \
            'fresh compiler output differs from the CP26 daemon lock and is provenance-only' >&2
    fi
}

build_runtime() {
    [[ -x "$prefix/bin/clang" ]] && device_libs_complete || {
        printf '%s\n' \
            'the OpenCL runtime requires the local compiler/device-libs; run --all or --compiler first' >&2
        return 1
    }
    mkdir -p "$runtime_build"
    run_clean /usr/bin/cmake -S "$runtime_source" -B "$runtime_build" -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DBUILD_SHARED_LIBS=ON \
        -DSELF_AMDGPU_RUNTIME_BUILD_TESTS=ON \
        -DSELF_AMDGPU_RUNTIME_BUILD_TOOLS=ON \
        -DSELF_AMDGPU_RUNTIME_BUILD_OPENCL=ON \
        -DSELF_AMDGPU_RUNTIME_OPENCL_PREFIX="$prefix" \
        -DSELF_AMDGPU_RUNTIME_OPENCL_GEM5="$gem5_binary" \
        -DSELF_AMDGPU_RUNTIME_OPENCL_GEM5_CONFIG="$gem5_config" \
        -DSELF_AMDGPU_RUNTIME_TEST_INSTALLED_PACKAGE=OFF
    assert_clean_prefix
    run_clean /usr/bin/cmake --build "$runtime_build" --parallel "$jobs"
    run_clean /usr/bin/ctest --test-dir "$runtime_build" --output-on-failure
    run_clean /usr/bin/cmake --install "$runtime_build"
    mkdir -p "$(/usr/bin/dirname "$runtime_endpoint")"
    install -m 0555 \
        "$runtime_build/tests/self_amdgpu_runtime_generic_dispatch_v2_endpoint_test" \
        "$runtime_endpoint"
    assert_clean_prefix
}

find_device_bc() {
    [[ -f "$device_lib_dir/ocml.bc" ]] && printf '%s' "$device_lib_dir/ocml.bc"
}

device_libs_complete() {
    local name
    for name in "${device_lib_names[@]}"; do
        [[ -f "$device_lib_dir/$name" ]] || return 1
    done
}

assert_device_libs() {
    local name
    for name in "${device_lib_names[@]}"; do
        [[ -f "$device_lib_dir/$name" ]] || {
            printf 'missing installed AMD device library: %s\n' \
                "$device_lib_dir/$name" >&2
            return 1
        }
    done
}

find_runtime_library() {
    find "$prefix/lib" "$prefix/lib64" -maxdepth 2 -type f \
        \( -name 'libself_amdgpu_runtime.a' -o -name 'libself_amdgpu_runtime.so*' \) \
        -print -quit 2>/dev/null || true
}

opencl_complete() {
    [[ -f "$opencl_library" && -x "$opencl_executable" \
        && -f "$opencl_source" && -x "$gem5_binary" \
        && -f "$gem5_config" ]]
}

write_manifest() {
    local runtime_library compiler_complete=false runtime_complete=false
    local opencl_ready=false
    runtime_library=$(find_runtime_library)
    if [[ -x "$prefix/bin/clang" && -x "$prefix/bin/clang++" \
        && -x "$prefix/bin/ld.lld" && -x "$prefix/bin/llvm-link" \
        && -x "$prefix/bin/llvm-objdump" && -x "$prefix/bin/opt" \
        && -x "$prefix/bin/FileCheck" && -f "$compiled_hsaco" \
        && -f "$installed_locked_hsaco" ]] && device_libs_complete; then
        compiler_complete=true
    fi
    if [[ -n "$runtime_library" && -x "$prefix/bin/sagr-handshake" \
        && -x "$prefix/bin/sagr-triton-hsaco-probe" \
        && -x "$runtime_endpoint" ]]; then
        runtime_complete=true
    fi
    if [[ "$runtime_complete" == true ]] && opencl_complete; then
        opencl_ready=true
    fi
    chmod 0644 "$manifest" 2>/dev/null || true
    PREFIX="$prefix" MANIFEST="$manifest" SETUP_SCHEMA="$setup_schema" \
    SETUP_SHA="$setup_sha" REQUESTED_MODE="$mode" \
    LLVM_HEAD="$llvm_head_expected" LLVM_TREE="$llvm_tree_expected" \
    ROCM_HEAD="$rocm_head_expected" ROCM_TREE="$rocm_tree_expected" \
    GEM5_HEAD="$gem5_head_expected" GEM5_TREE="$gem5_tree_expected" \
    RUNTIME_HEAD="$runtime_head_expected" RUNTIME_TREE="$runtime_tree_expected" \
    COMPILER_COMPLETE="$compiler_complete" RUNTIME_COMPLETE="$runtime_complete" \
    OPENCL_COMPLETE="$opencl_ready" ROOT_DIR="$root_dir" \
    DEVICE_LIB_DIR="$device_lib_dir" COMPILED_HSACO="$compiled_hsaco" \
    LOCKED_HSACO="$installed_locked_hsaco" LOCKED_EXPECTED_SHA="$locked_hsaco_sha" \
    RUNTIME_LIBRARY="$runtime_library" RUNTIME_ENDPOINT="$runtime_endpoint" \
    OPENCL_LIBRARY="$opencl_library" OPENCL_EXECUTABLE="$opencl_executable" \
    OPENCL_SOURCE="$opencl_source" GEM5_BINARY="$gem5_binary" \
    GEM5_CONFIG="$gem5_config" \
    /usr/bin/python3 -I <<'PY'
import hashlib
import json
import os
from pathlib import Path

def boolean(name):
    return os.environ[name] == "true"

def artifact(path):
    candidate = Path(path) if path else None
    if candidate is None or not candidate.is_file():
        return {"path": "", "sha256": ""}
    return {
        "path": str(candidate),
        "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    }

prefix = Path(os.environ["PREFIX"])
compiler = boolean("COMPILER_COMPLETE")
runtime = boolean("RUNTIME_COMPLETE")
opencl = boolean("OPENCL_COMPLETE")
compiled = artifact(os.environ["COMPILED_HSACO"] if compiler else "")
locked_expected = os.environ["LOCKED_EXPECTED_SHA"]
artifacts = {
    "activation": artifact(str(prefix / "activate")),
    "compiled_hsaco": compiled,
    "locked_executable_hsaco": artifact(os.environ["LOCKED_HSACO"] if compiler else ""),
    "runtime_library": artifact(os.environ["RUNTIME_LIBRARY"] if runtime else ""),
    "runtime_endpoint": artifact(os.environ["RUNTIME_ENDPOINT"] if runtime else ""),
    "runtime_handshake": artifact(str(prefix / "bin" / "sagr-handshake") if runtime else ""),
    "runtime_triton_probe": artifact(
        str(prefix / "bin" / "sagr-triton-hsaco-probe") if runtime else ""
    ),
    "opencl_library": artifact(os.environ["OPENCL_LIBRARY"] if opencl else ""),
    "opencl_executable": artifact(os.environ["OPENCL_EXECUTABLE"] if opencl else ""),
    "opencl_source": artifact(os.environ["OPENCL_SOURCE"] if opencl else ""),
}
if compiler:
    for candidate in sorted(Path(os.environ["DEVICE_LIB_DIR"]).glob("*.bc")):
        artifacts["device_bc:" + candidate.name] = artifact(str(candidate))
for tool in ("clang", "clang++", "lld", "ld.lld", "llvm-link", "llvm-objdump", "opt", "FileCheck"):
    key = "tool_" + tool.replace("+", "x").replace(".", "_").replace("-", "_")
    artifacts[key] = artifact(str(prefix / "bin" / tool) if compiler else "")
artifacts["compiled_hsaco"]["accepted_by_current_daemon"] = bool(
    compiled["sha256"] and compiled["sha256"] == locked_expected
)
payload = {
    "schema": "amdgpu-sim.rocm-prefix.v5",
    "setup_schema": int(os.environ["SETUP_SCHEMA"]),
    "setup_script_sha256": os.environ["SETUP_SHA"],
    "prefix": str(prefix),
    "requested_mode": os.environ["REQUESTED_MODE"],
    "sources": {
        "llvm-project": {"head": os.environ["LLVM_HEAD"], "tree": os.environ["LLVM_TREE"]},
        "rocm-systems": {"head": os.environ["ROCM_HEAD"], "tree": os.environ["ROCM_TREE"]},
        "gem5": {"head": os.environ["GEM5_HEAD"], "tree": os.environ["GEM5_TREE"]},
        "self-amdgpu-runtime": {"head": os.environ["RUNTIME_HEAD"], "tree": os.environ["RUNTIME_TREE"]},
    },
    "components": {
        "compiler": compiler,
        "device_libs": compiler,
        "runtime": runtime,
        "opencl": opencl,
    },
    "artifacts": artifacts,
    "managed_inputs": {
        "gem5_binary": artifact(os.environ["GEM5_BINARY"]),
        "gem5_config": artifact(os.environ["GEM5_CONFIG"]),
    },
    "locked_hsaco_sha256": locked_expected,
    "system_rocm_install": False,
    "production_umd_kmd": False,
}
path = Path(os.environ["MANIFEST"])
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.chmod(0o444)
temporary.replace(path)
PY
}

verify_manifest() {
    local expect=${1:-manifest} runtime_library
    runtime_library=$(find_runtime_library)
    EXPECT="$expect" MANIFEST="$manifest" PREFIX="$prefix" \
    SETUP_SCHEMA="$setup_schema" SETUP_SHA="$setup_sha" \
    LLVM_HEAD="$llvm_head_expected" LLVM_TREE="$llvm_tree_expected" \
    ROCM_HEAD="$rocm_head_expected" ROCM_TREE="$rocm_tree_expected" \
    GEM5_HEAD="$gem5_head_expected" GEM5_TREE="$gem5_tree_expected" \
    RUNTIME_HEAD="$runtime_head_expected" RUNTIME_TREE="$runtime_tree_expected" \
    LOCKED_SHA="$locked_hsaco_sha" DEVICE_LIB_DIR="$device_lib_dir" \
    COMPILED_HSACO="$compiled_hsaco" LOCKED_HSACO="$installed_locked_hsaco" \
    RUNTIME_LIBRARY="$runtime_library" RUNTIME_ENDPOINT="$runtime_endpoint" \
    OPENCL_LIBRARY="$opencl_library" OPENCL_EXECUTABLE="$opencl_executable" \
    OPENCL_SOURCE="$opencl_source" GEM5_BINARY="$gem5_binary" \
    GEM5_CONFIG="$gem5_config" \
    /usr/bin/python3 -I <<'PY'
import hashlib
import json
import os
from pathlib import Path

def require(condition, message):
    if not condition:
        raise SystemExit(f"manifest verification failed: {message}")

prefix = Path(os.environ["PREFIX"]).resolve()
manifest_path = Path(os.environ["MANIFEST"])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_sources = {
    "llvm-project": {"head": os.environ["LLVM_HEAD"], "tree": os.environ["LLVM_TREE"]},
    "rocm-systems": {"head": os.environ["ROCM_HEAD"], "tree": os.environ["ROCM_TREE"]},
    "gem5": {"head": os.environ["GEM5_HEAD"], "tree": os.environ["GEM5_TREE"]},
    "self-amdgpu-runtime": {"head": os.environ["RUNTIME_HEAD"], "tree": os.environ["RUNTIME_TREE"]},
}
require(data.get("schema") == "amdgpu-sim.rocm-prefix.v5", "schema")
require(data.get("setup_schema") == int(os.environ["SETUP_SCHEMA"]), "setup schema")
require(data.get("setup_script_sha256") == os.environ["SETUP_SHA"], "setup script hash")
require(data.get("prefix") == str(prefix), "prefix")
require(data.get("sources") == expected_sources, "source identities")
require(data.get("locked_hsaco_sha256") == os.environ["LOCKED_SHA"], "locked hash")
require(data.get("system_rocm_install") is False, "system install boundary")
require(data.get("production_umd_kmd") is False, "production runtime boundary")
requested_mode = data.get("requested_mode")
require(requested_mode in {"compiler", "runtime", "all"}, "requested mode")
expect = requested_mode if os.environ["EXPECT"] == "manifest" else os.environ["EXPECT"]
components = data.get("components", {})
require(
    set(components) == {"compiler", "device_libs", "runtime", "opencl"},
    "component keys",
)
require(components["compiler"] is components["device_libs"], "compiler/device-lib parity")
require(not components["opencl"] or components["runtime"], "OpenCL/runtime dependency")
if expect in {"compiler", "all"}:
    require(components["compiler"] is True, "compiler component")
if expect in {"runtime", "all"}:
    require(components["runtime"] is True, "runtime component")
    require(components["opencl"] is True, "OpenCL component")
require(any(components.values()), "empty profile")

expected = {"activation": str(prefix / "activate")}
compiler_paths = {
    "compiled_hsaco": os.environ["COMPILED_HSACO"],
    "locked_executable_hsaco": os.environ["LOCKED_HSACO"],
}
if components["compiler"]:
    for candidate in sorted(Path(os.environ["DEVICE_LIB_DIR"]).glob("*.bc")):
        compiler_paths["device_bc:" + candidate.name] = str(candidate)
for tool in ("clang", "clang++", "lld", "ld.lld", "llvm-link", "llvm-objdump", "opt", "FileCheck"):
    key = "tool_" + tool.replace("+", "x").replace(".", "_").replace("-", "_")
    compiler_paths[key] = str(prefix / "bin" / tool)
runtime_paths = {
    "runtime_library": os.environ["RUNTIME_LIBRARY"],
    "runtime_endpoint": os.environ["RUNTIME_ENDPOINT"],
    "runtime_handshake": str(prefix / "bin" / "sagr-handshake"),
    "runtime_triton_probe": str(prefix / "bin" / "sagr-triton-hsaco-probe"),
}
opencl_paths = {
    "opencl_library": os.environ["OPENCL_LIBRARY"],
    "opencl_executable": os.environ["OPENCL_EXECUTABLE"],
    "opencl_source": os.environ["OPENCL_SOURCE"],
}
expected.update({key: value if components["compiler"] else "" for key, value in compiler_paths.items()})
expected.update({key: value if components["runtime"] else "" for key, value in runtime_paths.items()})
expected.update({key: value if components["opencl"] else "" for key, value in opencl_paths.items()})
artifacts = data.get("artifacts", {})
require(set(artifacts) == set(expected), "artifact key set")
for name, expected_path in expected.items():
    artifact = artifacts[name]
    require(isinstance(artifact, dict), f"{name} descriptor")
    actual_path = artifact.get("path")
    digest = artifact.get("sha256")
    require(actual_path == expected_path, f"{name} path")
    if not expected_path:
        require(digest == "", f"{name} empty digest")
        continue
    candidate = Path(actual_path)
    require(candidate.is_file(), f"{name} file")
    try:
        candidate.resolve().relative_to(prefix)
    except ValueError:
        require(False, f"{name} escapes prefix")
    require(hashlib.sha256(candidate.read_bytes()).hexdigest() == digest, f"{name} digest")
compiled = artifacts["compiled_hsaco"]
require(
    compiled.get("accepted_by_current_daemon")
    is bool(compiled["sha256"] and compiled["sha256"] == os.environ["LOCKED_SHA"]),
    "compiled acceptance bit",
)
if components["compiler"]:
    require(
        artifacts["locked_executable_hsaco"]["sha256"] == os.environ["LOCKED_SHA"],
        "installed locked image",
    )
managed_expected = {
    "gem5_binary": os.environ["GEM5_BINARY"],
    "gem5_config": os.environ["GEM5_CONFIG"],
}
managed = data.get("managed_inputs", {})
require(set(managed) == set(managed_expected), "managed input key set")
for name, expected_path in managed_expected.items():
    descriptor = managed[name]
    require(descriptor.get("path") == expected_path, f"{name} path")
    candidate = Path(expected_path)
    require(candidate.is_file(), f"{name} file")
    require(
        hashlib.sha256(candidate.read_bytes()).hexdigest()
        == descriptor.get("sha256"),
        f"{name} digest",
    )
PY
}

verify_artifacts() {
    local expect=${1:-manifest} tool resource_dir
    verify_manifest "$expect"
    if [[ "$expect" == compiler || "$expect" == all \
        || -x "$prefix/bin/clang" ]]; then
        for tool in clang clang++ lld ld.lld llvm-link llvm-objdump opt FileCheck; do
            [[ -x "$prefix/bin/$tool" ]] || {
                printf 'missing local compiler tool: %s\n' "$prefix/bin/$tool" >&2
                return 1
            }
        done
        resource_dir=$(run_clean "$prefix/bin/clang" -print-resource-dir)
        [[ "$resource_dir" == "$prefix/"* && -f "$resource_dir/include/opencl-c.h" ]] || {
            printf 'invalid local Clang resource directory: %s\n' "$resource_dir" >&2
            return 1
        }
        assert_device_libs
        [[ -f "$compiled_hsaco" && -f "$installed_locked_hsaco" ]] || {
            printf '%s\n' 'compiler/device-lib/kernel component is incomplete' >&2
            return 1
        }
        /usr/bin/readelf -h "$compiled_hsaco" >/dev/null
        /usr/bin/readelf -h "$installed_locked_hsaco" >/dev/null
    fi
    if [[ "$expect" == runtime || "$expect" == all \
        || -x "$prefix/bin/sagr-handshake" ]]; then
        [[ -n "$(find_runtime_library)" && -x "$prefix/bin/sagr-handshake" \
            && -x "$prefix/bin/sagr-triton-hsaco-probe" \
            && -x "$runtime_endpoint" ]] && opencl_complete || {
            printf '%s\n' \
                'self-amdgpu-runtime/OpenCL product component is incomplete' >&2
            return 1
        }
    fi
}

run_opencl_smoke() {
    local smoke_dir stdout stderr
    smoke_dir=$(/usr/bin/mktemp -d "$prefix/tmp/opencl-direct.XXXXXX")
    chmod 0700 "$smoke_dir"
    stdout="$smoke_dir/stdout.jsonl"
    stderr="$smoke_dir/stderr.log"
    env -i HOME="$smoke_dir" PATH=/usr/bin:/bin LC_ALL=C \
        "$opencl_executable" "$opencl_source" >"$stdout" 2>"$stderr"
    STDOUT_PATH="$stdout" /usr/bin/python3 -I <<'PY'
import json
import os
from pathlib import Path

lines = Path(os.environ["STDOUT_PATH"]).read_text(encoding="utf-8").splitlines()
if not lines:
    raise SystemExit("OpenCL direct smoke produced no result")
result = json.loads(lines[-1])
required = {
    "schema": "self-amdgpu-runtime.opencl-vecadd.v1",
    "source_compiled": True,
    "gem5_execution": True,
    "output_correct": True,
    "mismatch_count": 0,
    "fallback_count": 0,
    "status": 0,
    "stage": "complete",
}
for key, expected in required.items():
    if result.get(key) != expected:
        raise SystemExit(
            f"OpenCL direct smoke mismatch: {key}={result.get(key)!r}, "
            f"expected {expected!r}"
        )
PY
    printf 'OpenCL direct executable verified: %s\n' "$opencl_executable"
    tail -n 1 "$stdout"
}

verify() {
    local expect=${1:-manifest}
    assert_source_locks
    assert_host_tools
    [[ -f "$manifest" ]] || {
        printf 'missing prefix manifest: %s\n' "$manifest" >&2
        return 1
    }
    [[ -f "$prefix/activate" ]] || {
        printf 'missing activation file: %s\n' "$prefix/activate" >&2
        return 1
    }
    assert_clean_prefix
    verify_artifacts "$expect"
    local source_hsaco_sha
    source_hsaco_sha=$(/usr/bin/sha256sum "$locked_hsaco" | awk '{print $1}')
    [[ "$source_hsaco_sha" == "$locked_hsaco_sha" ]] || {
        printf 'locked source HSACO hash mismatch: %s\n' "$source_hsaco_sha" >&2
        return 1
    }
    assert_source_locks
    printf 'ROCm simulator prefix verified: %s\n' "$prefix"
    printf 'source locks: llvm=%s rocm-systems=%s gem5=%s runtime=%s\n' \
        "$(git_head "$llvm_source")" "$(git_head "$rocm_source")" \
        "$(git_head "$gem5_source")" \
        "$(git_head "$runtime_source")"
    printf '%s\n' 'system install touched: no (/opt/rocm is never a target)'
}

assert_source_locks
assert_host_tools

if [[ "$mode" == verify ]]; then
    [[ -d "$prefix" ]] || {
        printf 'ROCm simulator prefix does not exist: %s\n' "$prefix" >&2
        exit 1
    }
    exec 9>"$prefix/.setup.lock"
    /usr/bin/flock -s -n 9 || {
        printf 'prefix is being modified by another setup process: %s\n' \
            "$prefix" >&2
        exit 1
    }
    verify manifest
    exit 0
fi

mkdir -p "$rocm_root" "$prefix" "$prefix/build" "$prefix/bin" \
    "$prefix/home" "$prefix/tmp" "$prefix/cache" "$prefix/config" "$prefix/data"
exec 9>"$prefix/.setup.lock"
/usr/bin/flock -n 9 || {
    printf 'another setup process owns prefix: %s\n' "$prefix" >&2
    exit 1
}
if [[ -f "$setup_stamp" ]]; then
    [[ "$(<"$setup_stamp")" == "$setup_sha" ]] || {
        printf '%s\n' 'setup script changed for this prefix; choose a new versioned prefix' >&2
        exit 1
    }
else
    printf '%s\n' "$setup_sha" > "$setup_stamp"
    chmod 0444 "$setup_stamp"
fi

case "$mode" in
    compiler)
        build_llvm
        build_device_libs
        compile_kernel
        write_activation
        write_manifest
        verify compiler
        ;;
    runtime)
        build_runtime
        write_activation
        write_manifest
        verify runtime
        run_opencl_smoke
        ;;
    all)
        build_llvm
        build_device_libs
        compile_kernel
        build_runtime
        write_activation
        write_manifest
        verify all
        run_opencl_smoke
        ;;
esac
