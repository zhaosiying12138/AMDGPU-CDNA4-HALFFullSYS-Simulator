#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

requested_prefix=${ROCM_PREFIX-}
requested_jobs=${ROCM_JOBS-}
for bootstrap_variable in ${!GIT_@}; do
    unset "$bootstrap_variable"
done
unset bootstrap_variable TAR_OPTIONS
unset BASH_ENV ENV CDPATH GLOBIGNORE PYTHONHOME PYTHONPATH PYTHONOPTIMIZE
unset LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
unset PIP_CONFIG_FILE PIP_TARGET PIP_PREFIX PIP_USER PIP_REQUIRE_VIRTUALENV
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST PIP_CACHE_DIR
unset -f git find grep awk install chmod mkdir dirname basename 2>/dev/null || true
export PATH=/usr/bin:/bin
export LC_ALL=C
IFS=$' \t\n'
umask 022

root_dir=$(cd "$(/usr/bin/dirname "$0")/.." && pwd -P)
llvm_source="$root_dir/projects/llvm-project"
device_source="$llvm_source/amd/device-libs"
rocm_source="$root_dir/projects/rocm-systems"
rocm_json_source="$rocm_source/projects/rocprofiler/plugin/json/json"
runtime_source="$root_dir/projects/self-amdgpu-runtime"
gem5_source="$root_dir/projects/gem5"
triton_source="$root_dir/projects/triton"
triton_plugin_source="$root_dir/plugins/triton/gemsim_amd"
triton_example="$root_dir/examples/triton/vecadd_correctness.py"
gem5_binary="$gem5_source/build/VEGA_X86/gem5.opt"
gem5_config="$gem5_source/configs/example/gemsim/host_dispatch.py"

setup_schema=8
llvm_head_expected=73f2a21fe16b34e35fd0e149564b8664e59da392
llvm_tree_expected=d589480097e8a30fd1df38435ccc9a9fca71f489
triton_llvm_head_expected=b010a18d2b648cab83c83967ff26b8fde11acdc6
triton_llvm_tree_expected=1cabfeea0f09b6983c8bb92f4161910fe2b7dca6
rocm_head_expected=92115a2941982a384de161be3f78cf9bff547027
rocm_tree_expected=28bf42b65f7aad25167180543dda69b5fc6caf58
triton_head_expected=cd513e2798db0f4675b3d1205c8e76eb3381a0b3
triton_tree_expected=944754ed44b5414f2b72fed267455abc9f6fc8c1
gem5_head_expected=82eab7b4888c2b414031ceaaa5fe142263cd3d90
gem5_tree_expected=27b224d5a4dcee75366b013785723f395a937855
runtime_head_expected=f9ef490668eb6b18bd72805cab2d2cb9140a6782
runtime_tree_expected=f34c4fb9e65a42d1c6cbfcd3d40cbc98e5df020b
locked_hsaco_sha=7b6a4d2bb7f9c4e7466bcf69f3110ecbfab54d07abd4c70b6bd96b6a6fb9de56
locked_hsaco="$rocm_source/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/gpuReadWrite_kernels.hsaco"
kernel_source="$rocm_source/projects/rdc/rdc_libs/rdc_modules/kernels/gpuReadWrite_kernels.cl"
triton_hsaco_sha=7308427e69dea6f320178c55863291d4d615338eb295a422a5ff7a2c2b8afa95
triton_example_sha=9523cc1553af670d1373a9307332e15c816050ecef975c7b4fbc29a0157ec84d
triton_json_sha_expected=7e6d24579f62d47e5998ead5c7989c115a7c18244164eca05fcd551ec0e9096c
python_dev_package=libpython3.14-dev
python_dev_version=3.14.4-1ubuntu0.1
python_dev_arch=amd64
python_dev_deb_name=libpython3.14-dev_3.14.4-1ubuntu0.1_amd64.deb
python_dev_deb_size=5987052
python_dev_deb_sha=3799b0822709454f33c89005a698d6adba811aeb31f2f468976f9c126e964e13
python_dev_deb_url=https://security.ubuntu.com/ubuntu/pool/main/p/python3.14/libpython3.14-dev_3.14.4-1ubuntu0.1_amd64.deb
python_dev_version_expected=3.14.4
python_dev_executable_sha=b8d8288faefdd300201f43fcf00f6f539a27218eeed3a3dff5ab10b9c4c99700
python_dev_python_h_sha=bc6e1b01ec1a37da58b81107effec928a6000186a6b93a8f9a9654f62aff5981
python_dev_patchlevel_sha=a73d192cc3e7a97d39f28933feac2f7e1be1962f1c2a682fe7ee2f6cc2dd4bee
python_dev_generic_pyconfig_sha=ca0e3a26b5f4dbaa14ef8cf62db1e17291045d4f92b4303625a47a1003b8b93c
python_dev_arch_pyconfig_sha=eec0c50c4157985d62d12230b4eca5bda4054a81225f2448db1da723d274c025
python_dev_include_tree_sha=35439d9b8d42e1a77f58aa5150f457e399055191d2735a6e2e66666102f36b54
python_dev_root_tree_sha=206c8d2af1c420abbe018aee7f030855d30ed1eff987591e35fc307f60ac5e35
pip_wheel_name=pip-26.1.2-py3-none-any.whl
pip_wheel_sha=382ff9f685ee3bc25864f820aa50505825f10f5458ffff07e30a6d96e5715cab
pip_wheel_url=https://files.pythonhosted.org/packages/5d/95/6b5cb3461ea5673ba0995989746db58eb18b91b54dbf331e72f569540946/pip-26.1.2-py3-none-any.whl
python_wheels=(
    'filelock==3.29.0|filelock-3.29.0-py3-none-any.whl|96f5f6344709aa1572bbf631c640e4ebeeb519e08da902c39a001882f30ac258|https://files.pythonhosted.org/packages/81/47/dd9a212ef6e343a6857485ffe25bba537304f1913bdbed446a23f7f592e1/filelock-3.29.0-py3-none-any.whl'
    'fsspec==2026.4.0|fsspec-2026.4.0-py3-none-any.whl|11ef7bb35dab8a394fde6e608221d5cf3e8499401c249bebaeaad760a1a8dec2|https://files.pythonhosted.org/packages/d5/0c/043d5e551459da400957a1395e0febbf771446ff34291afcbe3d8be2a279/fsspec-2026.4.0-py3-none-any.whl'
    'jinja2==3.1.6|jinja2-3.1.6-py3-none-any.whl|85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67|https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl'
    'markupsafe==3.0.3|markupsafe-3.0.3-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl|457a69a9577064c05a97c41f4e65148652db078a3a509039e64d3467b9e7ef97|https://files.pythonhosted.org/packages/41/3c/a36c2450754618e62008bf7435ccb0f88053e07592e6028a34776213d877/markupsafe-3.0.3-cp314-cp314-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl'
    'mpmath==1.3.0|mpmath-1.3.0-py3-none-any.whl|a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c|https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl'
    'nanobind==2.10.2|nanobind-2.10.2-py3-none-any.whl|6976c1b04b90481d2612b346485a3063818c6faa5077fe9d8bbc9b5fbe29c380|https://files.pythonhosted.org/packages/14/06/cb08965f985a5e1b9cb55ed96337c1f6daaa6b9cbdaeabe6bb3f7a1a11df/nanobind-2.10.2-py3-none-any.whl'
    'networkx==3.6.1|networkx-3.6.1-py3-none-any.whl|d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762|https://files.pythonhosted.org/packages/9e/c9/b2622292ea83fbb4ec318f5b9ab867d0a28ab43c5717bb85b0a5f6b3b0a4/networkx-3.6.1-py3-none-any.whl'
    'numpy==2.4.3|numpy-2.4.3-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl|679f2a834bae9020f81534671c56fd0cc76dd7e5182f57131478e23d0dc59e24|https://files.pythonhosted.org/packages/a9/7e/4f120ecc54ba26ddf3dc348eeb9eb063f421de65c05fc961941798feea18/numpy-2.4.3-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
    'packaging==26.3|packaging-26.3-py3-none-any.whl|d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c|https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl'
    'safetensors==0.8.0|safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl|fd6f3f93c9a0a7cc2788ee63fb763353d4bd2e89b0751bc78fcf7dda00bea774|https://files.pythonhosted.org/packages/28/50/f203ff3a3ddfe19308efc83c5a3a29ed02bf786732ec35e68bf9162f3365/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
    'setuptools==78.1.0|setuptools-78.1.0-py3-none-any.whl|3e386e96793c8702ae83d17b853fb93d3e09ef82ec62722e61da5cd22376dcd8|https://files.pythonhosted.org/packages/54/21/f43f0a1fa8b06b32812e0975981f4677d28e0f3271601dc88ac5a5b83220/setuptools-78.1.0-py3-none-any.whl'
    'sympy==1.14.0|sympy-1.14.0-py3-none-any.whl|e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5|https://files.pythonhosted.org/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl'
    'torch==2.13.0+cpu|torch-2.13.0+cpu-cp314-cp314-manylinux_2_28_x86_64.whl|d20fa53ee744502fa4c69818a720b05ca0d37abd055d4f6e66cae155114bc691|https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp314-cp314-manylinux_2_28_x86_64.whl'
    'typing-extensions==4.15.0|typing_extensions-4.15.0-py3-none-any.whl|f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548|https://files.pythonhosted.org/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl'
    'wheel==0.46.3|wheel-0.46.3-py3-none-any.whl|4b399d56c9d9338230118d705d9737a2a468ccca63d5e813e2a4fc7815d8bc4d|https://files.pythonhosted.org/packages/87/22/b76d483683216dde3d67cba61fb2444be8d5be289bf628c13fc0fd90e5f9/wheel-0.46.3-py3-none-any.whl'
)

# Keep the profile path neutral: the legacy dependency audit scans ldd output
# for simulator library names, including path components.
profile_name="gfx950-v${setup_schema}-llvm-${llvm_head_expected:0:12}-rocm-${rocm_head_expected:0:12}-triton-${triton_head_expected:0:12}-tllvm-${triton_llvm_head_expected:0:12}-sim-${gem5_head_expected:0:12}-runtime-${runtime_head_expected:0:12}"
rocm_root=$(/usr/bin/realpath -m -- "$root_dir/env/rocm")
prefix="$rocm_root/$profile_name"
[[ -n "$requested_prefix" ]] && prefix="$requested_prefix"
jobs=${requested_jobs:-$(/usr/bin/getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4')}
mode=verify

# Product commands are isolated from the schema-8 builder below. The default
# base profile remains its read-only compiler/Python/Triton foundation.
case "${1-}" in
    --print-base-prefix)
        (($# == 1)) || { printf '%s\n' '--print-base-prefix takes no arguments' >&2; exit 2; }
        printf '%s\n' "$prefix"
        exit 0
        ;;
    --print-prefix|--freeze-product|--product-runtime|--verify-product)
        product_mode=$1
        shift
        exec /usr/bin/python3 -I "$root_dir/tools/product_environment.py" \
            "$product_mode" --root "$root_dir" --base-prefix "$prefix" "$@"
        ;;
esac

usage() {
    cat >&2 <<'EOF'
usage: scripts/setup_rocm_env.sh [--verify-only|--all|--compiler|--runtime|--triton]
       [--print-base-prefix|--print-prefix|--freeze-product|--product-runtime|--verify-product]
       [--prefix PATH] [--jobs N]

The default is --verify-only. --all builds the pinned LLVM/Clang/lld,
device-libs, shared self-amdgpu-runtime, local libOpenCL, a CPU-only Python
venv, pinned Triton/LLVM, and the external gemsim_amd backend under a
repository-local env/rocm prefix. It never installs into /opt/rocm.
EOF
}

while (($#)); do
    case "$1" in
        --verify-only) mode=verify; shift ;;
        --all) mode=all; shift ;;
        --compiler) mode=compiler; shift ;;
        --runtime) mode=runtime; shift ;;
        --triton) mode=triton; shift ;;
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
    "$runtime_source" "$gem5_source" "$triton_source" \
    "$triton_plugin_source" "$triton_example" "$gem5_binary" \
    "$gem5_config" "$kernel_source" "$locked_hsaco"; do
    [[ -e "$required" ]] || {
        printf 'missing required source: %s\n' "$required" >&2
        exit 1
    }
done

git_head() {
    /usr/bin/git -C "$1" rev-parse HEAD
}

git_tree() {
    /usr/bin/git -C "$1" rev-parse 'HEAD^{tree}'
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
    dirty=$(/usr/bin/git -C "$repo" status --porcelain=v1 --untracked-files=all)
    [[ -z "$dirty" ]] || {
        printf '%s source tree is dirty:\n%s\n' "$label" "$dirty" >&2
        return 1
    }
}

assert_source_locks() {
    local json_sha
    assert_repo_lock llvm-project "$llvm_source" \
        "$llvm_head_expected" "$llvm_tree_expected"
    assert_repo_lock rocm-systems "$rocm_source" \
        "$rocm_head_expected" "$rocm_tree_expected"
    json_sha=$(tree_archive_sha "$rocm_json_source")
    [[ "$json_sha" == "$triton_json_sha_expected" ]] || {
        printf 'pinned Triton JSON source hash mismatch: got=%s expected=%s\n' \
            "$json_sha" "$triton_json_sha_expected" >&2
        return 1
    }
    assert_repo_lock triton "$triton_source" \
        "$triton_head_expected" "$triton_tree_expected"
    [[ "$(/usr/bin/git -C "$llvm_source" show -s --format='%H %T' \
        "$triton_llvm_head_expected")" == \
        "$triton_llvm_head_expected $triton_llvm_tree_expected" ]] || {
        printf '%s\n' 'pinned Triton LLVM commit/tree is unavailable' >&2
        return 1
    }
    assert_repo_lock gem5 "$gem5_source" \
        "$gem5_head_expected" "$gem5_tree_expected"
    assert_repo_lock self-amdgpu-runtime "$runtime_source" \
        "$runtime_head_expected" "$runtime_tree_expected"
    [[ "$(/usr/bin/sha256sum "$triton_example" | awk '{print $1}')" == \
        "$triton_example_sha" ]] || {
        printf '%s\n' 'Triton vecadd example hash does not match the CP28 lock' >&2
        return 1
    }
}

assert_host_tools() {
    local tool
    for tool in /usr/bin/cmake /usr/bin/ninja /usr/bin/cc /usr/bin/c++ \
        /usr/bin/clang /usr/bin/clang++ \
        /usr/bin/llvm-ar-21 /usr/bin/llvm-ranlib-21 \
        /usr/bin/python3 /usr/bin/sha256sum /usr/bin/readelf /usr/bin/realpath \
        /usr/bin/flock /usr/bin/file /usr/bin/git /usr/bin/tar /usr/bin/curl \
        /usr/bin/dpkg-deb /usr/bin/stat; do
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
    [[ -d "$triton_venv" ]] && roots+=("$triton_venv")
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
    local -a configured_roots=()
    [[ -d "$prefix/build" ]] && configured_roots+=("$prefix/build")
    [[ -d "$prefix/lib" ]] && configured_roots+=("$prefix/lib")
    [[ -d "$prefix/lib64" ]] && configured_roots+=("$prefix/lib64")
    [[ -d "$prefix/libexec" ]] && configured_roots+=("$prefix/libexec")
    forbidden=
    if ((${#configured_roots[@]})); then
        forbidden=$(grep -R -n -I -E \
            '/opt/rocm|/usr/local/cuda|\.triton/manual-llvm|/tmp/cp17|/miniforge3' \
            "${configured_roots[@]}" --include='CMakeCache.txt' \
            --include='*.cmake' --include='*.ninja' \
            --include='compile_commands.json' 2>/dev/null || true)
    fi
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
triton_llvm_source="$prefix/src/llvm-project-$triton_llvm_head_expected"
triton_json_source="$prefix/src/nlohmann-json-$rocm_head_expected"
triton_source_export="$prefix/src/triton-$triton_head_expected"
triton_plugin_export="$triton_source_export/external/gemsim_amd"
triton_source_stamp="$triton_source_export/.amdgpu-sim-source-lock"
triton_llvm_source_stamp="$triton_llvm_source/.amdgpu-sim-source-lock"
triton_json_source_stamp="$triton_json_source/.amdgpu-sim-source-lock"
triton_llvm_build="$prefix/build/triton-llvm-$triton_llvm_head_expected"
triton_llvm_cmake_cache="$triton_llvm_build/CMakeCache.txt"
triton_llvm_compile_commands="$triton_llvm_build/compile_commands.json"
triton_build="$prefix/build/triton-$triton_head_expected"
triton_cmake_cache="$triton_build/cmake/CMakeCache.txt"
triton_venv="$prefix/venv"
triton_python="$triton_venv/bin/python"
triton_cache="$prefix/cache/triton"
triton_wheelhouse="$prefix/cache/wheels"
triton_package_cache="$prefix/cache/packages"
triton_requirements="$prefix/cache/triton-requirements.txt"
python_dev_deb="$triton_package_cache/$python_dev_deb_name"
python_dev_root="$prefix/python-dev/cpython-$python_dev_version_expected-$python_dev_arch"
python_dev_include="$python_dev_root/include/python3.14"
python_dev_stamp="$python_dev_root/.amdgpu-sim-source-lock"
python_dev_python_h="$python_dev_include/Python.h"
python_dev_patchlevel="$python_dev_include/patchlevel.h"
python_dev_pyconfig="$python_dev_include/pyconfig.h"
python_dev_generic_pyconfig="$python_dev_root/provenance/generic-pyconfig.h"
python_dev_arch_pyconfig="$python_dev_root/provenance/x86_64-pyconfig.h"
triton_hsaco="$prefix/kernels/gfx950/triton/add_kernel.hsaco"
triton_smoke_result="$prefix/share/amdgpu-sim/triton/vecadd-result.json"
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
        printf '%s\n' 'for _amdgpu_sim_var in ${!TRITON_@} ${!AMDGCN_@} ${!SAGR_@} ${!CUDA_@} ${!HIP_@} ${!HSA_@} ${!ROCR_@} ${!ROCM_@} ${!PIP_@} ${!CONDA_@} ${!LD_@} ${!CMAKE_@} ${!PKG_CONFIG_@} ${!LLVM_@} ${!MLIR_@} ${!NVPTX_@} ${!PYTHON@} ${!GIT_@}; do'
        printf '%s\n' '    unset "$_amdgpu_sim_var"'
        printf '%s\n' 'done'
        printf '%s\n' 'unset _amdgpu_sim_var BASH_ENV ENV TAR_OPTIONS VIRTUAL_ENV VIRTUAL_ENV_PROMPT _CONDA_EXE _CE_CONDA _CE_M CUDACXX'
        printf '%s\n' 'unset DISABLE_LLVM_OPT DISABLE_MMA_V3 DISABLE_MMA_V5 USE_IR_LOC ALLOW_LHS_TMEM_LAYOUT_CONVERSION'
        printf '%s\n' 'unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH COMPILER_PATH GCC_EXEC_PREFIX'
        printf '%s\n' 'unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS'
        printf 'export ROCM_SIM_ROOT=%q\n' "$prefix"
        printf '%s\n' 'export HOME="$ROCM_SIM_ROOT/home"'
        printf '%s\n' 'export TMPDIR="$ROCM_SIM_ROOT/tmp"'
        printf '%s\n' 'export XDG_CACHE_HOME="$ROCM_SIM_ROOT/cache"'
        printf '%s\n' 'export XDG_CONFIG_HOME="$ROCM_SIM_ROOT/config"'
        printf '%s\n' 'export XDG_DATA_HOME="$ROCM_SIM_ROOT/data"'
        printf '%s\n' 'export ROCM_PATH="$ROCM_SIM_ROOT"'
        printf '%s\n' 'export HIP_PATH="$ROCM_SIM_ROOT"'
        printf '%s\n' 'export HSA_PATH="$ROCM_SIM_ROOT"'
        printf '%s\n' 'export PATH="$ROCM_SIM_ROOT/venv/bin:$ROCM_SIM_ROOT/bin:/usr/bin:/bin"'
        printf '%s\n' 'export LD_LIBRARY_PATH="$ROCM_SIM_ROOT/lib:$ROCM_SIM_ROOT/lib64"'
        printf '%s\n' 'export CMAKE_PREFIX_PATH="$ROCM_SIM_ROOT"'
        printf '%s\n' 'export PKG_CONFIG_PATH="$ROCM_SIM_ROOT/lib/pkgconfig:$ROCM_SIM_ROOT/lib64/pkgconfig"'
        printf 'export LLVM_SYSPATH=%q\n' "$triton_llvm_build"
        printf '%s\n' 'export TRITON_DEFAULT_BACKEND=gemsim_amd'
        printf '%s\n' 'export TRITON_CACHE_DIR="$ROCM_SIM_ROOT/cache/triton"'
        printf '%s\n' 'export VIRTUAL_ENV="$ROCM_SIM_ROOT/venv"'
        printf '%s\n' 'export PYTHONNOUSERSITE=1'
        printf '%s\n' 'export CC=/usr/bin/clang'
        printf '%s\n' 'export CXX=/usr/bin/clang++'
        printf '%s\n' 'export AR=/usr/bin/llvm-ar-21'
        printf '%s\n' 'export RANLIB=/usr/bin/llvm-ranlib-21'
        printf '%s\n' 'unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL'
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

download_pinned_artifact() {
    local destination=$1 expected_sha=$2 url=$3 label=$4
    local temporary actual_sha
    mkdir -p "$(/usr/bin/dirname "$destination")"
    if [[ -f "$destination" ]]; then
        actual_sha=$(/usr/bin/sha256sum "$destination" | awk '{print $1}')
        [[ "$actual_sha" == "$expected_sha" ]] || {
            printf 'cached %s hash mismatch: %s got=%s expected=%s\n' \
                "$label" "$destination" "$actual_sha" "$expected_sha" >&2
            return 1
        }
        return 0
    fi
    temporary="$destination.part.$$"
    /usr/bin/curl --proto '=https' --tlsv1.2 --fail --location \
        --retry 3 --output "$temporary" "$url"
    actual_sha=$(/usr/bin/sha256sum "$temporary" | awk '{print $1}')
    [[ "$actual_sha" == "$expected_sha" ]] || {
        printf 'downloaded %s hash mismatch: %s got=%s expected=%s\n' \
            "$label" "$destination" "$actual_sha" "$expected_sha" >&2
        /usr/bin/rm -f -- "$temporary"
        return 1
    }
    chmod 0444 "$temporary"
    mv -f "$temporary" "$destination"
}

download_pinned_wheel() {
    local filename=$1 expected_sha=$2 url=$3
    download_pinned_artifact "$triton_wheelhouse/$filename" \
        "$expected_sha" "$url" wheel
}

python_dev_headers_complete() {
    local actual package version architecture size root_sha include_sha
    local stamp_package stamp_version stamp_arch stamp_size stamp_deb_sha
    local stamp_url stamp_root_sha stamp_include_sha stamp_python_h
    local stamp_generic stamp_arch_config stamp_patchlevel stamp_extra
    [[ -f "$python_dev_deb" && -f "$python_dev_stamp" \
        && -f "$python_dev_python_h" && -f "$python_dev_patchlevel" \
        && -f "$python_dev_pyconfig" && -f "$python_dev_generic_pyconfig" \
        && -f "$python_dev_arch_pyconfig" ]] || return 1
    size=$(/usr/bin/stat -c '%s' "$python_dev_deb")
    actual=$(/usr/bin/sha256sum "$python_dev_deb" | awk '{print $1}')
    package=$(/usr/bin/dpkg-deb -f "$python_dev_deb" Package)
    version=$(/usr/bin/dpkg-deb -f "$python_dev_deb" Version)
    architecture=$(/usr/bin/dpkg-deb -f "$python_dev_deb" Architecture)
    [[ "$size" == "$python_dev_deb_size" && "$actual" == "$python_dev_deb_sha" \
        && "$package" == "$python_dev_package" \
        && "$version" == "$python_dev_version" \
        && "$architecture" == "$python_dev_arch" ]] || return 1
    IFS='|' read -r stamp_package stamp_version stamp_arch stamp_size \
        stamp_deb_sha stamp_url stamp_root_sha stamp_include_sha \
        stamp_python_h stamp_generic stamp_arch_config stamp_patchlevel \
        stamp_extra < "$python_dev_stamp" || return 1
    [[ -z "$stamp_extra" && "$stamp_package" == "$python_dev_package" \
        && "$stamp_version" == "$python_dev_version" \
        && "$stamp_arch" == "$python_dev_arch" \
        && "$stamp_size" == "$python_dev_deb_size" \
        && "$stamp_deb_sha" == "$python_dev_deb_sha" \
        && "$stamp_url" == "$python_dev_deb_url" \
        && "$stamp_root_sha" == "$python_dev_root_tree_sha" \
        && "$stamp_include_sha" == "$python_dev_include_tree_sha" \
        && "$stamp_python_h" == "$python_dev_python_h_sha" \
        && "$stamp_generic" == "$python_dev_generic_pyconfig_sha" \
        && "$stamp_arch_config" == "$python_dev_arch_pyconfig_sha" \
        && "$stamp_patchlevel" == "$python_dev_patchlevel_sha" ]] || return 1
    root_sha=$(tree_archive_sha "$python_dev_root")
    include_sha=$(tree_archive_sha "$python_dev_include")
    [[ "$root_sha" == "$python_dev_root_tree_sha" \
        && "$include_sha" == "$python_dev_include_tree_sha" \
        && "$(/usr/bin/sha256sum "$python_dev_python_h" | awk '{print $1}')" \
            == "$python_dev_python_h_sha" \
        && "$(/usr/bin/sha256sum "$python_dev_patchlevel" | awk '{print $1}')" \
            == "$python_dev_patchlevel_sha" \
        && "$(/usr/bin/sha256sum "$python_dev_pyconfig" | awk '{print $1}')" \
            == "$python_dev_arch_pyconfig_sha" \
        && "$(/usr/bin/sha256sum "$python_dev_generic_pyconfig" | awk '{print $1}')" \
            == "$python_dev_generic_pyconfig_sha" \
        && "$(/usr/bin/sha256sum "$python_dev_arch_pyconfig" | awk '{print $1}')" \
            == "$python_dev_arch_pyconfig_sha" \
        && -z "$(find "$python_dev_root" -type l -print -quit)" ]]
}

prepare_python_development_headers() {
    local extract temporary raw_generic raw_arch actual root_sha include_sha
    mkdir -p "$triton_package_cache" "$prefix/python-dev"
    download_pinned_artifact "$python_dev_deb" "$python_dev_deb_sha" \
        "$python_dev_deb_url" 'Python development package'
    [[ "$(/usr/bin/stat -c '%s' "$python_dev_deb")" == "$python_dev_deb_size" \
        && "$(/usr/bin/dpkg-deb -f "$python_dev_deb" Package)" == \
            "$python_dev_package" \
        && "$(/usr/bin/dpkg-deb -f "$python_dev_deb" Version)" == \
            "$python_dev_version" \
        && "$(/usr/bin/dpkg-deb -f "$python_dev_deb" Architecture)" == \
            "$python_dev_arch" ]] || {
        printf '%s\n' 'private Python development package identity mismatch' >&2
        return 1
    }
    if [[ ! -f "$python_dev_stamp" ]]; then
        [[ ! -e "$python_dev_root" ]] || {
            printf 'incomplete private Python development tree exists: %s\n' \
                "$python_dev_root" >&2
            return 1
        }
        extract=$(/usr/bin/mktemp -d "$prefix/tmp/python-dev-extract.XXXXXX")
        temporary="$prefix/python-dev/.cpython-$python_dev_version_expected-$python_dev_arch.tmp.$$"
        /usr/bin/mkdir -p "$temporary/include/python3.14" \
            "$temporary/provenance"
        /usr/bin/dpkg-deb -x "$python_dev_deb" "$extract"
        raw_generic="$extract/usr/include/python3.14"
        raw_arch="$extract/usr/include/x86_64-linux-gnu/python3.14"
        [[ "$(/usr/bin/sha256sum "$raw_generic/Python.h" | awk '{print $1}')" \
                == "$python_dev_python_h_sha" \
            && "$(/usr/bin/sha256sum "$raw_generic/patchlevel.h" | awk '{print $1}')" \
                == "$python_dev_patchlevel_sha" \
            && "$(/usr/bin/sha256sum "$raw_generic/pyconfig.h" | awk '{print $1}')" \
                == "$python_dev_generic_pyconfig_sha" \
            && "$(/usr/bin/sha256sum "$raw_arch/pyconfig.h" | awk '{print $1}')" \
                == "$python_dev_arch_pyconfig_sha" ]] || {
            printf '%s\n' 'private Python development header digest mismatch' >&2
            return 1
        }
        /usr/bin/cp -a "$raw_generic/." "$temporary/include/python3.14/"
        /usr/bin/install -m 0444 "$raw_generic/pyconfig.h" \
            "$temporary/provenance/generic-pyconfig.h"
        /usr/bin/install -m 0444 "$raw_arch/pyconfig.h" \
            "$temporary/provenance/x86_64-pyconfig.h"
        /usr/bin/install -m 0444 "$raw_arch/pyconfig.h" \
            "$temporary/include/python3.14/pyconfig.h"
        find "$temporary" -type d -exec /usr/bin/chmod 0755 {} +
        find "$temporary" -type f -exec /usr/bin/chmod 0444 {} +
        root_sha=$(tree_archive_sha "$temporary")
        include_sha=$(tree_archive_sha "$temporary/include/python3.14")
        [[ "$root_sha" == "$python_dev_root_tree_sha" \
            && "$include_sha" == "$python_dev_include_tree_sha" ]] || {
            printf 'normalized private Python header tree mismatch: root=%s include=%s\n' \
                "$root_sha" "$include_sha" >&2
            return 1
        }
        printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
            "$python_dev_package" "$python_dev_version" "$python_dev_arch" \
            "$python_dev_deb_size" "$python_dev_deb_sha" "$python_dev_deb_url" \
            "$python_dev_root_tree_sha" "$python_dev_include_tree_sha" \
            "$python_dev_python_h_sha" "$python_dev_generic_pyconfig_sha" \
            "$python_dev_arch_pyconfig_sha" "$python_dev_patchlevel_sha" > \
            "$temporary/.amdgpu-sim-source-lock"
        /usr/bin/chmod 0444 "$temporary/.amdgpu-sim-source-lock"
        /usr/bin/rm -rf -- "$extract"
        mv "$temporary" "$python_dev_root"
    fi
    python_dev_headers_complete || {
        printf '%s\n' 'private Python development headers are incomplete' >&2
        return 1
    }
}

triton_python_identity_complete() {
    local system_python actual
    [[ -x "$triton_python" ]] || return 1
    system_python=$(/usr/bin/realpath /usr/bin/python3)
    [[ "$system_python" == /usr/bin/python3.14 \
        && "$(/usr/bin/sha256sum "$system_python" | awk '{print $1}')" \
            == "$python_dev_executable_sha" \
        && "$(/usr/bin/sha256sum "$triton_python" | awk '{print $1}')" \
            == "$python_dev_executable_sha" ]] || return 1
    actual=$(run_clean "$triton_python" -I -c \
        'import sys, sysconfig; print(".".join(map(str, sys.version_info[:3])) + "|" + str(sysconfig.get_config_var("SOABI")))')
    [[ "$actual" == "$python_dev_version_expected|cpython-314-x86_64-linux-gnu" ]]
}

prepare_triton_python() {
    local record requirement filename sha url temporary
    prepare_python_development_headers
    mkdir -p "$triton_wheelhouse" "$prefix/cache"
    download_pinned_wheel "$pip_wheel_name" "$pip_wheel_sha" "$pip_wheel_url"
    temporary="$triton_requirements.tmp.$$"
    printf 'pip==26.1.2 --hash=sha256:%s\n' "$pip_wheel_sha" > "$temporary"
    for record in "${python_wheels[@]}"; do
        IFS='|' read -r requirement filename sha url <<<"$record"
        download_pinned_wheel "$filename" "$sha" "$url"
        printf '%s --hash=sha256:%s\n' "$requirement" "$sha" >> "$temporary"
    done
    chmod 0444 "$temporary"
    mv -f "$temporary" "$triton_requirements"

    if [[ ! -x "$triton_python" ]]; then
        run_clean /usr/bin/python3 -I -m venv --copies --without-pip \
            "$triton_venv"
    fi
    triton_python_identity_complete || {
        printf '%s\n' 'private Triton Python interpreter identity mismatch' >&2
        return 1
    }
    run_clean "$triton_python" -I -c \
        'import sys; sys.path.insert(0, sys.argv[1]); from pip._internal.cli.main import main; raise SystemExit(main(sys.argv[2:]))' \
        "$triton_wheelhouse/$pip_wheel_name" \
        --isolated install --no-index --no-deps --require-hashes \
        --find-links "$triton_wheelhouse" -r "$triton_requirements"
    run_clean "$triton_python" -I - <<'PY'
from importlib import metadata

expected = {
    "filelock": "3.29.0",
    "fsspec": "2026.4.0",
    "Jinja2": "3.1.6",
    "MarkupSafe": "3.0.3",
    "mpmath": "1.3.0",
    "nanobind": "2.10.2",
    "networkx": "3.6.1",
    "numpy": "2.4.3",
    "packaging": "26.3",
    "pip": "26.1.2",
    "safetensors": "0.8.0",
    "setuptools": "78.1.0",
    "sympy": "1.14.0",
    "torch": "2.13.0+cpu",
    "typing_extensions": "4.15.0",
    "wheel": "0.46.3",
}
for name, version in expected.items():
    actual = metadata.version(name)
    if actual != version:
        raise SystemExit(f"Python package lock mismatch: {name}={actual}, expected {version}")
PY
}

tree_archive_sha() {
    local directory=$1
    /usr/bin/tar --sort=name --mtime='@0' --owner=0 --group=0 \
        --numeric-owner --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='*.pyo' --exclude='.amdgpu-sim-source-lock' \
        -C "$directory" -cf - . | \
        /usr/bin/sha256sum | awk '{print $1}'
}

export_triton_sources() {
    local llvm_temporary json_temporary triton_temporary plugin_sha expected_stamp
    local llvm_export_sha json_export_sha triton_export_sha
    local stamp_head stamp_tree stamp_plugin stamp_example stamp_export stamp_extra
    mkdir -p "$prefix/src"
    if [[ ! -f "$triton_llvm_source_stamp" ]]; then
        [[ ! -e "$triton_llvm_source" ]] || {
            printf 'incomplete Triton LLVM source export exists: %s\n' \
                "$triton_llvm_source" >&2
            return 1
        }
        llvm_temporary="$prefix/src/.llvm-project-$triton_llvm_head_expected.tmp.$$"
        mkdir -p "$llvm_temporary"
        git -C "$llvm_source" archive "$triton_llvm_head_expected" | \
            /usr/bin/tar -x -C "$llvm_temporary"
        llvm_export_sha=$(tree_archive_sha "$llvm_temporary")
        printf '%s %s %s\n' "$triton_llvm_head_expected" \
            "$triton_llvm_tree_expected" "$llvm_export_sha" > \
            "$llvm_temporary/.amdgpu-sim-source-lock"
        chmod 0444 "$llvm_temporary/.amdgpu-sim-source-lock"
        mv "$llvm_temporary" "$triton_llvm_source"
    fi
    read -r stamp_head stamp_tree stamp_export stamp_extra < \
        "$triton_llvm_source_stamp"
    llvm_export_sha=$(tree_archive_sha "$triton_llvm_source")
    [[ -z "$stamp_extra" && "$stamp_head" == "$triton_llvm_head_expected" \
        && "$stamp_tree" == "$triton_llvm_tree_expected" \
        && "$stamp_export" == "$llvm_export_sha" ]] || {
        printf '%s\n' 'Triton LLVM source-export lock mismatch' >&2
        return 1
    }

    if [[ ! -f "$triton_json_source_stamp" ]]; then
        [[ ! -e "$triton_json_source" ]] || {
            printf 'incomplete Triton JSON source export exists: %s\n' \
                "$triton_json_source" >&2
            return 1
        }
        json_temporary="$prefix/src/.nlohmann-json-$rocm_head_expected.tmp.$$"
        mkdir -p "$json_temporary"
        /usr/bin/tar -C "$rocm_json_source" -cf - . | \
            /usr/bin/tar -x -C "$json_temporary"
        json_export_sha=$(tree_archive_sha "$json_temporary")
        [[ "$json_export_sha" == "$triton_json_sha_expected" ]] || {
            printf 'Triton JSON source hash mismatch: got=%s expected=%s\n' \
                "$json_export_sha" "$triton_json_sha_expected" >&2
            return 1
        }
        printf '%s %s %s\n' "$rocm_head_expected" "$rocm_tree_expected" \
            "$json_export_sha" > "$json_temporary/.amdgpu-sim-source-lock"
        chmod 0444 "$json_temporary/.amdgpu-sim-source-lock"
        mv "$json_temporary" "$triton_json_source"
    fi
    read -r stamp_head stamp_tree stamp_export stamp_extra < \
        "$triton_json_source_stamp"
    json_export_sha=$(tree_archive_sha "$triton_json_source")
    [[ -z "$stamp_extra" && "$stamp_head" == "$rocm_head_expected" \
        && "$stamp_tree" == "$rocm_tree_expected" \
        && "$stamp_export" == "$triton_json_sha_expected" \
        && "$json_export_sha" == "$triton_json_sha_expected" ]] || {
        printf '%s\n' 'Triton JSON source-export lock mismatch' >&2
        return 1
    }

    plugin_sha=$(tree_archive_sha "$triton_plugin_source")
    expected_stamp="$triton_head_expected $triton_tree_expected $plugin_sha $triton_example_sha"
    if [[ ! -f "$triton_source_stamp" ]]; then
        [[ ! -e "$triton_source_export" ]] || {
            printf 'incomplete Triton source export exists: %s\n' \
                "$triton_source_export" >&2
            return 1
        }
        triton_temporary="$prefix/src/.triton-$triton_head_expected.tmp.$$"
        mkdir -p "$triton_temporary"
        git -C "$triton_source" archive "$triton_head_expected" | \
            /usr/bin/tar -x -C "$triton_temporary"
        mkdir -p "$triton_temporary/external/gemsim_amd"
        /usr/bin/tar --exclude='__pycache__' --exclude='*.pyc' \
            --exclude='*.pyo' -C "$triton_plugin_source" -cf - . | \
            /usr/bin/tar -x -C "$triton_temporary/external/gemsim_amd"
        TRITON_EXPORT="$triton_temporary" /usr/bin/python3 -I <<'PY'
import os
from pathlib import Path

root = Path(os.environ["TRITON_EXPORT"])

def replace_once(path, old, new):
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one patch site in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")

replace_once(
    root / "CMakeLists.txt",
    "    LLVMPasses\n    LLVMNVPTXCodeGen\n    LLVMAMDGPUCodeGen\n",
    "    LLVMPasses\n    # AMD-only simulator build: NVPTX is intentionally absent.\n"
    "    LLVMAMDGPUCodeGen\n",
)
replace_once(
    root / "CMakeLists.txt",
    "  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})\n"
    "    add_subdirectory(third_party/${CODEGEN_BACKEND})\n"
    "  endforeach()\n\n"
    "  if (TRITON_BUILD_PROTON)\n",
    "  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})\n"
    "    add_subdirectory(third_party/${CODEGEN_BACKEND})\n"
    "  endforeach()\n\n"
    "  # Upstream common instrumentation links NVGPU/NVWS and NVIDIA LLVM\n"
    "  # helpers even in an AMD-only build. Build those compiler-internal\n"
    "  # providers without registering the NVIDIA backend or CUDA runtime.\n"
    "  list(FIND TRITON_CODEGEN_BACKENDS \"nvidia\" TRITON_NVIDIA_BACKEND_INDEX)\n"
    "  if(TRITON_NVIDIA_BACKEND_INDEX EQUAL -1)\n"
    "    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/nvidia/include)\n"
    "    include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/nvidia/include)\n"
    "    add_subdirectory(third_party/nvidia/include/Dialect/NVGPU)\n"
    "    add_subdirectory(third_party/nvidia/lib/Dialect/NVGPU/IR)\n"
    "    add_subdirectory(third_party/nvidia/include/Dialect/NVWS)\n"
    "    add_subdirectory(third_party/nvidia/lib/Dialect/NVWS)\n"
    "    add_subdirectory(third_party/nvidia/include/TritonNVIDIAGPUToLLVM)\n"
    "    add_subdirectory(third_party/nvidia/lib/TritonNVIDIAGPUToLLVM)\n"
    "  endif()\n\n"
    "  if (TRITON_BUILD_PROTON)\n",
)
replace_once(
    root / "CMakeLists.txt",
    "add_subdirectory(bin)\nadd_subdirectory(test)\n",
    "add_subdirectory(bin)\n"
    "if(TRITON_BUILD_PYTHON_MODULE AND TRITON_NVIDIA_BACKEND_INDEX EQUAL -1)\n"
    "  add_dependencies(triton-opt TritonNVIDIAGPUConversionPassIncGen)\n"
    "  add_dependencies(triton-reduce TritonNVIDIAGPUConversionPassIncGen)\n"
    "  add_dependencies(triton-lsp TritonNVIDIAGPUConversionPassIncGen)\n"
    "  add_dependencies(triton-tensor-layout TritonNVIDIAGPUConversionPassIncGen)\n"
    "  foreach(TRITON_BIN_TARGET triton-opt triton-reduce triton-lsp triton-tensor-layout)\n"
    "    target_compile_definitions(${TRITON_BIN_TARGET} PRIVATE\n"
    "      TRITON_BUILD_NVIDIA_BACKEND=0)\n"
    "  endforeach()\n"
    "endif()\n"
    "add_subdirectory(test)\n",
)
register_header = root / "bin/RegisterTritonDialects.h"
replace_once(
    register_header,
    '#pragma once\n#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"\n',
    '#pragma once\n#ifndef TRITON_BUILD_NVIDIA_BACKEND\n'
    '#define TRITON_BUILD_NVIDIA_BACKEND 1\n#endif\n\n'
    '#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"\n',
)
replace_once(
    register_header,
    '#include "nvidia/hopper/include/Transforms/Passes.h"\n'
    '#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"\n'
    '#include "nvidia/include/NVGPUToLLVM/Passes.h"\n'
    '#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"\n',
    '#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"\n'
    '#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"\n'
    '#if TRITON_BUILD_NVIDIA_BACKEND\n'
    '#include "nvidia/hopper/include/Transforms/Passes.h"\n'
    '#include "nvidia/include/NVGPUToLLVM/Passes.h"\n'
    '#endif\n',
)
replace_once(
    register_header,
    '  mlir::triton::registerConvertNVGPUToLLVMPass();\n',
    '#if TRITON_BUILD_NVIDIA_BACKEND\n'
    '  mlir::triton::registerConvertNVGPUToLLVMPass();\n'
    '#endif\n',
)
replace_once(
    register_header,
    '  mlir::registerNVHopperTransformsPasses();\n',
    '#if TRITON_BUILD_NVIDIA_BACKEND\n'
    '  mlir::registerNVHopperTransformsPasses();\n'
    '#endif\n',
)
llvm_cc = root / "python/src/llvm.cc"
replace_once(
    llvm_cc,
    '#include "llvm/Config/llvm-config.h"\n',
    '#include "llvm/Config/Targets.h"\n#include "llvm/Config/llvm-config.h"\n',
)
replace_once(
    llvm_cc,
    "      LLVMInitializeNVPTXTargetInfo();\n"
    "      LLVMInitializeNVPTXTarget();\n"
    "      LLVMInitializeNVPTXTargetMC();\n"
    "      LLVMInitializeNVPTXAsmPrinter();\n",
    "#if LLVM_HAS_NVPTX_TARGET\n"
    "      LLVMInitializeNVPTXTargetInfo();\n"
    "      LLVMInitializeNVPTXTarget();\n"
    "      LLVMInitializeNVPTXTargetMC();\n"
    "      LLVMInitializeNVPTXAsmPrinter();\n"
    "#endif\n",
)
replace_once(
    root / "setup.py",
    'backends = [*BackendInstaller.copy(["nvidia", "amd"]), *BackendInstaller.copy_externals()]',
    'backends = [*BackendInstaller.copy(["amd"]), *BackendInstaller.copy_externals()]',
)
PY
        triton_export_sha=$(tree_archive_sha "$triton_temporary")
        printf '%s %s\n' "$expected_stamp" "$triton_export_sha" > \
            "$triton_temporary/.amdgpu-sim-source-lock"
        chmod 0444 "$triton_temporary/.amdgpu-sim-source-lock"
        mv "$triton_temporary" "$triton_source_export"
    fi
    read -r stamp_head stamp_tree stamp_plugin stamp_example stamp_export \
        stamp_extra < "$triton_source_stamp"
    triton_export_sha=$(tree_archive_sha "$triton_source_export")
    [[ -z "$stamp_extra" && "$stamp_head" == "$triton_head_expected" \
        && "$stamp_tree" == "$triton_tree_expected" \
        && "$stamp_plugin" == "$plugin_sha" \
        && "$stamp_example" == "$triton_example_sha" \
        && "$stamp_export" == "$triton_export_sha" ]] || {
        printf '%s\n' 'Triton source/plugin export lock mismatch' >&2
        return 1
    }
}

triton_source_exports_complete() {
    local plugin_sha exported_plugin_sha llvm_export_sha json_export_sha triton_export_sha
    local stamp_head stamp_tree stamp_plugin stamp_example stamp_export stamp_extra
    [[ -d "$triton_llvm_source" && -f "$triton_llvm_source_stamp" \
        && -d "$triton_json_source" && -f "$triton_json_source_stamp" \
        && -d "$triton_source_export" && -f "$triton_source_stamp" \
        && -d "$triton_plugin_export" ]] || return 1

    read -r stamp_head stamp_tree stamp_export stamp_extra < \
        "$triton_llvm_source_stamp" || return 1
    llvm_export_sha=$(tree_archive_sha "$triton_llvm_source")
    [[ -z "$stamp_extra" && "$stamp_head" == "$triton_llvm_head_expected" \
        && "$stamp_tree" == "$triton_llvm_tree_expected" \
        && "$stamp_export" == "$llvm_export_sha" ]] || return 1

    read -r stamp_head stamp_tree stamp_export stamp_extra < \
        "$triton_json_source_stamp" || return 1
    json_export_sha=$(tree_archive_sha "$triton_json_source")
    [[ -z "$stamp_extra" && "$stamp_head" == "$rocm_head_expected" \
        && "$stamp_tree" == "$rocm_tree_expected" \
        && "$stamp_export" == "$triton_json_sha_expected" \
        && "$json_export_sha" == "$triton_json_sha_expected" ]] || return 1

    plugin_sha=$(tree_archive_sha "$triton_plugin_source")
    exported_plugin_sha=$(tree_archive_sha "$triton_plugin_export")
    [[ "$exported_plugin_sha" == "$plugin_sha" ]] || return 1
    read -r stamp_head stamp_tree stamp_plugin stamp_example stamp_export \
        stamp_extra < "$triton_source_stamp" || return 1
    triton_export_sha=$(tree_archive_sha "$triton_source_export")
    [[ -z "$stamp_extra" && "$stamp_head" == "$triton_head_expected" \
        && "$stamp_tree" == "$triton_tree_expected" \
        && "$stamp_plugin" == "$plugin_sha" \
        && "$stamp_example" == "$triton_example_sha" \
        && "$stamp_export" == "$triton_export_sha" ]]
}

triton_llvm_cache_value() {
    local key=$1
    /usr/bin/awk -F= -v key="$key" \
        'index($0, key ":") == 1 {print substr($0, index($0, "=") + 1); exit}' \
        "$triton_llvm_cmake_cache"
}

assert_triton_llvm_cmake_cache() {
    local key expected actual
    [[ -f "$triton_llvm_cmake_cache" \
        && -f "$triton_llvm_compile_commands" \
        && -f "$triton_llvm_build/build.ninja" ]] || {
        printf 'missing retained Triton LLVM build contract: %s\n' \
            "$triton_llvm_build" >&2
        return 1
    }
    while IFS='|' read -r key expected; do
        actual=$(triton_llvm_cache_value "$key")
        [[ "$actual" == "$expected" ]] || {
            printf 'Triton LLVM CMake cache mismatch: %s=%s expected=%s\n' \
                "$key" "$actual" "$expected" >&2
            return 1
        }
    done <<EOF
CMAKE_C_COMPILER|/usr/bin/clang
CMAKE_CXX_COMPILER|/usr/bin/clang++
CMAKE_LINKER|/usr/bin/ld.lld
CMAKE_MAKE_PROGRAM|/usr/bin/ninja
CMAKE_HOME_DIRECTORY|$triton_llvm_source/llvm
DEFAULT_ROCM_PATH|$prefix
LLVM_APPEND_VC_REV|OFF
LLVM_ENABLE_PROJECTS|mlir;lld;clang
LLVM_TARGETS_TO_BUILD|X86;AMDGPU
LLVM_USE_LINKER|lld
EOF
    /usr/bin/grep -Fq -- '-fuse-ld=lld' "$triton_llvm_build/build.ninja" || {
        printf '%s\n' 'Triton LLVM Ninja graph does not select lld' >&2
        return 1
    }
}

build_triton_llvm() {
    export_triton_sources
    if [[ -f "$triton_llvm_build/lib/cmake/llvm/LLVMConfig.cmake" \
        && -f "$triton_llvm_build/lib/cmake/mlir/MLIRConfig.cmake" \
        && -f "$triton_llvm_build/lib/libMLIROptLib.a" \
        && -f "$triton_llvm_build/lib/libMLIRReduceLib.a" \
        && -f "$triton_llvm_build/lib/libMLIRLspServerLib.a" \
        && -x "$triton_llvm_build/bin/llvm-tblgen" \
        && -x "$triton_llvm_build/bin/mlir-tblgen" ]]; then
        assert_triton_llvm_cmake_cache
        assert_clean_prefix
        return 0
    fi
    mkdir -p "$triton_llvm_build"
    run_clean /usr/bin/cmake -S "$triton_llvm_source/llvm" \
        -B "$triton_llvm_build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/clang \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
        -DCMAKE_LINKER=/usr/bin/ld.lld \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DDEFAULT_ROCM_PATH:PATH="$prefix" \
        -DLLVM_APPEND_VC_REV=OFF \
        -DLLVM_ENABLE_PROJECTS='mlir;lld;clang' \
        -DLLVM_TARGETS_TO_BUILD='X86;AMDGPU' \
        -DLLVM_USE_LINKER=lld \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_BUILD_TESTS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
        -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_CURL=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF -DLLVM_BUILD_LLVM_DYLIB=OFF \
        -DLLVM_LINK_LLVM_DYLIB=OFF \
        -DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON
    assert_triton_llvm_cmake_cache
    assert_clean_prefix
    run_clean /usr/bin/cmake --build "$triton_llvm_build" --parallel "$jobs"
    [[ -f "$triton_llvm_build/lib/cmake/llvm/LLVMConfig.cmake" \
        && -f "$triton_llvm_build/lib/cmake/mlir/MLIRConfig.cmake" \
        && -f "$triton_llvm_build/lib/libMLIROptLib.a" \
        && -f "$triton_llvm_build/lib/libMLIRReduceLib.a" \
        && -f "$triton_llvm_build/lib/libMLIRLspServerLib.a" \
        && -x "$triton_llvm_build/bin/llvm-tblgen" \
        && -x "$triton_llvm_build/bin/mlir-tblgen" ]] || {
        printf '%s\n' 'pinned Triton LLVM build is incomplete' >&2
        return 1
    }
    assert_triton_llvm_cmake_cache
    assert_clean_prefix
}

run_triton_clean() {
    env -i \
        HOME="$prefix/home" TMPDIR="$prefix/tmp" \
        XDG_CACHE_HOME="$prefix/cache" XDG_CONFIG_HOME="$prefix/config" \
        XDG_DATA_HOME="$prefix/data" SOURCE_DATE_EPOCH=0 \
        PATH="$triton_venv/bin:$prefix/bin:/usr/bin:/bin" \
        LD_LIBRARY_PATH="$prefix/lib:$prefix/lib64" \
        ROCM_SIM_ROOT="$prefix" ROCM_PATH="$prefix" HIP_PATH="$prefix" \
        HSA_PATH="$prefix" CMAKE_PREFIX_PATH="$prefix" \
        LLVM_SYSPATH="$triton_llvm_build" JSON_SYSPATH="$triton_json_source" \
        TRITON_PLUGIN_DIRS="$triton_plugin_export" \
        TRITON_DEFAULT_BACKEND=gemsim_amd \
        TRITON_OFFLINE_BUILD=1 TRITON_BUILD_PROTON=OFF \
        TRITON_BUILD_WITH_CCACHE=OFF TRITON_BUILD_WITH_CLANG_LLD=ON \
        TRITON_BUILD_DIR="$triton_build/cmake" \
        TRITON_APPEND_CMAKE_ARGS="-DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_LINKER=/usr/bin/ld.lld -DPython3_EXECUTABLE=$triton_python -DPython3_INCLUDE_DIR=$python_dev_include" \
        TRITON_CACHE_DIR="$triton_cache" \
        PYTHONNOUSERSITE=1 MAX_JOBS="$jobs" \
        CC=/usr/bin/clang CXX=/usr/bin/clang++ \
        AR=/usr/bin/llvm-ar-21 RANLIB=/usr/bin/llvm-ranlib-21 \
        "$@"
}

run_triton_installed_clean() {
    env -i \
        HOME="$prefix/home" TMPDIR="$prefix/tmp" \
        XDG_CACHE_HOME="$prefix/cache" XDG_CONFIG_HOME="$prefix/config" \
        XDG_DATA_HOME="$prefix/data" SOURCE_DATE_EPOCH=0 \
        PATH="$triton_venv/bin:$prefix/bin:/usr/bin:/bin" \
        LD_LIBRARY_PATH="$prefix/lib:$prefix/lib64" \
        ROCM_SIM_ROOT="$prefix" ROCM_PATH="$prefix" HIP_PATH="$prefix" \
        HSA_PATH="$prefix" CMAKE_PREFIX_PATH="$prefix" \
        LLVM_SYSPATH="$triton_llvm_build" \
        TRITON_DEFAULT_BACKEND=gemsim_amd TRITON_OFFLINE_BUILD=1 \
        TRITON_CACHE_DIR="$triton_cache" PYTHONNOUSERSITE=1 MAX_JOBS="$jobs" \
        CC=/usr/bin/clang CXX=/usr/bin/clang++ \
        AR=/usr/bin/llvm-ar-21 RANLIB=/usr/bin/llvm-ranlib-21 \
        "$@"
}

find_triton_library() {
    [[ -d "$triton_venv" ]] || return 0
    find "$triton_venv" -type f -path '*/triton/_C/libtriton.so' \
        -print -quit 2>/dev/null || true
}

find_triton_driver() {
    [[ -d "$triton_venv" ]] || return 0
    find "$triton_venv" -type f \
        -path '*/triton/backends/gemsim_amd/driver.py' \
        -print -quit 2>/dev/null || true
}

find_triton_compiler() {
    [[ -d "$triton_venv" ]] || return 0
    find "$triton_venv" -type f \
        -path '*/triton/backends/gemsim_amd/compiler.py' \
        -print -quit 2>/dev/null || true
}

python_wheels_complete() {
    local record requirement filename sha url actual actual_count
    actual_count=$(find "$triton_wheelhouse" -maxdepth 1 -type f \
        -name '*.whl' -print 2>/dev/null | wc -l)
    [[ "$actual_count" -eq $((${#python_wheels[@]} + 1)) ]] || return 1
    [[ -f "$triton_wheelhouse/$pip_wheel_name" ]] || return 1
    actual=$(/usr/bin/sha256sum "$triton_wheelhouse/$pip_wheel_name" | awk '{print $1}')
    [[ "$actual" == "$pip_wheel_sha" ]] || return 1
    for record in "${python_wheels[@]}"; do
        IFS='|' read -r requirement filename sha url <<<"$record"
        [[ -f "$triton_wheelhouse/$filename" ]] || return 1
        actual=$(/usr/bin/sha256sum "$triton_wheelhouse/$filename" | awk '{print $1}')
        [[ "$actual" == "$sha" ]] || return 1
    done
}

triton_cache_value() {
    local key=$1
    /usr/bin/awk -F= -v key="$key" \
        'index($0, key ":") == 1 {print substr($0, index($0, "=") + 1); exit}' \
        "$triton_cmake_cache"
}

assert_triton_cmake_cache() {
    local key expected actual
    [[ -f "$triton_cmake_cache" ]] || {
        printf 'missing retained Triton CMake cache: %s\n' \
            "$triton_cmake_cache" >&2
        return 1
    }
    while IFS='|' read -r key expected; do
        actual=$(triton_cache_value "$key")
        [[ "$actual" == "$expected" ]] || {
            printf 'Triton CMake cache mismatch: %s=%s expected=%s\n' \
                "$key" "$actual" "$expected" >&2
            return 1
        }
    done <<EOF
CMAKE_C_COMPILER|/usr/bin/clang
CMAKE_CXX_COMPILER|/usr/bin/clang++
CMAKE_LINKER|/usr/bin/ld.lld
CMAKE_MAKE_PROGRAM|/usr/bin/ninja
CMAKE_HOME_DIRECTORY|$triton_build/source
LLVM_SYSPATH|$triton_llvm_build
JSON_SYSPATH|$triton_json_source
Python3_EXECUTABLE|$triton_python
Python3_INCLUDE_DIR|$python_dev_include
TRITON_BUILD_PROTON|OFF
TRITON_CODEGEN_BACKENDS|amd
TRITON_OFFLINE_BUILD|1
EOF
}

triton_complete() {
    local library driver compiler actual_sha dynamic
    triton_source_exports_complete || return 1
    assert_triton_llvm_cmake_cache || return 1
    python_dev_headers_complete || return 1
    triton_python_identity_complete || return 1
    library=$(find_triton_library)
    driver=$(find_triton_driver)
    compiler=$(find_triton_compiler)
    [[ -x "$triton_python" && -n "$library" && -n "$driver" \
        && -n "$compiler" && -f "$triton_requirements" \
        && -f "$triton_hsaco" && -f "$triton_smoke_result" ]] || return 1
    assert_triton_cmake_cache || return 1
    python_wheels_complete || return 1
    dynamic=$(/usr/bin/readelf -d "$library" 2>/dev/null) || return 1
    if grep -Eiq 'libpython|\(RPATH\)|\(RUNPATH\)' <<<"$dynamic"; then
        printf '%s\n' 'Triton library links Python or carries a runtime search path' >&2
        return 1
    fi
    actual_sha=$(/usr/bin/sha256sum "$triton_hsaco" | awk '{print $1}')
    [[ "$actual_sha" == "$triton_hsaco_sha" ]] || return 1
}

build_triton() {
    local triton_library triton_driver build_source
    prepare_triton_python
    export_triton_sources
    build_triton_llvm
    mkdir -p "$triton_build"
    build_source="$triton_build/source"
    /usr/bin/rm -rf -- "$build_source"
    mkdir -p "$build_source"
    /usr/bin/tar -C "$triton_source_export" -cf - . | \
        /usr/bin/tar -x -C "$build_source"
    run_triton_clean "$triton_python" -I -m pip install \
        --isolated --no-index --no-build-isolation --no-deps --force-reinstall \
        "$build_source" || {
        printf 'failed Triton build source retained for diagnosis: %s\n' \
            "$build_source" >&2
        return 1
    }
    assert_triton_cmake_cache || {
        printf 'failed Triton build source retained for diagnosis: %s\n' \
            "$build_source" >&2
        return 1
    }
    /usr/bin/rm -rf -- "$build_source"
    triton_library=$(find_triton_library)
    triton_driver=$(find_triton_driver)
    [[ -n "$triton_library" && -n "$triton_driver" ]] || {
        printf '%s\n' 'Triton or gemsim_amd backend was not installed in the local venv' >&2
        return 1
    }
    assert_clean_prefix
}

run_triton_smoke() {
    local smoke_dir stdout stderr candidate candidate_sha selected= result_temporary
    smoke_dir=$(/usr/bin/mktemp -d "$prefix/tmp/triton-direct.XXXXXX")
    chmod 0700 "$smoke_dir"
    stdout="$smoke_dir/stdout.jsonl"
    stderr="$smoke_dir/stderr.log"
    run_triton_installed_clean "$triton_python" -I "$triton_example" \
        >"$stdout" 2>"$stderr"
    STDOUT_PATH="$stdout" STDERR_PATH="$stderr" /usr/bin/python3 -I <<'PY'
import json
import os
from pathlib import Path

lines = Path(os.environ["STDOUT_PATH"]).read_text(encoding="utf-8").splitlines()
if len(lines) != 1 or not lines[0].strip():
    raise SystemExit(f"Triton direct smoke must produce exactly one JSON line: {lines!r}")
if Path(os.environ["STDERR_PATH"]).stat().st_size != 0:
    raise SystemExit("Triton direct smoke produced stderr diagnostics")
result = json.loads(lines[0])
required = {
    "schema": "amdgpu-sim.triton-vecadd.v1",
    "backend": "gemsim_amd",
    "arch": "gfx950",
    "kernel": "add_kernel",
    "n_elements": 98432,
    "block_size": 1024,
    "program_count": 97,
    "launch_count": 2,
    "reuse": True,
    "output_correct": True,
    "mismatch_count": 0,
    "max_abs_error": 0.0,
    "fallback_count": 0,
}
for key, expected in required.items():
    if result.get(key) != expected:
        raise SystemExit(
            f"Triton direct smoke mismatch: {key}={result.get(key)!r}, "
            f"expected {expected!r}"
        )
launch_results = result.get("launch_results")
if not isinstance(launch_results, list) or len(launch_results) != 2:
    raise SystemExit("Triton direct smoke did not report two launch results")
for index, launch in enumerate(launch_results):
    launch_required = {
        "launch_index": index,
        "seed": index,
        "output_correct": True,
        "mismatch_count": 0,
        "max_abs_error": 0.0,
    }
    for key, expected in launch_required.items():
        if launch.get(key) != expected:
            raise SystemExit(
                f"Triton launch {index} mismatch: {key}={launch.get(key)!r}, "
                f"expected {expected!r}"
            )
PY
    while IFS= read -r candidate; do
        candidate_sha=$(/usr/bin/sha256sum "$candidate" | awk '{print $1}')
        if [[ "$candidate_sha" == "$triton_hsaco_sha" ]]; then
            selected=$candidate
            break
        fi
    done < <(find "$triton_cache" -type f -name 'add_kernel.hsaco' -print | sort)
    [[ -n "$selected" ]] || {
        printf 'normal Python run did not produce locked add_kernel HSACO %s\n' \
            "$triton_hsaco_sha" >&2
        return 1
    }
    mkdir -p "$(/usr/bin/dirname "$triton_hsaco")" \
        "$(/usr/bin/dirname "$triton_smoke_result")"
    install -m 0444 "$selected" "$triton_hsaco"
    result_temporary="$triton_smoke_result.tmp.$$"
    tail -n 1 "$stdout" > "$result_temporary"
    chmod 0444 "$result_temporary"
    mv -f "$result_temporary" "$triton_smoke_result"
    printf 'Triton Python vecadd verified: %s\n' "$triton_example"
    tail -n 1 "$stdout"
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
    local shared
    shared=$(find "$prefix/lib" "$prefix/lib64" -maxdepth 2 -type f \
        -name 'libself_amdgpu_runtime.so*' -print -quit 2>/dev/null || true)
    if [[ -n "$shared" ]]; then
        printf '%s\n' "$shared"
        return 0
    fi
    find "$prefix/lib" "$prefix/lib64" -maxdepth 2 -type f \
        -name 'libself_amdgpu_runtime.a' -print -quit 2>/dev/null || true
}

managed_runtime_complete() {
    local library symbols
    library=$(find_runtime_library)
    [[ -n "$library" ]] || return 1
    symbols=$(/usr/bin/readelf --wide --dyn-syms "$library" 2>/dev/null || true)
    [[ "$symbols" == *sagr_managed_kernel_launch* ]]
}

opencl_complete() {
    [[ -f "$opencl_library" && -x "$opencl_executable" \
        && -f "$opencl_source" && -x "$gem5_binary" \
        && -f "$gem5_config" ]]
}

write_manifest() {
    local runtime_library compiler_complete=false runtime_complete=false
    local opencl_ready=false triton_ready=false triton_library triton_driver
    local triton_compiler triton_plugin_sha
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
        && -x "$runtime_endpoint" ]] && managed_runtime_complete; then
        runtime_complete=true
    fi
    if [[ "$runtime_complete" == true ]] && opencl_complete; then
        opencl_ready=true
    fi
    if triton_complete; then
        triton_ready=true
    fi
    triton_library=$(find_triton_library)
    triton_driver=$(find_triton_driver)
    triton_compiler=$(find_triton_compiler)
    triton_plugin_sha=$(tree_archive_sha "$triton_plugin_source")
    chmod 0644 "$manifest" 2>/dev/null || true
    PREFIX="$prefix" MANIFEST="$manifest" SETUP_SCHEMA="$setup_schema" \
    SETUP_SHA="$setup_sha" REQUESTED_MODE="$mode" \
    LLVM_HEAD="$llvm_head_expected" LLVM_TREE="$llvm_tree_expected" \
    TRITON_LLVM_HEAD="$triton_llvm_head_expected" \
    TRITON_LLVM_TREE="$triton_llvm_tree_expected" \
    ROCM_HEAD="$rocm_head_expected" ROCM_TREE="$rocm_tree_expected" \
    TRITON_HEAD="$triton_head_expected" TRITON_TREE="$triton_tree_expected" \
    GEM5_HEAD="$gem5_head_expected" GEM5_TREE="$gem5_tree_expected" \
    RUNTIME_HEAD="$runtime_head_expected" RUNTIME_TREE="$runtime_tree_expected" \
    COMPILER_COMPLETE="$compiler_complete" RUNTIME_COMPLETE="$runtime_complete" \
    OPENCL_COMPLETE="$opencl_ready" TRITON_COMPLETE="$triton_ready" \
    ROOT_DIR="$root_dir" \
    DEVICE_LIB_DIR="$device_lib_dir" COMPILED_HSACO="$compiled_hsaco" \
    LOCKED_HSACO="$installed_locked_hsaco" LOCKED_EXPECTED_SHA="$locked_hsaco_sha" \
    RUNTIME_LIBRARY="$runtime_library" \
    RUNTIME_SONAME="$prefix/lib/libself_amdgpu_runtime.so.1" \
    RUNTIME_ENDPOINT="$runtime_endpoint" \
    OPENCL_LIBRARY="$opencl_library" OPENCL_EXECUTABLE="$opencl_executable" \
    OPENCL_SOURCE="$opencl_source" GEM5_BINARY="$gem5_binary" \
    GEM5_CONFIG="$gem5_config" TRITON_LIBRARY="$triton_library" \
    TRITON_DRIVER="$triton_driver" TRITON_COMPILER="$triton_compiler" \
    TRITON_PYTHON="$triton_python" TRITON_REQUIREMENTS="$triton_requirements" \
    PYTHON_DEV_DEB="$python_dev_deb" PYTHON_DEV_STAMP="$python_dev_stamp" \
    PYTHON_DEV_PYTHON_H="$python_dev_python_h" \
    PYTHON_DEV_PATCHLEVEL="$python_dev_patchlevel" \
    PYTHON_DEV_PYCONFIG="$python_dev_pyconfig" \
    PYTHON_DEV_GENERIC_PYCONFIG="$python_dev_generic_pyconfig" \
    PYTHON_DEV_ARCH_PYCONFIG="$python_dev_arch_pyconfig" \
    PYTHON_DEV_PACKAGE="$python_dev_package" \
    PYTHON_DEV_VERSION="$python_dev_version" \
    PYTHON_DEV_ARCH="$python_dev_arch" \
    PYTHON_DEV_DEB_NAME="$python_dev_deb_name" \
    PYTHON_DEV_DEB_SIZE="$python_dev_deb_size" \
    PYTHON_DEV_DEB_SHA="$python_dev_deb_sha" \
    PYTHON_DEV_DEB_URL="$python_dev_deb_url" \
    PYTHON_VERSION="$python_dev_version_expected" \
    PYTHON_EXECUTABLE_SHA="$python_dev_executable_sha" \
    PYTHON_INCLUDE_DIR="$python_dev_include" \
    PYTHON_INCLUDE_TREE_SHA="$python_dev_include_tree_sha" \
    PYTHON_ROOT_TREE_SHA="$python_dev_root_tree_sha" \
    TRITON_HSACO="$triton_hsaco" TRITON_HSACO_SHA="$triton_hsaco_sha" \
    TRITON_SMOKE_RESULT="$triton_smoke_result" \
    TRITON_LLVM_CONFIG="$triton_llvm_build/lib/cmake/llvm/LLVMConfig.cmake" \
    TRITON_MLIR_CONFIG="$triton_llvm_build/lib/cmake/mlir/MLIRConfig.cmake" \
    TRITON_LLVM_CMAKE_CACHE="$triton_llvm_cmake_cache" \
    TRITON_LLVM_COMPILE_COMMANDS="$triton_llvm_compile_commands" \
    TRITON_LLVM_DEFAULT_ROCM_PATH="$prefix" \
    TRITON_CMAKE_CACHE="$triton_cmake_cache" \
    TRITON_SOURCE_STAMP="$triton_source_stamp" \
    TRITON_LLVM_SOURCE_STAMP="$triton_llvm_source_stamp" \
    TRITON_JSON_SOURCE_STAMP="$triton_json_source_stamp" \
    TRITON_JSON_SHA="$triton_json_sha_expected" \
    TRITON_WHEELHOUSE="$triton_wheelhouse" \
    TRITON_EXAMPLE="$triton_example" \
    TRITON_EXAMPLE_SHA="$triton_example_sha" \
    TRITON_PLUGIN_SHA="$triton_plugin_sha" \
    TRITON_PLUGIN_CMAKE="$triton_plugin_source/CMakeLists.txt" \
    TRITON_PLUGIN_INIT="$triton_plugin_source/triton_gemsim_amd.cc" \
    TRITON_PLUGIN_NAME="$triton_plugin_source/backend/name.conf" \
    TRITON_PLUGIN_DRIVER="$triton_plugin_source/backend/driver.py" \
    TRITON_PLUGIN_COMPILER="$triton_plugin_source/backend/compiler.py" \
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
triton = boolean("TRITON_COMPLETE")
compiled = artifact(os.environ["COMPILED_HSACO"] if compiler else "")
locked_expected = os.environ["LOCKED_EXPECTED_SHA"]
artifacts = {
    "activation": artifact(str(prefix / "activate")),
    "compiled_hsaco": compiled,
    "locked_executable_hsaco": artifact(os.environ["LOCKED_HSACO"] if compiler else ""),
    "runtime_library": artifact(os.environ["RUNTIME_LIBRARY"] if runtime else ""),
    "runtime_soname": artifact(os.environ["RUNTIME_SONAME"] if runtime else ""),
    "runtime_endpoint": artifact(os.environ["RUNTIME_ENDPOINT"] if runtime else ""),
    "runtime_handshake": artifact(str(prefix / "bin" / "sagr-handshake") if runtime else ""),
    "runtime_triton_probe": artifact(
        str(prefix / "bin" / "sagr-triton-hsaco-probe") if runtime else ""
    ),
    "opencl_library": artifact(os.environ["OPENCL_LIBRARY"] if opencl else ""),
    "opencl_executable": artifact(os.environ["OPENCL_EXECUTABLE"] if opencl else ""),
    "opencl_source": artifact(os.environ["OPENCL_SOURCE"] if opencl else ""),
    "triton_python": artifact(os.environ["TRITON_PYTHON"] if triton else ""),
    "python_dev_deb": artifact(os.environ["PYTHON_DEV_DEB"] if triton else ""),
    "python_dev_stamp": artifact(os.environ["PYTHON_DEV_STAMP"] if triton else ""),
    "python_dev_python_h": artifact(
        os.environ["PYTHON_DEV_PYTHON_H"] if triton else ""
    ),
    "python_dev_patchlevel": artifact(
        os.environ["PYTHON_DEV_PATCHLEVEL"] if triton else ""
    ),
    "python_dev_pyconfig": artifact(
        os.environ["PYTHON_DEV_PYCONFIG"] if triton else ""
    ),
    "python_dev_generic_pyconfig": artifact(
        os.environ["PYTHON_DEV_GENERIC_PYCONFIG"] if triton else ""
    ),
    "python_dev_arch_pyconfig": artifact(
        os.environ["PYTHON_DEV_ARCH_PYCONFIG"] if triton else ""
    ),
    "triton_library": artifact(os.environ["TRITON_LIBRARY"] if triton else ""),
    "triton_driver": artifact(os.environ["TRITON_DRIVER"] if triton else ""),
    "triton_compiler": artifact(os.environ["TRITON_COMPILER"] if triton else ""),
    "triton_requirements": artifact(os.environ["TRITON_REQUIREMENTS"] if triton else ""),
    "triton_hsaco": artifact(os.environ["TRITON_HSACO"] if triton else ""),
    "triton_smoke_result": artifact(os.environ["TRITON_SMOKE_RESULT"] if triton else ""),
    "triton_llvm_config": artifact(os.environ["TRITON_LLVM_CONFIG"] if triton else ""),
    "triton_mlir_config": artifact(os.environ["TRITON_MLIR_CONFIG"] if triton else ""),
    "triton_llvm_cmake_cache": artifact(
        os.environ["TRITON_LLVM_CMAKE_CACHE"] if triton else ""
    ),
    "triton_llvm_compile_commands": artifact(
        os.environ["TRITON_LLVM_COMPILE_COMMANDS"] if triton else ""
    ),
    "triton_cmake_cache": artifact(os.environ["TRITON_CMAKE_CACHE"] if triton else ""),
    "triton_source_stamp": artifact(os.environ["TRITON_SOURCE_STAMP"] if triton else ""),
    "triton_llvm_source_stamp": artifact(
        os.environ["TRITON_LLVM_SOURCE_STAMP"] if triton else ""
    ),
    "triton_json_source_stamp": artifact(
        os.environ["TRITON_JSON_SOURCE_STAMP"] if triton else ""
    ),
}
if compiler:
    for candidate in sorted(Path(os.environ["DEVICE_LIB_DIR"]).glob("*.bc")):
        artifacts["device_bc:" + candidate.name] = artifact(str(candidate))
for tool in ("clang", "clang++", "lld", "ld.lld", "llvm-link", "llvm-objdump", "opt", "FileCheck"):
    key = "tool_" + tool.replace("+", "x").replace(".", "_").replace("-", "_")
    artifacts[key] = artifact(str(prefix / "bin" / tool) if compiler else "")
if triton:
    for candidate in sorted(Path(os.environ["TRITON_WHEELHOUSE"]).glob("*.whl")):
        artifacts["python_wheel:" + candidate.name] = artifact(str(candidate))
artifacts["compiled_hsaco"]["accepted_by_current_daemon"] = bool(
    compiled["sha256"] and compiled["sha256"] == locked_expected
)
payload = {
    "schema": "amdgpu-sim.rocm-prefix.v8",
    "setup_schema": int(os.environ["SETUP_SCHEMA"]),
    "setup_script_sha256": os.environ["SETUP_SHA"],
    "prefix": str(prefix),
    "requested_mode": os.environ["REQUESTED_MODE"],
    "sources": {
        "llvm-project": {"head": os.environ["LLVM_HEAD"], "tree": os.environ["LLVM_TREE"]},
        "triton-llvm-project": {"head": os.environ["TRITON_LLVM_HEAD"], "tree": os.environ["TRITON_LLVM_TREE"]},
        "rocm-systems": {"head": os.environ["ROCM_HEAD"], "tree": os.environ["ROCM_TREE"]},
        "triton": {"head": os.environ["TRITON_HEAD"], "tree": os.environ["TRITON_TREE"]},
        "gem5": {"head": os.environ["GEM5_HEAD"], "tree": os.environ["GEM5_TREE"]},
        "self-amdgpu-runtime": {"head": os.environ["RUNTIME_HEAD"], "tree": os.environ["RUNTIME_TREE"]},
    },
    "components": {
        "compiler": compiler,
        "device_libs": compiler,
        "runtime": runtime,
        "opencl": opencl,
        "python": triton,
        "triton": triton,
    },
    "artifacts": artifacts,
    "managed_inputs": {
        "gem5_binary": artifact(os.environ["GEM5_BINARY"]),
        "gem5_config": artifact(os.environ["GEM5_CONFIG"]),
        "triton_example": artifact(os.environ["TRITON_EXAMPLE"]),
        "triton_plugin_cmake": artifact(os.environ["TRITON_PLUGIN_CMAKE"]),
        "triton_plugin_init": artifact(os.environ["TRITON_PLUGIN_INIT"]),
        "triton_plugin_name": artifact(os.environ["TRITON_PLUGIN_NAME"]),
        "triton_plugin_driver": artifact(os.environ["TRITON_PLUGIN_DRIVER"]),
        "triton_plugin_compiler": artifact(os.environ["TRITON_PLUGIN_COMPILER"]),
    },
    "locked_hsaco_sha256": locked_expected,
    "locked_triton_hsaco_sha256": os.environ["TRITON_HSACO_SHA"],
    "locked_triton_example_sha256": os.environ["TRITON_EXAMPLE_SHA"],
    "triton_plugin_tree_sha256": os.environ["TRITON_PLUGIN_SHA"],
    "triton_build_contract": {
        "cc": "/usr/bin/clang",
        "cxx": "/usr/bin/clang++",
        "linker": "/usr/bin/ld.lld",
        "ninja": "/usr/bin/ninja",
        "json_source_sha256": os.environ["TRITON_JSON_SHA"],
        "triton_llvm_default_rocm_path": os.environ[
            "TRITON_LLVM_DEFAULT_ROCM_PATH"
        ],
        "triton_llvm_use_linker": "lld",
        "python_development_package": {
            "package": os.environ["PYTHON_DEV_PACKAGE"],
            "version": os.environ["PYTHON_DEV_VERSION"],
            "architecture": os.environ["PYTHON_DEV_ARCH"],
            "filename": os.environ["PYTHON_DEV_DEB_NAME"],
            "size": int(os.environ["PYTHON_DEV_DEB_SIZE"]),
            "sha256": os.environ["PYTHON_DEV_DEB_SHA"],
            "url": os.environ["PYTHON_DEV_DEB_URL"],
        },
        "python_executable": os.environ["TRITON_PYTHON"],
        "python_executable_sha256": os.environ["PYTHON_EXECUTABLE_SHA"],
        "python_include_dir": os.environ["PYTHON_INCLUDE_DIR"],
        "python_include_tree_sha256": os.environ["PYTHON_INCLUDE_TREE_SHA"],
        "python_root_tree_sha256": os.environ["PYTHON_ROOT_TREE_SHA"],
        "python_version": os.environ["PYTHON_VERSION"],
        "python_soabi": "cpython-314-x86_64-linux-gnu",
        "offline": True,
    },
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
    local expect=${1:-manifest} runtime_library triton_library triton_driver
    local triton_compiler triton_plugin_sha
    runtime_library=$(find_runtime_library)
    triton_library=$(find_triton_library)
    triton_driver=$(find_triton_driver)
    triton_compiler=$(find_triton_compiler)
    triton_plugin_sha=$(tree_archive_sha "$triton_plugin_source")
    EXPECT="$expect" MANIFEST="$manifest" PREFIX="$prefix" \
    SETUP_SCHEMA="$setup_schema" SETUP_SHA="$setup_sha" \
    LLVM_HEAD="$llvm_head_expected" LLVM_TREE="$llvm_tree_expected" \
    TRITON_LLVM_HEAD="$triton_llvm_head_expected" \
    TRITON_LLVM_TREE="$triton_llvm_tree_expected" \
    ROCM_HEAD="$rocm_head_expected" ROCM_TREE="$rocm_tree_expected" \
    TRITON_HEAD="$triton_head_expected" TRITON_TREE="$triton_tree_expected" \
    GEM5_HEAD="$gem5_head_expected" GEM5_TREE="$gem5_tree_expected" \
    RUNTIME_HEAD="$runtime_head_expected" RUNTIME_TREE="$runtime_tree_expected" \
    LOCKED_SHA="$locked_hsaco_sha" DEVICE_LIB_DIR="$device_lib_dir" \
    COMPILED_HSACO="$compiled_hsaco" LOCKED_HSACO="$installed_locked_hsaco" \
    RUNTIME_LIBRARY="$runtime_library" \
    RUNTIME_SONAME="$prefix/lib/libself_amdgpu_runtime.so.1" \
    RUNTIME_ENDPOINT="$runtime_endpoint" \
    OPENCL_LIBRARY="$opencl_library" OPENCL_EXECUTABLE="$opencl_executable" \
    OPENCL_SOURCE="$opencl_source" GEM5_BINARY="$gem5_binary" \
    GEM5_CONFIG="$gem5_config" TRITON_LIBRARY="$triton_library" \
    TRITON_DRIVER="$triton_driver" TRITON_COMPILER="$triton_compiler" \
    TRITON_PYTHON="$triton_python" TRITON_REQUIREMENTS="$triton_requirements" \
    PYTHON_DEV_DEB="$python_dev_deb" PYTHON_DEV_STAMP="$python_dev_stamp" \
    PYTHON_DEV_PYTHON_H="$python_dev_python_h" \
    PYTHON_DEV_PATCHLEVEL="$python_dev_patchlevel" \
    PYTHON_DEV_PYCONFIG="$python_dev_pyconfig" \
    PYTHON_DEV_GENERIC_PYCONFIG="$python_dev_generic_pyconfig" \
    PYTHON_DEV_ARCH_PYCONFIG="$python_dev_arch_pyconfig" \
    PYTHON_DEV_PACKAGE="$python_dev_package" \
    PYTHON_DEV_VERSION="$python_dev_version" \
    PYTHON_DEV_ARCH="$python_dev_arch" \
    PYTHON_DEV_DEB_NAME="$python_dev_deb_name" \
    PYTHON_DEV_DEB_SIZE="$python_dev_deb_size" \
    PYTHON_DEV_DEB_SHA="$python_dev_deb_sha" \
    PYTHON_DEV_DEB_URL="$python_dev_deb_url" \
    PYTHON_VERSION="$python_dev_version_expected" \
    PYTHON_EXECUTABLE_SHA="$python_dev_executable_sha" \
    PYTHON_INCLUDE_DIR="$python_dev_include" \
    PYTHON_INCLUDE_TREE_SHA="$python_dev_include_tree_sha" \
    PYTHON_ROOT_TREE_SHA="$python_dev_root_tree_sha" \
    TRITON_HSACO="$triton_hsaco" TRITON_HSACO_SHA="$triton_hsaco_sha" \
    TRITON_SMOKE_RESULT="$triton_smoke_result" \
    TRITON_LLVM_CONFIG="$triton_llvm_build/lib/cmake/llvm/LLVMConfig.cmake" \
    TRITON_MLIR_CONFIG="$triton_llvm_build/lib/cmake/mlir/MLIRConfig.cmake" \
    TRITON_LLVM_CMAKE_CACHE="$triton_llvm_cmake_cache" \
    TRITON_LLVM_COMPILE_COMMANDS="$triton_llvm_compile_commands" \
    TRITON_LLVM_DEFAULT_ROCM_PATH="$prefix" \
    TRITON_CMAKE_CACHE="$triton_cmake_cache" \
    TRITON_SOURCE_STAMP="$triton_source_stamp" \
    TRITON_LLVM_SOURCE_STAMP="$triton_llvm_source_stamp" \
    TRITON_JSON_SOURCE_STAMP="$triton_json_source_stamp" \
    TRITON_JSON_SHA="$triton_json_sha_expected" \
    TRITON_WHEELHOUSE="$triton_wheelhouse" \
    TRITON_EXAMPLE="$triton_example" \
    TRITON_EXAMPLE_SHA="$triton_example_sha" \
    TRITON_PLUGIN_SHA="$triton_plugin_sha" \
    TRITON_PLUGIN_CMAKE="$triton_plugin_source/CMakeLists.txt" \
    TRITON_PLUGIN_INIT="$triton_plugin_source/triton_gemsim_amd.cc" \
    TRITON_PLUGIN_NAME="$triton_plugin_source/backend/name.conf" \
    TRITON_PLUGIN_DRIVER="$triton_plugin_source/backend/driver.py" \
    TRITON_PLUGIN_COMPILER="$triton_plugin_source/backend/compiler.py" \
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
    "triton-llvm-project": {"head": os.environ["TRITON_LLVM_HEAD"], "tree": os.environ["TRITON_LLVM_TREE"]},
    "rocm-systems": {"head": os.environ["ROCM_HEAD"], "tree": os.environ["ROCM_TREE"]},
    "triton": {"head": os.environ["TRITON_HEAD"], "tree": os.environ["TRITON_TREE"]},
    "gem5": {"head": os.environ["GEM5_HEAD"], "tree": os.environ["GEM5_TREE"]},
    "self-amdgpu-runtime": {"head": os.environ["RUNTIME_HEAD"], "tree": os.environ["RUNTIME_TREE"]},
}
require(data.get("schema") == "amdgpu-sim.rocm-prefix.v8", "schema")
require(data.get("setup_schema") == int(os.environ["SETUP_SCHEMA"]), "setup schema")
require(data.get("setup_script_sha256") == os.environ["SETUP_SHA"], "setup script hash")
require(data.get("prefix") == str(prefix), "prefix")
require(data.get("sources") == expected_sources, "source identities")
require(data.get("locked_hsaco_sha256") == os.environ["LOCKED_SHA"], "locked hash")
require(
    data.get("locked_triton_hsaco_sha256") == os.environ["TRITON_HSACO_SHA"],
    "locked Triton hash",
)
require(
    data.get("locked_triton_example_sha256") == os.environ["TRITON_EXAMPLE_SHA"],
    "locked Triton example hash",
)
require(
    data.get("triton_plugin_tree_sha256") == os.environ["TRITON_PLUGIN_SHA"],
    "Triton plugin tree hash",
)
require(
    data.get("triton_build_contract")
    == {
        "cc": "/usr/bin/clang",
        "cxx": "/usr/bin/clang++",
        "linker": "/usr/bin/ld.lld",
        "ninja": "/usr/bin/ninja",
        "json_source_sha256": os.environ["TRITON_JSON_SHA"],
        "triton_llvm_default_rocm_path": os.environ[
            "TRITON_LLVM_DEFAULT_ROCM_PATH"
        ],
        "triton_llvm_use_linker": "lld",
        "python_development_package": {
            "package": os.environ["PYTHON_DEV_PACKAGE"],
            "version": os.environ["PYTHON_DEV_VERSION"],
            "architecture": os.environ["PYTHON_DEV_ARCH"],
            "filename": os.environ["PYTHON_DEV_DEB_NAME"],
            "size": int(os.environ["PYTHON_DEV_DEB_SIZE"]),
            "sha256": os.environ["PYTHON_DEV_DEB_SHA"],
            "url": os.environ["PYTHON_DEV_DEB_URL"],
        },
        "python_executable": os.environ["TRITON_PYTHON"],
        "python_executable_sha256": os.environ["PYTHON_EXECUTABLE_SHA"],
        "python_include_dir": os.environ["PYTHON_INCLUDE_DIR"],
        "python_include_tree_sha256": os.environ["PYTHON_INCLUDE_TREE_SHA"],
        "python_root_tree_sha256": os.environ["PYTHON_ROOT_TREE_SHA"],
        "python_version": os.environ["PYTHON_VERSION"],
        "python_soabi": "cpython-314-x86_64-linux-gnu",
        "offline": True,
    },
    "Triton build contract",
)
require(data.get("system_rocm_install") is False, "system install boundary")
require(data.get("production_umd_kmd") is False, "production runtime boundary")
requested_mode = data.get("requested_mode")
require(requested_mode in {"compiler", "runtime", "triton", "all"}, "requested mode")
expect = requested_mode if os.environ["EXPECT"] == "manifest" else os.environ["EXPECT"]
components = data.get("components", {})
require(
    set(components) == {"compiler", "device_libs", "runtime", "opencl", "python", "triton"},
    "component keys",
)
require(components["compiler"] is components["device_libs"], "compiler/device-lib parity")
require(not components["opencl"] or components["runtime"], "OpenCL/runtime dependency")
require(components["python"] is components["triton"], "Python/Triton parity")
require(not components["triton"] or components["runtime"], "Triton/runtime dependency")
if expect in {"compiler", "all"}:
    require(components["compiler"] is True, "compiler component")
if expect in {"runtime", "all"}:
    require(components["runtime"] is True, "runtime component")
    require(components["opencl"] is True, "OpenCL component")
if expect in {"triton", "all"}:
    require(components["compiler"] is True, "compiler dependency for Triton")
    require(components["runtime"] is True, "runtime dependency for Triton")
    require(components["python"] is True, "Python component")
    require(components["triton"] is True, "Triton component")
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
    "runtime_soname": os.environ["RUNTIME_SONAME"],
    "runtime_endpoint": os.environ["RUNTIME_ENDPOINT"],
    "runtime_handshake": str(prefix / "bin" / "sagr-handshake"),
    "runtime_triton_probe": str(prefix / "bin" / "sagr-triton-hsaco-probe"),
}
opencl_paths = {
    "opencl_library": os.environ["OPENCL_LIBRARY"],
    "opencl_executable": os.environ["OPENCL_EXECUTABLE"],
    "opencl_source": os.environ["OPENCL_SOURCE"],
}
triton_paths = {
    "triton_python": os.environ["TRITON_PYTHON"],
    "python_dev_deb": os.environ["PYTHON_DEV_DEB"],
    "python_dev_stamp": os.environ["PYTHON_DEV_STAMP"],
    "python_dev_python_h": os.environ["PYTHON_DEV_PYTHON_H"],
    "python_dev_patchlevel": os.environ["PYTHON_DEV_PATCHLEVEL"],
    "python_dev_pyconfig": os.environ["PYTHON_DEV_PYCONFIG"],
    "python_dev_generic_pyconfig": os.environ["PYTHON_DEV_GENERIC_PYCONFIG"],
    "python_dev_arch_pyconfig": os.environ["PYTHON_DEV_ARCH_PYCONFIG"],
    "triton_library": os.environ["TRITON_LIBRARY"],
    "triton_driver": os.environ["TRITON_DRIVER"],
    "triton_compiler": os.environ["TRITON_COMPILER"],
    "triton_requirements": os.environ["TRITON_REQUIREMENTS"],
    "triton_hsaco": os.environ["TRITON_HSACO"],
    "triton_smoke_result": os.environ["TRITON_SMOKE_RESULT"],
    "triton_llvm_config": os.environ["TRITON_LLVM_CONFIG"],
    "triton_mlir_config": os.environ["TRITON_MLIR_CONFIG"],
    "triton_llvm_cmake_cache": os.environ["TRITON_LLVM_CMAKE_CACHE"],
    "triton_llvm_compile_commands": os.environ["TRITON_LLVM_COMPILE_COMMANDS"],
    "triton_cmake_cache": os.environ["TRITON_CMAKE_CACHE"],
    "triton_source_stamp": os.environ["TRITON_SOURCE_STAMP"],
    "triton_llvm_source_stamp": os.environ["TRITON_LLVM_SOURCE_STAMP"],
    "triton_json_source_stamp": os.environ["TRITON_JSON_SOURCE_STAMP"],
}
expected.update({key: value if components["compiler"] else "" for key, value in compiler_paths.items()})
expected.update({key: value if components["runtime"] else "" for key, value in runtime_paths.items()})
expected.update({key: value if components["opencl"] else "" for key, value in opencl_paths.items()})
expected.update({key: value if components["triton"] else "" for key, value in triton_paths.items()})
if components["triton"]:
    for candidate in sorted(Path(os.environ["TRITON_WHEELHOUSE"]).glob("*.whl")):
        expected["python_wheel:" + candidate.name] = str(candidate)
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
if components["runtime"]:
    runtime_library = Path(artifacts["runtime_library"]["path"])
    runtime_soname = Path(artifacts["runtime_soname"]["path"])
    require(runtime_soname.is_symlink(), "runtime SONAME is not a symlink")
    require(runtime_soname.resolve() == runtime_library.resolve(), "runtime SONAME target")
    require(
        artifacts["runtime_soname"]["sha256"]
        == artifacts["runtime_library"]["sha256"],
        "runtime SONAME digest",
    )
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
if components["triton"]:
    require(
        artifacts["triton_python"]["sha256"] == os.environ["PYTHON_EXECUTABLE_SHA"],
        "private Python executable",
    )
    require(
        artifacts["python_dev_deb"]["sha256"] == os.environ["PYTHON_DEV_DEB_SHA"],
        "private Python development package",
    )
    require(
        artifacts["python_dev_python_h"]["sha256"]
        == "bc6e1b01ec1a37da58b81107effec928a6000186a6b93a8f9a9654f62aff5981",
        "private Python.h",
    )
    require(
        artifacts["python_dev_patchlevel"]["sha256"]
        == "a73d192cc3e7a97d39f28933feac2f7e1be1962f1c2a682fe7ee2f6cc2dd4bee",
        "private Python patchlevel",
    )
    require(
        artifacts["python_dev_pyconfig"]["sha256"]
        == "eec0c50c4157985d62d12230b4eca5bda4054a81225f2448db1da723d274c025",
        "normalized private Python pyconfig",
    )
    require(
        artifacts["python_dev_generic_pyconfig"]["sha256"]
        == "ca0e3a26b5f4dbaa14ef8cf62db1e17291045d4f92b4303625a47a1003b8b93c",
        "generic private Python pyconfig provenance",
    )
    require(
        artifacts["python_dev_arch_pyconfig"]["sha256"]
        == "eec0c50c4157985d62d12230b4eca5bda4054a81225f2448db1da723d274c025",
        "architecture private Python pyconfig provenance",
    )
    require(artifacts["triton_hsaco"]["sha256"] == os.environ["TRITON_HSACO_SHA"], "Triton image")
    smoke = json.loads(Path(os.environ["TRITON_SMOKE_RESULT"]).read_text(encoding="utf-8"))
    smoke_required = {
        "schema": "amdgpu-sim.triton-vecadd.v1",
        "backend": "gemsim_amd",
        "arch": "gfx950",
        "kernel": "add_kernel",
        "n_elements": 98432,
        "block_size": 1024,
        "program_count": 97,
        "launch_count": 2,
        "reuse": True,
        "output_correct": True,
        "mismatch_count": 0,
        "max_abs_error": 0.0,
        "fallback_count": 0,
    }
    for key, expected_value in smoke_required.items():
        require(smoke.get(key) == expected_value, f"Triton smoke {key}")
    launches = smoke.get("launch_results")
    require(isinstance(launches, list) and len(launches) == 2, "Triton launch results")
    for index, launch in enumerate(launches):
        require(launch.get("launch_index") == index, f"Triton launch {index} index")
        require(launch.get("seed") == index, f"Triton launch {index} seed")
        require(launch.get("output_correct") is True, f"Triton launch {index} correctness")
        require(launch.get("mismatch_count") == 0, f"Triton launch {index} mismatches")
        require(launch.get("max_abs_error") == 0.0, f"Triton launch {index} error")
managed_expected = {
    "gem5_binary": os.environ["GEM5_BINARY"],
    "gem5_config": os.environ["GEM5_CONFIG"],
    "triton_example": os.environ["TRITON_EXAMPLE"],
    "triton_plugin_cmake": os.environ["TRITON_PLUGIN_CMAKE"],
    "triton_plugin_init": os.environ["TRITON_PLUGIN_INIT"],
    "triton_plugin_name": os.environ["TRITON_PLUGIN_NAME"],
    "triton_plugin_driver": os.environ["TRITON_PLUGIN_DRIVER"],
    "triton_plugin_compiler": os.environ["TRITON_PLUGIN_COMPILER"],
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
if components["triton"]:
    require(
        artifacts["triton_driver"]["sha256"]
        == managed["triton_plugin_driver"]["sha256"],
        "installed Triton driver differs from reviewed plugin source",
    )
    require(
        artifacts["triton_compiler"]["sha256"]
        == managed["triton_plugin_compiler"]["sha256"],
        "installed Triton compiler differs from reviewed plugin source",
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
            && -x "$runtime_endpoint" ]] && managed_runtime_complete \
            && opencl_complete || {
            printf '%s\n' \
                'self-amdgpu-runtime/OpenCL product component is incomplete' >&2
            return 1
        }
    fi
    if [[ "$expect" == triton || "$expect" == all \
        || -f "$triton_source_stamp" || -d "$triton_venv" ]]; then
        triton_complete || {
            printf '%s\n' 'local CPU Python/Triton simulator component is incomplete' >&2
            return 1
        }
        run_triton_installed_clean "$triton_python" -I - <<'PY'
import ctypes
import importlib.util
import os
from importlib import metadata
from pathlib import Path

import torch
import triton

runtime = ctypes.CDLL(str(Path(os.environ["ROCM_SIM_ROOT"]) / "lib" / "libself_amdgpu_runtime.so.1"))
runtime.sagr_abi_version.argtypes = []
runtime.sagr_abi_version.restype = ctypes.c_uint32
if runtime.sagr_abi_version() != 0x00010008:
    raise SystemExit(f"managed runtime ABI mismatch: 0x{runtime.sagr_abi_version():08x}")
for symbol in (
    "sagr_managed_session_open",
    "sagr_managed_buffer_allocate",
    "sagr_managed_kernel_load",
    "sagr_managed_kernel_launch",
):
    if not hasattr(runtime, symbol):
        raise SystemExit(f"managed runtime symbol is missing: {symbol}")

expected = {
    "filelock": "3.29.0",
    "fsspec": "2026.4.0",
    "Jinja2": "3.1.6",
    "MarkupSafe": "3.0.3",
    "mpmath": "1.3.0",
    "nanobind": "2.10.2",
    "networkx": "3.6.1",
    "numpy": "2.4.3",
    "packaging": "26.3",
    "pip": "26.1.2",
    "safetensors": "0.8.0",
    "setuptools": "78.1.0",
    "sympy": "1.14.0",
    "torch": "2.13.0+cpu",
    "triton": "3.8.0",
    "typing_extensions": "4.15.0",
    "wheel": "0.46.3",
}
for name, version in expected.items():
    actual = metadata.version(name)
    if actual != version:
        raise SystemExit(f"installed package mismatch: {name}={actual}, expected {version}")
if torch.version.cuda is not None or torch.version.hip is not None:
    raise SystemExit("the local PyTorch wheel is not CPU-only")
backend_entries = {
    entry.name for entry in metadata.entry_points(group="triton.backends")
}
if backend_entries != {"amd", "gemsim_amd"}:
    raise SystemExit(f"unexpected Triton backend entry points: {sorted(backend_entries)}")
if importlib.util.find_spec("triton.backends.nvidia") is not None:
    raise SystemExit("the local Triton wheel contains the NVIDIA backend")
target = triton.runtime.driver.active.get_current_target()
if (target.backend, target.arch, target.warp_size) != ("gemsim_amd", "gfx950", 64):
    raise SystemExit(f"unexpected active Triton target: {target}")
PY
        run_clean "$prefix/bin/sagr-triton-hsaco-probe" \
            "$triton_hsaco" add_kernel >/dev/null
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
    STDOUT_PATH="$stdout" STDERR_PATH="$stderr" /usr/bin/python3 -I <<'PY'
import json
import os
from pathlib import Path

lines = Path(os.environ["STDOUT_PATH"]).read_text(encoding="utf-8").splitlines()
if len(lines) != 1 or not lines[0].strip():
    raise SystemExit(f"OpenCL direct smoke must produce exactly one JSON line: {lines!r}")
if Path(os.environ["STDERR_PATH"]).stat().st_size != 0:
    raise SystemExit("OpenCL direct smoke produced stderr diagnostics")
result = json.loads(lines[0])
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
    printf 'source locks: llvm=%s triton-llvm=%s triton=%s rocm-systems=%s gem5=%s runtime=%s\n' \
        "$(git_head "$llvm_source")" "$triton_llvm_head_expected" \
        "$(git_head "$triton_source")" \
        "$(git_head "$rocm_source")" "$(git_head "$gem5_source")" \
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
    triton)
        [[ -x "$prefix/bin/clang" ]] && device_libs_complete \
            && managed_runtime_complete || {
            printf '%s\n' \
                'Triton requires the local compiler, device-libs, and runtime; run --all first' >&2
            exit 1
        }
        build_triton
        write_activation
        run_triton_smoke
        write_manifest
        verify triton
        ;;
    all)
        build_llvm
        build_device_libs
        compile_kernel
        build_runtime
        build_triton
        write_activation
        run_opencl_smoke
        run_triton_smoke
        write_manifest
        verify all
        ;;
esac
