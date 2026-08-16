#!/usr/bin/env python3
"""Create and verify the repository-owned user-facing conda product."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable


SCHEMA = "amdgpu-sim.conda-product.v1"
IDENTITY_SCHEMA = "amdgpu-sim.conda-product-identity.v1"
ACTIVE_SCHEMA = "amdgpu-sim.active-conda-product.v1"
PREFIX_NAME = "product-v1-"
LOCK_RELATIVE = Path("config/conda-linux-64.lock")
PRODUCT_PLUGINS = {
    "triton-gemsim-amd": Path("plugins/triton/gemsim_amd"),
    "gemsim-vllm": Path("plugins/framework/gemsim_vllm"),
    "gemsim-ccl": Path("plugins/collectives/gemsim_ccl"),
}
PRODUCT_TOOLS = {
    "gemsim_live_registry": Path("scripts/gemsim_live_registry.py"),
    "gemsim_smi": Path("scripts/gemsim_smi.py"),
}
UPSTREAM_REPOSITORIES = {
    "pytorch": (Path("projects/pytorch"), "411e87a93704f547e5146c74c95fa11acf13d646"),
    "triton": (Path("projects/triton"), "cd513e2798db0f4675b3d1205c8e76eb3381a0b3"),
    "vllm": (Path("projects/vllm"), "8d9b52f7c2514490bdadfd5eb0c931e58625df2e"),
}
EXCLUDED_BASE_NAMES = {
    "__editable__.gemsim_ccl-0.1.0.pth",
    "__editable__.gemsim_vllm_plugin-0.1.0.pth",
    "gemsim_ccl-0.1.0.dist-info",
    "gemsim_vllm_plugin-0.1.0.dist-info",
}


class CondaProductError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def file_sha256(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise CondaProductError(f"expected an owned regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    if (metadata.st_size, metadata.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CondaProductError(f"file changed while hashing: {path}")
    return {"path": str(path), "bytes": metadata.st_size, "sha256": digest.hexdigest()}


def read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CondaProductError(f"invalid JSON: {path}") from error
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise CondaProductError(f"JSON is not a canonical object: {path}")
    return document, payload


def load_product_module(root: Path):
    path = root / "tools/product_environment.py"
    spec = importlib.util.spec_from_file_location("amdgpu_sim_product_environment", path)
    if spec is None or spec.loader is None:
        raise CondaProductError("could not load product environment module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root: Path, repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(root / repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        raise CondaProductError(f"git wrote stderr for {repository}")
    return result.stdout.strip()


def upstream_identity(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (relative, expected_head) in UPSTREAM_REPOSITORIES.items():
        head = git(root, relative, "rev-parse", "HEAD")
        status = git(root, relative, "status", "--porcelain=v1", "--untracked-files=all")
        if head != expected_head or status:
            raise CondaProductError(f"pinned upstream repository drifted: {name}")
        result[name] = {"path": str(root / relative), "head": head, "clean": True}
    return result


def active_native_product(root: Path) -> tuple[Path, dict[str, Any], bytes]:
    active, _ = read_canonical_json(root / "env/rocm/active-product")
    if active.get("schema") != "amdgpu-sim.active-product.v1":
        raise CondaProductError("active native product schema is invalid")
    prefix = Path(active.get("prefix", ""))
    manifest, payload = read_canonical_json(prefix / "manifest.json")
    if (
        manifest.get("schema") != "amdgpu-sim.product-prefix.v1"
        or manifest.get("prefix") != str(prefix)
        or manifest.get("product_id") != active.get("product_id")
        or hashlib.sha256(payload).hexdigest() != active.get("manifest_sha256")
    ):
        raise CondaProductError("active native product identity is invalid")
    for group, names in (
        ("artifacts", ("runtime_library", "opencl_library")),
        ("managed_inputs", ("gem5_binary", "gem5_config")),
    ):
        for name in names:
            expected = manifest[group][name]
            actual = file_sha256(Path(expected["path"]))
            if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
                raise CondaProductError(f"active native product artifact drifted: {name}")
    return prefix, manifest, payload


def tree_summary(
    root: Path,
    *,
    exclude: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if name not in {"__pycache__", ".pytest_cache"}
            and not (exclude and exclude((current_path / name).relative_to(root)))
        )
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root)
            if name.endswith((".pyc", ".pyo")) or (exclude and exclude(relative)):
                continue
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                record = file_sha256(path)
                core = {
                    "path": relative.as_posix(),
                    "kind": "regular",
                    "executable": bool(metadata.st_mode & stat.S_IXUSR),
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
                total += metadata.st_size
            elif stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path)
                core = {"path": relative.as_posix(), "kind": "symlink", "target": target}
            else:
                raise CondaProductError(f"special file in environment tree: {path}")
            digest.update(canonical_json(core))
            count += 1
    return {"file_count": count, "regular_bytes": total, "sha256": digest.hexdigest()}


def base_exclude(relative: Path) -> bool:
    if relative.parts and relative.parts[0] in EXCLUDED_BASE_NAMES:
        return True
    return relative.parts[:3] == ("triton", "backends", "gemsim_amd")


def plugin_identities(root: Path, product_module) -> dict[str, Any]:
    result = {}
    for name, relative in sorted(PRODUCT_PLUGINS.items()):
        descriptor = product_module.directory_source_set(root / relative)
        result[name] = {
            "path": str(root / relative),
            "file_count": descriptor["file_count"],
            "source_set_sha256": descriptor["source_set_sha256"],
        }
    return result


def tool_identities(root: Path) -> dict[str, Any]:
    return {
        name: file_sha256(root / relative)
        for name, relative in sorted(PRODUCT_TOOLS.items())
    }


def identity(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    product_module = load_product_module(root)
    native_prefix, native_manifest, native_payload = active_native_product(root)
    base_prefix = Path(native_manifest["base"]["prefix"])
    base_site = base_prefix / "venv/lib/python3.14/site-packages"
    lock = file_sha256(root / LOCK_RELATIVE)
    document = {
        "schema": IDENTITY_SCHEMA,
        "lock": lock,
        "builder": file_sha256(Path(__file__).resolve()),
        "native_product": {
            "prefix": str(native_prefix),
            "product_id": native_manifest["product_id"],
            "manifest_sha256": hashlib.sha256(native_payload).hexdigest(),
        },
        "base": {
            "prefix": str(base_prefix),
            "site_packages": str(base_site),
            "site_packages_source": tree_summary(base_site, exclude=base_exclude),
        },
        "plugins": plugin_identities(root, product_module),
        "tools": tool_identities(root),
        "upstream": upstream_identity(root),
        "python": {"implementation": "CPython", "version": "3.14.6", "abi": "cp314"},
    }
    return document, native_manifest


def product_id(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(document)).hexdigest()


def product_prefix(root: Path, identifier: str) -> Path:
    return root / "env/conda" / f"{PREFIX_NAME}{identifier}"


def conda_executable() -> str:
    executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not executable:
        fallback = Path.home() / "miniforge3/condabin/conda"
        if fallback.is_file():
            executable = str(fallback)
    if not executable:
        raise CondaProductError("conda executable is unavailable")
    return executable


def run(arguments: list[str], *, environment: dict[str, str] | None = None) -> None:
    subprocess.run(arguments, check=True, env=environment)


def product_environment(
    prefix: Path,
    native: dict[str, Any],
    state: Path,
) -> dict[str, str]:
    base = native["base"]["prefix"]
    home = state / "home"
    temporary = state / "tmp"
    cache = state / "triton-cache"
    for path in (home, temporary, cache):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return {
        "PATH": f"{prefix}/bin:{native['prefix']}/bin:{base}/bin:/usr/bin:/bin",
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "XDG_CACHE_HOME": str(state / "xdg-cache"),
        "ROCM_SIM_ROOT": native["prefix"],
        "ROCM_PATH": base,
        "HIP_PATH": native["prefix"],
        "HSA_PATH": native["prefix"],
        "HIP_PLATFORM": "amd",
        "HIP_CLANG_PATH": f"{base}/bin",
        "HSA_ENABLE_DXG_DETECTION": "0",
        # Fast copy is explicitly opt-in. Model mode requires both gates.
        "HSA_ENABLE_DTIF_FAST_COPY": "0",
        "SAGR_HSAKMT_MODEL_FAST_COPY": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "HSA_MODEL_LIB": f"{native['prefix']}/lib/libself_amdgpu_hsakmt_model.so.1",
        "HSA_MODEL_TOPOLOGY": f"{native['prefix']}/share/self-amdgpu-runtime/hsakmt-topology",
        "LD_LIBRARY_PATH": f"{native['prefix']}/lib:{base}/lib",
        "TRITON_DEFAULT_BACKEND": "gemsim_amd",
        "TRITON_CACHE_DIR": str(cache),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "LC_ALL": "C",
    }


def write_activation(prefix: Path, native: dict[str, Any], state: Path) -> None:
    base = native["base"]["prefix"]
    product = native["prefix"]
    activate = prefix / "etc/conda/activate.d/amdgpu-sim.sh"
    deactivate = prefix / "etc/conda/deactivate.d/amdgpu-sim.sh"
    activate.parent.mkdir(parents=True, exist_ok=True)
    deactivate.parent.mkdir(parents=True, exist_ok=True)
    variables = {
        "ROCM_SIM_ROOT": product,
        "ROCM_PATH": base,
        "HIP_PATH": product,
        "HSA_PATH": product,
        "HIP_PLATFORM": "amd",
        "HIP_CLANG_PATH": f"{base}/bin",
        "HSA_ENABLE_DXG_DETECTION": "0",
        "HSA_ENABLE_DTIF_FAST_COPY": "0",
        "SAGR_HSAKMT_MODEL_FAST_COPY": "0",
        "HSA_ENABLE_INTERRUPT": "0",
        "HSA_MODEL_LIB": f"{product}/lib/libself_amdgpu_hsakmt_model.so.1",
        "HSA_MODEL_TOPOLOGY": f"{product}/share/self-amdgpu-runtime/hsakmt-topology",
        "LD_LIBRARY_PATH": f"{product}/lib:{base}/lib",
        "PATH": f"{prefix}/bin:{product}/bin:{base}/bin:/usr/bin:/bin",
        "TRITON_DEFAULT_BACKEND": "gemsim_amd",
        "TRITON_CACHE_DIR": str(state / "triton-cache"),
        "PYTHONNOUSERSITE": "1",
    }
    lines = ["# Generated by tools/conda_product_environment.py."]
    restore = ["# Generated by tools/conda_product_environment.py."]
    for name, value in variables.items():
        lines.extend(
            [
                f'if [[ ${{{name}+x}} ]]; then export _AMDGPU_SIM_OLD_{name}="${{{name}}}"; else unset _AMDGPU_SIM_OLD_{name}; fi',
                f"export {name}={value}",
            ]
        )
        restore.extend(
            [
                f'if [[ ${{_AMDGPU_SIM_OLD_{name}+x}} ]]; then export {name}="${{_AMDGPU_SIM_OLD_{name}}}"; else unset {name}; fi',
                f"unset _AMDGPU_SIM_OLD_{name}",
            ]
        )
    activate.write_text("\n".join(lines) + "\n", encoding="ascii")
    deactivate.write_text("\n".join(restore) + "\n", encoding="ascii")


def install_smi_tools(root: Path, prefix: Path) -> dict[str, str]:
    libexec = prefix / "libexec/amdgpu-sim"
    libexec.mkdir(mode=0o755, parents=True, exist_ok=True)
    for name, relative in sorted(PRODUCT_TOOLS.items()):
        destination = libexec / f"{name}.py"
        shutil.copyfile(root / relative, destination)
        destination.chmod(0o444)
    executable = prefix / "bin/rocm-smi"
    executable.write_text(
        "#!/bin/sh\n"
        f'exec "{prefix}/bin/python" "{libexec}/gemsim_smi.py" "$@"\n',
        encoding="ascii",
    )
    executable.chmod(0o555)
    alias = prefix / "bin/gemsim-smi"
    alias.symlink_to("rocm-smi")
    return {"rocm_smi": str(executable), "gemsim_smi": str(alias)}


def install(root: Path) -> Path:
    document, native = identity(root)
    identifier = product_id(document)
    prefix = product_prefix(root, identifier)
    state = root / "env/conda-state" / identifier
    manifest_path = prefix / "amdgpu-sim-manifest.json"
    if manifest_path.is_file():
        verify(root, prefix)
        publish_active(root, prefix, identifier)
        return prefix
    if prefix.exists():
        raise CondaProductError(f"incomplete conda candidate requires manual inspection: {prefix}")
    prefix.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    run(
        [
            conda_executable(),
            "create",
            "--yes",
            "--offline",
            "--prefix",
            str(prefix),
            "--file",
            str(root / LOCK_RELATIVE),
        ]
    )
    base_site = Path(document["base"]["site_packages"])
    destination = prefix / "lib/python3.14/site-packages"
    shutil.rmtree(destination)
    run(["/usr/bin/cp", "-a", "--reflink=auto", str(base_site), str(destination)])
    for name in EXCLUDED_BASE_NAMES:
        path = destination / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    shutil.rmtree(destination / "triton/backends/gemsim_amd", ignore_errors=True)
    backend = root / PRODUCT_PLUGINS["triton-gemsim-amd"] / "backend"
    shutil.copytree(backend, destination / "triton/backends/gemsim_amd")
    wheelhouse = state / "wheelhouse"
    wheelhouse.mkdir(mode=0o700, parents=True)
    build_sources = state / "build-sources"
    build_sources.mkdir(mode=0o700)
    python = prefix / "bin/python"
    clean_env = product_environment(prefix, native, state)
    for name in ("gemsim-ccl", "gemsim-vllm"):
        source = build_sources / name
        shutil.copytree(
            root / PRODUCT_PLUGINS[name],
            source,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
        )
        product_module = load_product_module(root)
        snapshot = product_module.directory_source_set(source)
        expected = document["plugins"][name]
        if (
            snapshot["file_count"] != expected["file_count"]
            or snapshot["source_set_sha256"] != expected["source_set_sha256"]
        ):
            raise CondaProductError(f"plugin build snapshot differs from identity: {name}")
        run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheelhouse),
                str(source),
            ],
            environment=clean_env,
        )
    wheels = sorted(wheelhouse.glob("*.whl"))
    if len(wheels) != 2:
        raise CondaProductError("project wheel build did not produce exactly two wheels")
    run(
        [str(python), "-I", "-m", "pip", "install", "--no-deps", *map(str, wheels)],
        environment=clean_env,
    )
    smi_entries = install_smi_tools(root, prefix)
    write_activation(prefix, native, state)
    run([str(python), "-I", "-m", "pip", "check"], environment=clean_env)
    installed = tree_summary(prefix, exclude=lambda p: p.as_posix() == "amdgpu-sim-manifest.json")
    manifest = {
        "schema": SCHEMA,
        "product_id": identifier,
        "prefix": str(prefix),
        "identity": document,
        "native": {
            "prefix": native["prefix"],
            "base_prefix": native["base"]["prefix"],
            "runtime_library": native["artifacts"]["runtime_library"],
            "gem5_binary": native["managed_inputs"]["gem5_binary"],
            "gem5_config": native["managed_inputs"]["gem5_config"],
        },
        "state_root": str(state),
        "installed_tree": installed,
        "entry": {
            "python": str(python),
            "activate": str(prefix / "etc/conda/activate.d/amdgpu-sim.sh"),
            **smi_entries,
        },
    }
    manifest_path.write_bytes(canonical_json(manifest))
    verify(root, prefix)
    publish_active(root, prefix, identifier)
    return prefix


def verify(root: Path, prefix: Path | None = None) -> Path:
    if prefix is None:
        active, _ = read_canonical_json(root / "env/conda/active-product")
        if active.get("schema") != ACTIVE_SCHEMA:
            raise CondaProductError("active conda product schema is invalid")
        prefix = Path(active["prefix"])
    manifest, _ = read_canonical_json(prefix / "amdgpu-sim-manifest.json")
    current_identity, native = identity(root)
    identifier = product_id(current_identity)
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("product_id") != identifier
        or manifest.get("prefix") != str(prefix)
        or manifest.get("identity") != current_identity
        or prefix != product_prefix(root, identifier)
    ):
        raise CondaProductError("conda product identity drifted")
    expected_state = root / "env/conda-state" / identifier
    if manifest.get("state_root") != str(expected_state):
        raise CondaProductError("conda product state root is invalid")
    observed = tree_summary(prefix, exclude=lambda p: p.as_posix() == "amdgpu-sim-manifest.json")
    if observed != manifest.get("installed_tree"):
        raise CondaProductError("conda product contents drifted")
    python = prefix / "bin/python"
    script = """
import importlib.metadata as m
import pathlib
import sys
assert sys.prefix == sys.base_prefix
assert sys.version_info[:3] == (3, 14, 6)
for name, version in {
    'torch':'2.13.0+cpu', 'triton':'3.8.0',
    'vllm':'0.0.dev0+g8d9b52f7c2', 'gemsim-ccl':'0.1.0',
    'gemsim-vllm-plugin':'0.1.0',
}.items():
    assert m.version(name) == version, (name, m.version(name))
root = pathlib.Path(sys.prefix).resolve()
for name in ('torch','triton','vllm','gemsim_ccl','gemsim_vllm'):
    module = __import__(name)
    pathlib.Path(module.__file__).resolve().relative_to(root)
for pth in root.glob('lib/python3.14/site-packages/*.pth'):
    text = pth.read_text(errors='strict')
    assert '/plugins/' not in text and '__editable__' not in pth.name
import triton.backends.gemsim_amd.driver
"""
    environment = product_environment(prefix, native, expected_state)
    run([str(python), "-I", "-c", script], environment=environment)
    entry = manifest.get("entry")
    if not isinstance(entry, dict):
        raise CondaProductError("conda product entry points are invalid")
    for name, expected in (
        ("rocm_smi", prefix / "bin/rocm-smi"),
        ("gemsim_smi", prefix / "bin/gemsim-smi"),
    ):
        if entry.get(name) != str(expected) or not os.access(expected, os.X_OK):
            raise CondaProductError(f"conda product entry point is invalid: {name}")
    return prefix


def publish_active(root: Path, prefix: Path, identifier: str) -> None:
    manifest = file_sha256(prefix / "amdgpu-sim-manifest.json")
    active = {
        "schema": ACTIVE_SCHEMA,
        "product_id": identifier,
        "prefix": str(prefix),
        "manifest_sha256": manifest["sha256"],
    }
    path = root / "env/conda/active-product"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(canonical_json(active))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--install", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--print-prefix", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve(strict=True)
    try:
        if arguments.install:
            result = install(root)
        elif arguments.verify:
            result = verify(root)
        else:
            active, _ = read_canonical_json(root / "env/conda/active-product")
            if active.get("schema") != ACTIVE_SCHEMA:
                raise CondaProductError("active conda product schema is invalid")
            result = Path(active["prefix"])
    except (CondaProductError, OSError, subprocess.CalledProcessError) as error:
        print(f"conda product error: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
