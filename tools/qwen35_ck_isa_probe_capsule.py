#!/usr/bin/env python3
"""ISA/tiling probe for the gem5 CK-tile (flydsl) GEMM defect.

Companion to tools/qwen35_aiter_ck_repro_capsule.py (the committed oracle).
That capsule proved the machine's drifted tuned-GEMM table steers the decode
shapes (M in {1,2} x N in {5120,8192} x K=1024) onto the JIT flydsl kernel
``flydsl_gemm4_abf16_wbf16_bf16_t16x64x128_split_k2_block_m_warp1_
block_n_warp2_block_k_warp1_async_copyTrue_b_to_ldsTrue_..._gfx950``
whose gem5 execution is garbage (cos 0.000000 on all shapes).

This probe disambiguates WHERE the computation breaks, using inputs whose
correct output is decodable per-element, and dumps the JIT kernel's code
object (the ELF mapped in this process at the dispatch trace's
machine_code_va) for offline disassembly.

Probes (all through aiter.ops.flydsl.gemm_kernels.flydsl_hgemm with the
EXACT kernel config the table row selects, out= pre-filled with a sentinel):

  zero    a=ones(1,K), w[n,k]=1 (k<512) / 2 (k>=512)  -> every out[n]=1536
          tells whether the result is absent (sentinel), zeroed, one split-K
          half, or scrambled magnitude.
  onehot  a=e_{j0}, w[n,k] = (n%64) + 128*(k>=512)    -> out[n] decodes which
          B row/column stream each output element actually saw.
  m2      rows e_0 and e_511                            -> row mixing.

Exit 0 always (diagnostic capsule); results under --output-dir.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--output-dir", type=Path, default=Path(
    __import__("os").environ.get(
        "QWEN35_CK_ISA_PROBE_OUTPUT",
        str(Path(__file__).resolve().parents[1]
            / "artifacts/qwen35-ck-isa-probe-capsule/v1"))))
PARSER.add_argument("--dump-code", action="store_true", default=True)
ARGS = PARSER.parse_args()

ROOT = Path(__file__).resolve().parents[1]
DRIFTED_SRC = Path("/tmp/aiter_configs/bf16_tuned_gemm.csv")
ARGS.output_dir.mkdir(parents=True, exist_ok=True)
TABLE = ARGS.output_dir / "table_synthetic.csv"

# Same synthetic table construction as the oracle: drifted machine table plus
# explicit M in {1,2} rows for the engine decode shapes cloned from the
# drifted M=1 N=8192 K=1024 donor row.
shutil.copyfile(DRIFTED_SRC, TABLE)
import csv  # noqa: E402
with open(TABLE) as f:
    rows = list(csv.reader(f))
hdr, body = rows[0], rows[1:]
donor = next(r for r in body
             if r[0] == "gfx950" and r[2] == "1"
             and r[3] == "8192" and r[4] == "1024")
have = {tuple(r[:5]) for r in body}
for m in ("1", "2"):
    for n in ("5120", "8192"):
        key = ("gfx950", "256", m, n, "1024")
        if key in have:
            continue
        clone = list(donor)
        clone[2], clone[3], clone[4] = m, n, "1024"
        body.append(clone)
with open(TABLE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(hdr)
    w.writerows(body)
os.environ["AITER_CONFIG_GEMM_BF16"] = str(TABLE)

import torch  # noqa: E402

# Kernel config from the donor table row (verified against the dispatch trace:
# wg 128, lds 81920 = 4 stages of (16x128 + 64x128) bf16).
FLYDSL_CONFIG = dict(
    kernel_family=None, tile_m=16, tile_n=64, tile_k=128, split_k=2,
    block_m_warps=1, block_n_warps=2, block_k_warps=1,
    stages=4, async_copy=True, b_to_lds=True,
)

SENTINEL = 123.0
K = 1024
SPLIT_K = 2
KS = K // SPLIT_K  # 512


def run_flydsl(a, w, out):
    from aiter.ops.flydsl import gemm_kernels
    ref = (a.double() @ w.double().T)
    out = out.to("cuda")
    out.fill_(SENTINEL)
    gemm_kernels.flydsl_hgemm(a.to("cuda"), w.to("cuda"), out=out,
                              **FLYDSL_CONFIG)
    torch.cuda.synchronize()
    dev = out.cpu()
    diff = (dev.double() - ref).abs()
    cos = float(torch.nn.functional.cosine_similarity(
        dev.double().reshape(-1), ref.reshape(-1), dim=0))
    return dev, ref, diff, cos


def summarize(name, dev, ref, diff, cos, n_report=8):
    d = dev[0]
    r = ref[0]
    blocks = []
    for start in (0, 64, 5120 // 2, 5120 - 64):
        blocks.append({
            "n_range": [start, start + n_report],
            "got": [round(float(x), 3) for x in d[start:start + n_report]],
            "want": [round(float(x), 3) for x in r[start:start + n_report]],
        })
    uniq = torch.unique(d)
    return {
        "case": name,
        "cos": round(cos, 6),
        "mean_abs_diff": round(float(diff.mean()), 6),
        "max_abs_diff": round(float(diff.max()), 6),
        "n_unique_got": int(uniq.numel()),
        "unique_got_head": [round(float(x), 3) for x in uniq[:16]],
        "n_blocks": blocks,
    }


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO)

    from aiter.tuned_gemm import tgemm  # noqa: F401  (table load proof)
    from aiter.ops.flydsl import gemm_kernels

    report = {
        "schema": "amdgpu-sim.qwen35-ck-isa-probe.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": [],
    }

    skip_numeric = bool(int(os.environ.get("QWEN35_CK_PROBE_SKIP_NUMERIC",
                                           "0")))
    n = 5120

    # --- case zero: ones . (1|2 pattern) -> 1536 everywhere -------------
    if not skip_numeric:
        a = torch.ones(1, K, dtype=torch.bfloat16)
        w = torch.where(
            torch.arange(K).expand(n, K) < KS,
            torch.tensor(1.0), torch.tensor(2.0),
        ).to(torch.bfloat16)
        out = torch.empty(1, n, dtype=torch.bfloat16)
        dev, ref, diff, cos = run_flydsl(a, w, out)
        case = summarize("zero_ones_1536", dev, ref, diff, cos)
        # how many outputs equal the sentinel (writes lost), 0, or other values
        d0 = dev[0].float()
        case["n_sentinel"] = int((d0 == SENTINEL).sum())
        case["n_zero"] = int((d0 == 0.0).sum())
        for probe, label in ((512.0, "n_512_firsthalf_only"),
                             (1024.0, "n_1024_secondhalf_only"),
                             (1536.0, "n_1536_correct"),
                             (1659.0, "n_1659_sentinel_plus")):
            case[label] = int((d0 == probe).sum())
        report["cases"].append(case)
        print(f"[ckprobe] zero: {json.dumps(case)}", flush=True)

    # --- case onehot: a = e_{j0}; w[n,k] = (n%64) + 128*(k>=512) --------
    # ref[n] = n%64 + 128*(j0>=512). bf16-exact (<=255).
    if not skip_numeric:
        w = (torch.arange(n).unsqueeze(1) % 64
             + 128 * (torch.arange(K).unsqueeze(0) >= KS)).to(torch.bfloat16)
        onehot = {}
        for j0 in (0, 5, 255, 256, 511, 512, 640, 1023):
            a = torch.zeros(1, K, dtype=torch.bfloat16)
            a[0, j0] = 1.0
            out = torch.empty(1, n, dtype=torch.bfloat16)
            dev, ref, diff, cos = run_flydsl(a, w, out)
            d0 = dev[0].float()
            r0 = ref[0]
            ok_mask = (d0 - r0.float()).abs() <= 0.51
            entry = {
                "j0": j0, "cos": round(cos, 6),
                "n_exact": int(ok_mask.sum()),
                "got_head": [round(float(x), 2) for x in d0[:8]],
                "want_head": [round(float(x), 2) for x in r0[:8]],
                "n_sentinel": int((d0 == SENTINEL).sum()),
            }
            # which residue classes of n are right (tile fingerprint)
            wrong = (~ok_mask).nonzero().flatten()
            if wrong.numel():
                entry["wrong_n_mod64_hist"] = {
                    str(m): int(((wrong % 64) == m).sum()) for m in range(64)
                    if int(((wrong % 64) == m).sum()) > 0
                }
                entry["wrong_n_ranges"] = str(wrong[:16].tolist()) + " ... " + \
                    str(wrong[-8:].tolist())
            else:
                entry["wrong_n_mod64_hist"] = {}
                entry["wrong_n_ranges"] = ""
            onehot[j0] = entry
            print(f"[ckprobe] onehot j0={j0}: cos={cos:.4f} "
                  f"exact={entry['n_exact']}/{n}", flush=True)
        report["cases"].append({"case": "onehot", "by_j0": onehot})

        # --- case m2: rows e_0 and e_511 -------------------------------
        a = torch.zeros(2, K, dtype=torch.bfloat16)
        a[0, 0] = 1.0
        a[1, KS - 1] = 1.0
        out = torch.empty(2, n, dtype=torch.bfloat16)
        dev, ref, diff, cos = run_flydsl(a, w, out)
        case = summarize("m2_e0_e511", dev, ref, diff, cos)
        d0, d1 = dev[0].float(), dev[1].float()
        r0, r1 = ref[0].float(), ref[1].float()
        case["row0_exact"] = int(((d0 - r0).abs() <= 0.51).sum())
        case["row1_exact"] = int(((d1 - r1).abs() <= 0.51).sum())
        case["row0_equals_wanted_row1"] = int(((d0 - r1).abs() <= 0.51).sum())
        report["cases"].append(case)
        print(f"[ckprobe] m2: {json.dumps(case)}", flush=True)

    # --- oracle re-check through the same tgemm entry -------------------
    torch.manual_seed(0)
    a = (torch.randn(1, K) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, K) * 0.02).to(torch.bfloat16)
    got = tgemm.mm(a.to("cuda"), w.to("cuda")).cpu()
    ref = (a.double() @ w.double().T).to(torch.bfloat16)
    cos = float(torch.nn.functional.cosine_similarity(
        got.float().reshape(-1), ref.float().reshape(-1), dim=0))
    report["tgemm_oracle_cos"] = round(cos, 6)
    print(f"[ckprobe] tgemm oracle cos={cos:.6f}", flush=True)

    # --- dispatch-trace correlation + code object dump -------------------
    code_dumps = []
    if ARGS.dump_code:
        try:
            code_dumps = dump_code_objects()
        except Exception as exc:  # noqa: BLE001
            report["dump_error"] = repr(exc)
    report["code_dumps"] = code_dumps

    (ARGS.output_dir / "report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="ascii")
    print(f"[ckprobe] report written to {ARGS.output_dir}/report.json",
          flush=True)
    return 0


def dump_code_objects():
    """Dump gfx950 code objects mapped in this process.

    The dispatch trace (SAGR_MANAGED_RUN_ROOT/self-amdgpu-opencl-run.*/
    dispatch-trace.jsonl) records machine_code_va per kernel; GPU VA is host
    VA in this environment, so the ELF containing the kernel is findable by
    scanning backwards from that address for the AMDGPU ELF magic.
    """
    import struct

    dumps = []

    # machine_code_va values from this run's dispatch trace
    vas = []
    run_root = os.environ.get("SAGR_MANAGED_RUN_ROOT", "")
    if run_root:
        for run_dir in sorted(Path(run_root).glob("self-amdgpu-opencl-run.*")):
            trace = run_dir / "dispatch-trace.jsonl"
            if not trace.exists():
                continue
            for line in trace.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "native_execution_retired":
                    vas.append((int(rec["machine_code_va"]),
                                int(rec.get("fixed_shared_memory_bytes", 0)),
                                rec.get("grid")))
    if not vas:
        # Fall back: every readable mapping start that is an AMDGPU ELF.
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                start = int(parts[0].split("-")[0], 16)
                if "r" in parts[1]:
                    vas.append((start, -1, "maps-scan"))
    fd = os.open("/proc/self/mem", os.O_RDONLY)

    def elf_at(base):
        try:
            hdr = os.pread(fd, 80, base)
        except (OSError, OverflowError):
            return None
        if len(hdr) < 80 or hdr[:4] != b"\x7fELF" or hdr[4] != 2:
            return None
        (_type, e_machine, _v, _e, _p, _sh, _flags, e_ehsize, _phentsize,
         _phnum, shentsize, shnum, _shstrndx) = struct.unpack_from(
            "<HHIQQQIHHHHHH", hdr, 16)
        if e_machine != 224:  # EM_AMDGPU
            return None
        if shnum == 0 or shnum > 4096 or shentsize == 0 or _sh > (1 << 40):
            return None
        try:
            sh = os.pread(fd, shentsize * shnum, base + _sh)
        except (OSError, OverflowError):
            return None
        if len(sh) < shentsize * shnum:
            return None
        size = e_ehsize
        for i in range(shnum):
            fields = struct.unpack_from("<IIQQQQIIQQ", sh, i * shentsize)
            size = max(size, fields[4] + fields[5])  # sh_offset + sh_size
        # include the section header table itself
        size = max(size, _sh + shentsize * shnum)
        if size > (8 << 20):  # implausible for a kernel code object
            return None
        return size

    seen_bases = set()
    for va, lds, grid in vas:
        if lds == -1:
            base, size = va, elf_at(va)
            if not size:
                continue
        else:
            base = None
            size = None
            page = va & ~0xFFF
            # scan backwards up to 2MB for the AMDGPU ELF that owns va
            for back in range(512):
                cand = page - back * 0x1000
                if cand in seen_bases:
                    continue
                cand_size = elf_at(cand)
                if cand_size and cand <= va < cand + cand_size:
                    base, size = cand, cand_size
                    break
            if base is None:
                dumps.append({"va": hex(va), "lds": lds, "grid": grid,
                              "found": False})
                continue
        seen_bases.add(base)
        try:
            blob = os.pread(fd, size, base)
        except (OSError, OverflowError):
            dumps.append({"va": hex(va), "lds": lds, "grid": grid,
                          "found": True, "base": hex(base),
                          "read_error": True})
            continue
        if len(blob) != size:
            dumps.append({"va": hex(va), "lds": lds, "grid": grid,
                          "found": True, "base": hex(base),
                          "short_read": len(blob)})
            continue
        out = ARGS.output_dir / f"code_va{hex(va)}_base{hex(base)}.hsaco"
        out.write_bytes(blob)
        dumps.append({"va": hex(va), "lds": lds, "grid": grid, "found": True,
                      "base": hex(base), "bytes": size, "file": str(out)})
        print(f"[ckprobe] dumped code object {out} ({size} bytes)", flush=True)

    # The loaded code objects may not map their ELF headers into this
    # process (ROCr maps only the executable segment). The flydsl JIT keeps
    # the AMDGPU code object embedded in its host module (anonymous or
    # memfd mappings), so scan those for embedded EM_AMDGPU ELFs.
    regions = []
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            rng, perms = parts[0], parts[1]
            if "r" not in perms:
                continue
            path = parts[5] if len(parts) > 5 else ""
            if path.startswith("/") and "memfd" not in path \
                    and not path.startswith("/dev/shm"):
                continue
            start, end = (int(x, 16) for x in rng.split("-"))
            if end - start > 0:
                regions.append((start, end, path))

    CHUNK = 4 << 20
    count = 0
    for start, end, path in regions:
        pos = start
        while pos < end:
            try:
                chunk = os.pread(fd, min(CHUNK, end - pos), pos)
            except (OSError, OverflowError):
                break
            if not chunk:
                break
            off = chunk.find(b"\x7fELF")
            while off != -1:
                base = pos + off
                if base not in seen_bases:
                    size = elf_at(base)
                    if size:
                        seen_bases.add(base)
                        try:
                            blob = os.pread(fd, size, base)
                            out = ARGS.output_dir / \
                                f"embedded_base{hex(base)}.hsaco"
                            out.write_bytes(blob)
                            dumps.append({
                                "embedded": True, "base": hex(base),
                                "bytes": size, "mapping": path or "anon",
                                "file": str(out)})
                            print(f"[ckprobe] dumped embedded code object "
                                  f"{out} ({size} bytes)", flush=True)
                            count += 1
                        except (OSError, OverflowError):
                            pass
                off = chunk.find(b"\x7fELF", off + 1)
            pos += len(chunk)
        if count >= 12:
            break
    os.close(fd)
    return dumps


if __name__ == "__main__":
    raise SystemExit(main())
