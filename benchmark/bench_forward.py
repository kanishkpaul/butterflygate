"""Mechanism-agnostic forward-pass throughput benchmark.

Times the forward pass of *any* token-mixing block (attention, a linear-time
alternative, whatever) across a sweep of sequence lengths, with warmup, optional
bf16, and peak-VRAM tracking. This is the harness used to measure ButterflyGate
against baseline attention — it knows nothing about either; you hand it a module
factory and it reports tokens/second.

    python benchmark/bench_forward.py --lengths 512 1024 2048 4096

Note: the ButterflyGate module itself is not included in this repo (held back
pending write-up). Plug in your own `nn.Module` via `--factory`.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from dataclasses import dataclass, asdict


@dataclass
class Row:
    seq_len: int
    tokens_per_second: float
    wall_seconds: float
    peak_vram_gb: float
    ok: bool
    error: str = ""


def _load_factory(path: str):
    """`pkg.module:callable` -> a zero-arg callable returning an nn.Module."""
    mod_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(mod_name), attr)


def benchmark(factory, lengths, d_model=2048, dtype="bf16", warmup=2, iters=5):
    import torch  # imported lazily so `--help` works without torch installed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tdt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    rows: list[Row] = []

    for n in lengths:
        try:
            module = factory().to(device=device, dtype=tdt)
            x = torch.randn(1, n, d_model, device=device, dtype=tdt)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                for _ in range(warmup):
                    module(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
                end = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
                if device == "cuda":
                    start.record()
                import time

                t0 = time.perf_counter()
                for _ in range(iters):
                    module(x)
                if device == "cuda":
                    end.record()
                    torch.cuda.synchronize()
                    secs = start.elapsed_time(end) / 1000.0 / iters
                    vram = torch.cuda.max_memory_allocated() / 1e9
                else:
                    secs = (time.perf_counter() - t0) / iters
                    vram = 0.0

            rows.append(Row(n, n / secs, secs, round(vram, 3), True))
        except Exception as exc:  # noqa: BLE001 — record OOM/limit failures as data
            rows.append(Row(n, 0.0, 0.0, 0.0, False, f"{type(exc).__name__}: {exc}"))
            break  # a length that fails means longer ones will too

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default=None,
                    help="pkg.module:callable returning an nn.Module (your mixer block)")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096])
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", default=None, help="write results to this CSV")
    args = ap.parse_args()

    if args.factory is None:
        print("Provide --factory pkg.module:callable (the ButterflyGate module is "
              "not shipped here). See README.", file=sys.stderr)
        sys.exit(2)

    rows = benchmark(_load_factory(args.factory), args.lengths,
                     d_model=args.d_model, dtype=args.dtype)
    fieldnames = list(asdict(rows[0]).keys())
    print("\t".join(fieldnames))
    for r in rows:
        print("\t".join(str(asdict(r)[k]) for k in fieldnames))
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(asdict(r) for r in rows)


if __name__ == "__main__":
    main()
