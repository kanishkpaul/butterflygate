"""Render the throughput-ratio figure from a results CSV.

    python plots/plot_crossover.py results/gemma4_e2b_crossover.csv plots/crossover.png

Plots ButterflyGate / baseline throughput vs sequence length. Values > 1 mean
ButterflyGate is faster. Reads only the timing columns; nothing about the
mechanism is required or revealed.
"""

from __future__ import annotations

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str):
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            xs.append(int(row["seq_len"]))
            ys.append(float(row["bg_over_baseline"]))
    return xs, ys


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "results/gemma4_e2b_crossover.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "plots/crossover.png"
    xs, ys = load(src)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(1.0, color="#888", lw=1, ls="--", label="parity")
    ax.plot(xs, ys, "-o", color="#2563eb", lw=2, label="ButterflyGate / baseline")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}×", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], rotation=45, fontsize=8)
    ax.set_xlabel("sequence length (tokens, log scale)")
    ax.set_ylabel("throughput ratio (BG / baseline)")
    ax.set_title("ButterflyGate forward-pass throughput vs dense attention")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
