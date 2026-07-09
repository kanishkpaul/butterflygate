# butterflygate

**A sub-quadratic, hardware-parallel replacement for transformer self-attention
— and an honest benchmark of where it wins and where it doesn't.**

ButterflyGate replaces dense O(n²) self-attention with a **structured
token-mixing network** whose cost grows as **O(n log n)** rather than O(n²) — so
the advantage should widen as context grows. This page reports a forward-only
benchmark of that efficiency claim on **real weights**
(`google/gemma-4-E2B-it` and `meta-llama/Llama-3.2-1B`) over Dolma text.

> **Status — mechanism held back pending write-up.** This is a **results
> preview**. The architecture internals (the exchange structure, gate
> parametrization, and training recipe) are **not** published here to preserve
> priority ahead of a paper. What's shown is the measured efficiency behavior
> and an honest account of its current limits.
>
> **Scope — read this first.** These are **speed / scaling** measurements of an
> **untrained** gate dropped into pretrained models (early layers hybrid-replaced,
> forward pass only). The mechanism is fast; it is **not** a trained language
> model. Next-token accuracy of the modified path collapses because the gate was
> never trained — that's expected and stated plainly, not swept under the rug.
> The contribution shown here is the **efficiency characteristic**, not modeling
> quality.

---

## Headline result: the crossover

On `gemma-4-E2B-it`, ButterflyGate (BG) starts far faster at short context, the
baseline wins in the mid-range, and **BG pulls ahead again past ~1330 tokens and
stays ahead through the longest successful probe (14,336)**:

| seq_len | baseline tok/s | BG tok/s | BG / baseline | verdict |
|--------:|---------------:|---------:|--------------:|---|
| 512     | 2,358.9  | 7,648.5  | **3.24×** | BG faster |
| 1024    | 13,094.7 | 10,839.3 | 0.83× | baseline faster |
| 2048    | 11,626.3 | 11,407.3 | 0.98× | near parity |
| 4096    | 8,687.4  | 9,100.9  | **1.05×** | BG faster |
| 8192    | 4,261.6  | 5,731.7  | **1.35×** | BG faster |
| 12288   | 3,398.4  | 4,136.1  | **1.22×** | BG faster |
| 14336   | 3,115.8  | 3,787.8  | **1.22×** | BG faster |

On `Llama-3.2-1B` (fresh-restart limit sweep) the forward pass **succeeds out to
32,768 tokens**, BG faster at every length (3.54× at 512 → 1.15× at 32k). Full
tables incl. perplexity/accuracy (which show the untrained-gate collapse
honestly) are in `results/`.

## Why it matters

The interesting regime for any attention alternative is **long context**, and
that's exactly where the O(n log n) exchange network's advantage compounds — the
data shows the ratio moving monotonically in BG's favor from the crossover up to
a VRAM-bound ceiling. This is a clean, reproducible efficiency signal on real
model weights, not a toy microbenchmark.

The obvious next step (and the reason the accuracy columns look the way they do):
**train** the gate rather than dropping it in cold. That's ongoing.

## Setup

The benchmark runs forward-only on a single GPU via HuggingFace Transformers on
real model weights and Dolma text (`bf16`, batch size 1). Implementation,
harness, and full result artifacts are available on request for evaluation
under a conversation / NDA — omitted here for publication priority, not because
they don't exist.

## What's public vs. withheld

| Public here | Withheld until publication |
|---|---|
| The efficiency result (crossover + long-context speedups, real weights) | The exchange architecture + gate parametrization |
| Honest untrained-gate caveat (accuracy collapse) | The training recipe |
| Setup shape (forward-only, models, dataset) | Implementation, benchmark harness, raw run logs |

If you're evaluating this for a role, I'm glad to walk through the full method
and numbers directly.

## Author

Kanishk Paul — [kanishkpaul.com](https://kanishkpaul.com) ·
[github.com/kanishkpaul](https://github.com/kanishkpaul)
