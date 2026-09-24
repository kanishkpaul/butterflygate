# Benchmark methodology

A forward-only throughput comparison of ButterflyGate against baseline
self-attention on real pretrained weights. It measures **efficiency**, not
modeling quality (the gate is untrained — see the caveat).

## Protocol

This describes the private runner that produced the CSVs in `results/`, not
the harness in `benchmark/` (which does use warmup).

- **Mode:** forward pass only, `torch.no_grad()`, batch size 1, `bf16`.
- **Timing:** wall clock (`time.perf_counter()`) around all batches for one
  (variant, length) pair, with `torch.cuda.synchronize()` before and after.
  There were **no warmup passes**: the first timed pass was also the first
  forward pass.
- **Order:** lengths outer, variants inner, baseline before ButterflyGate.
  The model is reloaded for every row.
- **Metric:** tokens/second = tokens processed / wall seconds. The timed
  region also includes per-batch loss `.cpu()` syncs and a full-vocabulary
  `argmax` for next-token accuracy, for both variants.
- **Peak VRAM:** `torch.cuda.max_memory_allocated()`.
- **Hybrid placement:** the early transformer layers have their token-mixing
  path replaced; later layers are unchanged. Baseline = the unmodified model.

**Consequence.** Baseline at 512 tokens was the first timed pass of the whole
run and absorbed one-time CUDA and kernel start-up costs, which is why the
Gemma 512 row is an outlier. Short-context ratios should be rerun with
warmup, alternating variant order, and accuracy timed separately before they
are quoted.

## Models & data
- `google/gemma-4-E2B-it` and `meta-llama/Llama-3.2-1B`.
- Text from `allenai/dolma:v1_6-sample`.
- Long-context probes use fresh process restarts per length and stop at the
  first failure; a failed length is recorded as data, not retried.

## Limits observed
- Gemma-4-E2B: succeeds through 14,336 tokens; 16,384 fails from a fresh restart
  with a CUDA allocator assertion (VRAM ceiling on the test hardware).
- Llama-3.2-1B: succeeds through 32,768; 65,536 fails before timing.

## Honest caveat
The ButterflyGate path is an **untrained** drop-in, so next-token accuracy of
the modified model collapses. These numbers are a speed/scaling result only. The
architecture, gate parametrization, and training recipe are withheld pending a
write-up; this repo ships the mechanism-agnostic harness and the measured
timings.
