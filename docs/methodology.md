# Benchmark methodology

A forward-only throughput comparison of ButterflyGate against baseline
self-attention on real pretrained weights. It measures **efficiency**, not
modeling quality (the gate is untrained — see the caveat).

## Protocol
- **Mode:** forward pass only, `torch.no_grad()`, batch size 1, `bf16`.
- **Warmup:** discard the first passes, then time a fixed number of iterations
  per sequence length; on CUDA, time with events and `synchronize()`.
- **Metric:** tokens/second = `seq_len / mean_forward_seconds`. Peak VRAM via
  `torch.cuda.max_memory_allocated()`.
- **Hybrid placement:** the early transformer layers have their token-mixing
  path replaced; later layers are unchanged. Baseline = the unmodified model.

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
