# Limitations

## Speed and predictive quality are different outcomes

The replacement configurations are untrained. The Llama measurements show severe quality degradation. No completed fine-tuning result or quality recovery is included. A faster pass through a degraded model is not evidence of a useful attention replacement.

Gemma's recorded baseline perplexities are also very high. The 8B PG19 baseline perplexities are unusually low for interpretation as a representative dataset score. Both require validation of sample contents, model loading, and scoring. Preserve these observations, but do not infer benchmark-level quality, memorization, or an evaluation bug from the numbers alone.

## These are preliminary timing observations

There is no warmup, configuration order is fixed, and each row reloads the model. Timing includes transfers, loss synchronization, and available accuracy calculations. The 512-token Gemma baseline is an outlier; startup effects are a plausible explanation, not an experimentally isolated cause.

Most longer probes and all 8B rows evaluate one batch. There are no robust confidence intervals, latency distributions, or controlled device comparisons. Even the longer-context ratios remain provisional.

## Configuration comparisons are not isolated architectural ablations

Anonymous labels distinguish measurements while keeping implementation details private. Configurations can differ in more than one respect. A ratio is a comparison of the recorded configurations, not proof that one isolated design choice caused it.

Labels are scoped to an experiment. Variant-A in two model families does not assert identical implementation. Public results alone cannot establish the computational complexity, causal correctness, information coverage, or generality of the private method.

## Scope is forward evaluation

No generation latency, KV-cache behavior, training efficiency, downstream task accuracy, or production deployment performance is established here. Earlier public descriptions discussed asymptotic mixer scaling; these hybrid whole-model observations do not validate a whole-model sub-quadratic claim.

## Failure does not define a universal ceiling

Allocator assertions are runtime failures. Quota blocks are scheduling/resource limits. An unattempted length has no result. None establishes the maximum context length of the architecture.

## Reproducibility has a deliberate boundary

Readers can inspect tables, check public-file integrity, and regenerate every figure. The method, adapters, exact configurations, full-model runner, and training work remain private while research continues. Full independent reproduction of the measurements is therefore unavailable.
