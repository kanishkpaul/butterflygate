# Measurement protocol

This document describes the historical private full-model experiments represented in the public tables. The independent utility in `benchmark/` uses a different protocol and did not produce these rows.

## Shared conditions

- Forward-only evaluation of pretrained weights with untrained experimental replacements.
- Batch size 1 and `bf16`, using `eval()` and disabled gradient calculation.
- Model caching disabled; these are full-sequence passes, not cached token generation.
- Hugging Face ZeroGPU. The released records do not establish the physical accelerator model for each row; platform labels are not substituted for a verified device inventory.
- No warmup passes before timing. Each model/configuration/length row loads a model anew.
- Wall-clock timing with CUDA synchronization before and after the timed region.
- The timed region includes batch transfers, model forward/loss calculation, loss transfers to CPU, and, where recorded, next-token argmax and accuracy calculation. Tokenization and model loading occur outside it.
- Throughput is input tokens processed divided by wall time. `max_batches` is the requested maximum, not an independent repetition count. Losses are averaged over the evaluated batches; perplexity is the exponential of that mean loss.
- Peak memory is PyTorch peak allocated GPU memory, divided by 2^30. It is not total device occupancy, reserved memory, or mixer-only activation memory. Peak tracking is reset before model loading, so model allocations are included.

## Experiment-specific conditions

| Experiment | Date (UTC) | Data | Batch budget |
| :--- | :--- | :--- | :--- |
| Gemma E2B | 2026-05-24 | `allenai/dolma:v1_6-sample` | 3 for the primary short sweep; 1 for longer probes |
| Llama 1B | 2026-05-24 | `allenai/dolma:v1_6-sample` | 1 per row |
| Llama 8B | 2026-05-25 | `deepmind/pg19`, validation | 1 per row |

Document identities, token-stream hashes, package versions, and random seeds are not fully recorded in the released artifacts. Measurements at different lengths are not independent random samples of the datasets.

## Ordering and restarts

The full-model sweeps used baseline first. Configuration execution order within the 8B experiment is not represented by the CSV's sorted row order. Cold-run and ordering effects therefore apply to more than a single point.

Gemma's primary short sweep and longer probes are distinct runs. Fresh-restart Gemma probes coexist with sequential limit attempts; `run_id` preserves those boundaries. Llama 1B used fresh Space restarts for each length. The 8B sweep stopped producing measurements when ZeroGPU quota became insufficient.

## Failures and missing observations

Gemma's fresh 16,384-token probe and both Llama 1B configurations at 65,536 tokens failed before producing valid timings with a CUDA allocator/NVML assertion. This is recorded as an allocator failure, not a confirmed out-of-memory ceiling.

Some Gemma sequential probes continued after an earlier allocator failure. Their status is `allocator_failure_after_prior_failure`; they are not independent evidence about larger-context feasibility.

The 8B run completed all six configurations through 8,192 tokens and four at 16,384. Two attempts at 16,384 and four at 32,768 were quota-blocked; later configurations and lengths were not attempted after quota exhaustion. The dedicated perplexity phase and training phase produced no completed result. Quality metrics already present in the speed rows remain available.

## Provenance

[Source checksums](../results/provenance.csv) identify the private artifacts used for curation without exposing their filenames or repository history. A private source-to-public mapping is retained outside this repository. Public [SHA256SUMS](../results/SHA256SUMS) verify the released files.

These checksums provide artifact identity and integrity, not independent validation of the experiment. Historical values are preserved at source precision; no corrected rerun or estimated missing result has been substituted.
