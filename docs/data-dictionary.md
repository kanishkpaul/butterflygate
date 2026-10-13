# Data dictionary

## Measurements

The three `measurements.csv` files share a schema. `additional-probes.csv` uses the same schema.

| Column | Meaning |
| :--- | :--- |
| `run_id` | Public identifier for a source run; does not expose its private path |
| `variant` | `baseline` or an anonymous experiment-local configuration label |
| `seq_len` | Input sequence length in tokens |
| `batch_size` | Sequences per batch |
| `max_batches` | Requested maximum evaluated batches; not timing repetitions |
| `dtype` | Recorded numeric precision |
| `wall_seconds` | Total wall time for the timed evaluation region |
| `tokens_per_second` | Input tokens processed / wall time |
| `peak_memory_gib` | Peak allocated GPU memory in GiB, including model allocations |
| `mean_loss` | Recorded mean next-token loss |
| `perplexity` | Exponential of recorded mean loss; limited-sample observation |
| `token_accuracy` | Correct next-token argmax predictions / scored tokens, from 0 to 1 |
| `status` | Whether a measurement completed or failed |

Blank metrics are missing measurements, not zero. Gemma's historical rows contain loss and perplexity but do not contain token accuracy. Values are retained at source precision. Dates, model/dataset identifiers, and shared conditions are in each `metadata.json`.

## Status categories

| Status | Meaning |
| :--- | :--- |
| `measured` | Evaluation completed and produced timing/metric values |
| `allocator_failure` | Allocator/NVML assertion before valid timing |
| `allocator_failure_after_prior_failure` | Sequential attempt failed after an earlier allocator failure |
| `quota_blocked` | Requested execution was blocked by ZeroGPU quota |
| `not_attempted_after_quota` | No attempt was made after quota exhaustion |

The 8B `coverage.csv` records scheduled throughput coverage, including observations missing from `measurements.csv`. It does not describe architecture or information reachability.

## Provenance and checksums

`provenance.csv` associates anonymous source-artifact identifiers with experiment families and SHA-256 hashes. Original artifacts are private, so these are identifiers rather than publicly verifiable source evidence. `SHA256SUMS` covers the exported CSV and JSON files and can be verified locally.
