# Experiment index

Each experiment folder contains a readable report, full measurement rows, and essential conditions. Configuration labels are local to each folder.

| Folder | Model / data | Primary rows | Extra records |
| :--- | :--- | ---: | :--- |
| [gemma-dolma](gemma-dolma/README.md) | Gemma E2B / Dolma sample | 20 | 28 additional sequential-probe rows |
| [llama1b-dolma](llama1b-dolma/README.md) | Llama 3.2 1B / Dolma sample | 16 | Includes two allocator-failure rows |
| [llama8b-pg19](llama8b-pg19/README.md) | Llama 3.1 8B / PG19 validation | 34 | 54 scheduled throughput statuses |

The primary tables preserve missing metrics and unsuccessful rows. Additional Gemma probes are not pooled into an average or silently substituted into the main figure. [Protocol](../docs/measurement-protocol.md) · [Data dictionary](../docs/data-dictionary.md)

[Source artifact hashes](provenance.csv) · [Public file checksums](SHA256SUMS)

The old timing-only CSVs are superseded by these richer tables and remain available in the repository's earlier public history. The new layout is the canonical location for released measurements.
