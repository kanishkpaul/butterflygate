# Independent block benchmark

`bench_forward.py` is a mechanism-agnostic utility for timing a user-supplied PyTorch module on random activations. It includes warmup and peak-memory tracking. It did not produce the historical full-model result tables.

```bash
python -m pip install torch
python benchmark/bench_forward.py --factory my_package.mixers:make_block --lengths 512 1024 2048 4096
```

The factory must return a module that accepts a `[1, sequence_length, d_model]` tensor. Set `--d-model` to match it. The utility chooses CUDA when available and otherwise uses CPU; it has no MPS path. Use `--dtype fp32` for CPU checks. No default mixer or ButterflyGate implementation is supplied.

Its memory column uses decimal GB, unlike the historical tables' GiB. Utility outputs should not be pooled with the full-model observations. A failure ends its length sweep; that policy does not prove that all larger lengths would fail.
