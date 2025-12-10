## Part 1 – CIFAR-10 benchmark (airbench94_changed.py)

### How to run
From the repo root:
```bash
python airbench94_changed.py
```

### What it does
- Automatically downloads CIFAR-10 the first time it runs.
- Caches preprocessed data under `cifar10/` for faster re-runs.
- Trains a small CNN and prints mean / standard-deviation test accuracy and timing to the terminal.
- Raw logs are saved under:
  ```
  logs/<uuid>/log.pt
  ```

## Part 2 – NanoGPT benchmark (train_gpt.py)

### 1. Download and cache FineWeb-10B
From the repo root:
```bash
python cached_fineweb10B.py 9
```
This creates `data/fineweb10B/` with the pre-tokenized `.bin` shards.

### 2. Select an experiment
At the top of `train_gpt.py`, set:
```python
EXPERIMENT = "baseline"      # or "lazy_2", "lazy_4", "lazy_sched"
```

### 3. Run training
Single-GPU run:
```bash
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
(For multi-GPU, increase `--nproc_per_node` to the number of GPUs, e.g. 8 on an 8×H100 node.)

### Outputs
- Training logs are written to `logs/`
- Parsed CSV metrics for each run are stored in `logs/metrics/` and can be used to make the plots in the report
```
