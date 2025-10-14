# jukebox-hic

Lightweight Hi-C noise analysis toolkit that depends only on the Python [`hicstraw`](https://github.com/aidenlab/straw) bindings.

## Features
- Per-row noise estimates with optional subsampling simulations to gauge robustness.
- Works with `.hic`, `.cool`, `.mcool`, and `.scool` input matrices (SciPy-backed pipeline for cooler formats).
- Whole-genome noise tracks at requested resolutions.
- Simple plotting helpers for comparing noise distributions (log-scaled histograms).

## Installation
```bash
pip install jukebox-hic
```
Or clone the repository and install the package in editable mode:
```bash
pip install -e .
```

## Command Line Usage
### Sampled noise with subsampling summaries
```bash
jukebox-hic sample-noise \
  --hic sample.cool \
  --res 10000 \
  --subsample_ratios 1.0,0.75,0.5,0.25 \
  --norm balance \
  --out_dir outputs/10kb \
  --seed 42
```
Produces one `<chrom>_<res>.bedgraph` for the original map and a `subsample_summary.tsv`
table with mean/median/log noise, ACF, variability, and empty-bin ratios at each subsample.

### Full map noise
```bash
jukebox-hic full-noise \
  --hic sample.hic \
  --res 5000,10000 \
  --out_dir outputs/
```

### Plotting
```bash
jukebox-hic plot \
  --noise_bed outputs/chr1_10000_sampled.bedgraph \
  --out_png noise_density.png
```

## Python API
```python
from jukebox_hic import noise_sampling

noise_sampling.compute_sampled_noise(
    hic_path="sample.cool",
    res=10_000,
    sample_fraction=1.0,
    subsample_ratios=[1.0, 0.5, 0.25],
    norm="balance",
    out_dir="outputs",
    seed=42,
)
```

### Notes
- For `.mcool`, the CLI selects the group matching `--res` (e.g. `/resolutions/10000`). Provide `--cooler_path` for `.scool` cell groups or custom URIs.
- `--norm` accepts `none`, `balance` (cooler weights / KR for `.hic`), or a path to a bedgraph/vector defining custom bin weights.
