# jukebox-hic

Lightweight Hi-C noise analysis toolkit for `.hic` (via optional [`hicstraw`](https://github.com/aidenlab/straw)) and cooler formats.

## Features
- Per-row noise estimates with optional subsampling simulations to gauge robustness.
- Configurable row filtering based on distance-normalized signal and window occupancy.
- Works with `.hic`, `.cool`, `.mcool`, and `.scool` input matrices (SciPy-backed pipeline for cooler formats).
- Whole-genome noise tracks at requested resolutions.
- Simple plotting helpers for comparing noise distributions (log-scaled histograms).

## Installation
```bash
pip install jukebox-hic
```
To enable `.hic` inputs, install the optional extra that pulls in `hicstraw`:
```bash
pip install "jukebox-hic[straw]"
```
Or clone the repository and install the package in editable mode:
```bash
pip install -e .
# For optional runtime/memory profiling support
pip install -e .[profile]
```

## Command Line Usage
### Sampled noise with subsampling summaries
```bash
jukebox-hic sample-noise \
  --hic sample.cool \
  --res 10000 \
  --min_mean_score 0.9 \
  --min_nonzero_frac 0.05 \
  --subsample_ratios 1.0,0.75,0.5,0.25 \
  --norm balance \
  --out_dir outputs/10kb \
  --seed 42 \
  --profile outputs/10kb/profile.csv
```
Produces one `<chrom>_<res>.bedgraph` for the original map and a `subsample_summary.tsv`
table with mean/median/log noise, ACF, variability, and empty-bin ratios at each subsample.

### Full map noise
```bash
jukebox-hic full-noise \
  --hic sample.hic \
  --res 5000,10000 \
  --out_dir outputs/
# Add --cpu <N> to parallelize per (chrom, res) tasks
jukebox-hic full-noise \
  --hic sample.hic \
  --res 5000,10000 \
  --out_dir outputs/ \
  --cpu 4
# For .mcool/.scool inputs you can add: --cooler_path /resolutions/10000
```

### Plotting
```bash
jukebox-hic plot \
  --noise_bed outputs/chr1_10000_sampled.bedgraph \
  --out_png noise_density.png
```

### Build a blacklist from bedgraphs
```bash
jukebox-hic blacklist \
  --input outputs/10kb/*.bedgraph \
  --out blacklist.bed \
  --top_quantile 0.95
```
This command removes any rows with `NaN` noise values and blacklists the top 5% most
noisy intervals by default. Provide `--zscore_cutoff` to supply an explicit z-score
threshold instead of using the percentile.

## Python API
```python
from jukebox_hic import noise_sampling

noise_sampling.compute_sampled_noise(
    hic_path="sample.cool",
    res=10_000,
    sample_fraction=1.0,
    min_mean_score=0.9,
    min_nonzero_frac=0.05,
    subsample_ratios=[1.0, 0.5, 0.25],
    norm="balance",
    out_dir="outputs",
    seed=42,
)
```

### Notes
- For `.mcool`, the CLI selects the group matching `--res` (e.g. `/resolutions/10000`). Provide `--cooler_path` for `.scool` cell groups or custom URIs.
- `--norm` accepts `none`, `balance` (cooler weights / KR for `.hic`), or a path to a bedgraph/vector defining custom bin weights.
- All commands print a summary line with wall-clock time and memory usage; install the optional `profile` extra for psutil-backed RSS reporting.
- `--profile` writes per-chromosome runtime and memory metrics to a CSV for downstream plotting.
- `--cpu <N>` parallelizes `full-noise` by `(chromosome, resolution)` tasks (default 1, capped at available cores).
- Every CLI invocation now emits a `contacts_overview.tsv` summarizing per-chromosome totals at the coarsest resolution.

### If hic-straw fails to install
If `pip install "jukebox-hic[straw]"` fails (e.g. due to missing prebuilt wheels), convert the `.hic` file to a `.mcool` archive and use the cooler backend:
```bash
pip install hic2cool
hic2cool convert sample.hic sample.mcool --resolutions 10000
jukebox-hic sample-noise \
  --hic sample.mcool \
  --cooler_path /resolutions/10000 \
  --res 10000 \
  --out_dir outputs/10kb
```
The same `--cooler_path` argument selects groups inside `.scool` files.
