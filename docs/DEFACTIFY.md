# Original Defactify compatibility workflow

The group's replacement dataset is primary. Original Defactify is retained for
reference alignment and a possible optional comparison. Its sources are different:
real, SD2.1, SDXL, SD3, DALL-E 3 and Midjourney.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe scripts\download_dataset.py
.\.venv\Scripts\python.exe scripts\profile_dataset.py
```

The downloader pins Rajarshi-Roy-research/Defactify_Image_Dataset revision
1c3b714c9691eb6dd11c2c1e92eab1ac29584fa0, downloads 17 Parquet shards to data/raw,
and writes a manifest. It downloads raw Parquet directly, avoiding the old Windows
Arrow-cache path issue. Explicit --data-dir accepts an alternative destination.
The current machine's retained original snapshot is under data/hf/hub (see DATA_LAYOUT).

The profiler is offline. It accepts a snapshot root containing data/*.parquet or
the shard directory. It reads all headers/labels/counts and by default fully
RGB-converts/resizes five images per source/split (90 total). It is not a full
corruption or perceptual-duplicate scan. It exits nonzero for image/label errors.

Verified original dataset: 96,000 RGB JPEG rows; 42K/9K/45K published splits; 5:1
synthetic:real. Source resolutions differ; real photos are rectangular and variable.
The original exact-byte audit found duplicates and caption flags; do not treat
unreviewed original splits as approved for the replacement experiment.

Default paths resolve relative to the repo, but explicit relative options resolve
relative to the shell's working directory. --local-files-only requests cached
files only. --revision main intentionally changes the revision; use a new data
destination to avoid mixing leftovers. The original team notebook is unchanged
and has its own cache/working-directory assumptions.

The old data/a Arrow copy was removed only after verifying that all 96,000
image-byte/caption/label records match the retained Parquet data. It can be rebuilt
by the older notebook workflow if needed. Current scripts consume Parquet directly.
