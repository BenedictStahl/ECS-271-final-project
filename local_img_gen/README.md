# Generating AI images from MS COCO captions

Generates one image per MS COCO caption with a diffusers text-to-image model (default: `Qwen/Qwen-Image-2512`). Each image is saved as JPEG with a line in `metadata.jsonl` (file name, generator, prompt, COCO image ID, seed).

## Setup (RunPod, 80 GB GPU)

```bash
export HF_HOME=/workspace/hf_cache
pip install diffusers==0.43.0 transformers==5.23.0 accelerate==1.16.0 huggingface_hub==2.4.0 pillow==12.3.0
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip
```

## Run

```bash
python generate_images.py \
  --captions-file annotations/captions_train2017.json \
  --output-dir generated \
  --count 3000
```

Add `--limit 20` for a test run. Finished images are skipped, so the script can be restarted. Run `python generate_images.py --help` for all options.

For two machines, use `--order forward --limit N` on one and `--order reverse --limit M` on the other.
