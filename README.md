# ManwhaHDFyer

AI-powered manhwa upscaler built for readers who are tired of squinting at compressed webtoon panels. Takes low-quality CBZ chapters downloaded from manga reader apps and produces crisp, high-resolution output using Real-ESRGAN with 4x-UltraSharp.

**33 min/chapter → 6.6 min locally, ~1.5 min on cloud A100.** Zero temp files. 530MB output → 29MB. No doubled panels.

Built from scratch as a learning project in ML inference, image processing, and software architecture.

---

## Before / After

| Source (720px, compressed JPEG) | Upscaled (1440px, 2x, JPEG 92) |
|---|---|
| Blurry line art, compression artifacts, banding | Sharp lines, clean gradients, readable text |

---

## Features

- **GPU-accelerated upscaling** — FP16 inference on NVIDIA GPUs via Real-ESRGAN
- **Smart chunking** — Automatically detects tall scroll panels (720×4000+) and splits them into GPU-friendly chunks with feathered blending. Normal-ratio images go through direct upscale.
- **Zero temp files** — Images stream from source CBZ through GPU directly into output CBZ. No intermediate files touch disk.
- **Post-upscale sharpening** — Unsharp mask applied after the model runs, preserving line art crispness without amplifying source compression artifacts
- **Black area cleanup** — Near-black pixels flattened to pure black, eliminating patchiness from model hallucination on solid regions
- **Credit page skip** — Last N pages per chapter pass through without GPU processing
- **Resume support** — Skips chapters already processed. Safe to restart interrupted batch jobs.
- **Cloud-ready** — Same code runs locally or on Lambda.ai / Vast.ai GPU instances

---

## Architecture

```
ManwhaHDFyer/
├── core/                    ← The brain (delivery-agnostic)
│   ├── config.py            ← Dataclass-based settings (engine, output, paths)
│   ├── engine.py            ← Model loading, smart chunking, upscaling, sharpening
│   ├── extractor.py         ← CBZ → images (generator, no temp files)
│   └── packager.py          ← Images → CBZ (stream to zip, context manager)
├── pipeline/                ← Orchestration
│   └── batch.py             ← Chapter loop, logging, timing, skip logic
├── models/                  ← Weight files (gitignored, download manually)
├── Real-ESRGAN/             ← Cloned + patched inference library (gitignored)
├── legacy/                  ← Original MVP (preserved as baseline)
├── tests/                   ← Test stubs
├── main.py                  ← CLI entry point
└── requirements.txt
```

**Data flow for a single page:**
```
CBZ on disk
  → extractor reads zip entry → decodes to numpy array (in memory)
    → engine checks aspect ratio → direct or chunked upscale on GPU
      → post-sharpen → black cleanup
    → packager encodes to JPEG bytes → writes into output zip
      → memory freed, next page
→ Output CBZ on disk (only file written)
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- 8GB+ VRAM recommended (tested on RTX 4060 Laptop 8GB, A100 40GB)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/ManwhaHDFyer.git
cd ManwhaHDFyer
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS/WSL
# or
venv\Scripts\activate           # Windows
```

### 3. Install PyTorch with CUDA

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and get the install command for your CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Clone and install Real-ESRGAN

The pip version of `realesrgan` has broken dependencies as of 2025. Clone the repo and install in editable mode so you can patch issues:

```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -e .
cd ..
```

**Known fix required:** If you see `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`, edit the file that throws the error and replace:

```python
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```

with:

```python
from torchvision.transforms.functional import rgb_to_grayscale
```

### 5. Install remaining dependencies

```bash
pip install basicsr opencv-python Pillow
```

**Note:** If you get NumPy errors, downgrade to v1:

```bash
pip install "numpy<2"
```

### 6. Download the model weights

Download **4x-UltraSharp.pth** from [OpenModelDB](https://openmodeldb.info/models/4x-UltraSharp) and place it in the `models/` directory:

```
models/4x-UltraSharp.pth
```

---

## Usage

### Basic — one chapter test

```bash
python main.py --comic-folder "The Legend of the Northern Blade" --limit 1
```

### Full batch — all chapters

```bash
python main.py --comic-folder "The Legend of the Northern Blade"
```

### All options

```bash
python main.py \
  --comic-folder "My Manhwa Folder" \
  --scale 2 \
  --format jpg \
  --quality 92 \
  --limit 0
```

| Argument | Default | Description |
|---|---|---|
| `--comic-folder` | *required* | Path to folder containing CBZ files |
| `--scale` | 2 | Upscale factor (2 or 4) |
| `--format` | jpg | Output format: `jpg` or `png` |
| `--quality` | 92 | JPEG quality 1-100 (ignored for PNG) |
| `--limit` | 0 | Max chapters to process (0 = all) |

Results are saved to `results/` with the same filenames as the source CBZ files.

---

## Cloud Deployment (Lambda.ai / Vast.ai)

For large batches, rent a cloud GPU. Same code, bigger hardware.

### Quick start on Lambda

```bash
# SSH into your instance
ssh -i ~/.ssh/your-key.pem ubuntu@INSTANCE-IP

# Upload your project
scp -i ~/.ssh/your-key.pem -r ManwhaHDFyer/ ubuntu@INSTANCE-IP:~/

# On the instance: install dependencies
cd ~/ManwhaHDFyer
pip install basicsr opencv-python Pillow "numpy<2"
cd Real-ESRGAN && pip install -e . && cd ..

# Fix the torchvision import if needed (see Setup step 4)

# Run
python main.py --comic-folder "The Legend of the Northern Blade"

# Download results locally, then TERMINATE the instance
scp -i ~/.ssh/your-key.pem -r ubuntu@INSTANCE-IP:~/ManwhaHDFyer/results/ ./cloud_results/
```

### Performance benchmarks

| GPU | Per Chapter (35 pages) | Per Chapter (86 pages) | 93 Chapters | Cost |
|---|---|---|---|---|
| RTX 4060 Laptop (8GB) | ~6.6 min | ~15 min | ~11 hours | Free |
| A100 SXM4 (40GB) | ~1.5 min | ~4 min | ~2.5 hours | ~$3-5 |
| GH200 (96GB) | ~0.5 min | ~1.5 min | ~1 hour | ~$2-3 |

---

## How It Works

**The model:** 4x-UltraSharp is an RRDBNet (Residual-in-Residual Dense Block Network) trained using GAN (Generative Adversarial Network) methods. At inference time, only the generator runs — a deep CNN that transforms low-res pixels into high-res output. The model was trained at 4x scale; 2x output is achieved by downscaling the 4x result.

**Smart chunking:** Manhwa pages are typically 720×4000+ pixels — extreme aspect ratios that break standard tiling algorithms. The engine checks each image's aspect ratio: normal images go through Real-ESRGAN's built-in tiler, while tall panels are sliced into ~720px vertical chunks with 128px overlap, upscaled individually, then stitched with linear alpha blending across the overlap zones to eliminate seams.

**Post-processing:** After upscaling, a gentle unsharp mask recovers line art detail that the model softens. Near-black pixels (RGB all below 15) are clamped to pure black to eliminate patchiness in solid dark areas.

---

## Project History

Started as a 180-line monolithic script (preserved in `legacy/mvp_upscaler.py`) that took 33 minutes per chapter, wrote 580MB of temp files, and produced doubled panels from a buggy stitching algorithm. Rebuilt from scratch with modular architecture, generator-based streaming, feathered blending, and cloud deployment support.

---

## Roadmap

- [x] MVP upscaler (proof of concept)
- [x] Modular architecture (core/pipeline separation)
- [x] Feathered blending for chunk stitching
- [x] FP16 inference
- [x] Post-sharpen + black area fix
- [x] Batch processing with resume
- [x] Cloud GPU deployment
- [ ] A GUI comparer to compare upscaled vs original.
- [ ] Edge-aware sharpening (sharpen lines, preserve flat areas)
- [ ] GPU batching (multiple images per inference call)
- [ ] Write own inference wrapper (replace Real-ESRGAN dependency)
- [ ] Reverse proxy for real-time Mihon integration
- [ ] Queue-based processing with approval workflow
- [ ] Support for multiple models (anime-specific, photo, etc.)

---

## License

MIT
