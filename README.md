# AI-Generated Image Detector

Binary classifier (ML) trained to detect whether an image is **real** or **AI-generated**. Uses a ResNet18 backbone with ImageNet pretraining, fine-tuned on your own dataset.

## Setup

```bash
cd ai-image-detector
pip install -r requirements.txt
```

## Dataset

Organize images into two folders:

- `data/real/` — real (human-captured) images  
- `data/ai/` — AI-generated images (e.g. DALL·E, Midjourney, Stable Diffusion)

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`. Use at least a few hundred images per class for reasonable results.

## Train

```bash
python train.py --data data --epochs 10 --batch-size 32
```

Options:

- `--data` — path to folder containing `real/` and `ai/` (default: `data`)
- `--epochs` — number of training epochs (default: 10)
- `--batch-size` — batch size (default: 32)
- `--lr` — learning rate (default: 1e-3)
- `--val-split` — fraction used for validation (default: 0.2)
- `--save-dir` — where to save checkpoints (default: `checkpoints`)

Best model is saved as `checkpoints/best.pt`.

## Predict

Single image:

```bash
python predict.py path/to/image.jpg --checkpoint checkpoints/best.pt
```

All images in a folder:

```bash
python predict.py path/to/folder/ --checkpoint checkpoints/best.pt
```

Output format: `filename: real|ai (AI prob: 0.xxx)`.

## Notes

- Detection is **not perfect**; state-of-the-art benchmarks report ~58% accuracy on strong generators. Use as a signal, not ground truth.
- Better results come from more and more diverse data (many sources of real and AI images).
- For production, consider larger backbones (e.g. ResNet50), more epochs, and public datasets (e.g. COCOAI, VCT2) if you can use them.
