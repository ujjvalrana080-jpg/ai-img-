"""Predict whether an image is real or AI-generated."""
import argparse
from pathlib import Path

import torch
from PIL import Image

from dataset import get_transforms, CLASS_NAMES
from model import build_model, load_checkpoint


def load_image(path, transform, device):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    return x


def main():
    parser = argparse.ArgumentParser(description="Predict real vs AI-generated image")
    parser.add_argument("input", type=str, help="Image file or folder of images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(train=False)

    model = build_model(num_classes=2, pretrained=False).to(device)
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        load_checkpoint(ckpt, model, device=device)
    else:
        print(f"Checkpoint not found: {ckpt}. Run train.py first.")
        return

    model.eval()

    input_path = Path(args.input)
    if input_path.is_file():
        paths = [input_path]
    elif input_path.is_dir():
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
            paths.extend(input_path.glob(ext))
    else:
        print(f"Not found: {input_path}")
        return

    for path in paths:
        try:
            x = load_image(path, transform, device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
            pred = logits.argmax(dim=1).item()
            prob_ai = probs[1].item()
            label = CLASS_NAMES[pred]
            print(f"{path.name}: {label} (AI prob: {prob_ai:.3f})")
        except Exception as e:
            print(f"{path.name}: error - {e}")


if __name__ == "__main__":
    main()
