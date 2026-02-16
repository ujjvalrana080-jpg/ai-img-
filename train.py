"""Train a binary classifier: real vs AI-generated images."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RealVsAIDataset, get_transforms, CLASS_NAMES
from model import build_model, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train AI-generated image detector")
    parser.add_argument("--data", type=str, default="data", help="Root folder with real/ and ai/ subfolders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction for validation")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data folder not found: {data_path}")
        print("Create folders: data/real/ and data/ai/ and put images in them.")
        return

    train_dataset = RealVsAIDataset(
        args.data,
        transform=get_transforms(train=True),
    )
    val_dataset = RealVsAIDataset(
        args.data,
        transform=get_transforms(train=False),
    )
    n = len(train_dataset)
    if n == 0:
        print("No images found in data/real/ or data/ai/.")
        return

    val_size = int(n * args.val_split)
    train_size = n - val_size
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    train_ds = torch.utils.data.Subset(train_dataset, train_idx.tolist())
    val_ds = torch.utils.data.Subset(val_dataset, val_idx.tolist())
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch + 1,
                save_dir / "best.pt",
                metrics={"val_acc": val_acc, "val_loss": val_loss},
            )
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    save_checkpoint(
        model, optimizer, args.epochs,
        save_dir / "last.pt",
        metrics={"val_acc": val_acc, "val_loss": val_loss},
    )
    print(f"Training done. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
