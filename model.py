"""ResNet-based binary classifier for real vs AI-generated images."""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, pretrained=True, freeze_backbone_epochs=0):
    """Build ResNet18 backbone with binary classification head."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone


def save_checkpoint(model, optimizer, epoch, path, metrics=None):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if metrics is not None:
        state["metrics"] = metrics
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load checkpoint; optionally restore optimizer."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state.get("epoch", 0), state.get("metrics")
