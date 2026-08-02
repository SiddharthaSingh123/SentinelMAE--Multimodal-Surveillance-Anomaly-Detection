"""
evaluate.py
Evaluates the trained SentinelMAE model on the validation set.
Produces accuracy, per-class metrics, and confusion matrix.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from models.videomae import VideoMAEClassifier
from config import get_config


# ── Dataset ────────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    def __init__(self, processed_dir="data/processed/val"):
        self.clips = []
        self.labels = []
        self.classes = sorted([d.name for d in Path(processed_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_dir in Path(processed_dir).iterdir():
            if not class_dir.is_dir():
                continue
            for clip_path in class_dir.glob("*.pt"):
                self.clips.append(clip_path)
                self.labels.append(self.class_to_idx[class_dir.name])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = torch.load(self.clips[idx])   # (T, C, H, W)
        clip = clip.float() / 255.0
        label = self.labels[idx]
        return clip, label


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str):
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = VideoMAEClassifier(num_classes=cfg.num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.to(device)
    model.eval()

    # Load val dataset
    dataset = ClipDataset("data/processed/val")
    loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    classes = dataset.classes
    print(f"Val clips: {len(dataset)} | Classes: {classes}")

    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for clips, labels in tqdm(loader, desc="Evaluating"):
            clips  = clips.to(device)
            logits = model(clips)
            probs  = torch.softmax(logits, dim=-1)
            preds  = probs.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ── Metrics ──
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")
    print("\nPer-Class Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes,
                yticklabels=classes, cmap="Blues")
    plt.title(f"Confusion Matrix — Accuracy: {acc*100:.2f}%")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_model.pth")
    args = parser.parse_args()
    evaluate(args.checkpoint)