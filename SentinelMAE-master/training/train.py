"""
training/train.py
"""
import argparse, math, sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.videomae_finetune import build_model
from utils.config_loader import load_config
from utils.logger import get_logger
from utils.metrics import AverageMeter, accuracy


class VideoClipDataset(Dataset):
    def __init__(self, root, num_frames=16, image_size=224, num_classes=5):
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_classes = num_classes
        self.root = Path(root)
        self.samples = []
        if self.root.exists():
            for label, d in enumerate(sorted(p for p in self.root.iterdir() if p.is_dir())):
                for clip in sorted(d.glob("*.pt")):
                    self.samples.append((clip, label))
        self._random = len(self.samples) == 0

    def __len__(self):
        return 32 if self._random else len(self.samples)

    def __getitem__(self, idx):
        if self._random:
            return torch.randn(self.num_frames, 3, self.image_size, self.image_size), \
                   torch.randint(0, self.num_classes, ()).item()
        path, label = self.samples[idx]
        clip = torch.load(path)
        if clip.shape[0] != self.num_frames:
            idx_t = torch.linspace(0, clip.shape[0]-1, self.num_frames).long()
            clip = clip[idx_t]
        return clip, label


def get_lr(epoch, cfg):
    total = cfg["training"]["epochs"]
    warmup = cfg["scheduler"]["warmup_epochs"]
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(model, loader, criterion, optimizer, device, grad_clip, phase, logger, epoch):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    loss_m, acc_m = AverageMeter("Loss"), AverageMeter("Acc")
    pbar = tqdm(loader, desc=f"[{phase:5s} E{epoch:03d}]", leave=False)
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for clips, labels in pbar:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            n = clips.size(0)
            loss_m.update(loss.item(), n)
            acc_m.update(accuracy(logits, labels), n)
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg:.2f}%")
    logger.info(f"[{phase:5s} E{epoch:03d}]  loss={loss_m.avg:.4f}  acc={acc_m.avg:.2f}%")
    return loss_m.avg, acc_m.avg


def save_checkpoint(state, save_dir, filename):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, Path(save_dir) / filename)


def train(cfg, resume=None):
    logger = get_logger("sentinel.train")
    device_str = cfg["project"]["device"]
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else \
                     "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Device: {device}")
    torch.manual_seed(cfg["project"].get("seed", 42))

    dc = cfg["data"]
    train_ds = VideoClipDataset(dc["train_path"], dc["num_frames"], dc["image_size"], dc["num_classes"])
    val_ds   = VideoClipDataset(dc["val_path"],   dc["num_frames"], dc["image_size"], dc["num_classes"])
    train_loader = DataLoader(train_ds, batch_size=dc["batch_size"], shuffle=True,  num_workers=dc["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=dc["batch_size"]*2, shuffle=False, num_workers=dc["num_workers"])
    logger.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    cfg["project"]["device"] = device_str
    model = build_model(cfg)
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                      weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    start_epoch, best_val_acc = 0, 0.0
    if resume:
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_val_acc = ckpt["epoch"] + 1, ckpt["best_val_acc"]
        logger.info(f"Resumed from epoch {ckpt['epoch']}")

    tc = cfg["training"]
    for epoch in range(start_epoch, tc["epochs"]):
        lr_mult = get_lr(epoch, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = tc["learning_rate"] * lr_mult
        logger.info(f"Epoch {epoch:03d}  lr={optimizer.param_groups[0]['lr']:.6f}")
        run_epoch(model, train_loader, criterion, optimizer, device, tc["gradient_clip"], "train", logger, epoch)
        _, val_acc = run_epoch(model, val_loader, criterion, None, device, tc["gradient_clip"], "val", logger, epoch)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "best_val_acc": best_val_acc, "config": cfg}
        save_checkpoint(state, tc["save_dir"], "latest.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(state, tc["save_dir"], "best.pt")
            logger.info(f"  ✓ New best: {best_val_acc:.2f}%")
    logger.info(f"Done. Best val acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(load_config(args.config), resume=args.resume)