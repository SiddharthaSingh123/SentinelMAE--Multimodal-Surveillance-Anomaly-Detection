"""
models/videomae_finetune.py

Fine-tuning wrapper around VideoMAE (MCG-NJU/videomae-base) for
action/event classification with a custom classifier head.
"""

import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEConfig


class VideoMAEFineTuner(nn.Module):
    """
    VideoMAE backbone + classification head for video understanding.

    Args:
        backbone_name (str): HuggingFace model ID, e.g. 'MCG-NJU/videomae-base'
        num_classes   (int): Number of output classes
        dropout       (float): Dropout probability before classifier
        pretrained    (bool): Load pretrained weights from HuggingFace
    """

    def __init__(
        self,
        backbone_name: str = "MCG-NJU/videomae-base",
        num_classes: int = 5,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()

        if pretrained:
            self.backbone = VideoMAEModel.from_pretrained(backbone_name)
        else:
            config = VideoMAEConfig.from_pretrained(backbone_name)
            self.backbone = VideoMAEModel(config)

        hidden_size = self.backbone.config.hidden_size  # 768 for base

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, T, C, H, W)  — batch of video clips

        Returns:
            logits: (B, num_classes)
        """
        outputs = self.backbone(pixel_values=pixel_values)

        # Mean-pool over the sequence of patch tokens
        sequence_output = outputs.last_hidden_state        # (B, N, D)
        pooled = sequence_output.mean(dim=1)               # (B, D)

        logits = self.classifier(pooled)                   # (B, num_classes)
        return logits


# ─────────────────────────────────────────────────────────
#  Factory function — this is what main.py imports
# ─────────────────────────────────────────────────────────

def build_model(config: dict) -> VideoMAEFineTuner:
    """
    Build and return a VideoMAEFineTuner from the project config dict.

    Expected config structure (matches training/config.yaml):
        config['model']['backbone']   → HuggingFace model ID
        config['model']['pretrained'] → bool
        config['model']['dropout']    → float
        config['data']['num_classes'] → int
        config['project']['device']   → 'cpu' | 'cuda' | 'mps'

    Returns:
        model (VideoMAEFineTuner) moved to the configured device
    """
    model_cfg   = config.get("model", {})
    data_cfg    = config.get("data", {})
    project_cfg = config.get("project", {})

    backbone_name = model_cfg.get("backbone", "MCG-NJU/videomae-base")
    pretrained    = model_cfg.get("pretrained", True)
    dropout       = model_cfg.get("dropout", 0.1)
    num_classes   = data_cfg.get("num_classes", 5)
    device        = project_cfg.get("device", "cpu")

    print(f"[build_model] backbone={backbone_name}  "
          f"classes={num_classes}  pretrained={pretrained}  device={device}")

    model = VideoMAEFineTuner(
        backbone_name=backbone_name,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )

    model = model.to(device)
    return model


# ─────────────────────────────────────────────────────────
#  Quick smoke-test
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    dummy_config = {
        "project": {"device": "cpu"},
        "data":    {"num_classes": 5},
        "model":   {"backbone": "MCG-NJU/videomae-base",
                    "pretrained": False,     # skip download in test
                    "dropout": 0.1},
    }

    model = build_model(dummy_config)
    print(model)

    # VideoMAE-base expects 16 frames, 224×224
    B, T, C, H, W = 2, 16, 3, 224, 224
    dummy_input = torch.randn(B, T, C, H, W)
    logits = model(dummy_input)
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")   # expect (2, 5)