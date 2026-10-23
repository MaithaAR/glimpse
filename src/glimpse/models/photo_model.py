"""PhotoCarbModel: a pretrained ConvNeXt-Tiny (image network) with two small new outputs ("heads") for carbohydrate.

- Head 1 predicts log(1 + grams): logs tame the very large meals, and grams = exp(output) - 1.
- Head 2 predicts one of 4 carb bins (<30, 30-<60, 60-<90, >=90 g). Gram labels are noisy estimates; a coarse
  range is easier to learn and enough for spike risk (also asked for in the proposal feedback).
The backbone starts frozen (only the heads learn); its last stage can be unfrozen if inner folds show a gain."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

CARB_BIN_EDGES = (30.0, 60.0, 90.0)
STATE_VERSION = 1


def carb_bin(grams: torch.Tensor, edges: tuple[float, ...] = CARB_BIN_EDGES) -> torch.Tensor:
    """Bin index for grams; a value exactly on an edge goes to the upper bin (30 g -> bin 1)."""
    return torch.bucketize(grams, torch.tensor(edges, dtype=grams.dtype), right=True)


class PhotoCarbModel(nn.Module):
    """ConvNeXt-Tiny backbone (ImageNet classifier removed) -> 768 numbers per photo -> two heads."""

    def __init__(self, arch: str = "convnext_tiny.fb_in22k_ft_in1k", weights: Path | None = None,
                 dropout: float = 0.2, n_bins: int = len(CARB_BIN_EDGES) + 1) -> None:  # fmt: skip
        """Build the backbone without downloading anything; load the verified local weights file if given."""
        super().__init__()
        import timm

        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        if weights is not None:
            from safetensors.torch import load_file

            missing, unexpected = self.backbone.load_state_dict(load_file(str(weights)), strict=False)
            if missing or any(
                not k.startswith("head.fc") for k in unexpected
            ):  # only the ImageNet classifier is dropped
                raise ValueError(f"weights do not match {arch}: missing {missing[:3]}, unexpected {unexpected[:3]}")
        width = self.backbone.num_features
        self.reg_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))
        self.bin_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, n_bins))
        self.set_trainable(last_stage=False)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Photos (batch, 3, 224, 224) -> backbone summary (batch, 768)."""
        return self.backbone(images)

    def heads(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Summary -> (log(1+grams) of shape (batch,), bin scores of shape (batch, 4))."""
        return self.reg_head(emb).squeeze(-1), self.bin_head(emb)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full path used by the demo."""
        return self.heads(self.embed(images))

    def set_trainable(self, last_stage: bool) -> None:
        """Freeze the whole backbone (only heads learn), or also unfreeze its last stage and final norm."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in list(self.backbone.stages[-1].parameters()) + list(self.backbone.head.parameters()):
            p.requires_grad = last_stage
        for p in list(self.reg_head.parameters()) + list(self.bin_head.parameters()):
            p.requires_grad = True

    def trainable_state(self, backbone_sha256: str) -> dict:
        """Only the parts training can change (last stage, final norm, heads), tagged with the backbone fingerprint
        so they are never combined with different base weights."""
        keep = ("backbone.stages.3.", "backbone.head.", "reg_head.", "bin_head.")
        return {"version": STATE_VERSION, "backbone_sha256": backbone_sha256,
                "state": {k: v for k, v in self.state_dict().items() if k.startswith(keep)}}  # fmt: skip


def n_trainable(model: nn.Module) -> int:
    """Number of parameters that currently learn."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
