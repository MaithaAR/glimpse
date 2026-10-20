"""Fine-tuned photo model: pretrained ConvNeXt-Tiny -> served carbohydrate (log-grams) and a 4-bin carb class."""

from __future__ import annotations

import torch
from torch import nn

CARB_BIN_EDGES = (30.0, 60.0, 90.0)  # grams: <30, 30-60, 60-90, >90


def carb_bin(grams: torch.Tensor) -> torch.Tensor:
    """Bin index (0-3) for carbohydrate grams: <30, 30-<60, 60-<90, >=90 (an edge value goes to the upper bin)."""
    return torch.bucketize(grams, torch.tensor(CARB_BIN_EDGES, dtype=grams.dtype), right=True)


class PhotoCarbModel(nn.Module):
    """ConvNeXt-Tiny split into a frozen lower part (stem + stages 0-2) and a trainable upper part (stage 3 + pooling),
    followed by two small heads. Splitting lets the lower part run once per photo and be cached, so training on CPU
    stays fast while stage 3 can still be fine-tuned."""

    def __init__(self, arch: str = "convnext_tiny.fb_in22k_ft_in1k", pretrained: bool = True,
                 cache_dir: str | None = None, dropout: float = 0.2) -> None:  # fmt: skip
        """Load the backbone (pretrained weights from the local cache when `pretrained`) and add fresh heads."""
        super().__init__()
        import timm

        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, **kwargs)
        width = self.backbone.num_features
        self.reg_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, 1))
        self.bin_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, len(CARB_BIN_EDGES) + 1))

    def lower(self, images: torch.Tensor) -> torch.Tensor:
        """Frozen part: images (B, 3, 224, 224) -> stage-2 feature maps (B, 384, 14, 14)."""
        b = self.backbone
        x = b.stem(images)
        for stage in b.stages[:3]:
            x = stage(x)
        return x

    def upper(self, feats: torch.Tensor) -> torch.Tensor:
        """Trainable part: stage-2 maps -> pooled embedding (B, 768)."""
        b = self.backbone
        return b.forward_head(b.norm_pre(b.stages[3](feats)), pre_logits=True)

    def heads(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embedding -> (log1p carbs (B,), bin logits (B, 4))."""
        return self.reg_head(emb).squeeze(-1), self.bin_head(emb)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full path used at inference."""
        return self.heads(self.upper(self.lower(images)))

    def set_trainable(self, stage3: bool) -> None:
        """Freeze everything except the heads, and stage 3 too when `stage3` is True."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.stages[3].parameters():
            p.requires_grad = stage3
        for p in self.backbone.head.parameters():  # final LayerNorm before pooling output
            p.requires_grad = stage3
        for p in list(self.reg_head.parameters()) + list(self.bin_head.parameters()):
            p.requires_grad = True

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """Only the weights that training changes (stage 3, final norm, heads): small enough to commit."""
        keep = ("backbone.stages.3.", "backbone.head.", "reg_head.", "bin_head.")
        return {k: v for k, v in self.state_dict().items() if k.startswith(keep)}
