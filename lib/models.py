import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from pytorch_tcn import TCN as _TCN

# default eps (1e-5) NaN'd on near-constant channels
_BN_EPS = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(logits, targets, alpha=self.alpha,
                                  gamma=self.gamma, reduction="mean")


class SimpleCNN(nn.Module):
    def __init__(self, in_features=9, base_ch=32, dropout=0.3):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_features, base_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_ch, eps=_BN_EPS), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch * 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_ch * 2, eps=_BN_EPS), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(base_ch * 2, base_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_ch * 4, eps=_BN_EPS), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(base_ch * 4, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).squeeze(-1)
        return self.head(self.drop(x)).squeeze(-1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch, eps=_BN_EPS)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch, eps=_BN_EPS)
        self.drop = nn.Dropout(dropout)
        self.skip = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch, eps=_BN_EPS))
            if in_ch != out_ch or stride != 1 else nn.Identity()
        )

    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, in_features=9, base_ch=32, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, base_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_ch, eps=_BN_EPS), nn.ReLU(),
        )
        c1, c2, c3 = base_ch * 2, base_ch * 4, base_ch * 4
        self.blocks = nn.Sequential(
            ResBlock1D(base_ch, c1, stride=2, dropout=dropout),
            ResBlock1D(c1,      c2, stride=2, dropout=dropout),
            ResBlock1D(c2,      c3, stride=2, dropout=dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c3, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class TCN(nn.Module):
    def __init__(self, in_features, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        self.tcn = _TCN(in_features, channels, kernel_size=kernel_size,
                        dropout=dropout, causal=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x).squeeze(-1)


def build_model(name: str, in_features: int = 9, base_ch: int = 32,
                dropout: float = 0.3) -> nn.Module:
    name = name.lower()
    if name == "simple":
        return SimpleCNN(in_features=in_features, base_ch=base_ch, dropout=dropout)
    if name == "resnet":
        return ResNet1D(in_features=in_features, base_ch=base_ch, dropout=dropout)
    if name == "tcn":
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        channels = [c1, c1, c2, c2, c3, c3, c3, c3]
        return TCN(in_features=in_features, channels=channels, dropout=dropout)
    raise ValueError(f"Unknown model '{name}'. Choose 'simple', 'resnet', or 'tcn'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
