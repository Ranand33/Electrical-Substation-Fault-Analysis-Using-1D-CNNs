import os
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class PCMWindowDataset(Dataset):
    def __init__(self, X, y, pcm_ids, augment=False):
        self.X = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X
        self.y = torch.from_numpy(y.astype(np.int64)) if isinstance(y, np.ndarray) else y
        self.pcm_ids = torch.from_numpy(pcm_ids) if isinstance(pcm_ids, np.ndarray) else pcm_ids
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x = x + torch.randn_like(x) * 0.02
            x = x * torch.empty(1).uniform_(0.95, 1.05)
            if torch.rand(1).item() < 0.15:
                ch = torch.randint(0, x.shape[-1], (1,)).item()
                x[:, ch] = 0.0
            # Time-mask only on negatives: the failure label is anchored at the
            # window end, so masking that region on positives erases the signal.
            if torch.rand(1).item() < 0.2 and self.y[idx].item() == 0:
                t_max = max(2, x.shape[0] // 10)
                t_len = torch.randint(1, t_max, (1,)).item()
                t_start = torch.randint(0, max(1, x.shape[0] - t_len), (1,)).item()
                x[t_start:t_start + t_len, :] = 0.0
        return x, self.y[idx], self.pcm_ids[idx]


def load_split(data_dir: str, split: str,
               window_size: int = None, augment: bool = False) -> PCMWindowDataset:
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    ids = np.load(os.path.join(data_dir, f"pcm_{split}.npy"))
    ds = PCMWindowDataset(X, y, ids, augment=augment)
    if window_size and window_size < ds.X.shape[1]:
        ds.X = ds.X[:, -window_size:, :]
    return ds


def make_sampler(ds: PCMWindowDataset) -> WeightedRandomSampler:
    labels = ds.y.numpy()
    pcm_ids = ds.pcm_ids.numpy()
    _, inv = np.unique(pcm_ids * 2 + labels, return_inverse=True)
    counts = np.bincount(inv)
    weights = (1.0 / counts)[inv]
    return WeightedRandomSampler(
        torch.from_numpy(weights.astype(np.float32)),
        num_samples=len(weights),
        replacement=True,
    )


def apply_viol_filter(ds: PCMWindowDataset, threshold: float = 0.95,
                      verbose: bool = True) -> PCMWindowDataset:
    labels = ds.y.numpy()
    pcm_ids = ds.pcm_ids.numpy()
    rates = {int(p): float(labels[pcm_ids == p].mean())
             for p in np.unique(pcm_ids)}
    high = {p for p, r in rates.items() if r > threshold}

    if not high:
        if verbose:
            print(f"Violation-rate audit: no PCMs exceed {threshold:.0%} - all included.")
        return ds

    if verbose:
        print(f"High-violation PCMs excluded from training (>{threshold:.0%} labelled 1):")
        for p in sorted(high, key=lambda x: -rates[x]):
            n = int((pcm_ids == p).sum())
            print(f"  PCM {p:>5}  viol_rate={rates[p]:.1%}  n_windows={n}")

    keep = ~np.isin(pcm_ids, list(high))
    keep_t = torch.from_numpy(keep)
    filtered = PCMWindowDataset(
        ds.X[keep_t], ds.y[keep_t], ds.pcm_ids[keep_t], augment=ds.augment
    )
    if verbose:
        print(f"  Train after filter: {len(filtered):,} windows "
              f"({keep.sum():,} kept, {(~keep).sum():,} removed)\n")
    return filtered
