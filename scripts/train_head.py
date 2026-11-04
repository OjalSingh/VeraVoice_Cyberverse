"""Train the classifier head on extracted features and save a model bundle.

Usage: python scripts/train_head.py feats/train.npz feats/dev.npz --out model/
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veravoice.detector import save_bundle  # noqa: E402
from veravoice.head import LayerWeightedHead  # noqa: E402
from veravoice.metrics import compute_eer  # noqa: E402


def load(path: str):
    d = np.load(path)
    return torch.from_numpy(d["feats"].astype(np.float32)), torch.from_numpy(d["labels"]).float(), str(d["backbone"])


@torch.no_grad()
def score(head: nn.Module, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
    head.eval()
    return np.concatenate([torch.sigmoid(head(x[i : i + batch])).numpy() for i in range(0, len(x), batch)])


def train(x_tr, y_tr, x_dev, y_dev, epochs=30, lr=1e-3, batch=256, seed=0, log=print):
    torch.manual_seed(seed)
    head = LayerWeightedHead(num_layers=x_tr.shape[1], hidden_size=x_tr.shape[2])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    # Class-balanced loss: ASVspoof train is ~9:1 spoof to bona fide.
    pos = y_tr.mean().clamp(1e-3, 1 - 1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(1 - pos) / pos)
    best = (float("inf"), None, 0.5)
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            # Feature-space noise as light augmentation.
            xb = x_tr[b] + 0.01 * torch.randn_like(x_tr[b])
            loss = loss_fn(head(xb), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        eer, thr = compute_eer(score(head, x_dev), y_dev.numpy())
        log(f"epoch {ep + 1:2d}  loss {loss.item():.4f}  dev EER {eer * 100:.2f}%")
        if eer < best[0]:
            best = (eer, {k: v.clone() for k, v in head.state_dict().items()}, thr)
    head.load_state_dict(best[1])
    return head, best[0], best[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("dev")
    ap.add_argument("--out", default="model")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()

    x_tr, y_tr, backbone = load(a.train)
    x_dev, y_dev, _ = load(a.dev)
    print(f"train {tuple(x_tr.shape)}  dev {tuple(x_dev.shape)}")
    head, eer, thr = train(x_tr, y_tr, x_dev, y_dev, epochs=a.epochs)
    w = head.layer_weights().detach().numpy()
    print(f"best dev EER {eer * 100:.2f}% at threshold {thr:.3f}")
    print("layer weights:", " ".join(f"{v:.2f}" for v in w))
    save_bundle(a.out, head, backbone, thr, {"dev_eer": eer})
    print(f"saved bundle to {a.out}/")


if __name__ == "__main__":
    main()
