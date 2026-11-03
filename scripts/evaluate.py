"""Evaluate a bundle on one or more feature files; writes a JSON report with overall and per-attack EER.

Usage: python scripts/evaluate.py model/ feats/test_asvspoof.npz feats/test_itw.npz --report model/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veravoice.detector import CONFIG_FILE, HEAD_FILE  # noqa: E402
from veravoice.head import LayerWeightedHead  # noqa: E402
from veravoice.metrics import compute_eer  # noqa: E402


def evaluate(head, path: str, threshold: float) -> dict:
    d = np.load(path)
    x = torch.from_numpy(d["feats"].astype(np.float32))
    y = d["labels"]
    with torch.no_grad():
        s = torch.sigmoid(head(x)).numpy()
    eer, _ = compute_eer(s, y)
    pred = s > threshold
    out = {
        "clips": int(len(y)),
        "eer": eer,
        "accuracy_at_dev_threshold": float((pred == y).mean()),
        "per_attack_eer": {},
    }
    bona = y == 0
    for atk in sorted(set(d["attacks"][y == 1].tolist())):
        m = bona | (d["attacks"] == atk)
        if (y[m] == 1).any() and (y[m] == 0).any():
            out["per_attack_eer"][atk] = compute_eer(s[m], y[m])[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("feats", nargs="+")
    ap.add_argument("--report")
    a = ap.parse_args()

    with open(os.path.join(a.bundle, CONFIG_FILE)) as f:
        cfg = json.load(f)
    head = LayerWeightedHead(**cfg["head"])
    head.load_state_dict(torch.load(os.path.join(a.bundle, HEAD_FILE), weights_only=True))
    head.eval()

    report = {}
    for p in a.feats:
        name = os.path.splitext(os.path.basename(p))[0]
        report[name] = r = evaluate(head, p, cfg["threshold"])
        print(
            f"{name:16s} EER {r['eer'] * 100:6.2f}%   "
            f"acc@thr {r['accuracy_at_dev_threshold'] * 100:5.1f}%   ({r['clips']} clips)"
        )
        worst = sorted(r["per_attack_eer"].items(), key=lambda kv: -kv[1])[:3]
        if worst:
            print("   hardest attacks:", ", ".join(f"{k} {v * 100:.1f}%" for k, v in worst))
    if a.report:
        cfg["metrics"].update({k: v["eer"] for k, v in report.items()})
        with open(a.report, "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(a.bundle, CONFIG_FILE), "w") as f:
            json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
