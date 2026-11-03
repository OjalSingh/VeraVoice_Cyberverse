"""Build train/dev/test manifests from whatever datasets are mounted.

Usage (Kaggle): python scripts/build_manifests.py --asvspoof /kaggle/input/asvpoof-2019-dataset \
    --mlaad /kaggle/working/mlaad --itw /kaggle/input/in-the-wild --out manifests

Splits:
- train: ASVspoof 2019 LA train (attacks A01-A06) + MLAAD train split (modern TTS systems)
- dev:   ASVspoof 2019 LA dev (same attacks), used for early stopping and the threshold
- test_asvspoof: ASVspoof 2019 LA eval (A07-A19, mostly unseen attacks)
- test_mlaad: MLAAD TTS models held out from training (unseen systems), against held-out bona fide
- test_itw: In-the-Wild (real celebrity and politician deepfakes from the internet)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veravoice import data  # noqa: E402


def split_by_group(rows: list[dict], test_frac: float, key: str = "attack") -> tuple[list[dict], list[dict]]:
    """Deterministic split by group so a TTS system (or speaker) is never in both sides."""
    train, test = [], []
    for r in rows:
        bucket = zlib.crc32(r[key].encode()) % 1000 / 1000
        (test if bucket < test_frac else train).append(r)
    return train, test


def cap(rows: list[dict], n: int | None, seed: int = 0) -> list[dict]:
    if n is None or len(rows) <= n:
        return rows
    return random.Random(seed).sample(rows, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asvspoof", help="root of ASVspoof 2019 (folder containing LA/)")
    ap.add_argument("--mlaad", help="root of MLAAD audio (spoof)")
    ap.add_argument("--mailabs", help="root of M-AILABS audio (bona fide, MLAAD's source corpus)")
    ap.add_argument("--itw", help="root of In-the-Wild (folder with meta.csv)")
    ap.add_argument("--max-per-source", type=int, default=None, help="cap clips per source and split")
    ap.add_argument("--out", default="manifests")
    a = ap.parse_args()

    splits: dict[str, list[dict]] = {k: [] for k in ("train", "dev", "test_asvspoof", "test_mlaad", "test_itw")}

    if a.asvspoof:
        for split, proto, sub in [
            ("train", "ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
            ("dev", "ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
            ("test_asvspoof", "ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
        ]:
            proto_path = data.find_file(a.asvspoof, proto)
            audio_dir = data.find_file(a.asvspoof, sub)
            if not proto_path or not audio_dir:
                print(f"[warn] ASVspoof {split}: protocol or audio folder not found, skipping")
                continue
            rows = data.asvspoof2019_la(proto_path, os.path.join(audio_dir, "flac"))
            splits[split] += cap(rows, a.max_per_source)

    if a.mlaad:
        # Split by TTS system so the MLAAD test set only has models the head never saw.
        fake_tr, fake_te = split_by_group(data.folder(a.mlaad, 1, "mlaad"), 0.3)
        splits["train"] += cap(fake_tr, a.max_per_source)
        splits["test_mlaad"] += cap(fake_te, a.max_per_source)
    if a.mailabs:
        # Optional bona fide in the same languages as MLAAD, split by speaker/book folder.
        real_tr, real_te = split_by_group(data.folder(a.mailabs, 0, "mailabs"), 0.3)
        splits["train"] += cap(real_tr, a.max_per_source)
        splits["test_mlaad"] += cap(real_te, a.max_per_source)
    if splits["test_mlaad"] and not any(r["label"] == 0 for r in splits["test_mlaad"]):
        # Spoof-only MLAAD: score its unseen TTS systems against ASVspoof's held-out bona fide.
        splits["test_mlaad"] += [r for r in splits["test_asvspoof"] if r["label"] == 0]

    if a.itw:
        meta = data.find_file(a.itw, "meta.csv")
        if meta:
            splits["test_itw"] += cap(data.in_the_wild(meta), a.max_per_source)
        else:
            print("[warn] In-the-Wild meta.csv not found, skipping")

    os.makedirs(a.out, exist_ok=True)
    for name, rows in splits.items():
        if not rows:
            continue
        rows = [r for r in rows if os.path.exists(r["path"])]
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(rows, f)
        n_spoof = sum(r["label"] for r in rows)
        print(f"{name:14s} {len(rows):7d} clips  ({n_spoof} spoof, {len(rows) - n_spoof} bona fide)")


if __name__ == "__main__":
    main()
