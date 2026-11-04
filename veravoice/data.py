"""Manifest builders for the datasets VeraVoice trains and evaluates on.

A manifest is a list of dicts: {"path", "label" (1 = spoof, 0 = bona fide),
"source", "attack"}. ``attack`` lets us report EER per attack and keep unseen
attacks out of training.
"""

from __future__ import annotations

import csv
import glob
import os


def asvspoof2019_la(protocol_file: str, audio_dir: str, ext: str = ".flac") -> list[dict]:
    """Parse an ASVspoof 2019 LA CM protocol.

    Lines look like: ``LA_0079 LA_T_1138215 - - bonafide`` or
    ``LA_0079 LA_T_1271820 - A01 spoof``.
    """
    rows = []
    with open(protocol_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _speaker, utt, _, attack, key = parts[:5]
            rows.append(
                {
                    "path": os.path.join(audio_dir, utt + ext),
                    "label": 1 if key == "spoof" else 0,
                    "source": "asvspoof2019",
                    "attack": "bonafide" if attack == "-" else attack,
                }
            )
    return rows


def in_the_wild(meta_csv: str, audio_dir: str | None = None) -> list[dict]:
    """Parse the In-the-Wild dataset's meta.csv (columns: file, speaker, label)."""
    audio_dir = audio_dir or os.path.dirname(meta_csv)
    rows = []
    with open(meta_csv, newline="") as f:
        for r in csv.DictReader(f):
            label = r["label"].strip().lower()
            rows.append(
                {
                    "path": os.path.join(audio_dir, r["file"]),
                    "label": 0 if label in ("bona-fide", "bonafide", "real") else 1,
                    "source": "in_the_wild",
                    "attack": r.get("speaker", ""),
                }
            )
    return rows


def folder(root: str, label: int, source: str, patterns=("*.wav", "*.flac", "*.mp3")) -> list[dict]:
    """Every audio file under ``root`` with one label. Used for MLAAD (spoof) and M-AILABS (bona fide).

    The parent folder name is kept as ``attack`` so MLAAD's TTS model names show up in per-attack EER.
    """
    paths = []
    for p in patterns:
        paths += glob.glob(os.path.join(root, "**", p), recursive=True)
    return [
        {"path": p, "label": label, "source": source, "attack": os.path.basename(os.path.dirname(p))}
        for p in sorted(paths)
    ]


def find_file(root: str, name: str) -> str | None:
    """Locate a file anywhere under ``root`` (Kaggle dataset layouts vary)."""
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    return sorted(hits, key=len)[0] if hits else None
