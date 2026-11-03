import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

from veravoice import data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_asvspoof_protocol(tmp_path):
    p = tmp_path / "proto.txt"
    p.write_text("LA_0079 LA_T_1138215 - - bonafide\nLA_0079 LA_T_1271820 - A01 spoof\n\n")
    rows = data.asvspoof2019_la(str(p), "/audio")
    assert rows == [
        {"path": "/audio/LA_T_1138215.flac", "label": 0, "source": "asvspoof2019", "attack": "bonafide"},
        {"path": "/audio/LA_T_1271820.flac", "label": 1, "source": "asvspoof2019", "attack": "A01"},
    ]


def test_in_the_wild(tmp_path):
    (tmp_path / "meta.csv").write_text("file,speaker,label\n0.wav,Alec Guinness,spoof\n1.wav,Alec Guinness,bona-fide\n")
    rows = data.in_the_wild(str(tmp_path / "meta.csv"))
    assert [r["label"] for r in rows] == [1, 0]
    assert rows[0]["path"] == str(tmp_path / "0.wav")


def test_build_manifests_end_to_end(tmp_path):
    # Fake ASVspoof layout as it appears on Kaggle, with one level of nesting.
    la = tmp_path / "asv" / "LA" / "LA"
    proto_dir = la / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True)
    for split, proto, sub in [
        ("T", "ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
        ("D", "ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
        ("E", "ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
    ]:
        flac = la / sub / "flac"
        flac.mkdir(parents=True)
        lines = []
        for i, key in enumerate(["bonafide", "spoof"]):
            utt = f"LA_{split}_{i}"
            sf.write(flac / f"{utt}.flac", np.zeros(1600, np.float32), 16000)
            lines.append(f"LA_0001 {utt} - {'-' if key == 'bonafide' else 'A01'} {key}")
        (proto_dir / proto).write_text("\n".join(lines))
    # MLAAD-style spoof folders, one per TTS system.
    for model in [f"tts_{i}" for i in range(12)]:
        d = tmp_path / "mlaad" / "en" / model
        d.mkdir(parents=True)
        sf.write(d / "0.wav", np.zeros(1600, np.float32), 16000)
    out = tmp_path / "m"
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "build_manifests.py"),
            "--asvspoof",
            str(tmp_path / "asv"),
            "--mlaad",
            str(tmp_path / "mlaad"),
            "--out",
            str(out),
        ],
        check=True,
    )
    train = json.loads((out / "train.json").read_text())
    test_mlaad = json.loads((out / "test_mlaad.json").read_text())
    assert {r["source"] for r in train} == {"asvspoof2019", "mlaad"}
    # No TTS system appears on both sides of the MLAAD split.
    train_tts = {r["attack"] for r in train if r["source"] == "mlaad"}
    test_tts = {r["attack"] for r in test_mlaad if r["source"] == "mlaad"}
    assert train_tts and test_tts and not train_tts & test_tts
    # Spoof-only MLAAD is scored against held-out ASVspoof bona fide.
    assert any(r["label"] == 0 for r in test_mlaad)
    assert (out / "dev.json").exists() and not (out / "test_itw.json").exists()
