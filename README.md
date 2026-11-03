# VeraVoice

**Proof of human for voice.** VeraVoice scores a speech clip for signs that it was made by text-to-speech or a voice-cloning model rather than a person. Upload a file or record from your mic in the Streamlit app and get a verdict, a score, and a per-window breakdown for longer clips.

> Started as a hackathon MVP (Cyberverse). Version 0.2 replaces the original MFCC + SVM model, which mostly learned to tell gTTS apart from audiobooks, with a WavLM-based detector trained and evaluated on standard anti-spoofing benchmarks.

## Results

Equal error rate (EER, lower is better) on data the model never trained on. Numbers come from `model/report.json` and are filled in after each training run.

| Test set | What it measures | EER |
| --- | --- | --- |
| ASVspoof 2019 LA eval | 13 attack systems, 11 of them unseen in training | _pending first run_ |
| MLAAD (held-out TTS systems) | Modern TTS engines the model never saw | _pending first run_ |
| In-the-Wild | Real deepfakes of public figures scraped from the internet | _pending first run_ |

In-the-Wild is the honest number: models that score well on ASVspoof often drop sharply on it, so it's reported separately rather than averaged in.

## How it works

```
audio ──► mono, 16 kHz ──► trim silence, peak-normalize, 4 s windows
      ──► WavLM base+ (frozen, 13 hidden layers, mean-pooled)
      ──► learned softmax weights over layers ──► LayerNorm ──► MLP ──► spoof score
```

- **Frozen self-supervised backbone.** [WavLM base+](https://huggingface.co/microsoft/wavlm-base-plus) was pretrained on 94k hours of real speech, so synthetic speech looks out of distribution in its hidden states. Keeping it frozen means features are extracted once and the head trains in seconds.
- **Layer weighting.** Lower and middle layers carry the acoustic artefacts vocoders leave behind, while top layers are mostly about words. The head learns which layers to trust instead of using only the last one.
- **Shortcut-proof preprocessing.** Every clip, real or fake, goes through the same resampling, silence trimming and loudness normalization, so the model can't win by spotting a file format, a sample rate or leading silence (the failure mode of v0.1).
- **Splits by attack, not by clip.** ASVspoof eval mostly uses attacks absent from training, and MLAAD is split by TTS system, so test scores measure generalization to new generators.
- **Calibrated threshold.** The decision threshold is the EER point on the dev set, and scores near it are reported as "uncertain" instead of being forced into a verdict.

## Data

| Dataset | Role | Notes |
| --- | --- | --- |
| [ASVspoof 2019 LA](https://www.asvspoof.org/index2019.html) | train / dev / test | Bona fide VCTK speech and 19 TTS and voice-conversion attacks |
| [MLAAD](https://deepfake-total.com/mlaad) (English) | train / test | Clips from dozens of modern TTS systems, split by system |
| [In-the-Wild](https://deepfake-total.com/in_the_wild) | test only | 38 hours of real and fake speech of 58 public figures |

## Run the demo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app expects a trained bundle in `model/` (`head.pt` and `config.json`). The first run downloads WavLM base+ (about 380 MB) from Hugging Face.

## Retrain

Training runs on a free Kaggle GPU: open [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) on Kaggle, attach the datasets listed at the top, and Run all. It takes about an hour, and you download `veravoice_model.zip` at the end.

The same steps work anywhere with the data on disk:

```bash
python scripts/build_manifests.py --asvspoof /data/asvspoof2019 --mlaad /data/mlaad --itw /data/in_the_wild --out manifests
python scripts/extract_embeddings.py manifests/train.json feats/train.npz --random-crop
python scripts/extract_embeddings.py manifests/dev.json feats/dev.npz
python scripts/extract_embeddings.py manifests/test_itw.json feats/test_itw.npz
python scripts/train_head.py feats/train.npz feats/dev.npz --out model
python scripts/evaluate.py model feats/test_itw.npz --report model/report.json
```

## Project layout

```
veravoice/        audio preprocessing, WavLM embedder, head, EER metric, dataset parsers, Detector
scripts/          build_manifests, extract_embeddings, train_head, evaluate
notebooks/        Kaggle training notebook
tests/            pytest suite (runs offline with a tiny random WavLM)
app.py            Streamlit demo
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && pytest -q
```

CI runs lint and tests on every push and pull request.

## Limitations

- Scores are evidence, not proof. Phone-line audio, heavy compression and background music all hurt accuracy.
- Training data is mostly English. Hindi and Tamil clips are not covered by the benchmarks above and should be treated as unverified.
- New voice-cloning systems appear every month. Expect the In-the-Wild EER to drift, and retrain with newer MLAAD releases.

## License

MIT, see [LICENSE](LICENSE).
