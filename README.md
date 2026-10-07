# Acoustic language identification: MLP, LSTM and GRU

Completed experimental experiment with fresh local-recording evaluation. This is a small-data model comparison, not a dependable general-purpose language detector. Code was repaired in a publication copy; original project and local recordings are retained. Team recordings and shared project are not claimed as solely authored work.

## Run

Install `requirements.txt` in Python 3.12. Run `python -m pytest -q`, then:

```text
python experiment.py --data-dir /path/to/recordings --cache-dir /path/to/local-cache --output /path/to/results.json --epochs 30
```

For the classical comparison `--epochs` does not change the fixed 1,500-step scratch optimization; it controls neural training only. WAV files can be arranged in folders such as `speaker-a english`, `speaker-a urdu`, and `speaker-a mixed`. Recognized suffixes also include eng/en, ur, mix/ue, and ar/arabic. For other layouts, pass `--manifest manifest.csv` with `path,label,speaker` columns; paths must be relative to `--data-dir`. Labels and speaker IDs must be nonempty. Every speaker must have recordings in each class. At least three speakers are required.

No recordings, audio features, model binaries, or plots are committed. The JSON metrics and feature/dataset fingerprint are included; exact reproduction requires the same locally retained recordings. No new recordings are downloaded. Audio is resampled to mono 16 kHz, capped at the first 30 seconds, and converted to 13 MFCCs (512-sample FFT, 256-sample hop). WAV input must be 0.15–60 seconds, mono/stereo and under 32 MB. Silent/invalid files and conflicting duplicate recordings fail explicitly; exact duplicates are removed. Scaling is fitted on training data only.

## Fresh results: held-out speakers

The measured dataset has 268 recordings, three speakers, and English, Urdu and mixed speech. Each of the three speakers is used once as the held-out test speaker. Results pooled across those test predictions:

- MLP: 39.55% accuracy, 0.353 macro F1
- LSTM: 35.45% accuracy, 0.354 macro F1
- GRU: 34.70% accuracy, 0.347 macro F1

The neural experiment reimplements the original architecture families in PyTorch: 78 MFCC/delta mean/std features feed a 256/128/64 MLP with batch normalization and dropout; 13-feature sequences feed a 64-unit LSTM or GRU, valid-length mean pooling, dropout and a 32-unit head. Padding cannot contribute to pooled features. One remaining speaker trains each fold and the other selects the checkpoint by validation loss, with early stopping after six unimproved epochs. This leaves very little training diversity. The original four-class, 873-recording dataset and Arabic evaluation were not available: these fresh three-class results do not reproduce that earlier run.

These results show weak transfer across speakers. Three voices, recording conditions and possible shared utterances are too limited to support broad deployment claims; speaker separation does not by itself control channel or phrase confounding. Mixed speech is one whole-clip class, not word-level code-switch detection. Models use no large pretrained speech representation.

## Notebook and verification

The notebook follows the same setup → experiment → results pattern as the repaired project repositories. It reads the included numeric metrics and runs an implementation smoke check; retraining uses the command above with your local recordings. Outputs are stripped. `VERIFICATION.json` records executed tests and the full fresh training run. Tests cover gradient correctness, noncontiguous labels, finite probabilities, invalid inputs, speaker separation, real feature extraction and single-file preprocessing; neural checks also verify invariance to padded frames.
