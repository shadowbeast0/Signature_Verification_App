# Signature Verification

**Live demo: https://shadowbeast0.github.io/Signature_Verification_App/**

Enrol a person's signatures, then check whether a new signature really belongs to them. The Siamese CNN runs entirely in the browser: nothing is uploaded anywhere.

## How it works

1. **Enrol** a person with at least three genuine signatures, drawn on screen or uploaded as photos/scans.
2. **Verify** a new signature against the person it claims to belong to.

Every signature goes through the same pipeline used in training:

```
image ─► grayscale ─► denoise + Otsu threshold ─► crop to the ink ─► pad to square
      ─► 128×128 ─► Canny edges (20, 220) ─► Siamese CNN branch ─► 128-d embedding
```

A signature is **verified** only when both hold:

- it is about as similar to the person's enrolled samples as those samples are to each other (writer-specific threshold), and
- it is closer to the claimed person than to anyone else enrolled **and** to a built-in background set of 12 other writers.

The background set means a forged or different signature is still rejected when only one person is enrolled.

## Accuracy

The CNN (4 conv blocks → 512 → 128, trained on the Kaggle signature verification dataset with Canny edges) reached 94% pair accuracy on the dataset's test split (see `model/signature-verification-model.ipynb`).

On synthetic signatures (different names in different handwriting styles, with random tilt, scale and pen jitter):

| Setup | Genuine accepted | Different writers rejected |
|---|---|---|
| Several people enrolled | 24 / 24 | 60 / 60 |
| Only one person enrolled | 25 / 30 | 136 / 150 |

Results on real signatures depend on how consistently a person signs. Enrol 4–5 samples written the way you normally sign, on plain light paper.

## Run it

```bash
python -m http.server 8000 --directory web
```

Open http://localhost:8000. Signatures are stored in the browser (IndexedDB); use **People → Download backup** to move them to another device.

## Project layout

| Path | Contents |
|---|---|
| `web/` | The app: `preprocess.js` (OpenCV-compatible pipeline), `model.js` (CNN inference), `verify.js` (decision rule), `store.js` (IndexedDB) |
| `web/model/` | Trained weights (float16), layer manifest and background writer set |
| `model/` | Original Keras model and training notebook |
| `tools/` | Weight export, NumPy reference implementation, background set builder, synthetic check |
| `tests/` | Browser pipeline must match OpenCV and the NumPy reference |

```bash
pip install -r requirements.txt
python tools/export_weights.py
python tools/build_cohort.py
python tools/synthetic_check.py
npm test
```
