import io
import json
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model" / "signature_model_siamese.keras"
OUT = ROOT / "web" / "model"

LAYERS = [
    ("conv1", "functional/layers/conv2d", ["kernel", "bias"]),
    ("bn1", "functional/layers/batch_normalization", ["gamma", "beta", "mean", "variance"]),
    ("conv2", "functional/layers/conv2d_1", ["kernel", "bias"]),
    ("bn2", "functional/layers/batch_normalization_1", ["gamma", "beta", "mean", "variance"]),
    ("conv3", "functional/layers/conv2d_2", ["kernel", "bias"]),
    ("bn3", "functional/layers/batch_normalization_2", ["gamma", "beta", "mean", "variance"]),
    ("conv4", "functional/layers/conv2d_3", ["kernel", "bias"]),
    ("bn4", "functional/layers/batch_normalization_3", ["gamma", "beta", "mean", "variance"]),
    ("fc1", "functional/layers/dense", ["kernel", "bias"]),
    ("fc2", "functional/layers/dense_1", ["kernel", "bias"]),
    ("head", "dense", ["kernel", "bias"]),
]


def load():
    archive = zipfile.ZipFile(SOURCE)
    weights = h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r")
    tensors = {}
    for name, path, parts in LAYERS:
        for i, part in enumerate(parts):
            tensors[f"{name}.{part}"] = np.asarray(weights[f"layers/{path}/vars/{i}"], dtype=np.float32)
    return tensors


def main():
    tensors = load()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, blobs, offset = [], [], 0
    for key, value in tensors.items():
        data = value.astype("<f2").tobytes()
        manifest.append({"name": key, "shape": list(value.shape), "offset": offset, "length": value.size})
        blobs.append(data)
        offset += len(data)
    (OUT / "weights.bin").write_bytes(b"".join(blobs))
    (OUT / "manifest.json").write_text(json.dumps({"dtype": "float16", "tensors": manifest}), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
