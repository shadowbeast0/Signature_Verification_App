import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIZE = 128


def load_weights():
    manifest = json.loads((ROOT / "web" / "model" / "manifest.json").read_text(encoding="utf-8"))
    raw = np.frombuffer((ROOT / "web" / "model" / "weights.bin").read_bytes(), dtype="<f2")
    weights = {}
    for item in manifest["tensors"]:
        start = item["offset"] // 2
        weights[item["name"]] = raw[start:start + item["length"]].astype(np.float32).reshape(item["shape"])
    return weights


def clean(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    points = cv2.findNonZero(ink)
    if points is None:
        return np.full((SIZE, SIZE), 255, np.uint8)
    x, y, w, h = cv2.boundingRect(points)
    margin = max(4, int(0.04 * max(w, h)))
    crop = ink[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]
    return 255 - crop


def preprocess(gray):
    paper = clean(gray)
    small = cv2.resize(paper, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 20, 220)
    return edges.astype(np.float32) / 255.0


def conv(x, kernel, bias):
    h, w, _ = x.shape
    kh, kw, _, out = kernel.shape
    cols = np.stack([x[i:h - kh + 1 + i, j:w - kw + 1 + j, :] for i in range(kh) for j in range(kw)], axis=2)
    return np.maximum(np.tensordot(cols, kernel.reshape(kh * kw, -1, out), axes=([2, 3], [0, 1])) + bias, 0)


def batchnorm(x, w, name):
    return (x - w[f"{name}.mean"]) / np.sqrt(w[f"{name}.variance"] + 1e-3) * w[f"{name}.gamma"] + w[f"{name}.beta"]


def pool(x):
    h, w, c = x.shape
    x = x[: h // 2 * 2, : w // 2 * 2]
    return x.reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))


def embed(image, w):
    x = image[:, :, None]
    for i in range(1, 5):
        x = pool(batchnorm(conv(x, w[f"conv{i}.kernel"], w[f"conv{i}.bias"]), w, f"bn{i}"))
    x = np.maximum(x.reshape(-1) @ w["fc1.kernel"] + w["fc1.bias"], 0)
    return np.maximum(x @ w["fc2.kernel"] + w["fc2.bias"], 0)


def different_probability(a, b, w):
    merged = np.concatenate([np.abs(a - b), a * b])
    return float(1 / (1 + np.exp(-(merged @ w["head.kernel"][:, 0] + w["head.bias"][0]))))
