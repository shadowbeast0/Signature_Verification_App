import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reference import SIZE, clean, embed, load_weights
from synthetic_check import FONTS, NAMES, signature

ROOT = Path(__file__).resolve().parents[1]


def square(paper):
    h, w = paper.shape
    side = max(h, w)
    canvas = np.full((side, side), 255, np.uint8)
    canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = paper
    return canvas


def main():
    weights = load_weights()
    cases = []
    for i, (name, font) in enumerate(zip(NAMES, FONTS)):
        gray = signature(name, font, 100 + i)
        small = cv2.resize(square(clean(gray)), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(small, 20, 220).astype(np.float32) / 255
        cases.append({
            "width": gray.shape[1],
            "height": gray.shape[0],
            "gray": base64.b64encode(gray.tobytes()).decode(),
            "edges": base64.b64encode(edges.astype(np.uint8).tobytes()).decode(),
            "embedding": [round(float(v), 5) for v in embed(edges, weights)],
        })
    (ROOT / "tests" / "fixture.json").write_text(json.dumps(cases), encoding="utf-8")


if __name__ == "__main__":
    main()
