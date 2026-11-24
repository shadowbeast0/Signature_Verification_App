import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_fixture import square
from reference import SIZE, clean, embed, load_weights
from synthetic_check import signature

ROOT = Path(__file__).resolve().parents[1]
FONTS = ["FRSCRIPT.TTF", "GIGI.TTF", "ITCKRIST.TTF", "KUNSTLER.TTF", "SCRIPTBL.TTF", "mvboli.ttf", "segoescb.ttf"]
NAMES = ["Meera Iyer", "Karan Mehta", "Ritika Bose", "Sourav Dutta", "Neha Kapoor", "Arnab Saha", "Pooja Nair", "Imran Khan",
         "Ishita Roy", "Debojit Pal", "Tanvi Joshi", "Rohit Verma"]


def prepare(gray):
    small = cv2.resize(square(clean(gray)), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.Canny(small, 20, 220).astype(np.float32) / 255


def main():
    weights = load_weights()
    groups = [[embed(prepare(signature(name, font, 2000 + i * 11 + j)), weights) for j in range(3)]
              for i, (name, font) in enumerate(zip(NAMES, FONTS * 2))]
    centre = np.mean([v for group in groups for v in group], axis=0)
    data = {
        "centre": [round(float(v), 5) for v in centre],
        "writers": [[[round(float(v), 5) for v in vector] for vector in group] for group in groups],
    }
    (ROOT / "web" / "model" / "cohort.json").write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
