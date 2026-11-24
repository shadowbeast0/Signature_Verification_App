import itertools
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from reference import embed, different_probability, load_weights, preprocess

FONTS = ["LHANDW.TTF", "segoesc.ttf", "Inkfree.ttf", "FREESCPT.TTF", "VLADIMIR.TTF", "BRUSHSCI.TTF"]
NAMES = ["Arjeesh Palai", "Rahul Sharma", "Priya Das", "Ankit Roy", "Sneha Ghosh", "Vikram Sen"]


def signature(name, font, seed):
    rng = random.Random(seed)
    face = ImageFont.truetype(f"C:/Windows/Fonts/{font}", rng.randint(54, 66))
    canvas = Image.new("L", (900, 260), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((rng.randint(20, 60), rng.randint(40, 80)), name, font=face, fill=rng.randint(0, 60))
    canvas = canvas.rotate(rng.uniform(-6, 6), fillcolor=255, resample=Image.BICUBIC)
    canvas = canvas.resize((int(900 * rng.uniform(0.85, 1.15)), int(260 * rng.uniform(0.85, 1.15))))
    noise = np.asarray(canvas, dtype=np.float32) + np.random.default_rng(seed).normal(0, 6, canvas.size[::-1])
    return np.clip(noise, 0, 255).astype(np.uint8)


def main():
    weights = load_weights()
    people = {(name, font): name for name, font in zip(NAMES, FONTS)}
    samples = {key: [embed(preprocess(signature(key[0], key[1], seed)), weights) for seed in range(4)] for key in people}
    same, different = [], []
    for key, vectors in samples.items():
        same += [different_probability(a, b, weights) for a, b in itertools.combinations(vectors, 2)]
    for (k1, v1), (k2, v2) in itertools.combinations(samples.items(), 2):
        different += [different_probability(a, b, weights) for a in v1[:2] for b in v2[:2]]
    same, different = np.array(same), np.array(different)
    print(f"same writer      p(different): mean {same.mean():.2f}  max {same.max():.2f}")
    print(f"different writer p(different): mean {different.mean():.2f}  min {different.min():.2f}")
    print(f"old app (Verified if p>=0.5): accepts {np.mean(different >= 0.5):.0%} of different-writer pairs, {np.mean(same >= 0.5):.0%} of same-writer pairs")
    print(f"fixed  (Verified if p<0.5):   accepts {np.mean(different < 0.5):.0%} of different-writer pairs, {np.mean(same < 0.5):.0%} of same-writer pairs")


if __name__ == "__main__":
    main()
