export const SIZE = 128;

export function toGray(rgba, width, height) {
  const gray = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
    const alpha = rgba[p + 3] / 255;
    const r = rgba[p] * alpha + 255 * (1 - alpha);
    const g = rgba[p + 1] * alpha + 255 * (1 - alpha);
    const b = rgba[p + 2] * alpha + 255 * (1 - alpha);
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return gray;
}

const reflect = (i, n) => {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - i - 2;
  return i;
};

function gaussianBlur5(src, w, h) {
  const sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8;
  const k = [-2, -1, 0, 1, 2].map((x) => Math.exp(-(x * x) / (2 * sigma * sigma)));
  const sum = k.reduce((a, b) => a + b, 0);
  const kernel = k.map((v) => v / sum);
  const tmp = new Float32Array(w * h);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let acc = 0;
      for (let t = -2; t <= 2; t += 1) acc += kernel[t + 2] * src[y * w + reflect(x + t, w)];
      tmp[y * w + x] = acc;
    }
  }
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let acc = 0;
      for (let t = -2; t <= 2; t += 1) acc += kernel[t + 2] * tmp[reflect(y + t, h) * w + x];
      out[y * w + x] = Math.min(255, Math.max(0, Math.round(acc)));
    }
  }
  return out;
}

function otsu(img) {
  const hist = new Float64Array(256);
  for (const v of img) hist[v] += 1;
  const total = img.length;
  let sumAll = 0;
  for (let i = 0; i < 256; i += 1) sumAll += i * hist[i];
  let sumB = 0;
  let weightB = 0;
  let best = 0;
  let threshold = 0;
  for (let t = 0; t < 256; t += 1) {
    weightB += hist[t];
    if (!weightB) continue;
    const weightF = total - weightB;
    if (!weightF) break;
    sumB += t * hist[t];
    const meanB = sumB / weightB;
    const meanF = (sumAll - sumB) / weightF;
    const between = weightB * weightF * (meanB - meanF) ** 2;
    if (between > best) { best = between; threshold = t; }
  }
  return threshold;
}

function morph(mask, w, h, erode) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let value = erode ? 255 : 0;
      for (let dy = -1; dy <= 0; dy += 1) {
        for (let dx = -1; dx <= 0; dx += 1) {
          const yy = erode ? y + dy : y - dy;
          const xx = erode ? x + dx : x - dx;
          const inside = yy >= 0 && yy < h && xx >= 0 && xx < w;
          const v = inside ? mask[yy * w + xx] : (erode ? 255 : 0);
          value = erode ? Math.min(value, v) : Math.max(value, v);
        }
      }
      out[y * w + x] = value;
    }
  }
  return out;
}

export function cleanAndCrop(gray, w, h) {
  const blur = gaussianBlur5(gray, w, h);
  const t = otsu(blur);
  let ink = new Uint8Array(w * h);
  for (let i = 0; i < ink.length; i += 1) ink[i] = blur[i] > t ? 0 : 255;
  ink = morph(morph(ink, w, h, true), w, h, false);
  let minX = w;
  let minY = h;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (!ink[y * w + x]) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return null;
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const margin = Math.max(4, Math.floor(0.04 * Math.max(bw, bh)));
  const x0 = Math.max(0, minX - margin);
  const y0 = Math.max(0, minY - margin);
  const x1 = Math.min(w, maxX + 1 + margin);
  const y1 = Math.min(h, maxY + 1 + margin);
  const cw = x1 - x0;
  const ch = y1 - y0;
  const side = Math.max(cw, ch);
  const square = new Uint8Array(side * side).fill(255);
  const ox = Math.floor((side - cw) / 2);
  const oy = Math.floor((side - ch) / 2);
  for (let y = 0; y < ch; y += 1) {
    for (let x = 0; x < cw; x += 1) square[(oy + y) * side + ox + x] = 255 - ink[(y0 + y) * w + x0 + x];
  }
  return { image: square, side };
}

export function resizeArea(src, sw, sh, dw, dh) {
  const out = new Uint8Array(dw * dh);
  const sx = sw / dw;
  const sy = sh / dh;
  for (let y = 0; y < dh; y += 1) {
    const fy0 = y * sy;
    const fy1 = fy0 + sy;
    for (let x = 0; x < dw; x += 1) {
      const fx0 = x * sx;
      const fx1 = fx0 + sx;
      let acc = 0;
      let area = 0;
      for (let yy = Math.floor(fy0); yy < Math.ceil(fy1); yy += 1) {
        const wy = Math.min(fy1, yy + 1) - Math.max(fy0, yy);
        if (wy <= 0) continue;
        for (let xx = Math.floor(fx0); xx < Math.ceil(fx1); xx += 1) {
          const wx = Math.min(fx1, xx + 1) - Math.max(fx0, xx);
          if (wx <= 0) continue;
          acc += src[Math.min(sh - 1, yy) * sw + Math.min(sw - 1, xx)] * wx * wy;
          area += wx * wy;
        }
      }
      out[y * dw + x] = Math.round(acc / area);
    }
  }
  return out;
}

export function canny(img, w, h, low, high) {
  const at = (x, y) => img[reflect(y, h) * w + reflect(x, w)];
  const dx = new Int32Array(w * h);
  const dy = new Int32Array(w * h);
  const mag = new Int32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
      const gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
      const i = y * w + x;
      dx[i] = gx;
      dy[i] = gy;
      mag[i] = Math.abs(gx) + Math.abs(gy);
    }
  }
  const TG22 = 13573;
  const state = new Uint8Array(w * h);
  const stack = [];
  const m = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : mag[y * w + x]);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = y * w + x;
      const v = mag[i];
      if (v <= low) continue;
      const xs = dx[i];
      const ys = dy[i];
      const ax = Math.abs(xs);
      const ay = Math.abs(ys) << 15;
      const tg22x = ax * TG22;
      let isMax = false;
      if (ay < tg22x) {
        isMax = v > m(x - 1, y) && v >= m(x + 1, y);
      } else {
        const tg67x = tg22x + (ax << 16);
        if (ay > tg67x) {
          isMax = v > m(x, y - 1) && v >= m(x, y + 1);
        } else {
          const s = (xs ^ ys) < 0 ? -1 : 1;
          isMax = v > m(x - s, y - 1) && v > m(x + s, y + 1);
        }
      }
      if (!isMax) continue;
      if (v > high) { state[i] = 2; stack.push(i); } else state[i] = 1;
    }
  }
  while (stack.length) {
    const i = stack.pop();
    const x = i % w;
    const y = (i - x) / w;
    for (let yy = y - 1; yy <= y + 1; yy += 1) {
      for (let xx = x - 1; xx <= x + 1; xx += 1) {
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx;
        if (state[j] === 1) { state[j] = 2; stack.push(j); }
      }
    }
  }
  const out = new Float32Array(w * h);
  for (let i = 0; i < out.length; i += 1) out[i] = state[i] === 2 ? 1 : 0;
  return out;
}

export function preprocessGray(gray, w, h) {
  const cropped = cleanAndCrop(gray, w, h);
  if (!cropped) return null;
  const small = resizeArea(cropped.image, cropped.side, cropped.side, SIZE, SIZE);
  return { edges: canny(small, SIZE, SIZE, 20, 220), small };
}
