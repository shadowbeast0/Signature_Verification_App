function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

export function decodeWeights(manifest, buffer) {
  const halves = new Uint16Array(buffer);
  const weights = {};
  for (const tensor of manifest.tensors) {
    const start = tensor.offset / 2;
    const values = new Float32Array(tensor.length);
    for (let i = 0; i < tensor.length; i += 1) values[i] = halfToFloat(halves[start + i]);
    weights[tensor.name] = { shape: tensor.shape, values };
  }
  return weights;
}

function convReluBnPool(input, h, w, cin, weights, conv, bn) {
  const kernel = weights[`${conv}.kernel`].values;
  const bias = weights[`${conv}.bias`].values;
  const cout = bias.length;
  const oh = h - 2;
  const ow = w - 2;
  const out = new Float32Array(oh * ow * cout);
  for (let y = 0; y < oh; y += 1) {
    for (let x = 0; x < ow; x += 1) {
      const base = (y * ow + x) * cout;
      for (let o = 0; o < cout; o += 1) out[base + o] = bias[o];
      for (let ky = 0; ky < 3; ky += 1) {
        for (let kx = 0; kx < 3; kx += 1) {
          const inBase = ((y + ky) * w + (x + kx)) * cin;
          const kBase = (ky * 3 + kx) * cin * cout;
          for (let c = 0; c < cin; c += 1) {
            const v = input[inBase + c];
            if (v === 0) continue;
            const row = kBase + c * cout;
            for (let o = 0; o < cout; o += 1) out[base + o] += v * kernel[row + o];
          }
        }
      }
    }
  }
  const gamma = weights[`${bn}.gamma`].values;
  const beta = weights[`${bn}.beta`].values;
  const mean = weights[`${bn}.mean`].values;
  const variance = weights[`${bn}.variance`].values;
  const scale = gamma.map((g, i) => g / Math.sqrt(variance[i] + 1e-3));
  for (let i = 0; i < out.length; i += 1) {
    const o = i % cout;
    out[i] = (Math.max(0, out[i]) - mean[o]) * scale[o] + beta[o];
  }
  const ph = Math.floor(oh / 2);
  const pw = Math.floor(ow / 2);
  const pooled = new Float32Array(ph * pw * cout);
  for (let y = 0; y < ph; y += 1) {
    for (let x = 0; x < pw; x += 1) {
      for (let o = 0; o < cout; o += 1) {
        let best = -Infinity;
        for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) best = Math.max(best, out[((2 * y + dy) * ow + 2 * x + dx) * cout + o]);
        }
        pooled[(y * pw + x) * cout + o] = best;
      }
    }
  }
  return { data: pooled, h: ph, w: pw, c: cout };
}

function denseRelu(input, weights, name) {
  const kernel = weights[`${name}.kernel`].values;
  const bias = weights[`${name}.bias`].values;
  const units = bias.length;
  const out = Float32Array.from(bias);
  for (let i = 0; i < input.length; i += 1) {
    const v = input[i];
    if (v === 0) continue;
    const row = i * units;
    for (let j = 0; j < units; j += 1) out[j] += v * kernel[row + j];
  }
  for (let j = 0; j < units; j += 1) out[j] = Math.max(0, out[j]);
  return out;
}

export function embed(edges, weights) {
  let layer = { data: edges, h: 128, w: 128, c: 1 };
  for (let i = 1; i <= 4; i += 1) layer = convReluBnPool(layer.data, layer.h, layer.w, layer.c, weights, `conv${i}`, `bn${i}`);
  return denseRelu(denseRelu(layer.data, weights, "fc1"), weights, "fc2");
}

export function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na * nb) + 1e-9);
}

export async function loadModel(baseUrl) {
  const [manifest, buffer] = await Promise.all([
    fetch(new URL("model/manifest.json", baseUrl)).then((r) => r.json()),
    fetch(new URL("model/weights.bin", baseUrl)).then((r) => r.arrayBuffer()),
  ]);
  return decodeWeights(manifest, buffer);
}
