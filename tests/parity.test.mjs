import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { preprocessGray } from "../web/preprocess.js";
import { cosine, decodeWeights, embed } from "../web/model.js";

const fixture = JSON.parse(readFileSync(new URL("./fixture.json", import.meta.url), "utf8"));
const manifest = JSON.parse(readFileSync(new URL("../web/model/manifest.json", import.meta.url), "utf8"));
const buffer = readFileSync(new URL("../web/model/weights.bin", import.meta.url));
const weights = decodeWeights(manifest, buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
const bytes = (b64) => new Uint8Array(Buffer.from(b64, "base64"));

test("edge maps match OpenCV", () => {
  for (const item of fixture) {
    const { edges } = preprocessGray(bytes(item.gray), item.width, item.height);
    const expected = bytes(item.edges);
    let agree = 0;
    for (let i = 0; i < expected.length; i += 1) agree += (edges[i] > 0.5) === (expected[i] > 0) ? 1 : 0;
    assert.ok(agree / expected.length > 0.985, `pixel agreement ${(agree / expected.length).toFixed(4)}`);
  }
});

test("network matches the reference implementation on the same input", () => {
  for (const item of fixture) {
    const input = Float32Array.from(bytes(item.edges), (v) => (v ? 1 : 0));
    assert.ok(cosine(embed(input, weights), item.embedding) > 0.999);
  }
});

test("full pipeline embeddings stay close to the reference", () => {
  for (const item of fixture) {
    const { edges } = preprocessGray(bytes(item.gray), item.width, item.height);
    assert.ok(cosine(embed(edges, weights), item.embedding) > 0.97);
  }
});
