import { test } from "node:test";
import assert from "node:assert/strict";
import { buildSequence, confidenceFromProbs, pyJsonDumps, renderOptions, softmax, tempBucket, toInternal, type SpecialIds } from "../src/sequence.js";

const ids: SpecialIds = { cls: 1, sep: 2, mask: 3, pad: 0, maskTok: "[MASK]" };
// one id per whitespace-separated word, so lengths are easy to reason about
const encode = (s: string) =>
  s
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => 100 + w.length);

test("pyJsonDumps matches Python json.dumps(ensure_ascii=False)", () => {
  assert.equal(pyJsonDumps({ subject: "返金", n: 3, ok: true, x: null, xs: [1, "a"] }), '{"subject": "返金", "n": 3, "ok": true, "x": null, "xs": [1, "a"]}');
  assert.equal(pyJsonDumps("plain"), '"plain"');
});

test("renderOptions: choice with/without descriptions, score levels, noul defaults", () => {
  assert.deepEqual(renderOptions(toInternal({ type: "choice", instructions: "", criteria: { a: "first", b: null } })), ["a: first", "b"]);
  assert.deepEqual(renderOptions(toInternal({ type: "choice", instructions: "", criteria: ["x", "y"] })), ["x", "y"]);
  assert.deepEqual(renderOptions(toInternal({ type: "score", instructions: "", criteria: ["low", "high"] })), ["level 0: low", "level 1: high"]);
  assert.deepEqual(renderOptions(toInternal({ type: "noul", instructions: "" })), ["false: no, the statement does not hold", "true: yes, the statement holds"]);
});

test("buildSequence layout: [CLS] head [SEP] ([MASK] opt)* [SEP] state [SEP]", () => {
  const q = toInternal({ type: "choice", instructions: "which one", criteria: ["a", "bb"] });
  const { ids: seq, markers } = buildSequence(encode, ids, "hello world", q, 512, 192);
  // head = "choice question: which one" -> 4 words
  assert.equal(seq[0], 1);
  assert.equal(seq[5], 2);
  assert.deepEqual(markers, [6, 8]);
  assert.equal(seq[6], 3);
  assert.equal(seq[8], 3);
  assert.deepEqual(seq.slice(-4), [2, 105, 105, 2]);
});

test("buildSequence truncates the state to max_len and drops markers past it", () => {
  const q = toInternal({ type: "noul", instructions: "is it" });
  const { ids: seq, markers } = buildSequence(encode, ids, "w ".repeat(1000), q, 64, 32);
  assert.equal(seq.length, 64);
  assert.equal(seq[63], 2);
  assert.equal(markers.length, 2);
});

test("buildSequence scrubs the mask token from user text", () => {
  const q = toInternal({ type: "noul", instructions: "x [MASK] y" });
  const { ids: seq } = buildSequence(encode, ids, "state [MASK] here", q, 128, 64);
  assert.equal(seq.filter((v) => v === 3).length, 2);
});

// Harvested from the differential run that replaced tempBucket's nested ternary: the generator
// (which inputs decide the answer) and the property (where each bucket must start and stop).
// The old implementation is gone, so these pin the boundaries directly instead of comparing.
test("tempBucket bucket boundaries and qtype prefix", () => {
  const expected: [number, string][] = [
    [-1, "2"],
    [0, "2"],
    [2, "2"],
    [3, "3-5"],
    [5, "3-5"],
    [6, "6-10"],
    [10, "6-10"],
    [11, "11+"],
    [1000, "11+"],
  ];
  expected.forEach(([k, size]) => {
    assert.equal(tempBucket(0, k), `choice:${size}`, `k=${k}`);
  });

  // every qtype name reaches the key
  assert.equal(tempBucket(0, 2), "choice:2");
  assert.equal(tempBucket(1, 2), "score:2");
  assert.equal(tempBucket(2, 2), "noul:2");

  // non-integer k falls in the bucket its comparisons put it in, not a rounded one
  assert.equal(tempBucket(0, 2.5), "choice:3-5");
  assert.equal(tempBucket(0, 10.5), "choice:11+");

  // NaN fails every `<=`, so it lands in the final bucket
  assert.equal(tempBucket(0, NaN), "choice:11+");
  assert.equal(tempBucket(0, Infinity), "choice:11+");
  assert.equal(tempBucket(0, -Infinity), "choice:2");
});

test("tempBucket / softmax / confidence", () => {
  assert.equal(tempBucket(0, 2), "choice:2");
  assert.equal(tempBucket(1, 4), "score:3-5");
  assert.equal(tempBucket(0, 8), "choice:6-10");
  assert.equal(tempBucket(2, 20), "noul:11+");
  const p = softmax([1, 1, 1]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b) - 1) < 1e-12);
  assert.ok(Math.abs(confidenceFromProbs(p)) < 1e-12);
  assert.equal(confidenceFromProbs([1, 0]), 1);
});
