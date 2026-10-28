// Runs the JS feature pipeline on a JSON file of cases and prints the results as JSON.
// Used by backend/tests/test_feature_parity.py.  Usage: node scripts/features_cli.mjs <cases.json>
import { readFileSync } from "node:fs";
import { extract, resample, staticVector } from "../src/lib/features.js";

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = cases.map(({ frames, left_handed }) => {
  const ex = extract(frames, !!left_handed);
  const seq = resample(ex.feats, ex.t, ex.handMask);
  return {
    feats: ex.feats.map((r) => Array.from(r)),
    hand_mask: ex.handMask,
    body: ex.body,
    seq: seq && seq.map((r) => Array.from(r)),
    static: ex.feats.map((r) => staticVector(r)),
  };
});
process.stdout.write(JSON.stringify(out));
