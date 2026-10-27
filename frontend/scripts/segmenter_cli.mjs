// Feeds frames through the auto-segmenter and prints [startT, endT, n] per emitted segment.
// Used by backend/tests/test_segmenter.py.  Usage: node scripts/segmenter_cli.mjs <frames.json> <static|dynamic>
import { readFileSync } from "node:fs";
import { Segmenter } from "../src/lib/segmenter.js";

const frames = JSON.parse(readFileSync(process.argv[2], "utf8"));
const seg = new Segmenter(process.argv[3] || "dynamic");
const out = [];
for (const f of frames) {
  const s = seg.push(f);
  if (s) out.push([s[0].t, s[s.length - 1].t, s.length]);
}
process.stdout.write(JSON.stringify(out));
