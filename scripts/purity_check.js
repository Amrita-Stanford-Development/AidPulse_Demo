// Cross-check the demo's retrieval against the manuscript's own reported
// Ophthalmology numbers (Reviewer_Package baseline_comparison_metadata.csv:
// VAE Purity@5=0.5814, MRR=0.7249, P@1=0.5975, n=236, leave-one-out).
const fs = require('fs');
const path = require('path');
const { cosineSim } = require('../nn.js');

const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../data/ophthalmology_corpus.json')));
const n = data.length;

let purity5Sum = 0, p1Sum = 0, mrrSum = 0;
const perClass = {};

for (let i = 0; i < n; i++) {
  const scored = [];
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    scored.push({ j, sim: cosineSim(data[i].emb, data[j].emb) });
  }
  scored.sort((a, b) => b.sim - a.sim);

  const top5 = scored.slice(0, 5);
  const purity5 = top5.filter((s) => data[s.j].cad_type === data[i].cad_type).length / 5;
  const p1 = data[scored[0].j].cad_type === data[i].cad_type ? 1 : 0;
  let rr = 0;
  for (let r = 0; r < scored.length; r++) {
    if (data[scored[r].j].cad_type === data[i].cad_type) { rr = 1 / (r + 1); break; }
  }
  purity5Sum += purity5; p1Sum += p1; mrrSum += rr;

  const cls = data[i].cad_type;
  perClass[cls] = perClass[cls] || { n: 0, purity5: 0 };
  perClass[cls].n += 1;
  perClass[cls].purity5 += purity5;
}

console.log(`n=${n}`);
console.log(`Purity@5 = ${(purity5Sum / n).toFixed(4)}  (manuscript reports 0.5814)`);
console.log(`P@1      = ${(p1Sum / n).toFixed(4)}  (manuscript reports 0.5975)`);
console.log(`MRR      = ${(mrrSum / n).toFixed(4)}  (manuscript reports 0.7249)`);
console.log('\nPer-class Purity@5 (shows where "far away" cases come from):');
for (const [cls, v] of Object.entries(perClass)) {
  console.log(`  ${cls.padEnd(22)} n=${v.n}  Purity@5=${(v.purity5 / v.n).toFixed(3)}`);
}
