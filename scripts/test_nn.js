// Correctness check: does the JS forward pass reproduce the real production
// embeddings (computed once in Python/PyTorch) for known corpus patients?
// Run: node scripts/test_nn.js
const fs = require('fs');
const path = require('path');
const { encode } = require('../nn.js');

const model = JSON.parse(fs.readFileSync(path.join(__dirname, '../data/cardiology_model.json')));
const corpus = JSON.parse(fs.readFileSync(path.join(__dirname, '../data/cardiology_corpus.json')));
const groundTruth = JSON.parse(fs.readFileSync(path.join(__dirname, '_ground_truth_sample.json')));

const byId = new Map(corpus.map((p) => [p.id, p]));

let maxErr = 0;
for (const gt of groundTruth) {
  const patient = byId.get(gt.id);
  if (!patient) { console.log(`MISSING patient ${gt.id}`); process.exitCode = 1; continue; }
  const jsEmb = encode(patient.x, model);
  const err = Math.max(...jsEmb.map((v, i) => Math.abs(v - gt.emb[i])));
  maxErr = Math.max(maxErr, err);
  console.log(`id=${gt.id}  max abs diff = ${err.toExponential(3)}`);
  console.log('  js:', jsEmb.map((v) => v.toFixed(4)).join(', '));
  console.log('  py:', gt.emb.map((v) => v.toFixed(4)).join(', '));
}
console.log(maxErr < 1e-3 ? `PASS (max diff ${maxErr.toExponential(3)})` : `FAIL (max diff ${maxErr.toExponential(3)})`);
if (maxErr >= 1e-3) process.exitCode = 1;
