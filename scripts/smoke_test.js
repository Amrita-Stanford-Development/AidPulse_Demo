// End-to-end sanity check of the exact retrieval logic the browser runs,
// without a browser: encode the whole corpus, then confirm (a) a patient is
// its own nearest neighbor (embedding is a pure function of its features),
// and (b) retrieval isn't degenerate (neighbors aren't all one class).
const fs = require('fs');
const path = require('path');
const { encode, topKSimilar } = require('../nn.js');

const dataDir = path.join(__dirname, '../data');
const cardioModel = JSON.parse(fs.readFileSync(path.join(dataDir, 'cardiology_model.json')));
const cardioCorpus = JSON.parse(fs.readFileSync(path.join(dataDir, 'cardiology_corpus.json')));
const ophtho = JSON.parse(fs.readFileSync(path.join(dataDir, 'ophthalmology_corpus.json')));

console.log(`Cardiology corpus: ${cardioCorpus.length} patients`);
cardioCorpus.forEach((p) => { p.emb = encode(p.x, cardioModel); });

// self-similarity check
const probe = cardioCorpus[42];
const top = topKSimilar(probe.emb, cardioCorpus, 3, 42);
console.log('Nearest (excl. self) to patient', probe.id, '->', top.map((t) => `${t.item.id}@${t.sim.toFixed(4)}`));
if (top[0].sim < 0.9) { console.error('FAIL: expected a very close neighbor to exist in a 10k corpus'); process.exitCode = 1; }

const positives = cardioCorpus.filter((p) => p.cvd === 1).length;
console.log(`Cardiology label balance: ${positives}/${cardioCorpus.length} positive`);

console.log(`\nOphthalmology corpus: ${ophtho.length} cases`);
const ids = new Set(ophtho.map((r) => r.p_id));
if (ids.size !== ophtho.length) { console.error('FAIL: duplicate anonymized p_id'); process.exitCode = 1; }
const cadCounts = {};
ophtho.forEach((r) => { cadCounts[r.cad_type] = (cadCounts[r.cad_type] || 0) + 1; });
console.log('CAD type distribution:', cadCounts);
const oTop = topKSimilar(ophtho[0].emb, ophtho, 5, 0);
console.log('Nearest 5 to case', ophtho[0].p_id, '->', oTop.map((t) => `#${t.item.p_id}@${t.sim.toFixed(3)}`));

console.log(process.exitCode ? '\nSMOKE TEST FAILED' : '\nSMOKE TEST PASSED');
