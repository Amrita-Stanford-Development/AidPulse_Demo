// Shared helpers for showing the intermediate computation to a reviewer:
// the actual numeric vectors at each step, plus the actual code that ran,
// fetched live from nn.js so this can never drift out of sync with it.
let _nnSource = null;
async function getNnSource() {
  if (!_nnSource) _nnSource = await fetch('nn.js').then((r) => r.text());
  return _nnSource;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

function vecTableHtml(pairs, decimals = 4) {
  const rows = pairs
    .map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${typeof v === 'number' ? v.toFixed(decimals) : v}</td></tr>`)
    .join('');
  return `<table class="vec-table">${rows}</table>`;
}

function simTableHtml(rows, decimals = 4) {
  const body = rows
    .map((r) => `<tr><td>${escapeHtml(r.label)}</td><td>${r.sim.toFixed(decimals)}</td></tr>`)
    .join('');
  return `<table class="vec-table"><tr><th>Case</th><th>Cosine similarity</th></tr>${body}</table>`;
}

async function codeDetailsHtml(summary) {
  const src = await getNnSource();
  return `<details class="code-block"><summary>${escapeHtml(summary)}</summary><pre><code>${escapeHtml(src)}</code></pre></details>`;
}

function stepsWrapperHtml(innerHtml) {
  return `<details class="steps">
    <summary>Show the computation, step by step</summary>
    ${innerHtml}
  </details>`;
}

if (typeof module !== 'undefined') {
  module.exports = { getNnSource, escapeHtml, vecTableHtml, simTableHtml, codeDetailsHtml, stepsWrapperHtml };
}
