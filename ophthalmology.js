(function () {
  let data = null;

  function cardHtml(item, sim) {
    return `<div class="card">
      <span class="sim">${(sim * 100).toFixed(1)}% similar</span>
      <dl>
        <div class="row"><dt>Case</dt><dd>#${item.p_id}</dd></div>
        <div class="row"><dt>Age</dt><dd>${item.age_decade}</dd><dt>Gender</dt><dd>${item.gender}</dd></div>
        <div class="row"><dt>CAD type</dt><dd><b>${item.cad_type}</b></dd></div>
      </dl>
    </div>`;
  }

  async function renderSteps(query, top) {
    const embRows = query.emb.map((v, i) => [`z[${i}]`, v]);
    const simRows = top.map((t) => ({
      label: `Case #${t.item.p_id} — ${t.item.age_decade}, ${t.item.gender}, ${t.item.cad_type}`,
      sim: t.sim,
    }));
    const code = await codeDetailsHtml('Show the exact code that ran (nn.js — cosine similarity only; no encoder is shipped here)');

    document.getElementById('ophtho-steps').innerHTML = stepsWrapperHtml(`
      <h3>1. The selected case's real embedding</h3>
      <p class="note">This 6-dimensional vector is exactly what the real trained VAE encoder produced for this real (anonymized) patient — computed once, offline, in the original research pipeline. It is <b>not</b> recomputed here: this demo does not ship that encoder's weights, so a new hypothetical case can't be embedded, only one of these 236 already-embedded real cases can be selected.</p>
      ${vecTableHtml(embRows)}
      ${code}

      <h3>2. Ranked against the other 235 cases by cosine similarity</h3>
      ${simTableHtml(simRows)}
    `);
  }

  function onRun() {
    const select = document.getElementById('ophtho-select');
    const idx = Number(select.value);
    const query = data[idx];
    const top = topKSimilar(query.emb, data, 5, idx);

    const matches = top.filter((t) => t.item.cad_type === query.cad_type).length;
    document.getElementById('ophtho-summary').innerHTML =
      `Query: <b>#${query.p_id} — ${query.age_decade}, ${query.gender}, ${query.cad_type}</b>. ` +
      `Nearest 5 of 236: <b>${matches}</b> share the same CAD type.`;

    document.getElementById('ophtho-results').innerHTML = top
      .map((t) => cardHtml(t.item, t.sim))
      .join('');

    renderSteps(query, top);
  }

  async function init() {
    data = await fetch('data/ophthalmology_corpus.json').then((r) => r.json());
    const select = document.getElementById('ophtho-select');
    select.innerHTML = data
      .map((item, i) => `<option value="${i}">#${item.p_id} — ${item.age_decade}, ${item.gender}, ${item.cad_type}</option>`)
      .join('');
    document.getElementById('ophtho-run').addEventListener('click', onRun);
  }

  window.addEventListener('DOMContentLoaded', init);
})();
