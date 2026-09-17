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
