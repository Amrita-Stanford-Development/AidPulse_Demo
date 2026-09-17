(function () {
  const CHOL_LABEL = { 1: 'Normal', 2: 'Above normal', 3: 'Well above normal' };

  let model = null;
  let corpus = null;
  let nameIdx = null;

  function get(x, name) { return x[nameIdx[name]]; }

  function computeFeatures(form) {
    const age = Number(form.age.value);
    const gender = Number(form.gender.value);
    const height = Number(form.height.value);
    const weight = Number(form.weight.value);
    const ap_hi = Number(form.ap_hi.value);
    const ap_lo = Number(form.ap_lo.value);
    const cholesterol = Number(form.cholesterol.value);
    const gluc = Number(form.gluc.value);
    const smoke = form.smoke.checked ? 1 : 0;
    const alco = form.alco.checked ? 1 : 0;
    const active = form.active.checked ? 1 : 0;

    const bmi = weight / (height / 100) ** 2;
    const dict = {
      age_years: age, height, weight, bmi, ap_hi, ap_lo,
      pulse_pressure: ap_hi - ap_lo,
      map: (ap_hi + 2 * ap_lo) / 3,
      cholesterol, gluc, gender, smoke, alco, active,
      age_group_30_40: age >= 30 && age < 40 ? 1 : 0,
      age_group_40_50: age >= 40 && age < 50 ? 1 : 0,
      age_group_50_60: age >= 50 && age < 60 ? 1 : 0,
      age_group_60_plus: age >= 60 ? 1 : 0,
      lifestyle_risk: smoke + alco + (1 - active),
      metabolic_risk: cholesterol - 1 + (gluc - 1),
    };
    return model.feature_names.map((n) => dict[n]);
  }

  function cardHtml(x, cvd, sim) {
    return `<div class="card">
      <span class="sim">${(sim * 100).toFixed(1)}% similar</span>
      <dl>
        <div class="row"><dt>Age</dt><dd>${Math.round(get(x, 'age_years'))}</dd><dt>Gender</dt><dd>${get(x, 'gender') === 1 ? 'Male' : 'Female'}</dd></div>
        <div class="row"><dt>BMI</dt><dd>${get(x, 'bmi').toFixed(1)}</dd><dt>BP</dt><dd>${get(x, 'ap_hi')}/${get(x, 'ap_lo')}</dd></div>
        <div class="row"><dt>Cholesterol</dt><dd>${CHOL_LABEL[get(x, 'cholesterol')]}</dd></div>
        <div class="row"><dt>Glucose</dt><dd>${CHOL_LABEL[get(x, 'gluc')]}</dd></div>
        <div class="row"><dt>Smoker</dt><dd>${get(x, 'smoke') ? 'Yes' : 'No'}</dd><dt>Active</dt><dd>${get(x, 'active') ? 'Yes' : 'No'}</dd></div>
        <div class="row"><dt>Outcome</dt><dd><b>${cvd ? 'CVD Positive' : 'CVD Negative'}</b></dd></div>
      </dl>
    </div>`;
  }

  function onSubmit(e) {
    e.preventDefault();
    if (!model || !corpus) return;
    const x = computeFeatures(e.target);
    const queryEmb = encode(x, model);
    const top = topKSimilar(queryEmb, corpus, 10);

    const positives = top.filter((t) => t.item.cvd === 1).length;
    const summary = document.getElementById('cardio-summary');
    summary.innerHTML = `Nearest 10 of 10,254: <b>${positives}</b> CVD Positive, <b>${10 - positives}</b> CVD Negative.`;

    document.getElementById('cardio-results').innerHTML = top
      .map((t) => cardHtml(t.item.x, t.item.cvd, t.sim))
      .join('');
  }

  async function init() {
    [model, corpus] = await Promise.all([
      fetch('data/cardiology_model.json').then((r) => r.json()),
      fetch('data/cardiology_corpus.json').then((r) => r.json()),
    ]);
    nameIdx = {};
    model.feature_names.forEach((n, i) => { nameIdx[n] = i; });
    corpus.forEach((p) => { p.emb = encode(p.x, model); });
    document.getElementById('cardio-form').addEventListener('submit', onSubmit);
  }

  window.addEventListener('DOMContentLoaded', init);
})();
