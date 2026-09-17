// Tiny VAE-encoder forward pass, reimplemented from the real trained weights.
// Mirrors pipeline_lib.py's SplitLossVAE.encode() in eval mode exactly:
// scale continuous features -> concat with binary features (cont then bin,
// per cont_idx/bin_idx) -> Linear->BatchNorm(running stats)->ReLU, twice ->
// Linear(mu). No sampling, no decoder: get_embeddings() only ever uses mu.

function linear(x, layer) {
  const out = new Array(layer.b.length);
  for (let o = 0; o < layer.W.length; o++) {
    let s = layer.b[o];
    const row = layer.W[o];
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}

function batchNormRelu(x, bn, eps = 1e-5) {
  return x.map((v, i) => {
    const norm = ((v - bn.mean[i]) / Math.sqrt(bn.var[i] + eps)) * bn.gamma[i] + bn.beta[i];
    return norm > 0 ? norm : 0;
  });
}

// rawFeatures must be in model.feature_names order, unscaled.
function encode(rawFeatures, model) {
  const contScaled = model.cont_idx.map(
    (idx, k) => (rawFeatures[idx] - model.scaler_mean[k]) / model.scaler_scale[k]
  );
  const bin = model.bin_idx.map((idx) => rawFeatures[idx]);
  const x = contScaled.concat(bin);
  let h = batchNormRelu(linear(x, model.enc1), model.bn1);
  h = batchNormRelu(linear(h, model.enc2), model.bn2);
  return linear(h, model.mu);
}

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

// Top-k most similar entries to queryEmb among items (each item has `.emb`),
// excluding an item at excludeIndex (used when the query is itself a corpus member).
function topKSimilar(queryEmb, items, k, excludeIndex = -1) {
  const scored = items
    .map((item, i) => ({ item, i, sim: cosineSim(queryEmb, item.emb) }))
    .filter((s) => s.i !== excludeIndex);
  scored.sort((a, b) => b.sim - a.sim);
  return scored.slice(0, k);
}

if (typeof module !== 'undefined') {
  module.exports = { linear, batchNormRelu, encode, cosineSim, topKSimilar };
}
