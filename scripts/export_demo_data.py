"""
export_demo_data.py
====================
One-time generator for this demo's data/ folder. NOT run by site visitors and
not run automatically — it reads from the private AidPulse research repo and
writes only derived, review-safe JSON into ../data/.

Run from a machine that has the private repo checked out:
    /path/to/LLM_Patient_Similarity/venv/bin/python3 export_demo_data.py \
        --source /path/to/LLM_Patient_Similarity

What it exports and why each is safe to publish:

  data/cardiology_model.json
    The REAL trained encoder (weights + scaler) for Cardiology, from
    Reviewer_Package/Models/cardiology/engine.pkl — already cleared for
    reviewer release, trained on the public Kaggle Cardiovascular Disease
    dataset. Encoder only (no decoder weights are read or exported).

  data/cardiology_corpus.json
    The same 10,254-patient public corpus behind that model's embeddings.npz,
    with raw (public) feature values so retrieved neighbors can be shown with
    real details. Embeddings are NOT shipped — the page computes them
    client-side from this data and the weights above, so query and corpus
    embeddings always come from the exact same code path.

  data/ophthalmology_corpus.json
    Real embeddings for the real 236-patient AIMS Kochi cohort (from
    Cohort_v2/Embeddings/metadata/embeddings.npz), which is what Table 5 in
    the manuscript evaluates. NO model weights (encoder or decoder) are
    exported for this department, and NO raw clinical values are exported —
    only the embedding vector plus a coarse age-decade bucket, gender, and
    CAD-type label. The real REDCap Record ID is replaced with a shuffled
    anonymous index with no relation to the source order, so nothing here
    can be linked back to a specific hospital record.
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd


def export_cardiology(source: Path, out_dir: Path):
    engine_path = source / 'Reviewer_Package' / 'Models' / 'cardiology' / 'engine.pkl'
    emb_path = source / 'Reviewer_Package' / 'Models' / 'cardiology' / 'embeddings.npz'
    with open(engine_path, 'rb') as f:
        engine = pickle.load(f)
    emb = np.load(emb_path)
    corpus_ids = set(int(i) for i in emb['patient_ids'])

    sd = engine['model_state_dict']

    def lin(name):
        return {'W': sd[f'{name}.weight'].tolist(), 'b': sd[f'{name}.bias'].tolist()}

    def bn(name):
        return {
            'gamma': sd[f'{name}.weight'].tolist(),
            'beta': sd[f'{name}.bias'].tolist(),
            'mean': sd[f'{name}.running_mean'].tolist(),
            'var': sd[f'{name}.running_var'].tolist(),
        }

    model_json = {
        'architecture_note': 'Linear(enc1)->BatchNorm->ReLU -> Linear(enc2)->BatchNorm->ReLU -> Linear(mu). '
                              'Eval mode: BatchNorm uses running stats, Dropout is a no-op.',
        'feature_names': engine['feature_names'],
        'cont_idx': engine['cont_idx'],
        'bin_idx': engine['bin_idx'],
        'scaler_mean': engine['scaler'].mean_.tolist(),
        'scaler_scale': engine['scaler'].scale_.tolist(),
        'enc1': lin('encoder.0'), 'bn1': bn('encoder.1'),
        'enc2': lin('encoder.4'), 'bn2': bn('encoder.5'),
        'mu': lin('fc_mu'),
    }
    (out_dir / 'cardiology_model.json').write_text(json.dumps(model_json))
    print(f"  wrote cardiology_model.json ({(out_dir / 'cardiology_model.json').stat().st_size / 1024:.0f} KB)")

    # ---- Raw corpus: reproduce the exact cleaning from load_cardio_clean(),
    # then keep only rows whose id is in the model's real embeddings.npz corpus.
    df = pd.read_csv(source / 'Primary' / 'Data' / 'cardio_train.csv', sep=';')
    df = df[
        (df['ap_hi'] > 0) & (df['ap_hi'] < 250) &
        (df['ap_lo'] > 0) & (df['ap_lo'] < 200) &
        (df['ap_lo'] < df['ap_hi'])
    ]
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)]
    df = df[(df['height'] >= 140) & (df['height'] <= 220) & (df['weight'] >= 40) & (df['weight'] <= 200)]
    df['age_years'] = df['age'] / 365.25
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    df['age_group_30_40'] = ((df['age_years'] >= 30) & (df['age_years'] < 40)).astype(int)
    df['age_group_40_50'] = ((df['age_years'] >= 40) & (df['age_years'] < 50)).astype(int)
    df['age_group_50_60'] = ((df['age_years'] >= 50) & (df['age_years'] < 60)).astype(int)
    df['age_group_60_plus'] = (df['age_years'] >= 60).astype(int)
    df['lifestyle_risk'] = df['smoke'] + df['alco'] + (1 - df['active'])
    df['metabolic_risk'] = (df['cholesterol'] - 1) + (df['gluc'] - 1)
    df['gender_bin'] = (df['gender'] == 2).astype(float)

    df = df[df['id'].isin(corpus_ids)].reset_index(drop=True)
    assert len(df) == len(corpus_ids), f"corpus mismatch: {len(df)} rows vs {len(corpus_ids)} embedding ids"

    col_for_feature = {'gender': 'gender_bin'}  # engine calls it 'gender', df column is 'gender_bin'
    feat_cols = [col_for_feature.get(name, name) for name in engine['feature_names']]

    corpus = [
        {'id': int(row['id']), 'cvd': int(row['cardio']), 'x': [float(row[c]) for c in feat_cols]}
        for _, row in df.iterrows()
    ]
    (out_dir / 'cardiology_corpus.json').write_text(json.dumps(corpus))
    print(f"  wrote cardiology_corpus.json ({len(corpus)} patients, "
          f"{(out_dir / 'cardiology_corpus.json').stat().st_size / 1024:.0f} KB)")


def export_ophthalmology(source: Path, out_dir: Path):
    emb_path = source / 'Cohort_v2' / 'Embeddings' / 'metadata' / 'embeddings.npz'
    csv_path = source / 'Cohort_v2' / 'Data' / 'Metadata.csv'
    emb = np.load(emb_path)
    label_map = {0: 'Asymptomatic', 1: 'Chronic Stable Angina', 2: 'ACS'}

    df = pd.read_csv(csv_path)[['Record ID', 'Age', 'Gender']].set_index('Record ID')

    rows = []
    for i in range(len(emb['patient_ids'])):
        record_id = int(emb['patient_ids'][i])
        age = df.loc[record_id, 'Age']
        gender = df.loc[record_id, 'Gender']
        decade = 'Unknown' if pd.isna(age) else f"{int(age) // 10 * 10}s"
        rows.append({
            'cad_type': label_map[int(emb['labels'][i])],
            'age_decade': decade,
            'gender': gender if isinstance(gender, str) else 'Unknown',
            'emb': [float(v) for v in emb['embeddings'][i]],
        })

    rng = random.Random(42)
    rng.shuffle(rows)  # break any link between anonymized order and source order
    for i, row in enumerate(rows, start=1):
        row['p_id'] = i

    (out_dir / 'ophthalmology_corpus.json').write_text(json.dumps(rows))
    print(f"  wrote ophthalmology_corpus.json ({len(rows)} cases, "
          f"{(out_dir / 'ophthalmology_corpus.json').stat().st_size / 1024:.0f} KB)")
    print("  NOTE: no model weights and no raw clinical values were read for export "
          "beyond Age/Gender, which are reduced to a decade bucket and category before writing.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='Path to the private LLM_Patient_Similarity repo root')
    args = ap.parse_args()
    source = Path(args.source).resolve()
    out_dir = Path(__file__).resolve().parent.parent / 'data'
    out_dir.mkdir(exist_ok=True)

    print("Exporting Cardiology (real weights + real public corpus)...")
    export_cardiology(source, out_dir)
    print("Exporting Ophthalmology (embeddings only, anonymized, no weights, no raw values)...")
    export_ophthalmology(source, out_dir)
    print("Done.")
