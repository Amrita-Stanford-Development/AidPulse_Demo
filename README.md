# AidPulse — Reviewer Retrieval Demo

A small, standalone, static web page letting a reviewer submit an input and see
a real trained AidPulse retrieval model's output, in the browser, with no
server and no LLM. Built in response to a specific peer-review comment asking
for exactly that.

**Live demo:** _add your Vercel/GitHub Pages URL here after deploying_

## What's here and why it's safe to publish

| Department | Model weights | Reference data | Input |
|---|---|---|---|
| Cardiology | Real, already cleared for reviewer release (public Kaggle-sourced) | Real 10,254-patient public corpus | Free-text — type any hypothetical patient |
| Ophthalmology (AIMS Kochi) | **None shipped** (no encoder, no decoder) | Real embeddings for the real 236-patient private cohort, IDs anonymized, no raw values, coarse age-decade/gender/CAD-type only | Pick one of the 236 anonymized cases |

Full reasoning is in `scripts/export_demo_data.py`'s docstring. Short version:
Cardiology's model and data were already deemed safe to share publicly (public
dataset, and the weights are already sitting in the manuscript's reviewer
package). Ophthalmology's underlying model and the raw AIMS Kochi records are
not — treated the same way the manuscript's own reviewer package already
treats them (patent-pending application, non-public clinical dataset) — so
nothing here can be run through a decoder or joined back to a real patient
record.

## What's deliberately excluded

- The AidPulse application itself (Flask backend, templates, patent-pending).
- The LLM/clinical-reasoning step (Ollama-based) — this demo is retrieval only.
- The Ophthalmology model's weights, in either direction (encoder or decoder).
- Any raw AIMS Kochi patient value. Only a decade-bucketed age, gender, and
  3-class CAD-type label are shown per case.

## Running locally

No build step. Any static file server works:

```
python3 -m http.server 8000
# open http://localhost:8000
```

## Regenerating `data/`

`scripts/export_demo_data.py` reads from the private research repo (not
included here) and writes the JSON files in `data/`. Not runnable standalone
in this repo — same convention as the manuscript's own `Reviewer_Package`.

```
/path/to/LLM_Patient_Similarity/venv/bin/python3 scripts/export_demo_data.py \
    --source /path/to/LLM_Patient_Similarity
```

## Tests

- `node scripts/test_nn.js` — checks the from-scratch JS forward pass against
  real embeddings computed once in PyTorch (must match to float precision).
- `node scripts/smoke_test.js` — end-to-end sanity check of the retrieval
  logic used by the page (self-similarity, label balance, no duplicate
  anonymized IDs).

## Deploying

Static site, no config needed. Push this folder to a GitHub repo and either
enable GitHub Pages, or import it in Vercel (framework preset: "Other" /
static) — no build command, output directory is the repo root.
