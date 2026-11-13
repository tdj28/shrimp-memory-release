# Memory and logic with coexisting cycles in continuous language-model feedback: code and evidence

This repository accompanies the article *Memory and logic with coexisting cycles in continuous language-model
feedback* (T. D. Jones, arXiv link to be added). It contains what another researcher needs to check and rerun the
register experiments: the instrument and runners, the audits and analyses, the plans written before each run, the
receipts of every result, the steering vector and reference inputs, and tests.

**Status: release candidate, not yet published.** The data archive link and the results index are still to come.

## What the loop is

A pretrained language model (Qwen2.5-0.5B-Instruct) is fed the expected embedding of its own next-token distribution,
`x_{n+1} = E^T softmax(z_alpha([P; x_n]) / T)`, with a fixed prompt prefix `P`, a steering vector added after layer 12
with strength `alpha`, and temperature `T`. The prefix is in `tools/validate_qwen_map.py` (`MESSAGES`,
`PREFIX_IDS`); the steering vector is `inputs/vectors.npz` (key `philosophy`). All runs use effective float64
arithmetic on one CPU thread.

## Contents

| Path | What it holds |
|---|---|
| `tools/` | the instrument (`validate_qwen_map.py`, `benchmark_qwen_batch.py` and the runner chain), runners, audits, analyses and figure scripts |
| `tests/` | model-free tests of the runners, audits and launcher (fake backends and synthetic archives) |
| `docs/` | the plan of each experiment, written before its runs, with dated revisions and results |
| `docs/receipts/` | the receipt of each result: inputs bound by SHA-256, outcomes, model-call counts |
| `inputs/` | the steering vector, the two reference cycles and their manifest, and the two earlier result files the runners qualify their inputs against |

Not included: the research history of the private study repository, the materials of the article's prose tests
(Section V: the sentence bank, note texts and calibration sentences), the model weights, and the raw run data, which
is too large for a repository and will be archived separately.

## Environments

- Model runs: Python 3.12, `requirements-run.txt` (torch 2.14.1 CPU build, transformers 5.19.0).
- Audits, analyses, figures and tests, which make no model call: `requirements-audit.txt`.

The model is Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`; the instrument refuses
weights, config or steering vector whose SHA-256 differs from the values in `tools/validate_qwen_map.py`.

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --cache-dir ./hf-cache
```

## Tests

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest -q tests
```

## License

Code: MIT (`LICENSE`). Documents, receipts, inputs and data: CC BY 4.0 (`LICENSE-docs-and-data.md`).
