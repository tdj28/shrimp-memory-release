# Memory and logic with coexisting cycles in continuous language-model feedback: code and evidence

This repository accompanies the article *Memory and logic with coexisting cycles in continuous language-model
feedback* (T. D. Jones, arXiv link to be added). It contains what another researcher needs to check and rerun the
register experiments: the instrument and runners, the audits and analyses, the plans written before each run, the
receipts of every result, the steering vector and reference inputs, and tests.

**Status: release candidate, not yet published.**

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
| `inputs/` | the steering vector, the two reference cycles and their manifest, the P8 reference states, and the two earlier runs (`continued-hold/`, `recovery/`) whose results, metadata and trajectories the runners verify before any call |

Not included: the research history of the private study repository; the code and materials of the article's
Section V (the message exchange, the prose tests with their sentence bank, note texts and calibration sentences, and
the Pythia route); the model weights; and the full raw run data (about 60 GB), whose files are identified by SHA-256 in
the receipts and are available from the author on reasonable request.

## Environments

- Model runs: Python 3.12, `requirements-run.txt` (torch 2.14.1 CPU build, transformers 5.19.0).
- Audits, analyses, figures and tests, which make no model call: `requirements-audit.txt`.

The model is Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`; the instrument refuses
weights, config or steering vector whose SHA-256 differs from the values in `tools/validate_qwen_map.py`.

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --cache-dir ./hf-cache
```

## Reproduce the full-adder table

From the repository root, run `python tools/replay_adder_example.py` with NumPy installed.
The example includes the original final twelve states of each hold in both lanes for all four digit pairs.
It checks the archived blocks and reference cycles, scores period and unique cycle membership,
prints all 24 measured readouts, and compares them with the Boolean truth table.
The small extract covers the readouts. Checking continuity throughout the moves requires the full trajectories.
The index identifies each original block and its run manifest.

## Tests

Run each test module in its own process. Some fixtures are archive-sized, and pytest keeps every temporary folder
until a session ends:

```bash
for f in tests/test_*.py; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest -q "$f"
done
```

From a fresh clone this gives 2,145 passed and 107 skipped. 49 tests in nine short-pulse suites
(`*write0_phase_panel*`, `*carried72*`, `test_run_public_short_write0_*`) also check pinned historical artifacts at
their original absolute paths. They fail without the data archive mounted there, exactly as they do in the private
study repository.

## Checking the inputs without a model call

The runners verify every input against its SHA-256 before they load a model. A preflight does this and stops:

```bash
I=inputs
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -u tools/run_public_lane_layout.py --shard 12 --shards 16 --cache-dir ./hf-cache \
  --vectors $I/vectors.npz --references $I/reference-cycles.npz --reference-manifest $I/reference-manifest.json \
  --source-hold-results $I/continued-hold/results.json --recovery-results $I/recovery/results.json \
  --references-sha256 d96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0 \
  --reference-manifest-sha256 17b8e826a63a01d442877b6cf837272f760f49f8f4ba26aa6192294f47b6853f \
  --recovery-results-sha256 5de613795c4bc10e0584072fd7cb12586de4db869286d15cbbdc7f57dab37ef5 \
  --output-dir runs/preflight-lanes-12 --preflight-only
```

Without `--preflight-only` the same command loads the model and runs the job.

The full sets go through the launcher. It requires a clean checkout at the commit you name, a new run folder, and the
run environment's interpreter in `PY`. It admits jobs by available memory, about 7 GiB each, and stops only its own
jobs:

```bash
PY=$(command -v python) tools/launch_full_adder_and_lane_layout.sh preflight "$PWD" "$(git rev-parse HEAD)" runs/pre-adder-and-lanes
PY=$(command -v python) tools/launch_full_adder_and_lane_layout.sh preflight "$PWD" "$(git rev-parse HEAD)" runs/pre-single-lane single-lane
```

Replace `preflight` with `execute` to run them: 8,304 calls for the adder and two-lane sets, 2,496 for the single-lane
set. `inputs/` and `hf-cache/` are the defaults (`SHRIMP_INPUTS`, `SHRIMP_HF_CACHE`). `runs/` and `hf-cache/` are
git-ignored, so the checkout stays clean.

## License

Code: MIT (`LICENSE`). Documents, receipts, inputs and data: CC BY 4.0 (`LICENSE-docs-and-data.md`).

## Prospective write-boundary panel

Use the same thread settings, environment, cache and input arguments as the lane preflight above.
Replace the runner with `tools/check_p6_writes.py`, omit the two shard arguments and
`--preflight-only`, and pass `--protocol boundary --output-dir runs/write-boundary`.
The scalar runner uses 1,944 model calls for six pulse/hold rows. The runner reports a nonzero
exit if a predicted fate differs or is unresolved. Check the execution status of every row,
then run `python tools/audit_write_boundary.py runs/write-boundary runs/boundary-score.json`
from the same working directory to recompute the scores and distinguish a scientific miss from an
interrupted experiment. Keep the full raw outputs locally.
