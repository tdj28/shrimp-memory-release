# Memory and logic with coexisting cycles in continuous language-model feedback: code and evidence

This repository accompanies the article *Memory and logic with coexisting cycles in continuous language-model
feedback* (T. D. Jones, arXiv link to be added). It holds the instrument and runners, the audits and analyses, the
plans written before each run, the receipts of the results, the steering vector and reference inputs, and tests. With
the model weights, the runners rerun the register experiments. Without them, a clone can check every input, replay
the full-adder readouts from saved states and run the model-free tests. What a clone cannot check is listed under
[Provenance](#provenance).

**Version:** 21 November 2026, accompanying the first arXiv version of the article.

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
| `inputs/adder-example/` | the final twelve states of every hold of the full-adder run, both lanes, with an index of their original blocks |

Not included: the research history of the private study repository; the programs and materials of the message
exchange and of the sentence test that the article's Discussion describes, and of the study's other prose
experiments, with every sentence and note text; the model weights; and the full raw run data (about 60 GB), whose
files are identified by SHA-256 in the receipts and are available from the author on reasonable request. One analysis
touches that material: the register-branch analysis (`docs/public-register-branch.md`) reads the saved write-1 stages
of one run of the sentence test, in which each note selected an exact write, and its receipt binds those files by
path and SHA-256. The notes, the programs and that run's own receipt are not here. The toggle-sweep figure and its map
classes (`tools/plot_public_toggle_sweep.py`, `tools/classify_public_toggle_sweep.py`) also read the fixed-start scans
of the earlier study of this loop (Ref. 1 of the article), passed as `--map-root`.

## Environments

- Model runs: Python 3.12, `requirements-run.txt` (torch 2.14.1 CPU build, transformers 5.19.0).
- Audits, analyses, figures and tests, which make no model call: `requirements-audit.txt`.

The model is Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`. The instrument refuses
weights or a steering vector whose SHA-256 differs from `WEIGHTS_SHA256` or `VECTOR_SHA256` in
`tools/validate_qwen_map.py`, and a model config that differs from `CONFIG_SHA256` in `tools/check_p6_hold.py`. With
the run environment installed:

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --cache-dir ./hf-cache
```

## Reproduce the full-adder table

From the repository root, run `python tools/replay_adder_example.py` with NumPy installed. The example holds the
original final twelve states of each hold in both lanes for all four digit pairs. It checks the saved blocks and the
reference cycles, scores the period and unique cycle membership of each, prints the eight rows (24 readouts) and
compares them with the full-adder truth table. The small extract covers the readouts only; checking continuity through
the moves needs the full trajectories. The index names each original block and the SHA-256 of its run manifest,
which `docs/receipts/public-full-adder.json` also records; the manifests themselves are in the raw data.

## Tests

Run each test module in its own process, with its temporary folders on disk. Some fixtures are archive-sized (over
1 GB for one module), and pytest keeps every temporary folder until a session ends:

```bash
mkdir -p runs/pytest
for f in tests/test_*.py; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -m pytest -q --basetemp "runs/pytest/$(basename "$f" .py)" "$f"
done
```

The 40 modules took about 3.3 hours on a shared machine. From a fresh clone they give 2,145 passed and 156 skipped.
Of the skips, 49 are tests in nine short-pulse suites (`*write0_phase_panel*`, `*carried72*`,
`test_run_public_short_write0_*`) that check pinned historical artifacts at their original absolute paths; they run
where the data archive is mounted at those paths.

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

## Prospective write-boundary panel

Use the same thread settings, environment, cache and input arguments as the lane preflight above. Replace the runner
with `tools/check_p6_writes.py`, omit the two shard arguments and `--preflight-only`, and pass
`--protocol boundary --output-dir runs/write-boundary`. This runner has no preflight mode: it loads the model at once
and stops itself at 1,944 model calls, 1,200 s or 20 GiB. The six pulse/hold rows use all 1,944 calls, and the runner
exits nonzero if a predicted fate differs or is unresolved. Check the execution status of every row, then run
`python tools/audit_write_boundary.py runs/write-boundary runs/boundary-score.json --references inputs/reference-cycles.npz`
from the same working directory. It recomputes the scores from the saved holds, checks every raw file the rows cite and
tells a scientific miss from an interrupted experiment. The plan, its provenance and the measured outcome are in
`docs/prospective-write-boundary.md`.

## Provenance

- **Frozen commits are in the private study repository.** Where a receipt names a frozen or source commit
  (`frozen_commit`, `source_commit`), the commit is in the private study repository, whose history is not part of
  this release. So this repository's own history does not show when a plan was frozen, and the audits that read
  source files from a frozen commit with `git show` cannot run from a clone, even with the raw data. Most receipts
  also record the SHA-256 of the source files a run executed, which can be compared with the files here.
- **Two versions of `tools/check_p6_writes.py`.** The boundary protocol was added after the earlier receipts were
  written. Five receipts (`fresh-readout`, `qwen-batch-trajectory`, `transverse-amplitude`, `transverse-intermediate`,
  `transverse-readout`) pin the earlier file (SHA-256 `2ac2b805…`), which is the version in the first commit of this
  repository (`git log -- tools/check_p6_writes.py`). The boundary receipt pins the current one.
- **Original paths.** Receipts, run inputs and some tools keep the absolute paths of the original machines (`/data2`,
  `/archive`, and `/home/ubuntu` on rented hosts). They are records: most are bound by SHA-256, and editing them
  would break the chain of hashes. Pass data folders as arguments.
- **Files named but not here.** Some plans and receipts name files that are not included: receipts of experiments
  outside the article, review notes, and tools of other experiments. They are kept as written.

## License

Code: MIT (`LICENSE`). Documents, receipts, inputs and data: CC BY 4.0 (`LICENSE-docs-and-data.md`).
