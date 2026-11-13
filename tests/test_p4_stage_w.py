"""Stage W preparation enforces the Stage R contract (Codex Task 9, D1). Synthetic archive-format files only; no
model calls. The arrays are exact period-4 cycles, a software fixture, not a model trajectory."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_public_p4_register as P  # noqa: E402
import task8_audit_common as C  # noqa: E402


def save(p, v):
    p.write_text(json.dumps(v, indent=1) + "\n")


def binding(p):
    return {"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def manifest(d):
    save(d / "manifest.json", {"files": [binding(p) for p in sorted(d.iterdir())
                                         if p.is_file() and p.name != "manifest.json"]})


def cycles():
    return np.random.default_rng(7).normal(size=(2, 4, 896)) * 0.01


def build(root, steps=400, alpha=P.ALPHA):
    """inputs-R (as stage-r writes it) and a returned stage R shard (as the pair instrument writes it)."""
    R = cycles()
    inputs, d = root / "inputs-R", root / "R" / "shard-00"
    inputs.mkdir(parents=True)
    d.mkdir(parents=True)
    start = np.stack([R[0, 0], R[1, 0]])
    np.savez(inputs / "starts-R.npz", states=start[None])
    save(inputs / "design-R.json", P.stage_r_design())
    save(inputs / "inputs-R.json", {"design_sha256": C.sha(inputs / "design-R.json"),
                                    "starts_sha256": C.sha(inputs / "starts-R.npz")})
    save(d / "design.json", {"profile": "public-p4-register", "stage_set": "R",
                             "design": binding(inputs / "design-R.json"), "starts": binding(inputs / "starts-R.npz")})
    save(d / "backend-readiness.json", {"mode": "native64", "torch_threads": 1, "torch_interop_threads": 1,
                                        "thread_environment": {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                                                 "OPENBLAS_NUM_THREADS",
                                                                                 "NUMEXPR_NUM_THREADS")}})
    save(d / "model-inputs.json", {"config": {"sha256": "c"}, "vectors": {"sha256": "v"}, "weights": {"sha256": "w"}})
    save(d / "source.json", {"source_commit": "f" * 40, "working_tree_status": ""})
    np.save(d / "group-000-start.npy", start)
    ctrl = {"alpha": alpha, "log10_temperature": P.controls(P.H4)["log10_temperature"]}
    np.save(d / "stage-00-initial.npy", start)
    st = dict(stage_id=0, name="relax", group_id=0, group="p4-relax-H4", local_stage_id=0, steps=steps,
              attempted_steps=steps, returned_steps=steps, completed_steps=steps, status="completed",
              failed_outputs=[], cache_comparison={"passed": True}, final_scalar_cache_unchanged=True,
              controls=ctrl, control_bits=C.literal_bits(ctrl), blocks=[],
              initial=dict(binding(d / "stage-00-initial.npy"), shape=[2, 896], dtype="float64",
                           array_sha256=hashlib.sha256(start.tobytes()).hexdigest()))
    y = np.stack([np.stack([R[0, t % 4], R[1, t % 4]]) for t in range(1, steps + 1)])
    for s0 in range(0, steps, 12):
        z = y[s0:s0 + 12]
        f = d / f"stage-00-block-{s0 + len(z):03d}.npz"
        np.savez_compressed(f, states=z, token_ids=np.zeros((len(z), 2), dtype=np.int64),
                            first_state_index=np.int64(s0 + 1), last_state_index=np.int64(s0 + len(z)), **ctrl)
        st["blocks"].append(binding(f))
    calls = C.expected_calls([dict(stage_id=0, stage="relax", group_id=0, group="p4-relax-H4", steps=steps)],
                             "public-p4-register")
    for c in calls:
        c["status"] = "returned"
        if c["category"] == "prefix":
            c["dtype_observations"] = {"norm": {"input": "torch.float64"}}
        else:
            c.update(precision_passed=True, outputs_valid=True, input_unchanged=True, cache_unchanged=True,
                     expanded_cache_valid=True, **({"state_carried": True} if c["category"] == "update" else {}))
    group = dict(group_id=0, name="p4-relax-H4", stage_ids=[0], start_index=0, status="completed",
                 diagnostics_passed=True)
    save(d / "result.json", dict(profile="public-p4-register", status="completed", shard=0, shards=1, groups=[group],
                                 stages=[st], calls=calls, model_calls=len(calls)))
    manifest(d)
    return d, inputs, R


def prepare(root, d, inputs):
    out = root / "W"
    return P.stage_w(argparse.Namespace(r_shard=d, inputs_r=inputs, output_dir=out)), out


def edit(d, change):
    r = json.loads((d / "result.json").read_text())
    change(r)
    save(d / "result.json", r)
    manifest(d)


def rejected(root, d, inputs):
    with pytest.raises(SystemExit, match="does not meet its contract"):
        prepare(root, d, inputs)
    out = root / "W"
    report = json.loads((out / "preparation-failed.json").read_text())
    assert report["status"] == "rejected" and report["problems"]
    assert not any((out / f).exists() for f in ("references-p4.npz", "design-W.json", "starts-W.npz"))
    return report["problems"]


def test_valid_stage_r_prepares_stage_w(tmp_path):
    d, inputs, R = build(tmp_path)
    rc, out = prepare(tmp_path, d, inputs)
    assert rc == 0
    with np.load(out / "references-p4.npz") as z:
        assert np.array_equal(z["states"], R)
    report = json.loads((out / "references-report.json").read_text())
    assert report["R1"]["holds"] and report["stage_r_contract"] == "passed"


def test_short_run_is_rejected(tmp_path):
    """Codex's probe (a): an eight-update run formerly produced references with R1 true."""
    d, inputs, _ = build(tmp_path, steps=8)
    rejected(tmp_path, d, inputs)


def test_wrong_controls_are_rejected(tmp_path):
    """Codex's probe (b): alpha changed by 0.01 on a full 400-update run."""
    d, inputs, _ = build(tmp_path, alpha=P.ALPHA + 0.01)
    assert any("controls" in q for q in rejected(tmp_path, d, inputs))


def test_invalid_manifest_is_rejected(tmp_path):
    d, inputs, _ = build(tmp_path)
    (d / "model-inputs.json").write_text("{}\n")                    # changed after the manifest was written
    assert any("hash mismatch" in q for q in rejected(tmp_path, d, inputs))


def test_wrong_start_is_rejected(tmp_path):
    d, inputs, R = build(tmp_path)
    np.save(d / "group-000-start.npy", np.stack([R[1, 0], R[0, 0]]))
    manifest(d)
    assert any("start" in q for q in rejected(tmp_path, d, inputs))


def test_tampered_prepared_inputs_are_rejected(tmp_path):
    d, inputs, _ = build(tmp_path)
    design = P.stage_r_design()
    design["groups"][0]["stages"][0]["steps"] = 8
    save(inputs / "design-R.json", design)                           # no longer the file inputs-R.json binds
    rejected(tmp_path, d, inputs)


def test_run_from_other_inputs_is_rejected(tmp_path):
    d, inputs, _ = build(tmp_path)
    r = json.loads((d / "design.json").read_text())
    r["starts"]["sha256"] = "0" * 64
    save(d / "design.json", r)
    manifest(d)
    assert any("does not bind" in q for q in rejected(tmp_path, d, inputs))


def test_nonfinite_block_is_rejected(tmp_path):
    d, inputs, _ = build(tmp_path)
    p = d / "stage-00-block-120.npz"
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["states"][3, 1, 9] = np.nan
    np.savez_compressed(p, **arrays)
    edit(d, lambda r: r["stages"][0]["blocks"][9].update(binding(p)))
    assert any("finite float64" in q for q in rejected(tmp_path, d, inputs))


@pytest.mark.parametrize("fault", ["update_steps_all_one", "wrong_group_name", "incomplete_status", "two_stages",
                                   "cache_changed", "input_gate_missing", "expansion_failed"])
def test_record_faults_are_rejected(tmp_path, fault):
    d, inputs, _ = build(tmp_path)

    def change(r):
        if fault == "update_steps_all_one":
            for c in r["calls"]:
                if c["category"] == "update":
                    c["step"] = 1
        elif fault == "wrong_group_name":
            r["groups"][0]["name"] = "p4-hold-only"
        elif fault == "incomplete_status":
            r["stages"][0]["status"] = "stopped"
        elif fault == "two_stages":
            r["stages"].append(dict(r["stages"][0], stage_id=1))
        elif fault == "cache_changed":                       # N1: saved per-call gates
            r["calls"][200]["cache_unchanged"] = False
        elif fault == "input_gate_missing":
            r["calls"][2].pop("input_unchanged")
        elif fault == "expansion_failed":
            r["calls"][300]["expanded_cache_valid"] = False
    edit(d, change)
    rejected(tmp_path, d, inputs)


def test_malformed_result_is_a_recorded_rejection(tmp_path):
    d, inputs, _ = build(tmp_path)
    (d / "result.json").write_text("[1, 2")
    manifest(d)
    rejected(tmp_path, d, inputs)
