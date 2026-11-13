"""Strict shared stage loader (Codex Task 9, D2). Synthetic block files only; no model calls."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import analyze_public_register_branch as arb  # noqa: E402
import analyze_public_write_passage as W  # noqa: E402

A, T = 0.07293697809562831, -0.15395151820007266


def block(run, stage, first, last, *, name=None, lanes=2, alpha=A, logt=T, dtype=np.float64, nan=False):
    y = np.random.default_rng(1000 * stage + first).normal(size=(last - first + 1, lanes, 896)).astype(dtype)
    if nan:
        y[-1, 0, 5] = np.nan
    f = run / f"stage-{stage:02d}-block-{name if name is not None else f'{last:03d}'}.npz"
    np.savez(f, states=y, first_state_index=first, last_state_index=last, alpha=alpha, log10_temperature=logt)
    return f, y


def record(run, stages):
    """stages: (stage_id, steps, completed_steps, status, files, (alpha, log10_temperature))."""
    recs = [{"stage_id": sid, "steps": steps, "completed_steps": done, "status": status,
             "controls": {"alpha": c[0], "log10_temperature": c[1]},
             "blocks": [{"path": f.name, "bytes": f.stat().st_size,
                         "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]}
            for sid, steps, done, status, files, c in stages]
    (run / "result.json").write_text(json.dumps({"stages": recs}))


def two_blocks(run):
    f1, y1 = block(run, 0, 1, 12)
    f2, y2 = block(run, 0, 13, 24)
    return f1, f2, np.concatenate([y1, y2])


def test_valid_stage_against_its_record(tmp_path):
    f1, f2, y = two_blocks(tmp_path)
    record(tmp_path, [(0, 24, 24, "completed", [f1, f2], (A, T))])
    states, a, t, files = arb.load_stage(tmp_path, 0)
    assert np.array_equal(states, y) and (a, t) == (A, T) and files == [f1, f2]
    assert np.array_equal(arb.load_stage(tmp_path, 0, 24)[0], y)
    with pytest.raises(SystemExit, match="expected 36"):
        arb.load_stage(tmp_path, 0, 36)


def test_duplicate_update_range_is_rejected(tmp_path):
    """Codex's reproduction: -012 and -024 both declaring updates 1..12 were returned as 24 updates."""
    block(tmp_path, 0, 1, 12)
    block(tmp_path, 0, 1, 12, name="024")
    with pytest.raises(SystemExit, match="does not name its last update"):
        arb.load_stage(tmp_path, 0, 24)


def test_overlap_and_gap_and_missing_start_are_rejected(tmp_path):
    for k, ranges in enumerate([((1, 12), (7, 18)), ((1, 12), (25, 36)), ((13, 24),)]):
        run = tmp_path / f"case-{k}"
        run.mkdir()
        for first, last in ranges:
            block(run, 0, first, last)
        with pytest.raises(SystemExit, match="starts at update"):
            arb.load_stage(run, 0, partial=True)


def test_lexical_order_is_not_chronology(tmp_path):
    """Unpadded names sort 108 before 96; the loader orders by stored index and returns the true path."""
    f1, y1 = block(tmp_path, 0, 1, 96, name="96")
    f2, y2 = block(tmp_path, 0, 97, 108, name="108")
    states, _, _, files = arb.load_stage(tmp_path, 0, 108)
    assert np.array_equal(states, np.concatenate([y1, y2])) and files == [f1, f2]


@pytest.mark.parametrize("bad", [dict(nan=True), dict(dtype=np.float32)])
def test_nonfinite_or_reduced_precision_block_is_rejected(tmp_path, bad):
    block(tmp_path, 0, 1, 12)
    block(tmp_path, 0, 13, 24, **bad)
    with pytest.raises(SystemExit, match="not a finite float64 block"):
        arb.load_stage(tmp_path, 0, 24)


def test_mixed_controls_and_lane_shapes_are_rejected(tmp_path):
    a, b = tmp_path / "controls", tmp_path / "lanes"
    a.mkdir(), b.mkdir()
    block(a, 0, 1, 12)
    block(a, 0, 13, 24, alpha=A + 1e-12)
    with pytest.raises(SystemExit, match="mixes controls"):
        arb.load_stage(a, 0, 24)
    block(b, 0, 1, 12)
    block(b, 0, 13, 24, lanes=1)
    with pytest.raises(SystemExit, match="lane shape changes"):
        arb.load_stage(b, 0, 24)


def test_unreadable_block_is_rejected(tmp_path):
    np.savez(tmp_path / "stage-00-block-012.npz", states=np.zeros((12, 2, 896)))
    with pytest.raises(SystemExit, match="not a readable stage block"):
        arb.load_stage(tmp_path, 0, 12)


def test_files_must_be_exactly_the_recorded_blocks(tmp_path):
    f1, f2, _ = two_blocks(tmp_path)
    record(tmp_path, [(0, 12, 12, "completed", [f1], (A, T))])          # f2 is on disk but not recorded
    with pytest.raises(SystemExit, match="differ from the record"):
        arb.load_stage(tmp_path, 0)


def test_recorded_digest_must_match_the_bytes_read(tmp_path):
    f1, f2, _ = two_blocks(tmp_path)
    record(tmp_path, [(0, 24, 24, "completed", [f1, f2], (A, T))])
    r = json.loads((tmp_path / "result.json").read_text())
    r["stages"][0]["blocks"][1]["sha256"] = "0" * 64
    (tmp_path / "result.json").write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="SHA-256 differs"):
        arb.load_stage(tmp_path, 0)


def test_record_controls_must_match_the_blocks(tmp_path):
    f1, f2, _ = two_blocks(tmp_path)
    record(tmp_path, [(0, 24, 24, "completed", [f1, f2], (A, T + 1e-9))])
    with pytest.raises(SystemExit, match="block controls differ from the record"):
        arb.load_stage(tmp_path, 0)


def test_incomplete_record_needs_explicit_partial_mode(tmp_path):
    f1, f2, y = two_blocks(tmp_path)
    record(tmp_path, [(0, 36, 24, "stopped", [f1, f2], (A, T))])
    with pytest.raises(SystemExit, match="record stopped"):
        arb.load_stage(tmp_path, 0)
    assert np.array_equal(arb.load_stage(tmp_path, 0, partial=True)[0], y)


def test_without_a_record_the_length_must_be_stated(tmp_path):
    _, _, y = two_blocks(tmp_path)
    with pytest.raises(SystemExit, match="no saved stage manifest"):
        arb.load_stage(tmp_path, 0)
    assert np.array_equal(arb.load_stage(tmp_path, 0, 24)[0], y)
    assert np.array_equal(arb.load_stage(tmp_path, 0, partial=True)[0], y)
    with pytest.raises(SystemExit, match="expected 36"):
        arb.load_stage(tmp_path, 0, 36)


@pytest.mark.parametrize("kwargs", [dict(expected_steps=True), dict(expected_steps=0), dict(expected_steps=12.0),
                                    dict(expected_steps=24, partial=True)])
def test_expected_length_argument_is_validated(tmp_path, kwargs):
    two_blocks(tmp_path)
    with pytest.raises(ValueError):
        arb.load_stage(tmp_path, 0, **kwargs)


def test_stage_count_requires_stages_zero_to_n(tmp_path):
    block(tmp_path, 0, 1, 12)
    block(tmp_path, 2, 1, 12)
    with pytest.raises(SystemExit, match="are not 0..1"):
        arb.stage_count(tmp_path)
    ok = tmp_path / "ok"
    ok.mkdir()
    f0, _ = block(ok, 0, 1, 12)
    f1, _ = block(ok, 1, 1, 12)
    record(ok, [(0, 12, 12, "completed", [f0], (A, T)), (1, 12, 12, "completed", [f1], (A, T))])
    assert arb.stage_count(ok) == 2
    r = json.loads((ok / "result.json").read_text())
    r["stages"] = r["stages"][:1]
    (ok / "result.json").write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="are not 0..0"):
        arb.stage_count(ok)


def test_write_passage_loader_is_the_strict_one(tmp_path):
    f1, f2, y = two_blocks(tmp_path)
    record(tmp_path, [(0, 24, 24, "completed", [f1, f2], (A, T))])
    assert np.array_equal(W.load_stage(tmp_path, 0, A, T), y)
    with pytest.raises(SystemExit, match="differ from the design"):
        W.load_stage(tmp_path, 0, A + 1e-12, T)
    block(tmp_path, 0, 1, 12, name="036")                                 # a stale duplicate appears
    with pytest.raises(SystemExit):
        W.load_stage(tmp_path, 0, A, T)
