"""The README tables are generated from the recorded results; these tests pin the generator."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from relmon import render_readme as rr

ROOT = Path(__file__).resolve().parents[1]


def _stream(cond, sev, detected=1.0, steady=0.02, clear=0.05):
    return {
        "condition": cond,
        "severity": sev,
        "n_reps": 20,
        "fallback_frac_bad": 0.6,
        "fallback_frac_bad_ci": [0.5, 0.7],
        "normal_frac_bad": 0.3,
        "fallback_frac_clear": clear,
        "fallback_frac_clear_ci": [0.0, 0.1],
        "fallback_frac_clear_steady": steady,
        "fallback_frac_clear_steady_ci": [0.0, 0.05],
        "carryover_frac": 0.05,
        "detected_frac": detected,
        "delay_s": 4.0,
        "delay_s_ci": [3.0, 5.0],
    }


def _rep(a, e):
    return {
        "auc": a,
        "auc_ci": [a - 0.03, a + 0.03],
        "ece": e,
        "ece_ci": [e - 0.01, e + 0.02],
        "n_clips": 150,
        "n_boot_used": 1000,
    }


def _fake():
    return {
        "report": {
            "quality_only": _rep(0.87, 0.07),
            "health_only": _rep(0.90, 0.08),
            "combined": _rep(0.94, 0.05),
        },
        "auc_differences": {
            "combined - quality_only": {"diff": 0.07, "ci": [0.03, 0.11]},
            "combined - health_only": {"diff": 0.04, "ci": [0.01, 0.07]},
        },
        "per_condition_auc_combined": {"clear": {"auc": 0.93}, "rain": {"auc": 0.98}},
        "reliable_recall_threshold": 0.3,
        "setting_stats": {
            "clear@0.0": {"recall": 0.46, "precision": 0.55},
            "rain@0.5": {"recall": 0.55, "precision": 0.5},
            "rain@0.75": {"recall": 0.02, "precision": 0.06},
        },
        "stream_summary": [_stream("rain", 0.5, detected=0.8), _stream("rain", 0.75)],
    }


def test_blocks_contain_the_recorded_numbers():
    d = _fake()
    frame = rr.block_frame_level(d)
    assert "| **combined** | 0.940 (0.910 to 0.970) | 0.050 (0.040 to 0.070) |" in frame
    assert "- combined minus quality only: +0.070 (+0.030 to +0.110)" in frame
    assert "Per-condition AUC of the combined model: clear 0.93, rain 0.98." in frame
    streams = rr.block_streams(d)
    assert "| rain 0.75 | fails |" in streams and "| rain 0.5 | copes |" in streams
    assert "Mean over 20 replicate streams" in streams
    det = rr.block_detector(d)
    assert "| clear | 0.46 / 0.55 |" in det and "| rain |  | 0.55 / 0.50 | 0.02 / 0.06 |" in det


def test_verdict_rules():
    d = _fake()
    assert rr.verdicts(d)[:3] == ("supported", "supported", "supported")
    bad = copy.deepcopy(d)
    bad["auc_differences"]["combined - health_only"]["ci"] = [-0.01, 0.07]
    assert rr.verdicts(bad)[0] == "not supported"
    bad = copy.deepcopy(d)
    bad["stream_summary"][1]["detected_frac"] = 0.8  # a failing setting reacted too rarely
    assert rr.verdicts(bad)[1] == "partially supported"
    bad = copy.deepcopy(d)
    bad["stream_summary"][0]["fallback_frac_clear"] = 0.2
    assert rr.verdicts(bad)[2] == "partially supported"
    steady_only = copy.deepcopy(d)
    steady_only["stream_summary"][0]["fallback_frac_clear_steady"] = (
        0.2  # recovery-inclusive is lower
    )
    assert rr.verdicts(steady_only)[2] == "supported"
    only_copes = copy.deepcopy(d)
    only_copes["stream_summary"] = [only_copes["stream_summary"][0]]
    assert rr.verdicts(only_copes)[1] == "partially supported"  # no failing stream, nothing shown
    assert "no stream in which the detector fails" in rr.block_claims(only_copes)


def test_claims_block_states_the_verdicts_and_numbers():
    claims = rr.block_claims(_fake())
    assert "**supported**: AUC 0.940 (combined) vs 0.870 (quality) and 0.900 (health)" in claims
    assert "the smallest paired difference is +0.040 (+0.010 to +0.070)" in claims


def test_render_replaces_every_block_and_is_idempotent():
    text = "".join(f"<!-- BEGIN:{n} -->\nPLACEHOLDER\n<!-- END:{n} -->\n" for n in rr.BLOCKS)
    once = rr.render(text, _fake())
    assert "PLACEHOLDER" not in once and rr.render(once, _fake()) == once


def test_readme_matches_the_recorded_results():
    out = subprocess.run(
        [sys.executable, "-m", "relmon.render_readme", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    d = json.loads((ROOT / rr.RESULTS).read_text(encoding="utf-8"))
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{d['n_test_frames']:,} frames, {round(100 * d['reliable_frac'])}% reliable" in text


def test_every_marker_pair_in_the_readme_has_a_generator():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for n in rr.BLOCKS:
        assert f"<!-- BEGIN:{n} -->" in text and f"<!-- END:{n} -->" in text


@pytest.mark.parametrize("threshold", [rr.REACT_MIN, rr.FALSE_MAX])
def test_verdict_thresholds_are_the_documented_ones(threshold):
    doc = rr.__doc__ or ""
    assert f"{int(threshold * 100)}%" in doc or f"{threshold:.2f}" in doc
