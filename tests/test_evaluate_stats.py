"""Known-answer tests for the interval and stream-statistics helpers in relmon.evaluate."""
import numpy as np
import pytest

from relmon.evaluate import (
    auc,
    bootstrap_ci,
    cluster_bootstrap_auc_difference,
    cluster_bootstrap_metrics,
    ece,
    stream_stats,
    summarize_streams,
)


def _stream(bad_modes, first_clear_fallback=(), last_clear_fallback=()):
    """200 clear + 160 bad + 200 clear frames, all NORMAL except the given overrides."""
    seg = np.array([0] * 200 + [1] * 160 + [2] * 200)
    modes = np.array(["normal"] * 560, dtype=object)
    modes[200:360] = bad_modes
    for i in first_clear_fallback:
        modes[i] = "fallback"
    for i in last_clear_fallback:
        modes[360 + i] = "fallback"
    return modes, seg


def test_stream_stats_known_answer():
    bad = ["normal"] * 30 + ["fallback"] * 100 + ["degraded"] * 30
    modes, seg = _stream(bad, first_clear_fallback=range(0, 80), last_clear_fallback=range(0, 40))
    s = stream_stats(modes, seg, fps=10.0, warmup_s=8.0)
    assert s["fallback_frac_bad"] == pytest.approx(100 / 160)
    assert s["normal_frac_bad"] == pytest.approx(30 / 160)
    assert s["delay_s"] == pytest.approx(3.0)
    # first clear segment: 80 warm-up frames dropped (120 left); second clear segment: all 200
    assert s["fallback_frac_clear"] == pytest.approx(40 / 320)
    # steady state drops the first 80 frames of both: the 40 recovery frames fall inside the warm-up
    assert s["fallback_frac_clear_steady"] == 0.0


def test_stream_stats_without_any_fallback_has_no_delay():
    modes, seg = _stream(["normal"] * 160)
    s = stream_stats(modes, seg)
    assert s["delay_s"] is None and s["fallback_frac_bad"] == 0.0 and s["normal_frac_bad"] == 1.0


def test_warmup_applies_to_the_first_clear_segment_only_for_the_headline_fraction():
    modes, seg = _stream(["normal"] * 160, first_clear_fallback=range(0, 80))
    s = stream_stats(modes, seg)
    assert s["fallback_frac_clear"] == 0.0 and s["fallback_frac_clear_steady"] == 0.0
    modes, seg = _stream(["normal"] * 160, first_clear_fallback=range(80, 200))
    s = stream_stats(modes, seg)
    assert s["fallback_frac_clear"] == pytest.approx(120 / 320)
    assert s["fallback_frac_clear_steady"] == pytest.approx(120 / 240)


def test_bootstrap_ci_properties():
    assert bootstrap_ci([0.3]) is None and bootstrap_ci([None, 0.3]) is None
    assert bootstrap_ci([0.4, 0.4, 0.4]) == pytest.approx([0.4, 0.4])
    vals = list(np.linspace(0.0, 1.0, 12))
    lo, hi = bootstrap_ci(vals, seed=3)
    assert lo < np.mean(vals) < hi
    assert bootstrap_ci(vals, seed=3) == [lo, hi]
    lo2, hi2 = bootstrap_ci(vals * 8, seed=3)
    assert (hi2 - lo2) < (hi - lo)


def _clips(separation, n_clips=12, n_frames=40, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.array([0.0, 1.0]), n_clips * n_frames // 2)
    p = np.clip(0.5 + separation * (y - 0.5) + rng.normal(0, 0.15, len(y)), 0.001, 0.999)
    ids = np.repeat(np.arange(n_clips), n_frames)
    return p, y, ids


def test_cluster_bootstrap_brackets_a_clean_signal_and_is_deterministic():
    p, y, ids = _clips(separation=1.0)
    out = cluster_bootstrap_metrics(p, y, ids, n_boot=200, seed=1)
    lo, hi = out["auc_ci"]
    assert 0.0 <= lo <= auc(p, y) <= hi <= 1.0
    assert out["n_clips"] == 12 and 0.0 <= out["ece_ci"][0] <= out["ece_ci"][1] <= 1.0
    assert cluster_bootstrap_metrics(p, y, ids, n_boot=200, seed=1) == out
    p2, y2, ids2 = _clips(separation=0.3)
    lo2, hi2 = cluster_bootstrap_metrics(p2, y2, ids2, n_boot=200, seed=1)["auc_ci"]
    assert hi2 - lo2 > hi - lo and lo2 <= auc(p2, y2) <= hi2
    p3, y3, ids3 = _clips(separation=0.3, n_clips=48)
    lo3, hi3 = cluster_bootstrap_metrics(p3, y3, ids3, n_boot=200, seed=1)["auc_ci"]
    assert hi3 - lo3 < hi2 - lo2          # more clips, narrower interval


def test_cluster_bootstrap_covers_the_true_auc_of_pure_noise():
    covered = 0
    for seed in range(40):
        p, y, ids = _clips(separation=0.0, seed=seed)
        lo, hi = cluster_bootstrap_metrics(p, y, ids, n_boot=200, seed=1)["auc_ci"]
        covered += lo <= 0.5 <= hi
    assert covered / 40 >= 0.8


def test_ece_of_perfect_calibration_is_small():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(float)
    assert ece(p, y) < 0.03


def test_summarize_streams_averages_detected_delays_only():
    reps = [dict(fallback_frac_bad=0.5, normal_frac_bad=0.1, fallback_frac_clear=0.0,
                 fallback_frac_clear_steady=0.0, delay_s=3.0),
            dict(fallback_frac_bad=0.7, normal_frac_bad=0.2, fallback_frac_clear=0.1,
                 fallback_frac_clear_steady=0.0, delay_s=5.0),
            dict(fallback_frac_bad=0.0, normal_frac_bad=1.0, fallback_frac_clear=0.0,
                 fallback_frac_clear_steady=0.0, delay_s=None)]
    s = summarize_streams(reps, n_boot=200)
    assert s["n_reps"] == 3 and s["detected_frac"] == pytest.approx(2 / 3)
    assert s["delay_s"] == pytest.approx(4.0)
    assert s["fallback_frac_bad"] == pytest.approx(0.4)
    assert s["fallback_frac_bad_ci"][0] <= 0.4 <= s["fallback_frac_bad_ci"][1]
    single = summarize_streams(reps[:1], n_boot=200)
    assert single["fallback_frac_bad_ci"] is None and single["delay_s"] == 3.0


def test_carryover_is_not_counted_as_a_reaction():
    bad = ["fallback"] * 40 + ["normal"] * 120
    modes, seg = _stream(bad, last_clear_fallback=())
    modes[199] = "fallback"          # monitor already in FALLBACK in the frame before the onset
    s = stream_stats(modes, seg)
    assert s["carryover"] is True and s["delay_s"] is None
    modes2, seg2 = _stream(bad)
    s2 = stream_stats(modes2, seg2)
    assert s2["carryover"] is False and s2["delay_s"] == 0.0


def test_summarize_streams_reports_carryover_and_detection_separately():
    base = dict(fallback_frac_bad=0.5, normal_frac_bad=0.1, fallback_frac_clear=0.0,
                fallback_frac_clear_steady=0.0)
    reps = [dict(base, carryover=False, delay_s=4.0), dict(base, carryover=True, delay_s=None),
            dict(base, carryover=False, delay_s=None), dict(base, carryover=False, delay_s=6.0)]
    s = summarize_streams(reps, n_boot=100)
    assert s["carryover_frac"] == 0.25 and s["detected_frac"] == 0.5 and s["delay_s"] == 5.0


def test_auc_difference_interval_is_paired():
    p_good, y, ids = _clips(separation=1.0, seed=4)
    p_bad, _, _ = _clips(separation=0.3, seed=4)
    d = cluster_bootstrap_auc_difference(p_good, p_bad, y, ids, n_boot=200, seed=1)
    assert d["diff"] == pytest.approx(auc(p_good, y) - auc(p_bad, y))
    assert 0.0 < d["ci"][0] < d["ci"][1]       # better on every resample, interval not degenerate
    same = cluster_bootstrap_auc_difference(p_bad, p_bad, y, ids, n_boot=200, seed=1)
    assert same["diff"] == 0.0 and same["ci"] == [0.0, 0.0]
    assert cluster_bootstrap_auc_difference(p_good, p_bad, y, ids, n_boot=200, seed=1) == d
