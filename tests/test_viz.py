from pathlib import Path

import pytest

from relmon.viz import render_monitor_gif

RESULTS = Path(__file__).parents[1] / "docs" / "results" / "eval_results.json"


@pytest.mark.skipif(not RESULTS.exists(), reason="result file not present")
def test_monitor_gif_smoke(tmp_path):
    out = tmp_path / "m.gif"
    info = render_monitor_gif(out, RESULTS, every=120)
    assert out.exists() and out.stat().st_size > 1000 and info["frames"] >= 2
