# Reproducibility

Development seeds are `1000+`, reported seeds `0-5` (frames) and `0-3` (closed loop).

| result | command | approx. time |
|---|---|---|
| detector behaviour, reliability metrics, mode streams (`docs/results/eval_results.json`) | `relmon-eval --dev-seeds 3 --test-seeds 6 --out results/eval` | 12 min, 1 core |
| closed-loop study (`docs/results/sumo_summary.json`) | `relmon-sumo --sumocfg examples/sumo/cross_heavy.sumocfg --severity 1.0 --seeds 4 --jobs 3 --out results/sumo` | 5 min, 3 cores |
| figure (`docs/results/monitor_streams.png`) | `python -c "from relmon.plots import plot_streams; plot_streams('results/eval/results.json', 'docs/results/monitor_streams.png')"` | seconds |
| animation (`docs/assets/monitor.gif`) | `relmon-gif --out docs/assets/monitor.gif --results docs/results/eval_results.json` | 1 min |

Environment: Python 3.12, versions in `requirements-lock.txt`, SUMO 1.28.0. Re-running reproduces the
figures exactly on the same versions. `make all` runs ruff, mypy and pytest; CI does the same on
Python 3.10-3.12.
