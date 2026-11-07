# perception-reliability-monitor

<!-- banner -->
<p align="center"><img src="docs/assets/banner.svg" alt="perception-reliability-monitor" width="860"></p>


**Ground-truth-free reliability monitoring and graceful degradation for camera-based traffic
perception.**

A signal controller that trusts a camera at 2 a.m. in the rain is only as safe as that camera's
worst moment. This package answers one question *without labels at run time*: **can the current
perception output be trusted right now?** It turns image-quality statistics and detector-side health
signals into a calibrated reliability score `r(t)` in [0, 1] and a three-state mode (NORMAL,
DEGRADED, FALLBACK) with hysteresis, so that a controller can switch to a fixed-time plan when the
camera cannot be trusted.

[![ci](https://github.com/soheylfalahzade/perception-reliability-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/soheylfalahzade/perception-reliability-monitor/actions)
![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![typing](https://img.shields.io/badge/mypy-checked-blue)

<p align="center"><img src="docs/assets/pipeline.svg" alt="pipeline animation" width="100%"></p>

<p align="center"><img src="docs/assets/monitor.gif" alt="data-driven animation" width="90%"></p>

<sub>Animation: a stream that starts clear, enters rain (dashed lines) and returns to clear. Left: the degraded frame with ground-truth (green) and detected (red, dashed) boxes. Right: reliability r(t) with mode shading. The first seconds show the warm-up of the health windows. Regenerate with `relmon-gif`.</sub>

```mermaid
flowchart LR
    A[Frame] --> B[Quality features<br/>brightness, contrast, sharpness,<br/>noise, saturation, obstruction]
    A --> C[Detector + tracker]
    C --> D[Health features<br/>mean score, count ratio,<br/>track fragmentation]
    B --> E[Logistic reliability model]
    D --> E
    E --> F[EMA + hysteresis<br/>state machine]
    F --> G{mode}
    G -->|NORMAL / DEGRADED| H[Camera-driven control]
    G -->|FALLBACK| I[Fixed-time plan]
```

## Components

| module | role |
|---|---|
| `quality.py` | seven reference-free features, expressed relative to a clear-weather calibration (zero when nothing changed): log brightness, contrast, sharpness, noise, dynamic range, saturated-pixel fraction, persistent lens-obstruction fraction |
| `health.py` | greedy IoU tracker and rolling detector-health features: mean confidence, count ratio against a clear reference, track births per detection |
| `monitor.py` | `ReliabilityModel` (standardised logistic regression, IRLS) and `ReliabilityMonitor` (EMA, asymmetric dwell times: fall back quickly, recover cautiously) |
| `scene.py`, `degrade.py` | synthetic fixed-camera road scene with exact boxes; night, rain, fog, glare, lens dirt and camera blackout at adjustable severity |
| `detector.py` | a deliberately simple classical detector (background differencing); stand-in for any perception module whose quality really depends on the pixels |
| `dataset.py`, `evaluate.py` | labelled clips, continuous clear -> bad -> clear streams, metrics |
| `sumo_control.py` | closed-loop SUMO study with four control policies |

The monitor is detector-agnostic: it needs frames and the detector's boxes and scores.

## Installation (Ubuntu / WSL2)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[sumo]"    # optional, for the closed-loop study
pytest
```

## Usage

```bash
relmon-eval --dev-seeds 3 --test-seeds 6 --out results/eval          # about 12 min, one core
relmon-sumo --sumocfg examples/sumo/cross_heavy.sumocfg --severity 1.0 --seeds 4 --jobs 3 \
            --out results/sumo                                        # needs the sumo extra
```

```python
from relmon import ReliabilityMonitor, MonitorConfig, QualityExtractor, DetectorHealth
quality.calibrate(clear_frames)                        # once, in clear weather
x = np.concatenate([quality.extract(frame), health.update(detections)])
r, mode = monitor.update(x)                            # per frame
```

## Evaluation protocol

**Data.** Frames are rendered from a synthetic scene (320 x 180, three lanes, moving vehicles with
exact boxes), degraded by five conditions plus a camera blackout at severities 0.25-1.0, and passed
through the reference detector. The label of a frame is *reliable* if the rolling (30-frame) recall
is at least 0.6 times the camera's own clear-weather recall (0.30 for a clear recall of 0.49).
Labels are used only for training and evaluation, never at run time.

**Protocol.** Models are trained on 3 development seeds (1000+) and evaluated on 6 disjoint test
seeds (14,901 frames, 63% reliable). Streams for the mode analysis are continuous (clear 20 s, bad
16 s, clear 20 s) so that health and obstruction state persist, and every setting is replicated on
independent streams (`--stream-reps`).

## Results (synthetic images)

**What the reference detector does under each condition** (recall / precision over test clips):

<!-- BEGIN:detector -->
| condition | 0.25 | 0.5 | 0.75 | 1.0 |
| --- | --- | --- | --- | --- |
| clear | 0.46 / 0.55 |  |  |  |
| night | 0.53 / 0.54 | 0.68 / 0.59 | 0.62 / 0.59 | 0.54 / 0.44 |
| rain | 0.48 / 0.54 | 0.55 / 0.50 | 0.02 / 0.06 | 0.01 / 0.02 |
| fog | 0.67 / 0.60 | 0.59 / 0.55 | 0.75 / 0.59 | 0.46 / 0.61 |
| glare | 0.15 / 0.19 | 0.14 / 0.25 | 0.06 / 0.12 | 0.01 / 0.01 |
| dirt | 0.55 / 0.51 | 0.50 / 0.33 | 0.32 / 0.22 | 0.28 / 0.19 |
| blackout | 0.60 / 0.54 | 0.64 / 0.58 | 0.52 / 0.55 | 0.04 / 0.11 |
<!-- END:detector -->

Night, fog and moderate blackout do not hurt this detector (it normalises each frame
photometrically); rain, glare, dirt and a full outage do.

**Frame-level reliability prediction** (held-out):

<!-- BEGIN:frame_level -->
| features | AUC (95% interval) | ECE (95% interval) |
| --- | --- | --- |
| image quality only | 0.873 (0.824 to 0.914) | 0.070 (0.054 to 0.111) |
| detector health only | 0.901 (0.865 to 0.935) | 0.078 (0.052 to 0.114) |
| **combined** | 0.938 (0.911 to 0.962) | 0.049 (0.033 to 0.084) |

Intervals resample whole clips (150 test clips, 1000 resamples), because the frames of one clip are strongly dependent. Paired AUC difference on the same resampled clips:

- combined minus quality only: +0.065 (+0.036 to +0.098)
- combined minus health only: +0.037 (+0.008 to +0.065)

Per-condition AUC of the combined model: clear 0.93, night 0.84, rain 0.98, fog 0.88, glare 0.82, dirt 0.83, blackout 0.95.
<!-- END:frame_level -->

**Mode behaviour on continuous streams** (clear 20 s, bad 16 s, clear 20 s, replicated):

<!-- BEGIN:streams -->
| stream | detector | FALLBACK in bad | NORMAL in bad (mean) | FALLBACK in clear | in clear, after recovery | reacted | delay [s] | carry-over |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| night 0.5 | copes | 0.00 (0.00 to 0.00) | 0.87 | 0.03 (0.01 to 0.06) | 0.04 (0.00 to 0.08) | 0.00 | - | 0.05 |
| night 0.75 | copes | 0.00 (0.00 to 0.00) | 0.89 | 0.05 (0.01 to 0.10) | 0.04 (0.01 to 0.08) | 0.00 | - | 0.00 |
| rain 0.5 | copes | 0.46 (0.32 to 0.60) | 0.20 | 0.11 (0.06 to 0.18) | 0.06 (0.02 to 0.11) | 0.70 | 6.7 (5.0 to 8.7) | 0.10 |
| rain 0.75 | fails | 0.80 (0.79 to 0.81) | 0.14 | 0.14 (0.10 to 0.21) | 0.03 (0.00 to 0.09) | 1.00 | 3.2 (3.0 to 3.4) | 0.00 |
| fog 0.5 | copes | 0.00 (0.00 to 0.00) | 0.96 | 0.02 (0.00 to 0.05) | 0.01 (0.00 to 0.03) | 0.00 | - | 0.00 |
| fog 0.75 | copes | 0.00 (0.00 to 0.01) | 0.87 | 0.05 (0.01 to 0.10) | 0.05 (0.01 to 0.10) | 0.00 | - | 0.10 |
| glare 0.5 | fails | 0.53 (0.47 to 0.59) | 0.27 | 0.16 (0.11 to 0.21) | 0.05 (0.00 to 0.10) | 1.00 | 7.3 (6.3 to 8.2) | 0.00 |
| glare 0.75 | fails | 0.66 (0.60 to 0.72) | 0.21 | 0.21 (0.15 to 0.27) | 0.06 (0.01 to 0.11) | 0.95 | 5.8 (4.9 to 6.7) | 0.05 |
| dirt 0.5 | copes | 0.04 (0.00 to 0.09) | 0.74 | 0.03 (0.00 to 0.07) | 0.03 (0.00 to 0.07) | 0.10 | 4.9 (2.7 to 7.1) | 0.00 |
| dirt 0.75 | copes | 0.20 (0.09 to 0.33) | 0.44 | 0.08 (0.02 to 0.15) | 0.04 (0.00 to 0.11) | 0.50 | 9.5 (6.8 to 12.0) | 0.00 |
| blackout 0.5 | copes | 0.03 (0.00 to 0.07) | 0.70 | 0.03 (0.01 to 0.06) | 0.01 (0.00 to 0.03) | 0.15 | 2.8 (0.5 to 6.5) | 0.00 |
| blackout 0.75 | copes | 0.00 (0.00 to 0.01) | 0.90 | 0.02 (0.00 to 0.05) | 0.02 (0.00 to 0.05) | 0.00 | - | 0.10 |
| blackout 1 | fails | 0.78 (0.74 to 0.82) | 0.17 | 0.18 (0.13 to 0.23) | 0.04 (0.02 to 0.08) | 0.85 | 4.2 (3.9 to 4.5) | 0.15 |

Mean over 20 replicate streams per setting, with 95% bootstrap intervals over replicates. *FALLBACK in clear* counts the first clear segment after an 8 s warm-up and the whole second one, so slow recovery counts as false fallback; *after recovery* applies the warm-up to both. *Reacted* is the share of replicates in which the monitor entered FALLBACK during the bad segment without having been in FALLBACK just before it; *carry-over* is the share that entered the bad segment already in FALLBACK. The delay is measured from the onset over replicates that reacted. *detector* marks whether the detector's relative recall in that setting is below the reliability threshold.
<!-- END:streams -->

![streams](docs/results/monitor_streams.png)

Rows marked *fails* are the settings where the detector's relative recall is below the reliability
threshold; rows marked *copes* with a non-zero *reacted* share are reactions to degradations that the
detector tolerates, that is, false alarms in terms of detector failure. Fallbacks after a bad episode
are false fallbacks as well, because detector-health windows and the obstruction average recover
slowly.

## Closed-loop study in SUMO

A two-phase intersection (780 and 640 veh/h) is controlled by: **fixed** (30 s / 30 s), **oracle**
(adaptive rule on true queues), **blind** (the same rule on camera-estimated queues), and
**monitor** (blind, but fixed-time while any camera is in FALLBACK). Camera estimates are drawn from
the *measured* count ratios of the detector under each condition; monitor inputs are drawn from the
measured feature banks. The adaptive rule (minimum green 15 s, switch margin 3 vehicles) was tuned on
development seeds so that the oracle beats fixed time; reported seeds are 0-3. Mean travel time (s),
severity 1.0, both cameras degraded unless marked:

| scenario | fixed | oracle | blind | monitor |
|---|---|---|---|---|
| clear | 63.3 | 57.6 | 57.6 | 57.6 |
| rain | 63.3 | 57.6 | 57.9 | 63.3 |
| glare | 63.3 | 57.6 | 57.8 | 63.3 |
| dirt | 63.3 | 57.6 | 57.6 | 59.2 |
| blackout | 63.3 | 57.6 | 57.7 | 63.3 |
| rain, W camera only | 63.3 | 57.6 | 57.9 | 63.3 |
| blackout, W camera only | 63.3 | 57.6 | 57.7 | 63.3 |
| intermittent rain | 63.3 | 57.6 | 57.8 | 59.6 |
| **worst case over 12 scenarios** | 63.3 | 57.6 | 57.9 | 63.3 |

**This is a negative result for the monitor in this setting.** Even under the worst conditions the
blind policy stays within 0.3 s of the oracle, so the monitor's fallback to fixed time only costs
about 5.7 s (9%) in the conditions it flags. The measured count distortions (roughly 0.4x-2.5x of the
true count, with similar bias on both approaches and a rule that only compares queues) are too mild
to hurt this controller, and the blackout noise leaves false detections, so counts do not collapse
to zero. The monitor detects the failures (see above); this experiment does not show that detecting
them improves control. Controllers that depend on absolute counts, on per-approach capacity, or on
detector-triggered actuation would be more exposed; testing them is future work.

## Claims and evidence

<!-- BEGIN:claims_monitor -->
| claim | evidence | file |
| --- | --- | --- |
| Image quality and detector health jointly predict reliability | **supported**: AUC 0.938 (combined) vs 0.873 (quality) and 0.901 (health); the smallest paired difference is +0.037 (+0.008 to +0.065) | `docs/results/eval_results.json` |
| The monitor reacts when the detector fails | **partially supported**: in the 4 streams where the detector fails the monitor reacted in 85% to 100% of replicates, with a mean delay of 3.2 to 7.3 s; reaction elsewhere is in the stream table | `docs/results/eval_results.json` |
| The monitor limits false fallbacks | **partially supported**: up to 0.21 of clear frames in FALLBACK, counting the recovery after the bad episode (glare 0.75); after the recovery up to 0.06 (rain 0.5) | `docs/results/eval_results.json` |
<!-- END:claims_monitor -->

| claim | evidence | file |
|---|---|---|
| Monitoring improves closed-loop control | **not supported**: blind control is within 0.3 s of oracle; fallback costs about 5.7 s | `docs/results/sumo_summary.json` |

## Threats to validity

- *Construct*: images are synthetic and the reference detector is a simple classical method; deep detectors fail differently.
- *Internal*: the reliable label depends on a relative threshold (0.6 x clear recall); monitor thresholds and dwell times were not tuned.
- *External*: one scene geometry; closed-loop camera behaviour resamples measured frame statistics independently in time.
- *Statistical*: 6 test seeds for frame-level metrics, with intervals that resample whole clips, and 4 seeds for the closed loop, without intervals; the mode statistics use replicate streams with intervals over replicates. Replicates of one setting share the scene and the trained model, so their intervals do not cover the variation between trained models.

## Data and code availability

All code, configuration, seeds and result files behind the figures and tables are in this repository.
No external datasets are used. Formal definitions: [`docs/METHODS.md`](docs/METHODS.md); exact
commands, seeds and runtimes: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation

See [`CITATION.cff`](CITATION.cff).

## Limitations and scope

- Images are **synthetic** and the reference detector is deliberately simple; no real adverse-weather
  footage was used. Conclusions about *which* conditions are harmful belong to this detector, not to
  deep detectors, which fail differently.
- Condition severities are generated, not measured; the closed-loop camera model resamples measured
  frame statistics independently in time.
- The reliable/unreliable label depends on the chosen fraction (0.6) of clear-weather recall.
- Monitor thresholds and dwell times were set a priori and not tuned for false-fallback rate.
- The scene has a single camera geometry.

## Calibrating the fallback threshold

The thresholds in `MonitorConfig` were set a priori. To fix the false-fallback rate on healthy footage instead,
record feature streams in good conditions and let the monitor pick the most sensitive threshold that stays
under a target:

```python
from relmon.monitor import calibrate_fallback_threshold, fallback_rate
cfg = calibrate_fallback_threshold(model, healthy_streams, target=0.01)   # <= 1 % of healthy frames in FALLBACK
print(cfg.fallback_below, fallback_rate(model, cfg, healthy_streams))
```

## Roadmap

Real adverse-weather footage (e.g. public night/rain subsets) with a modern detector; a controller
that is sensitive to absolute counts; recalibration of the monitor thresholds against a target
false-fallback rate; per-camera online recalibration.

## License

MIT. See [LICENSE](LICENSE).
