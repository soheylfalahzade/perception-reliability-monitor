# Methods

## 1. Quality features (reference-free)
Calibrated once on clear frames: mean brightness `b`, contrast `c` (grey-level std), sharpness `s`
(variance of the Laplacian), noise `n` (1.4826 x median absolute Laplacian deviation), dynamic range `d`
(p95 - p05) and a per-block gradient map (8 x 8 blocks). At run time the features are
`log(b/b0), log(c/c0), log(s/s0), log(n/n0)`, the saturated-pixel fraction (grey >= 250),
`log(d/d0)` and the *obstruction fraction*: share of blocks whose slowly averaged gradient (EMA 0.05)
falls below 85% of the clear baseline.

## 2. Detector-health features
`mean_score` (mean detection confidence over the last 240 detections), `count_ratio` (mean count over
60 frames divided by the clear reference count) and `fragmentation` (new track IDs per detection from a
greedy IoU tracker over 60 frames).

## 3. Reliability model
Standardised features `z` and a logistic model `r_raw = sigmoid(w . z + b)` fitted by L2-regularised
Newton/IRLS (`lambda = 1`). Label: a frame is *reliable* if the rolling 30-frame recall of the reference
detector is at least `0.6 x` the camera's clear-weather recall. Labels are used for fitting and
evaluation only.

## 4. Monitor
`r_t = 0.15 r_raw + 0.85 r_{t-1}`. The mode goes to FALLBACK when `r < 0.35` for 3 s, to DEGRADED when
`r < 0.65` for 3 s, and returns to NORMAL when `r > 0.55` for 6 s (asymmetric dwell times: fall back
quickly, recover cautiously). Thresholds and dwell times were set a priori.

## 5. Metrics
Frame-level AUC (rank-based, ties averaged) and expected calibration error (10 equal-width bins);
mode behaviour on continuous clear -> bad -> clear streams: fraction of the bad segment in FALLBACK/NORMAL,
fraction of clear frames (after an 8 s warm-up) in FALLBACK, and onset-to-FALLBACK delay.

## 6. Closed-loop study
Two-phase intersection, policies `fixed` (30 s / 30 s), `oracle` (adaptive rule on true halting counts),
`blind` (same rule on camera-estimated counts), `monitor` (blind unless any camera is in FALLBACK).
Adaptive rule: switch after a minimum green of 15 s if the other queue exceeds the current one by more
than 3 vehicles, or the current is empty and the other is not; maximum green 60 s. Camera counts are
`n_true x ratio / ratio_clear` with `ratio` resampled from the measured detector count ratios for the
current condition; monitor inputs are feature vectors resampled from the measured banks.
