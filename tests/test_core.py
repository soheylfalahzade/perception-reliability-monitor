import numpy as np

from relmon.degrade import CONDITIONS, Degrader
from relmon.detector import BackgroundBlobDetector, iou, match_recall
from relmon.health import DetectorHealth, IouTracker
from relmon.monitor import Mode, MonitorConfig, ReliabilityModel, ReliabilityMonitor
from relmon.quality import FEATURE_NAMES, QualityExtractor
from relmon.scene import Scene, spawn_traffic, visible_box


def _frames(n=120, seed=0):
    sc = Scene(0)
    rng = np.random.default_rng(seed)
    return sc, [sc.render(v) for v in spawn_traffic(rng, n, 0.08)]


def test_quality_features_are_zero_in_clear_and_react_to_degradation():
    sc, fr = _frames()
    q = QualityExtractor()
    q.calibrate([f[0] for f in fr[::2]])
    deg = Degrader(1)
    base = q.extract(fr[5][0])
    assert len(base) == len(FEATURE_NAMES) and np.abs(base[[0, 1, 5]]).max() < 0.5
    dark = q.extract(deg.apply(fr[5][0], fr[5][1], "night", 1.0))
    assert dark[0] < -0.8                      # log-brightness drops sharply
    fog = q.extract(deg.apply(fr[5][0], fr[5][1], "fog", 1.0))
    assert fog[1] < -0.15                      # contrast collapses
    glare = q.extract(deg.apply(fr[5][0], fr[5][1], "glare", 1.0))
    assert glare[4] > base[4] + 0.01           # saturated pixels appear


def test_obstruction_feature_grows_with_persistent_dirt():
    sc, fr = _frames()
    q = QualityExtractor()
    q.calibrate([f[0] for f in fr[::2]])
    deg = Degrader(2)
    last = None
    for img, light in fr[:100]:
        last = q.extract(deg.apply(img, light, "dirt", 1.0))
    assert last[6] > 0.04


def test_degradations_are_identity_at_zero_severity_and_valid_images():
    sc, fr = _frames(10)
    deg = Degrader(3)
    img, light = fr[3]
    for c in CONDITIONS:
        out = deg.apply(img, light, c, 0.0)
        assert out.dtype == np.uint8 and np.array_equal(out, img)
        out1 = deg.apply(img, light, c, 1.0)
        assert out1.shape == img.shape and out1.dtype == np.uint8


def test_detector_finds_large_vehicles_in_clear_weather():
    sc, fr = _frames(200)
    det = BackgroundBlobDetector()
    det.calibrate([f[0] for f in fr[::4]])
    rng = np.random.default_rng(5)
    tp = n = 0
    for v in list(spawn_traffic(rng, 400, 0.08))[40::3]:
        img, _ = sc.render(v)
        gt = [b for b in (visible_box(x) for x in v) if b and b[3] - b[1] >= 30]
        a, b, _ = match_recall(gt, det(img))
        tp, n = tp + a, n + b
    assert n > 5 and tp / n > 0.7


def test_iou_and_tracker_ids():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and iou((0, 0, 5, 5), (6, 6, 9, 9)) == 0.0
    from relmon.detector import Box

    tr = IouTracker()
    a = tr.update([Box(10, 10, 30, 30, 1.0)])
    b = tr.update([Box(12, 11, 32, 31, 1.0)])
    assert a == b and tr.births == 1
    h = DetectorHealth(reference_count=1.0)
    for _ in range(5):
        f = h.update([Box(10, 10, 30, 30, 0.8)])
    assert abs(f[0] - 0.8) < 1e-9 and abs(f[1] - 1.0) < 1e-9


def _toy_model():
    rng = np.random.default_rng(0)
    good = rng.normal(0.0, 0.3, (400, 3))
    bad = rng.normal(-2.0, 0.5, (400, 3))
    x = np.vstack([good, bad])
    y = np.concatenate([np.ones(400), np.zeros(400)])
    return ReliabilityModel().fit(x, y)


def test_reliability_model_separates_and_monitor_hysteresis():
    m = _toy_model()
    assert m.predict(np.zeros(3)) > 0.9 and m.predict(np.full(3, -2.0)) < 0.1
    mon = ReliabilityMonitor(m, MonitorConfig(fps=10.0, dwell_down_s=1.0, dwell_up_s=2.0))
    for _ in range(30):
        assert mon.update(np.zeros(3))[1] == Mode.NORMAL
    modes = [mon.update(np.full(3, -2.0))[1] for _ in range(60)]
    assert modes[-1] == Mode.FALLBACK
    first_fb = modes.index(Mode.FALLBACK)
    assert first_fb >= 10                        # dwell time respected (not instantaneous)
    rec = [mon.update(np.zeros(3))[1] for _ in range(120)]
    assert rec[-1] == Mode.NORMAL
    assert Mode.NORMAL not in rec[:15]           # cautious return
