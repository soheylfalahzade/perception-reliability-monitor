"""Closed-loop study: perception-driven signal control with and without the reliability monitor.

A two-phase intersection is controlled by (a) a fixed-time plan, (b) a queue-adaptive rule fed
by *true* queues (oracle), (c) the same rule fed by camera-estimated queues (blind), and
(d) the camera-driven rule gated by the reliability monitor (fallback to fixed time).
Camera-estimated queues are drawn from the measured detector behaviour on degraded images
(count ratios per condition) and the monitor consumes feature vectors drawn from the same banks.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from relmon.monitor import Mode, MonitorConfig, ReliabilityModel, ReliabilityMonitor

MIN_GREEN_S, MAX_GREEN_S, MARGIN = 15.0, 60.0, 3.0
POLICIES = ("fixed", "oracle", "blind", "monitor")
APPROACHES = {"S": "in_S_0", "W": "in_W_0"}
GREEN = {"S": "GGrr", "W": "rrGG"}
YELLOW = {"S": "yyrr", "W": "rryy"}


@dataclass(frozen=True)
class Scenario:
    """Camera conditions over time: segments ``(t_start, t_end, condition, severity)``."""

    name: str
    segments: tuple[tuple[float, float, str, float], ...]
    affected: tuple[str, ...] = ("S", "W")   # approaches whose camera is degraded


def scenarios(sev: float = 0.75, horizon: float = 1800.0) -> list[Scenario]:
    out = [Scenario("clear", ((0.0, horizon, "clear", 0.0),))]
    out += [Scenario(c, ((0.0, horizon, c, sev),))
            for c in ("night", "rain", "fog", "glare", "dirt", "blackout")]
    out += [Scenario(f"{c}_W_only", ((0.0, horizon, c, sev),), ("W",))
            for c in ("rain", "glare", "dirt", "blackout")]
    out.append(Scenario("intermittent_rain", ((0.0, 500.0, "clear", 0.0),
                                              (500.0, 1100.0, "rain", sev),
                                              (1100.0, horizon, "clear", 0.0))))
    return out


class Camera:
    """Per-approach perception model drawing from measured banks."""

    def __init__(self, banks: dict, model: ReliabilityModel, scenario: Scenario,
                 rng: np.random.Generator, approach: str) -> None:
        self.banks, self.sc, self.rng, self.approach = banks, scenario, rng, approach
        clear = np.array(banks["clear@0.0"]["count_ratio_bank"])
        self.f_clear = float(clear.mean())
        self.monitor = ReliabilityMonitor(model, MonitorConfig(fps=2.0))

    def _key(self, t: float) -> str:
        if self.approach not in self.sc.affected:
            return "clear@0.0"
        for a, b, c, s in self.sc.segments:
            if a <= t < b:
                return "clear@0.0" if c == "clear" else f"{c}@{s}"
        return "clear@0.0"

    def observe(self, t: float, n_true: int) -> int:
        """Camera-estimated queue (count) for ``n_true`` real halting vehicles."""
        bank = self.banks[self._key(t)]["count_ratio_bank"]
        ratio = float(bank[self.rng.integers(len(bank))]) / self.f_clear
        return max(0, int(round(n_true * ratio)))

    def monitor_step(self, t: float) -> Mode:
        """Advance the monitor with a feature vector drawn from the current condition's bank."""
        bank = self.banks[self._key(t)]["feature_bank"]
        return self.monitor.update(np.asarray(bank[self.rng.integers(len(bank))]))[1]


def run_one(args: tuple) -> dict:
    """Simulate one (scenario, policy, seed) run and return travel-time statistics."""
    import traci

    sumocfg, banks, model_json, sc, policy, seed, horizon = args
    model = ReliabilityModel(np.array(model_json["w"]), np.array(model_json["m"]),
                             np.array(model_json["s"]))
    cams = {k: Camera(banks, model, sc, np.random.default_rng(seed * 7 + i), k)
            for i, k in enumerate(APPROACHES)}
    label = f"{sc.name}_{policy}_{seed}"
    traci.start(["sumo", "-c", sumocfg, "--seed", str(seed), "--no-step-log", "true",
                 "--no-warnings", "true", "--end", str(horizon)], label=label)
    c = traci.getConnection(label)
    depart: dict[str, float] = {}
    travel: list[tuple[float, float]] = []     # (arrival time, travel time)
    phase, p_start, yellow_until, pending = "S", 0.0, -1.0, None
    c.trafficlight.setRedYellowGreenState("C", GREEN[phase])
    fallback_steps = steps = 0
    mode_state = {k: Mode.NORMAL for k in APPROACHES}
    try:
        while c.simulation.getTime() < horizon:
            c.simulationStep()
            t = c.simulation.getTime()
            steps += 1
            for vid in c.simulation.getDepartedIDList():
                depart[vid] = t
            for vid in c.simulation.getArrivedIDList():
                travel.append((t, t - depart.pop(vid, t)))
            for k, cam in cams.items():
                mode_state[k] = cam.monitor_step(t)
            if pending is not None:
                if t >= yellow_until:
                    phase, p_start, pending = pending, t, None
                    c.trafficlight.setRedYellowGreenState("C", GREEN[phase])
                continue
            true_q = {k: c.lane.getLastStepHaltingNumber(lane) for k, lane in APPROACHES.items()}
            fb = any(m == Mode.FALLBACK for m in mode_state.values())
            fallback_steps += int(policy == "monitor" and fb)
            if policy == "oracle":
                q = true_q
            elif policy in ("blind", "monitor"):
                q = {k: cams[k].observe(t, true_q[k]) for k in APPROACHES}
            other = "W" if phase == "S" else "S"
            held = t - p_start
            if policy == "fixed" or (policy == "monitor" and fb):
                switch = held >= 30.0
            else:
                switch = held >= MAX_GREEN_S or (held >= MIN_GREEN_S and (
                    q[other] > q[phase] + MARGIN or (q[phase] == 0 and q[other] > 0)))
            if switch:
                pending, yellow_until = other, t + 3.0
                c.trafficlight.setRedYellowGreenState("C", YELLOW[phase])
    finally:
        c.close()
    tt = np.array([v for _, v in travel]) if travel else np.array([np.nan])
    return {"scenario": sc.name, "policy": policy, "seed": seed, "mean_travel_s": float(tt.mean()),
            "p95_travel_s": float(np.percentile(tt, 95)), "n_arrived": len(travel),
            "fallback_frac": fallback_steps / max(steps, 1)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``relmon-sumo``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sumocfg", default="examples/sumo/cross.sumocfg")
    ap.add_argument("--eval-dir", default="results/eval")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--horizon", type=float, default=1800.0)
    ap.add_argument("--severity", type=float, default=0.75)
    ap.add_argument("--out", default="results/sumo")
    args = ap.parse_args(argv)
    ev = Path(args.eval_dir)
    banks = json.loads((ev / "banks.json").read_text())
    res = json.loads((ev / "results.json").read_text())
    model_json = {"w": res["combined_weights"], "m": res["combined_mean"],
                  "s": res["combined_scale"]}
    jobs = [(args.sumocfg, banks, model_json, sc, pol, seed, args.horizon)
            for sc in scenarios(args.severity, args.horizon) for pol in POLICIES
            for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run_one, jobs))
    import pandas as pd

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)
    piv = df.groupby(["scenario", "policy"])["mean_travel_s"].mean().unstack()[list(POLICIES)]
    summary = {"mean_travel_s": piv.round(2).to_dict("index"),
               "worst_case_s": {p: float(piv[p].max()) for p in POLICIES},
               "fallback_frac": df[df.policy == "monitor"].groupby("scenario")["fallback_frac"]
               .mean().round(3).to_dict(), "seeds": args.seeds}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
