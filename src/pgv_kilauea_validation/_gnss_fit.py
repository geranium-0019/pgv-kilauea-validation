"""Retained robust GNSS endpoint estimator; see THIRD_PARTY.md."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
COMPONENTS = ("east", "north", "up")
def iso_utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Settings:
    window_minutes: float = 30
    sensitivity_windows_minutes: tuple = (15, 30, 60)
    min_points: int = 6
    min_points_each_side: int = 2
    min_span_fraction: float = 0.7
    max_gap_seconds: float = 600
    max_nearest_seconds: float = 150
    huber_k: float = 1.345
    sigma_multiplier: float = 1.0
    step_bic_gain: float = 10

    def __post_init__(self):
        positive = (self.window_minutes, self.max_gap_seconds,
                    self.max_nearest_seconds, self.huber_k, self.step_bic_gain,
                    self.sigma_multiplier)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Window, gap, tolerance and fit parameters must be positive")
        if not 0 < self.min_span_fraction <= 1:
            raise ValueError("Invalid span fraction")
        if self.min_points < 6 or self.min_points_each_side < 2:
            raise ValueError("At least six points and two on each side are required")
        if not self.sensitivity_windows_minutes or any(
            not np.isfinite(v) or v <= 0 for v in self.sensitivity_windows_minutes
        ):
            raise ValueError("Sensitivity windows must be positive")


def _solve(x, y, weights):
    root_w = np.sqrt(weights)
    return np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0]


def robust_line(hours, values, sigmas, cfg):
    """IRLS local line; position at the target is the intercept."""
    x = np.column_stack((np.ones(len(hours)), hours))
    base_w = 1.0 / sigmas**2
    beta = _solve(x, values, base_w)
    for _ in range(50):
        z = (values - x @ beta) / sigmas
        scale = max(1.4826 * np.median(np.abs(z - np.median(z))), 1e-8)
        robust_w = np.minimum(1.0, cfg.huber_k * scale / np.maximum(np.abs(z), 1e-12))
        weights = base_w * robust_w
        updated = _solve(x, values, weights)
        if np.max(np.abs(updated - beta)) < 1e-7:
            beta = updated
            break
        beta = updated
    # Linear influence matrix with robust weights held fixed.
    influence = np.linalg.solve(x.T @ (weights[:, None] * x), x.T * weights)
    # Propagate the supplied sigmas, conditional on independent observations.
    se = float(np.sqrt(np.sum((influence[0] * sigmas)**2)))
    return x, beta, weights, robust_w, se


def step_diagnostic(hours, values, sigmas):
    """Heuristic BIC scan for a jump within a window; not an eruption label.

    Use the original observations, before robust downweighting, so that a real
    step is not silently treated as a collection of outliers. A single point
    cannot form its own segment (at least three points on either side).
    """
    n = len(hours)
    x = np.column_stack((np.ones(n), hours))
    w = 1 / sigmas**2
    beta = _solve(x, values, w)
    rss0 = max(float(np.sum(w * (values - x @ beta)**2)), 1e-16)
    best_gain, best_step = 0.0, 0.0
    for split in range(3, n - 2):
        design = np.column_stack((x, np.arange(n) >= split))
        b = _solve(design, values, w)
        rss = max(float(np.sum(w * (values - design @ b)**2)), 1e-16)
        gain = n * np.log(rss0 / rss) - np.log(n)
        residual_scale = 1.4826 * np.median(np.abs(values - design @ b))
        if gain > best_gain and abs(b[2]) > 3 * max(residual_scale, 1e-6):
            best_gain, best_step = float(gain), float(b[2])
    return best_gain, best_step


def estimate_position(observations, target, window_minutes, cfg):
    """Return a flat, CSV-ready record. Failed estimates retain their reason."""
    half = window_minutes * 60
    times = observations[:, 0]
    left = int(np.searchsorted(times, target - half, side="left"))
    right = int(np.searchsorted(times, target + half, side="right"))
    obs = observations[left:right]
    row = dict(window_minutes=window_minutes, status="ok", n_points=len(obs),
               n_before=int(np.sum(obs[:, 0] < target)),
               n_after=int(np.sum(obs[:, 0] > target)),
               first_utc=iso_utc(obs[0, 0]) if len(obs) else None,
               last_utc=iso_utc(obs[-1, 0]) if len(obs) else None,
               span_minutes=float(np.ptp(obs[:, 0]) / 60) if len(obs) else 0.0,
               nearest_offset_seconds=float(np.min(np.abs(obs[:, 0] - target))) if len(obs) else None,
               max_gap_seconds=float(np.max(np.diff(obs[:, 0]))) if len(obs) > 1 else None)
    for component in COMPONENTS:
        for field in ("position_mm", "slope_mm_hour", "se_mm",
                      "interval_low_mm", "interval_high_mm",
                      "residual_rms_mm", "downweighted_points", "effective_points",
                      "step_bic_gain", "step_mm"):
            row[f"{component}_{field}"] = None
    if len(obs) < cfg.min_points:
        row["status"] = "no_data" if not len(obs) else "insufficient_points"
    elif min(row["n_before"], row["n_after"]) < cfg.min_points_each_side:
        row["status"] = "one_sided"
    elif row["nearest_offset_seconds"] > cfg.max_nearest_seconds:
        row["status"] = "gap_at_acquisition"
    elif row["max_gap_seconds"] > cfg.max_gap_seconds:
        row["status"] = "gap_in_window"
    elif np.ptp(obs[:, 0]) < cfg.min_span_fraction * 2 * half:
        row["status"] = "short_span"
    if row["status"] != "ok":
        return row

    hours = (obs[:, 0] - target) / 3600
    for j, component in enumerate(COMPONENTS):
        values, sigmas = obs[:, j + 1], obs[:, j + 4]
        x, beta, weights, robust_w, se = robust_line(hours, values, sigmas, cfg)
        residual = values - x @ beta
        effective = float(weights.sum()**2 / np.sum(weights**2))
        gain, jump = step_diagnostic(hours, values, sigmas)
        values_out = dict(position_mm=float(beta[0]), slope_mm_hour=float(beta[1]),
                          se_mm=se,
                          interval_low_mm=float(beta[0] - cfg.sigma_multiplier * se),
                          interval_high_mm=float(beta[0] + cfg.sigma_multiplier * se),
                          residual_rms_mm=float(np.sqrt(np.mean(residual**2))),
                          downweighted_points=int(np.sum(robust_w < 0.5)),
                          effective_points=effective, step_bic_gain=gain, step_mm=jump)
        row.update({f"{component}_{k}": v for k, v in values_out.items()})
    if row["up_step_bic_gain"] > cfg.step_bic_gain:
        row["status"] = "possible_discontinuity"
    elif row["up_effective_points"] < 4:
        row["status"] = "low_effective_points"
    return row
