"""Retained numerical PGV functions; see THIRD_PARTY.md for provenance."""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def deg2m(ref_lat: float) -> tuple[float, float]:
    """Return (deg2m_lat, deg2m_lon) at a given latitude."""
    d2m_lat = 111_320.0
    d2m_lon = 111_320.0 * np.cos(np.radians(ref_lat))
    return d2m_lat, d2m_lon


def pixel_to_lonlat(gt: tuple, rows: np.ndarray, cols: np.ndarray):
    """Convert pixel (row, col) → (lon, lat) via geotransform."""
    lon = gt[0] + cols * gt[1] + rows * gt[2]
    lat = gt[3] + cols * gt[4] + rows * gt[5]
    return lon, lat


def build_caldera_mask(
    gt: tuple,
    h: int,
    w: int,
    ref_lat: float,
    ref_lon: float,
    volcano_radius_m: float = 3000.0,
) -> np.ndarray:
    """Build caldera mask: circular region within radius from reference point."""
    d2m_lat, d2m_lon = deg2m(ref_lat)

    rows, cols = np.mgrid[0:h, 0:w]
    lons, lats = pixel_to_lonlat(gt, rows, cols)
    dy = (lats - ref_lat) * d2m_lat
    dx = (lons - ref_lon) * d2m_lon
    dist = np.sqrt(dx**2 + dy**2)

    return dist <= volcano_radius_m


def compute_wrapped_gradient(
    phase: np.ndarray,
    coh: np.ndarray,
    coh_thresh: float,
    gt: tuple,
    d2m_lon: float,
    d2m_lat: float,
    step_px: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute wrapped phase gradient in geographic (east, north) directions.

    Central differences on complex exponential handle 2π wrapping.
    Uses Jacobian of the GeoTransform to convert pixel-space gradients
    to geographic (east, north) directions, handling image rotation.

    Returns (grad_east, grad_north, grad_dir) in [rad/m, rad/m, rad].
    """
    s = step_px
    h, w = phase.shape

    # Pixel-space gradients (wrapped, no spacing normalization for direction)
    # +col direction
    g_col = np.full((h, w), np.nan)
    g_col[:, s:-s] = (
        np.angle(np.exp(1j * (phase[:, 2 * s:] - phase[:, :-2 * s])))
        / (2 * s)
    )
    # +row direction
    g_row = np.full((h, w), np.nan)
    g_row[s:-s, :] = (
        np.angle(np.exp(1j * (phase[2 * s:, :] - phase[:-2 * s, :])))
        / (2 * s)
    )

    # Mask low coherence + NaN phase
    invalid = (coh < coh_thresh) | np.isnan(phase)
    g_col[invalid] = np.nan
    g_row[invalid] = np.nan

    # Jacobian: pixel (col, row) → geographic (east_m, north_m)
    a = gt[1] * d2m_lon   # dx/dcol
    b = gt[2] * d2m_lon   # dx/drow
    c = gt[4] * d2m_lat   # dy/dcol
    d = gt[5] * d2m_lat   # dy/drow
    det = a * d - b * c

    # Geographic gradient via J^{-T}: [g_east, g_north] = J^{-T} [g_col, g_row]
    grad_east = (d * g_col - c * g_row) / det
    grad_north = (-b * g_col + a * g_row) / det

    grad_dir = np.arctan2(grad_north, grad_east)
    return grad_east, grad_north, grad_dir


def do_vote_directional(
    x_m: np.ndarray,
    y_m: np.ndarray,
    dirs: np.ndarray,
    *,
    vote_sp: float,
    xn: float,
    xx: float,
    yn: float,
    yx: float,
    vnx: int,
    vny: int,
    ray_min: float,
    ray_max: float,
    n_ray_steps: int,
    ray_spread: float,
    smooth_sigma_m: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast vote rays separately for forward and backward directions.

    Parameters
    ----------
    weights : 1-D array, optional
        Per-pixel vote weight. If None, uniform weight (1.0) is used.

    Returns (vote_forward, vote_backward) both Gaussian-smoothed.
    """
    vm_fwd = np.zeros((vny, vnx))
    vm_bwd = np.zeros((vny, vnx))
    ray_dists = np.linspace(ray_min, ray_max, n_ray_steps)
    cos_d = np.cos(dirs)
    sin_d = np.sin(dirs)
    cos_p = -sin_d   # perpendicular
    sin_p = cos_d

    w_px = weights if weights is not None else np.ones(len(dirs))

    for d in ray_dists:
        width_m = d * ray_spread
        n_side = max(0, int(width_m / vote_sp))

        for sgn, vm in [(1.0, vm_fwd), (-1.0, vm_bwd)]:
            cx = x_m + sgn * cos_d * d
            cy = y_m + sgn * sin_d * d
            ci = ((cx - xn) / vote_sp).astype(int)
            ri = ((cy - yn) / vote_sp).astype(int)
            ok = (ci >= 0) & (ci < vnx) & (ri >= 0) & (ri < vny)
            np.add.at(vm, (ri[ok], ci[ok]), w_px[ok])

            for s in range(1, n_side + 1):
                for ps in (1.0, -1.0):
                    ci_s = ((cx + ps * s * vote_sp * cos_p - xn)
                            / vote_sp).astype(int)
                    ri_s = ((cy + ps * s * vote_sp * sin_p - yn)
                            / vote_sp).astype(int)
                    ok_s = ((ci_s >= 0) & (ci_s < vnx)
                            & (ri_s >= 0) & (ri_s < vny))
                    side_w = 1.0 - s / (n_side + 1)
                    np.add.at(vm, (ri_s[ok_s], ci_s[ok_s]), w_px[ok_s] * side_w)

    sigma = smooth_sigma_m / vote_sp
    return (ndimage.gaussian_filter(vm_fwd, sigma=sigma),
            ndimage.gaussian_filter(vm_bwd, sigma=sigma))
