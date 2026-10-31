"""Publication figures from verified PGV and GNSS output artifacts."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter, map_coordinates

from .io import as_bool, digest, read_csv, write_json
from .pgv import core, compute, load_maps
from .analysis import CLASSES

HALO = [patheffects.withStroke(linewidth=1.5, foreground="white")]
STYLE = [
    ("not-selected", "Other timing", "#858585", "o"),
    ("single-eruption", "Co-eruptive", "#D55E00", "o"),
    ("pre-eruptive-24h", "Pre-eruptive", "#0072B2", "^"),
]


def setup(config, font=None, font_dir=None):
    if font_dir:
        files = sorted(Path(font_dir).glob("*.ttf")) + sorted(
            Path(font_dir).glob("*.otf")
        )
        if not files:
            raise FileNotFoundError("No fonts in " + str(font_dir))
        for p in files:
            font_manager.fontManager.addfont(str(p))
    family = font or config.values["figures"]["font"]
    try:
        path = font_manager.findfont(
            font_manager.FontProperties(family=family), fallback_to_default=False
        )
    except ValueError as exc:
        raise ValueError(
            'Font unavailable. Supply --font-dir or choose --font "DejaVu Sans".'
        ) from exc
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.pad": 1,
            "ytick.major.pad": 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return dict(family=family, file=path, sha256=digest(path))


def terrain(config, x, y):
    ref = config.values["reference"]
    dlat, dlon = core.deg2m(ref["lat"])
    step = x[1] - x[0]
    transform = from_origin(
        ref["lon"] + (x[0] - step / 2) / dlon,
        ref["lat"] + (y[-1] + step / 2) / dlat,
        step / dlon,
        step / dlat,
    )
    with rasterio.open(config.path("dem")) as ds:
        if (
            ds.shape == (len(y), len(x))
            and ds.crs.to_epsg() == 4326
            and np.allclose(tuple(ds.transform), tuple(transform), rtol=0, atol=1e-12)
        ):
            z = ds.read(1).astype(float)
        else:
            z = np.full((len(y), len(x)), np.nan)
            reproject(
                rasterio.band(ds, 1),
                z,
                src_transform=ds.transform,
                src_crs=ds.crs,
                src_nodata=ds.nodata,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    if not np.isfinite(z).all():
        raise ValueError("DEM does not cover voting grid")
    shade = 0.3 + 0.68 * LightSource(315, 40).hillshade(
        z, vert_exag=1.5, dx=step, dy=step
    )
    rings = []
    for feature in json.loads(config.path("hmmr_geojson").read_text())["features"]:
        g = feature["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            p = np.asarray(poly[0])
            rings.append(
                (
                    (p[:, 0] - ref["lon"]) * dlon / 1000,
                    (p[:, 1] - ref["lat"]) * dlat / 1000,
                )
            )
    extent = (
        (x[0] - step / 2) / 1000,
        (x[-1] + step / 2) / 1000,
        (y[0] - step / 2) / 1000,
        (y[-1] + step / 2) / 1000,
    )
    return shade[::-1], rings, extent


def save(config, fig, name):
    folder = config.output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(
            folder / (name + "." + ext),
            dpi=config.values["figures"]["dpi"],
            facecolor="white",
        )
    plt.close(fig)


def frequency(config):
    with np.load(config.output / "frequency/maps.npz") as z:
        maps = {k: z[k].copy() for k in z.files}
    x, y = maps["x_m"], maps["y_m"]
    shade, rings, extent = terrain(config, x, y)
    rois = {}
    for t in ["ASC", "DESC"]:
        with np.load(config.roi_path(t)) as z:
            rois[t] = z["roi"].copy()
    model = json.loads(config.path("source_model_json").read_text())
    station = model["station"]
    ref = config.values["reference"]
    dlat, dlon = core.deg2m(ref["lat"])
    bx = (station["lon"] - ref["lon"]) * dlon / 1000
    by = (station["lat"] - ref["lat"]) * dlat / 1000
    width, height = 90, 158
    panel = 29.5
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad((0, 0, 0, 0))
    norm = Normalize(0, 100)
    for i, cls in enumerate(CLASSES):
        for j, track in enumerate(["ASC", "DESC"]):
            row = i * 2 + j
            bottom = 119.5 - row * 31.5
            for k, d in enumerate(["fwd", "bwd"]):
                ax = fig.add_axes(
                    [
                        (15 + k * 36) / width,
                        bottom / height,
                        panel / width,
                        panel / height,
                    ]
                )
                key = f"{cls}__{track}__{d}"
                n = int(maps[key + "__n"])
                a = maps[key + "__count"]
                ax.imshow(
                    shade,
                    origin="lower",
                    extent=extent,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="bilinear",
                    rasterized=True,
                )
                if n:
                    ax.imshow(
                        np.ma.masked_equal(a / n * 100, 0),
                        origin="lower",
                        extent=extent,
                        cmap=cmap,
                        norm=norm,
                        alpha=0.82,
                        interpolation="nearest",
                        rasterized=True,
                    )
                for rx, ry in rings:
                    ax.plot(rx, ry, c="black", lw=0.6, path_effects=HALO)
                ax.contour(
                    x / 1000,
                    y / 1000,
                    rois[track].astype(float),
                    [0.5],
                    colors=["#0077BB"],
                    linewidths=0.6,
                    linestyles="--",
                )
                ax.plot(
                    bx,
                    by,
                    "^",
                    color="#009E73",
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    markersize=3.6,
                    zorder=10,
                )
                if row == 0 and k == 0:
                    ax.annotate(
                        "BYRL",
                        (bx, by),
                        xytext=(-2, 5),
                        textcoords="offset points",
                        ha="right",
                        fontsize=6.5,
                        color="#006B4F",
                        zorder=11,
                        path_effects=HALO,
                    )
                    ax.plot(
                        [-2.7, -1.7], [-2.8, -2.8], c="black", lw=0.8, path_effects=HALO
                    )
                    ax.text(
                        -2.2, -2.4, "1 km", ha="center", fontsize=5.7, path_effects=HALO
                    )
                ax.text(
                    0.025,
                    0.97,
                    f"({chr(97 + i * 4 + j * 2 + k)})",
                    transform=ax.transAxes,
                    va="top",
                    fontweight="bold",
                    fontsize=7,
                    path_effects=HALO,
                )
                ax.text(
                    0.97,
                    0.97,
                    f"n = {n}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=6.5,
                    path_effects=HALO,
                )
                ax.set(
                    xlim=extent[:2],
                    ylim=extent[2:],
                    xticks=[-3, 0, 3],
                    yticks=[-3, 0, 3],
                    aspect="equal",
                )
                if k:
                    ax.tick_params(labelleft=False)
                if row < 3:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("East (km)", labelpad=1)
                if row == 0:
                    ax.set_title(
                        "Forward\n(deflation)" if k == 0 else "Backward\n(inflation)",
                        fontsize=7.5,
                        pad=4,
                    )
            text = "Pre-eruptive (≤24 h)" if i == 0 else "Co-eruptive"
            fig.text(
                2.2 / width,
                (bottom + panel / 2) / height,
                text + " | " + track,
                ha="center",
                va="center",
                rotation=90,
                fontsize=7,
                fontweight="bold",
            )
    fig.text(
        8 / width,
        87 / height,
        "North (km)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7,
    )
    handles = [
        Line2D([], [], c="black", lw=0.7),
        Line2D([], [], c="#0077BB", lw=0.7, ls="--"),
    ]
    fig.legend(
        handles,
        ["HMMR reference area", "Geometry-predicted PGV region"],
        loc="center",
        bbox_to_anchor=(0.53, 13.5 / height),
        ncol=1,
        fontsize=6.5,
        frameon=False,
        handlelength=1.9,
        labelspacing=0.25,
        borderaxespad=0,
    )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=fig.add_axes([15 / width, 6.5 / height, 65.5 / width, 2 / height]),
        orientation="horizontal",
        ticks=[0, 20, 40, 60, 80, 100],
        alpha=0.82,
    )
    cb.ax.tick_params(length=2, pad=1, labelsize=6.5)
    cb.set_label("PGV threshold-exceedance frequency (%)", fontsize=7, labelpad=1.5)
    save(config, fig, "eruption_frequency")


def gnss(config):
    window = config.values["gnss"]["primary_window_minutes"]
    rows = read_csv(config.output / "gnss" / f"pairs_{window}min.csv")
    if not rows:
        raise ValueError("No GNSS-valid pairs")
    x = np.array([float(r["di"]) for r in rows])
    y = np.array([float(r["gnss_amplitude_mm"]) for r in rows])
    se = np.array([float(r["gnss_se_mm"]) for r in rows])
    fig, ax = plt.subplots(figsize=(90 / 25.4, 80 / 25.4))
    fig.subplots_adjust(left=0.195, right=0.985, bottom=0.195, top=0.975)
    dx = np.ptp(x) or 1.0
    dy = np.max(y + se) - np.min(y - se) or 1.0
    xmin, xmax = x.min() - 0.07 * dx, x.max() + 0.07 * dx
    ymin, ymax = np.min(y - se) - 0.06 * dy, np.max(y + se) + 0.06 * dy
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_axisbelow(True)
    if xmax > 0 and ymax > 0:
        ax.add_patch(
            Rectangle(
                (0, 0), xmax, ymax, facecolor="#E8F3E9", edgecolor="none", zorder=-5
            )
        )
    if xmin < 0 and ymin < 0:
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                -xmin,
                -ymin,
                facecolor="#E8F3E9",
                edgecolor="none",
                zorder=-5,
            )
        )
    ax.grid(color=".89", lw=0.4)
    ax.axhline(0, c=".35", lw=0.7)
    ax.axvline(0, c=".35", lw=0.7)
    counts = {}
    for j, (cls, text, c, marker) in enumerate(STYLE):
        ix = np.array([r["analysis_class"] == cls for r in rows])
        counts[cls] = int(ix.sum())
        ax.errorbar(
            x[ix],
            y[ix],
            yerr=se[ix],
            fmt="none",
            ecolor=c,
            alpha=0.35,
            elinewidth=0.5,
            capsize=0,
            zorder=2 + 2 * j,
        )
        ax.scatter(
            x[ix],
            y[ix],
            c=c,
            marker=marker,
            s=16 if marker == "o" else 22,
            edgecolor="white",
            linewidth=0.3,
            zorder=3 + 2 * j,
        )
    correct = sum(as_bool(r["sign_match"]) for r in rows)
    n = len(rows)
    ax.text(
        0.035,
        0.965,
        f"Sign agreement\n{correct}/{n} ({correct / n:.1%})",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=2),
    )
    ax.set_xlabel("DI", labelpad=2.5)
    ax.set_ylabel("GNSS model-aligned displacement (mm)", labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    handles = [
        Line2D(
            [],
            [],
            marker=m,
            color=c,
            ls="",
            markersize=4.3,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label=f"{text} ({counts[cls]})",
        )
        for cls, text, c, m in reversed(STYLE)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.055),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=0.85,
        handletextpad=0.25,
        columnspacing=0.7,
        borderaxespad=0,
    )
    save(config, fig, "gnss_agreement")


def method(config, pair=None):
    pair = pair or config.values["figures"]["method_pair"]
    rows = {r["pair_id"]: r for r in config.pairs(eligible_only=True)}
    if pair not in rows:
        raise ValueError("Method figure must use an eligible pair")
    obs = compute(config, rows[pair], raw=True)
    p = config.values["pgv"]
    ref = config.values["reference"]
    smooth = np.array(
        [
            gaussian_filter(a, p["smooth_sigma_m"] / p["vote_spacing_m"])
            for a in obs["result"]
        ]
    )
    saved = load_maps(config, rows[pair])
    for i, k in enumerate(["fwd", "bwd"]):
        np.testing.assert_allclose(smooth[i], saved[k], rtol=0, atol=1e-10)
    x, y = obs["x_m"], obs["y_m"]
    shade, _, extent = terrain(config, x, y)
    fig, axes = plt.subplots(1, 5, figsize=(180 / 25.4, 52 / 25.4))
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.26, top=0.88, wspace=0.12)
    for i, (ax, title) in enumerate(
        zip(
            axes,
            [
                "(a) Input",
                "(b) Vote map",
                "(c) Null map",
                "(d) Difference",
                "(e) Result",
            ],
        )
    ):
        ax.imshow(
            shade,
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="bilinear",
            rasterized=True,
        )
        ax.set(
            xlim=extent[:2],
            ylim=extent[2:],
            xticks=[-3, 0, 3],
            yticks=[-3, 0, 3],
            aspect="equal",
        )
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.set_xlabel("East (km)", fontsize=6.5, labelpad=1)
        if i == 0:
            ax.set_ylabel("North (km)", fontsize=6.5, labelpad=1)
        else:
            ax.tick_params(labelleft=False)
    phase, coh, gt = obs["phase"], obs["coherence"], obs["gt"]
    h, w = phase.shape
    dlat, dlon = core.deg2m(ref["lat"])
    iext = (
        (gt[0] - ref["lon"]) * dlon / 1000,
        (gt[0] + w * gt[1] - ref["lon"]) * dlon / 1000,
        (gt[3] + h * gt[5] - ref["lat"]) * dlat / 1000,
        (gt[3] - ref["lat"]) * dlat / 1000,
    )
    finite = np.isfinite(phase) & np.isfinite(coh) & (coh >= 0) & (coh <= 1)
    hue = np.mod((np.where(finite, phase, 0) + np.pi) / (2 * np.pi), 1.0)
    rr, cc = np.indices(phase.shape, dtype=float)
    east = (gt[0] + (cc + 0.5) * gt[1] + (rr + 0.5) * gt[2] - ref["lon"]) * dlon
    north = (gt[3] + (cc + 0.5) * gt[4] + (rr + 0.5) * gt[5] - ref["lat"]) * dlat
    hillshade = map_coordinates(
        shade,
        [(north - y[0]) / (y[1] - y[0]), (east - x[0]) / (x[1] - x[0])],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    hsv = np.stack([hue, np.sqrt(np.where(finite, coh, 0)), hillshade], axis=-1)
    rgba = np.concatenate(
        [matplotlib.colors.hsv_to_rgb(hsv), finite[:, :, None].astype(float)], axis=-1
    )
    axes[0].imshow(
        rgba, origin="upper", extent=iext, interpolation="nearest", rasterized=True
    )
    legend = matplotlib.colors.ListedColormap(
        matplotlib.colors.hsv_to_rgb(
            np.column_stack([np.linspace(0, 1, 256), np.ones(256), np.ones(256)])
        )
    )
    ims = [plt.cm.ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap=legend)]
    actual = obs["actual"].sum(axis=0)
    null = obs["null"].sum(axis=0)
    limit = float(np.percentile(np.concatenate([actual.ravel(), null.ravel()]), 99))
    for ax, a in zip(axes[1:3], [actual, null]):
        ims.append(
            ax.imshow(
                a,
                origin="lower",
                extent=extent,
                cmap="inferno",
                vmin=0,
                vmax=limit,
                interpolation="nearest",
                rasterized=True,
            )
        )
    difference = actual - null
    dlim = float(np.percentile(np.abs(difference), 97)) or 1.0
    ims.append(
        axes[3].imshow(
            difference,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-dlim,
            vmax=dlim,
            interpolation="nearest",
            rasterized=True,
        )
    )
    positive = np.maximum(smooth, 0)
    rgba = np.zeros((*positive.shape[1:], 4))
    fwd = positive[0] >= positive[1]
    rgba[fwd, :3] = matplotlib.colors.to_rgb("#3366CC")
    rgba[~fwd, :3] = matplotlib.colors.to_rgb("#EE8833")
    rgba[:, :, 3] = 0.85 * positive.max(axis=0) / (float(positive.max()) or 1.0)
    axes[4].imshow(
        rgba, origin="lower", extent=extent, interpolation="nearest", rasterized=True
    )
    for i, im in enumerate(ims):
        pos = axes[i].get_position()
        cb = fig.colorbar(
            im,
            cax=fig.add_axes([pos.x0, 0.135, pos.width, 0.025]),
            orientation="horizontal",
            extend="neither" if i == 0 else "both" if i == 3 else "max",
        )
        cb.ax.tick_params(length=1.5, pad=1, labelsize=5.5)
        if i == 0:
            cb.set_ticks([-np.pi, 0, np.pi])
            cb.set_ticklabels(["−π", "0", "π"])
            cb.set_label("Phase (rad)", fontsize=5.7, labelpad=1)
        else:
            cb.ax.xaxis.set_major_locator(MaxNLocator(3))
            cb.set_label(
                "Weighted votes" if i < 3 else "Vote difference",
                fontsize=5.7,
                labelpad=1,
            )
    pos = axes[4].get_position()
    fig.legend(
        handles=[
            Patch(color="#3366CC", label="Forward"),
            Patch(color="#EE8833", label="Backward"),
        ],
        loc="center",
        bbox_to_anchor=((pos.x0 + pos.x1) / 2, 0.125),
        fontsize=5.8,
        ncol=1,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.2,
        frameon=False,
    )
    save(config, fig, "pgv_processing")
    write_json(
        config.output / "figures/method.json",
        dict(
            pair_id=pair,
            PGV_reproduced=True,
            input_display="HSV: phase hue, sqrt(coherence) saturation, terrain hillshade value",
        ),
    )


def render(config, font=None, font_dir=None, pair=None):
    font_info = setup(config, font, font_dir)
    frequency(config)
    gnss(config)
    method(config, pair)
    folder = config.output / "figures"
    write_json(
        folder / "manifest.json",
        dict(
            font=font_info,
            config_sha256=digest(config.file),
            output_sha256={
                p.name: digest(p)
                for p in folder.glob("*")
                if p.suffix in [".pdf", ".png"]
            },
            frequency="Directional vote > 2 spatial SD; descriptive, not a calibrated significance level",
            gnss="First/third quadrant shading denotes sign agreement; errors are formal SE",
        ),
    )
