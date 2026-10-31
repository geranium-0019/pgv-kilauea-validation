"""Five-minute GNSS coverage, robust endpoints, and source-direction WLS."""

from datetime import datetime, timedelta, timezone
import gzip
import json
import numpy as np

from ._gnss_fit import Settings, estimate_position
from .io import read_csv, write_csv, write_json

COMPONENTS = ("east", "north", "up")
EPOCH = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


def utc(value):
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("Acquisition time must include UTC offset")
    return d.astimezone(timezone.utc)


def expected_times(start, end, offset=18):
    offset = timedelta(seconds=offset)
    step = timedelta(seconds=300)
    first = EPOCH + ((start + offset - EPOCH) // step) * step - offset
    if first < start:
        first += step
    result = []
    while first <= end:
        result.append(first.timestamp())
        first += step
    return np.array(result, dtype=float)


def load_observations(config):
    g = config.values["gnss"]
    year = g["gps_calendar_year"]
    offset = g["gps_minus_utc_seconds"]
    station = g.get("station", "BYRL")
    arrays = []
    invalid = 0
    files = sorted(config.path("gnss_directory").glob(f"{station}.{year}.*.kenv.gz"))
    required = [
        "site",
        "sec-J2000",
        "year",
        "mm",
        "dd",
        "s-day",
        "___e-ref(m)",
        "___n-ref(m)",
        "___v-ref(m)",
        "sig_e(m)",
        "sig_n(m)",
        "sig_v(m)",
    ]
    low = datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()
    high = datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp()
    for path in files:
        with gzip.open(path, "rt") as f:
            header = next(f).split()
            cols = [header.index(k) for k in required]
            raw = np.loadtxt(f, dtype=str, usecols=cols, ndmin=2)
        if not len(raw):
            continue
        if np.any(raw[:, 0] != station):
            raise ValueError("Unexpected GNSS station")
        values = raw[:, 1:].astype(float)
        gps = EPOCH.timestamp() + values[:, 0]
        if np.any((gps < low) | (gps >= high)):
            raise ValueError(
                "GNSS fixed UTC offset only applies to configured GPS calendar year"
            )
        y, m, d = values[0, 1:4].astype(int)
        if not np.all(values[:, 1:4] == (y, m, d)):
            raise ValueError("Multiple dates in GNSS daily file")
        expected = (
            datetime(y, m, d, tzinfo=timezone.utc) - EPOCH
        ).total_seconds() + values[:, 4]
        if not np.allclose(values[:, 0], expected, rtol=0, atol=1e-6):
            raise ValueError("Inconsistent GNSS timestamps")
        enu = values[:, 5:11] * 1000
        valid = np.isfinite(enu).all(axis=1) & (enu[:, 3:] > 0).all(axis=1)
        invalid += int((~valid).sum())
        arrays.append(np.column_stack((gps[valid] - offset, enu[valid])))
    if not arrays:
        raise FileNotFoundError(
            "No GNSS daily files: " + str(config.path("gnss_directory"))
        )
    obs = np.concatenate(arrays)
    obs = obs[np.argsort(obs[:, 0])]
    if np.any(np.diff(obs[:, 0]) <= 0):
        raise ValueError("Duplicate GNSS timestamps")
    return obs, dict(
        files=len(files), valid_observations=len(obs), invalid_observations=invalid
    )


def endpoints(config):
    obs, report = load_observations(config)
    settings = Settings(**config.values["gnss"]["estimator"])
    rows = []
    for image in read_csv(config.path("acquisitions_csv")):
        target = utc(image["acquisition_utc"])
        for window in config.values["gnss"]["windows_minutes"]:
            dt = timedelta(minutes=window)
            expected = expected_times(
                target - dt, target + dt, config.values["gnss"]["gps_minus_utc_seconds"]
            )
            actual = obs[
                (obs[:, 0] >= target.timestamp() - window * 60)
                & (obs[:, 0] <= target.timestamp() + window * 60),
                0,
            ]
            missing = np.setdiff1d(expected, actual)
            if len(missing):
                row = dict(
                    status="missing_window",
                    n_points=len(actual),
                    **{
                        c + "_" + suffix: None
                        for c in COMPONENTS
                        for suffix in ["position_mm", "se_mm"]
                    },
                )
            else:
                row = estimate_position(obs, target.timestamp(), window, settings)
            row.update(
                image_id=image["image_id"],
                acquisition_utc=image["acquisition_utc"],
                window_minutes=window,
                expected_count=len(expected),
                missing_count=len(missing),
            )
            rows.append(row)
    write_csv(config.output / "gnss/image_estimates.csv", rows)
    write_json(
        config.output / "gnss/coverage.json",
        dict(**report, endpoints=len(rows), interpolated=False),
    )
    return rows


def fit_amplitude(displacement, variance, direction):
    d = np.asarray(displacement, dtype=float)
    v = np.asarray(variance, dtype=float)
    g = np.asarray(direction, dtype=float)
    if (
        d.shape != (3,)
        or v.shape != (3,)
        or g.shape != (3,)
        or not np.isfinite([d, v, g]).all()
        or np.any(v <= 0)
    ):
        raise ValueError("Expected three finite components and positive variances")
    if not np.isclose(np.linalg.norm(g), 1, atol=1e-10):
        raise ValueError("Source response must be a unit direction")
    information = float(np.sum(g * g / v))
    if information <= 0:
        raise ValueError("No information for source-direction fit")
    weights = g / v / information
    return float(weights @ d), float(1 / np.sqrt(information))


def compare(config, di_rows, recompute_endpoints=False):
    estimates = (
        endpoints(config)
        if recompute_endpoints
        else read_csv(config.path("endpoint_estimates_csv"))
    )
    lookup = {(r["image_id"], int(r["window_minutes"])): r for r in estimates}
    if len(lookup) != len(estimates):
        raise ValueError("Duplicate GNSS endpoint estimate")
    model = json.loads(config.path("source_model_json").read_text())
    depth = str(model["depth_m"])
    g = np.asarray(model["unit_model_vectors"][depth])
    di = {r["pair_id"]: r for r in di_rows}
    summaries = []
    for window in config.values["gnss"]["windows_minutes"]:
        records = []
        exclusions = []
        for row in config.pairs(eligible_only=True):
            endpoints_window = [
                lookup[(row[r + "_image_id"], window)] for r in ["master", "slave"]
            ]
            check30 = [lookup[(row[r + "_image_id"], 30)] for r in ["master", "slave"]]
            check = (
                endpoints_window + check30
                if config.values["gnss"]["require_complete_30min_windows"]
                else endpoints_window
            )
            if any(int(e["missing_count"]) for e in check):
                exclusions.append(
                    dict(
                        pair_id=row["pair_id"],
                        window_minutes=window,
                        reason="missing_GNSS_window",
                    )
                )
                continue
            delta = np.array(
                [
                    float(endpoints_window[1][c + "_position_mm"])
                    - float(endpoints_window[0][c + "_position_mm"])
                    for c in COMPONENTS
                ]
            )
            variance = np.array(
                [
                    sum(float(e[c + "_se_mm"]) ** 2 for e in endpoints_window)
                    for c in COMPONENTS
                ]
            )
            amplitude, se = fit_amplitude(delta, variance, g)
            index = float(di[row["pair_id"]]["di"])
            records.append(
                dict(
                    pair_id=row["pair_id"],
                    track=row["track"],
                    analysis_class=row["analysis_class"],
                    window_minutes=window,
                    di=index,
                    forward_mean=di[row["pair_id"]]["forward_mean"],
                    backward_mean=di[row["pair_id"]]["backward_mean"],
                    gnss_amplitude_mm=amplitude,
                    gnss_se_mm=se,
                    sign_match=bool(index * amplitude > 0),
                )
            )
        records.sort(key=lambda r: r["pair_id"])
        write_csv(config.output / "gnss" / f"pairs_{window}min.csv", records)
        write_csv(
            config.output / "gnss" / f"excluded_{window}min.csv",
            exclusions,
            fields=["pair_id", "window_minutes", "reason"],
        )
        k = sum(r["sign_match"] for r in records)
        n = len(records)
        summaries.append(
            dict(
                window_minutes=window,
                n_pairs=n,
                n_agree=k,
                agreement_fraction=k / n if n else None,
            )
        )
    write_csv(config.output / "gnss/summary.csv", summaries)
    write_json(config.output / "gnss/model.json", model)
    return summaries
