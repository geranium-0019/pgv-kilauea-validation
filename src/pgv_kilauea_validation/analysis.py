"""Paper cohorts, directional frequency maps, raw DI, and reference checks."""

import json
import numpy as np

from .gnss import utc
from .io import as_bool, atomic_path, read_csv, write_csv, write_json
from .pgv import load_maps

CLASSES = ("pre-eruptive-24h", "single-eruption")


def classify_interval(first, last, episodes):
    if first >= last:
        raise ValueError("Invalid acquisition interval")
    episodes = sorted(episodes, key=lambda e: e[0])
    if any(a >= b for a, b in episodes) or any(
        episodes[i - 1][1] > episodes[i][0] for i in range(1, len(episodes))
    ):
        raise ValueError("Invalid or overlapping eruption catalog intervals")
    if not episodes:
        return "not-selected"
    overlaps = [(a, b) for a, b in episodes if min(last, b) > max(first, a)]
    covered = episodes[0][0] <= first and last <= episodes[-1][1]
    if not covered:
        return "not-selected"
    if len(overlaps) == 1:
        a, b = overlaps[0]
        if first < a and b < last:
            return "single-eruption"
    if not overlaps:
        previous = [(a, b) for a, b in episodes if b <= first]
        following = [(a, b) for a, b in episodes if a >= last]
        if (
            previous
            and following
            and 0 < (following[0][0] - last).total_seconds() <= 24 * 3600
        ):
            return "pre-eruptive-24h"
    return "not-selected"


def classification(config):
    episodes = [
        (utc(r["start_utc"]), utc(r["end_utc"]))
        for r in read_csv(config.path("eruptions_csv"))
    ]
    result = []
    for row in config.pairs():
        category = classify_interval(
            utc(row["master_acquisition_utc"]),
            utc(row["slave_acquisition_utc"]),
            episodes,
        )
        if category != row["analysis_class"]:
            raise ValueError(
                "Temporal class disagrees with manifest: " + row["pair_id"]
            )
        result.append(
            dict(
                pair_id=row["pair_id"],
                track=row["track"],
                analysis_class=category,
                quality_eligible=as_bool(row["quality_eligible"]),
            )
        )
    write_csv(config.output / "classification.csv", result)
    return result


def aggregate(config):
    rois = {}
    grids = {}
    counts = {}
    di = []
    thresholds = []
    x = y = None
    for track in config.values["roi_files"]:
        with np.load(config.roi_path(track), allow_pickle=False) as z:
            rois[track] = {k: z[k].copy() for k in ["roi", "x_m", "y_m"]}
        if not rois[track]["roi"].any():
            raise ValueError("Empty DI region")
    for row in config.pairs(eligible_only=True):
        maps = load_maps(config, row)
        roi = rois[row["track"]]
        if not (
            np.array_equal(maps["x_m"], roi["x_m"])
            and np.array_equal(maps["y_m"], roi["y_m"])
        ):
            raise ValueError("PGV and DI region grids differ")
        if x is None:
            x, y = maps["x_m"], maps["y_m"]
        if not np.array_equal(x, maps["x_m"]) or not np.array_equal(y, maps["y_m"]):
            raise ValueError("PGV grids differ between pairs")
        f, b = [float(maps[k][roi["roi"]].mean()) for k in ["fwd", "bwd"]]
        di.append(
            dict(
                pair_id=row["pair_id"],
                track=row["track"],
                analysis_class=row["analysis_class"],
                forward_mean=f,
                backward_mean=b,
                di=b - f,
                roi_cells=int(roi["roi"].sum()),
            )
        )
        cls = row["analysis_class"]
        track = row["track"]
        if cls not in CLASSES:
            continue
        counts[(cls, track)] = counts.get((cls, track), 0) + 1
        for direction in ["fwd", "bwd"]:
            key = f"{cls}__{track}__{direction}"
            a = maps[direction]
            threshold = config.values["pgv"]["occurrence_k_sigma"] * float(
                a.std(ddof=0)
            )
            grids.setdefault(key, np.zeros(a.shape, dtype=np.uint32))
            grids[key] += a > threshold
            thresholds.append(
                dict(
                    pair_id=row["pair_id"],
                    analysis_class=cls,
                    track=track,
                    direction=direction,
                    threshold=threshold,
                )
            )
    if x is None:
        raise ValueError("No eligible pairs")
    arrays = dict(x_m=x, y_m=y)
    for cls in CLASSES:
        for track in config.values["roi_files"]:
            n = counts.get((cls, track), 0)
            for direction in ["fwd", "bwd"]:
                key = f"{cls}__{track}__{direction}"
                a = grids.get(key, np.zeros((len(y), len(x)), dtype=np.uint32))
                arrays[key + "__count"] = a
                arrays[key + "__n"] = n
    with atomic_path(config.output / "frequency/maps.npz") as p:
        with p.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    write_csv(config.output / "di.csv", di)
    write_csv(config.output / "frequency/thresholds.csv", thresholds)
    summary = [
        dict(analysis_class=cls, track=track, n_pairs=counts.get((cls, track), 0))
        for cls in CLASSES
        for track in config.values["roi_files"]
    ]
    write_csv(config.output / "frequency/cohort.csv", summary)
    return di


def verify(config, compare_maps=True):
    ref = config.path("reference_dir")
    expected = json.loads((ref / "expected.json").read_text())
    tol = expected["absolute_tolerance"]
    pairs = config.pairs()
    eligible = config.pairs(eligible_only=True)
    if (
        len(pairs) != expected["input_pairs"]
        or len(eligible) != expected["eligible_pairs"]
    ):
        raise ValueError("Paper cohort size differs")
    with (
        np.load(config.output / "frequency/maps.npz") as actual,
        np.load(ref / "frequency.npz") as old,
    ):
        np.testing.assert_array_equal(actual["x_m"], old["x_m"])
        np.testing.assert_array_equal(actual["y_m"], old["y_m"])
        for key in actual.files:
            if key in ["x_m", "y_m"]:
                continue
            np.testing.assert_array_equal(actual[key], old["quality_37__" + key])
    classes = {c: sum(r["analysis_class"] == c for r in eligible) for c in CLASSES}
    if (
        classes[CLASSES[0]] != expected["frequency_pre"]
        or classes[CLASSES[1]] != expected["frequency_co"]
    ):
        raise ValueError("Frequency cohort differs")
    details = []
    for window in config.values["gnss"]["windows_minutes"]:
        actual = read_csv(config.output / "gnss" / f"pairs_{window}min.csv")
        old = {r["pair_id"]: r for r in read_csv(ref / f"gnss_{window}min.csv")}
        if {r["pair_id"] for r in actual} != set(old):
            raise ValueError("GNSS pair membership differs")
        fields = {
            "di": "di_raw_orbit_roi",
            "forward_mean": "forward_raw_orbit_roi",
            "backward_mean": "backward_raw_orbit_roi",
            "gnss_amplitude_mm": "gnss_source_amplitude_mm",
            "gnss_se_mm": "gnss_source_se_mm",
        }
        maximum = 0.0
        for row in actual:
            previous = old[row["pair_id"]]
            for key, oldkey in fields.items():
                err = abs(float(row[key]) - float(previous[oldkey]))
                maximum = max(maximum, err)
                if err > tol:
                    raise ValueError(
                        f"GNSS/DI reference mismatch: {row['pair_id']} {key} {err}"
                    )
            if as_bool(row["sign_match"]) != as_bool(
                previous["match_di_raw_orbit_roi"]
            ):
                raise ValueError("Sign classification differs")
        correct = sum(as_bool(r["sign_match"]) for r in actual)
        if correct != expected[f"gnss_{window}min_agreement"]:
            raise ValueError("Sign agreement count differs")
        details.append(
            dict(
                window_minutes=window,
                n=len(actual),
                agree=correct,
                max_absolute_error=maximum,
            )
        )
    mapdir = config.data / "reference_maps"
    map_error = 0.0
    n_maps = 0
    if compare_maps and mapdir.exists():
        for row in eligible:
            current = load_maps(config, row)
            with np.load(mapdir / (row["pair_id"] + ".npz")) as old:
                for direction in ["fwd", "bwd"]:
                    err = float(np.abs(current[direction] - old[direction]).max())
                    map_error = max(map_error, err)
                    np.testing.assert_allclose(
                        current[direction], old[direction], rtol=0, atol=tol
                    )
            n_maps += 1
    endpoint_file = config.output / "gnss/image_estimates.csv"
    endpoint_max = None
    if endpoint_file.exists():
        old = {
            (r["image_id"], r["window_minutes"]): r
            for r in read_csv(config.path("endpoint_estimates_csv"))
        }
        endpoint_max = 0.0
        recomputed = read_csv(endpoint_file)
        if {(r["image_id"], r["window_minutes"]) for r in recomputed} != set(old):
            raise ValueError("GNSS endpoint membership differs")
        for row in recomputed:
            prev = old[row["image_id"], row["window_minutes"]]
            if int(row["missing_count"]) != int(prev["missing_count"]):
                raise ValueError("Endpoint coverage changed")
            if int(row["missing_count"]):
                continue
            for c in ["east", "north", "up"]:
                for suffix in ["position_mm", "se_mm"]:
                    error = abs(
                        float(row[c + "_" + suffix]) - float(prev[c + "_" + suffix])
                    )
                    endpoint_max = max(endpoint_max, error)
                    if error > 1e-8:
                        raise ValueError("GNSS raw-data endpoint estimate differs")
    report = dict(
        state="verified",
        input_pairs=len(pairs),
        eligible_pairs=len(eligible),
        frequency_counts_exact=True,
        gnss=details,
        pgv_maps_compared=n_maps,
        max_map_absolute_error=map_error,
        raw_gnss_endpoint_max_error=endpoint_max,
    )
    write_json(config.output / "verification.json", report)
    return report
