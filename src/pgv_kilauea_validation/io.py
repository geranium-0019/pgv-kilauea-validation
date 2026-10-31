"""Configuration, checked input files, and atomic artifact writes."""

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import os
import yaml


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix="." + path.stem + "-", suffix=path.suffix, dir=path.parent
    )
    os.close(fd)
    temp = Path(name)
    try:
        yield temp
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as p:
        p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with atomic_path(path) as p:
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def as_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError("Invalid boolean: " + str(value))


@dataclass
class Config:
    file: Path
    root: Path
    values: dict
    output: Path
    data: Path

    @classmethod
    def load(cls, path, output=None):
        path = Path(path).resolve()
        v = yaml.safe_load(path.read_text())
        root = (path.parent / v.get("base_dir", ".")).resolve()

        def resolve(s):
            p = Path(s)
            return p if p.is_absolute() else root / p

        out = Path(output).resolve() if output else resolve(v["output_dir"]).resolve()
        data = resolve(v["data_dir"]).resolve()
        if out == root or out == data or out in data.parents or data in out.parents:
            raise ValueError("Output must be separate from project root and input data")
        pgv = v["pgv"]
        if not 0 <= pgv["coherence_threshold"] <= 1 or pgv["gradient_step_px"] < 1:
            raise ValueError("Invalid gradient settings")
        for key in [
            "radius_m",
            "vote_spacing_m",
            "extent_m",
            "ray_max_m",
            "ray_steps",
            "target_vote_points",
        ]:
            if pgv[key] <= 0:
                raise ValueError("Invalid positive setting: " + key)
        if not 0 <= pgv["ray_min_m"] < pgv["ray_max_m"] or pgv["smooth_sigma_m"] < 0:
            raise ValueError("Invalid ray/smoothing settings")
        if pgv["weight"] not in ["coherence_squared", "coherence", "uniform"]:
            raise ValueError("Unknown vote weighting")
        return cls(path, root, v, out, data)

    def path(self, key):
        p = Path(self.values[key])
        return p if p.is_absolute() else self.root / p

    def roi_path(self, track):
        p = Path(self.values["roi_files"][track])
        return p if p.is_absolute() else self.root / p

    def pairs(self, eligible_only=False):
        rows = read_csv(self.path("pairs_csv"))
        if len({r["pair_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate pair ID")
        for r in rows:
            if Path(r["pair_id"]).name != r["pair_id"]:
                raise ValueError("Unsafe pair ID")
            if r["track"] not in self.values["roi_files"]:
                raise ValueError("Unknown track")
            minimum = self.values.get("quality", {}).get("minimum_valid_fraction", 0.0)
            if as_bool(r["quality_eligible"]) and float(r["valid_fraction"]) <= minimum:
                raise ValueError(
                    "Eligible pair does not exceed minimum valid fraction: "
                    + r["pair_id"]
                )
        return (
            [r for r in rows if as_bool(r["quality_eligible"])]
            if eligible_only
            else rows
        )

    def input_path(self, row, key):
        p = Path(row[key + "_path"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("Input manifest paths must be relative to data_dir")
        return self.data / p
