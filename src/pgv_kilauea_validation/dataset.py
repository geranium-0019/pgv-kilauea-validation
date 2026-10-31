"""Attach external rasters without modifying or committing the input dataset."""

import shutil
from pathlib import Path
from .io import as_bool, digest, write_json


def attach(config, source, mode="symlink"):
    source = Path(source).resolve()
    links = {
        "images": source / "data/kirauea/images",
        "dem.tif": source / "data/kirauea/dem/kilauea_2019_pgv_grid.tif",
        "gnss": source / "data/kirauea/gnss/BYRL.2025.kenv",
        "reference_maps": source / "val_kirauea/raw/pgv",
    }
    missing = [str(p) for p in links.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source inputs: " + ", ".join(missing))
    config.data.mkdir(parents=True, exist_ok=True)
    for name, src in links.items():
        dst = config.data / name
        if dst.exists() or dst.is_symlink():
            if dst.resolve() == src.resolve():
                continue
            raise FileExistsError(
                "Input destination exists; refusing to replace: " + str(dst)
            )
        if mode == "symlink":
            dst.symlink_to(src, target_is_directory=src.is_dir())
        elif src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    record = dict(
        source=str(source), mode=mode, links={k: str(v) for k, v in links.items()}
    )
    write_json(config.data / "import.json", record)
    return record


def validate(config, hashes=False):
    rows = config.pairs()
    checked = []
    missing = []
    for row in rows:
        for key in ["phase", "coherence"]:
            p = config.input_path(row, key)
            if not p.is_file():
                missing.append(str(p))
                continue
            if hashes and digest(p) != row[key + "_sha256"]:
                raise ValueError("Input hash mismatch: " + str(p))
        checked.append(
            dict(pair_id=row["pair_id"], eligible=as_bool(row["quality_eligible"]))
        )
    if missing:
        raise FileNotFoundError("Input files missing: " + ", ".join(missing[:5]))
    for name in [
        "source_model_json",
        "endpoint_estimates_csv",
        "acquisitions_csv",
        "eruptions_csv",
        "dem",
        "hmmr_geojson",
    ]:
        if not config.path(name).is_file():
            raise FileNotFoundError(config.path(name))
    for t in config.values["roi_files"]:
        if not config.roi_path(t).is_file():
            raise FileNotFoundError(config.roi_path(t))
    return dict(
        input_pairs=len(rows),
        eligible_pairs=sum(r["eligible"] for r in checked),
        hashes_verified=hashes,
    )
