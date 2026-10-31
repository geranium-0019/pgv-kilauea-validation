# PGV Kīlauea Validation

Reproducible Phase Gradient Voting (PGV), eruption-timing aggregation, and GNSS sign validation for Capella Space interferograms of Kīlauea. This repository starts from geocoded wrapped phase and coherence; interferogram generation belongs to [capella-insar](https://github.com/geranium-0019/capella-insar).

The paper configuration contains 164 input pairs, 127 independently audited quality-eligible pairs, 37 pre-/co-eruptive frequency-map pairs, and 120 GNSS-valid pairs. The primary index is the mean null-corrected Backward vote minus the mean Forward vote within each track's prescribed region. No temporal standardization or displacement-amplitude cutoff is used.

## Install

Python 3.10 or later, on Linux, WSL, or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
pgv-kilauea doctor
```

Dependencies use rasterio for GeoTIFF I/O; GDAL Python bindings are not required. `requirements-repro.txt` records the versions used for the local numerical reproduction. Run commands below from the repository root; an external YAML can be selected with `--config`.

## Attach the existing local paper data

```bash
pgv-kilauea prepare-paper --source /path/to/graduation_thesis
pgv-kilauea validate --hashes
```

The default mode creates links under ignored `data/` to the existing images, DEM, five-minute GNSS archive, and reference PGV maps. It does not modify the source project. Use `--mode copy` for an independent local copy. These large inputs are not included in the Git repository. Small configuration/metadata, GNSS endpoint estimates, and independent numerical reference tables are included.

## Full reproduction

```bash
pgv-kilauea run
pgv-kilauea verify
```

`run` recalculates all 127 PGV maps, recomputes GNSS endpoint estimates from the raw five-minute archive, forms DI and frequency maps, and renders three figures. Existing PGV outputs are reused only when their input hashes and numerical settings match. No changes are made to the audited cohort based on the results. `verify` compares all available reference PGV maps, exact frequency counts, DI values, GNSS amplitudes, and the 103/120 primary sign agreement.

On environments with native numerical crashes, use the optional single-CPU mode **before the command**:

```bash
pgv-kilauea --safe-cpu run
pgv-kilauea --safe-cpu verify
```

This mode records CPU affinity, reduced NumPy SIMD features, thread settings, and dependency versions. It is an execution workaround, not a change to the scientific settings.

For Aptos figures, supply your installed font directory (font files are not redistributed):

```bash
pgv-kilauea figures --font Aptos --font-dir /path/to/Aptos
```

The default portable font is DejaVu Sans. Aptos is embedded in the exported PDFs when selected.

## Individual stages

```bash
pgv-kilauea classify
pgv-kilauea pgv --pair hawaii_20250421_20250424_v3d --geotiff
pgv-kilauea pgv
pgv-kilauea evaluate --recompute-endpoints
pgv-kilauea figures
pgv-kilauea verify
```

`evaluate` without `--recompute-endpoints` uses the bundled endpoint estimates, which permits repeating the spatial/GNSS evaluation without the raw GNSS archive. `run --cached-endpoints` provides the same shortcut explicitly. `--output /path/to/new-output` selects a separate experiment directory. `--overwrite` is available for the PGV/run commands; otherwise conflicting output provenance causes an error.

## Outputs

```text
output/
  pgv/                 Per-pair maps and checked provenance JSON
  geotiff/             Optional georeferenced Forward/Backward maps
  classification.csv   Timing classes independently checked against the catalog
  di.csv               Raw orbit-region directional means and DI
  frequency/           Threshold counts, denominators, and pair thresholds
  gnss/                Endpoint estimates, exclusions, amplitudes, and summary
  figures/             PDF/PNG frequency, GNSS, and processing-example figures
  verification.json    Numerical comparison with the paper reference
  run_status.json      Command settings, environment, progress, and errors
  pgv_status.json      Per-pair progress
```

See [scientific methods](docs/methods.md), [input format](docs/inputs.md), and [reproduction notes](docs/reproduction.md). Numerical function provenance is documented in [THIRD_PARTY.md](THIRD_PARTY.md). Historical temporal-normalization experiments, SAR preprocessing intermediates, and unrelated volcano/simulation workflows are not part of this package.

## Tests

```bash
python -m unittest discover -s tests -v
```
