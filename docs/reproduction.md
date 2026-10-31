# Reproduction and operation

The [local acceptance record](reproduction_result.json) records the 2026-10-31 full reproduction: all 127 original PGV maps, both GNSS-window comparisons, and fresh raw-data endpoint estimates matched with maximum numerical difference zero. Eighteen unit tests passed. Figures were rendered using locally installed Aptos.

## Environment

Create an isolated virtual environment; do not inherit an existing Conda environment's site-packages. Rasterio wheels provide their required geospatial runtime. A broken external PROJ installation can prevent CRS identification even when a GeoTIFF is valid.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install -e .
pgv-kilauea doctor
```

The version pins document the tested Python 3.12 environment. Other supported Python versions may require compatible dependency versions instead. The base package declares broader supported versions; numerical verification remains the acceptance check.

## End-to-end run

```bash
pgv-kilauea prepare-paper --source /path/to/graduation_thesis
pgv-kilauea validate --hashes
pgv-kilauea run
pgv-kilauea verify
```

For a single diagnostic pair:

```bash
pgv-kilauea pgv --pair hawaii_20250421_20250424_v3d --geotiff
```

This does not produce the complete cohort figures; run the full `pgv` stage before `evaluate` or `figures`. The method figure independently recalculates that pair's raw vote and null maps and checks its smoothed result against the saved PGV output.

## Resume and experiments

Progress is written to `output/pgv_status.json` and printed as `[completed/total] pair_id`. Completed map artifacts contain a SHA256 checksum and a numerical signature including input hashes, seed, configuration, and computation source hashes. A matching artifact is reused; an altered or conflicting artifact raises an error. `validate --hashes` explicitly rechecks current source images, including those with cached outputs.

Ctrl+C stops processing. A subsequent identical command resumes completed pairs. Output writes use temporary files and atomic replacement. Concurrent commands targeting the same output directory are prevented by a file lock. These are foreground commands; for a long unattended run, use a terminal multiplexer such as tmux.

Choose a separate experiment output directory before changing settings:

```bash
pgv-kilauea --config configs/kilauea.yaml --output output/alternative pgv
```

`--overwrite` replaces PGV outputs explicitly. Verification is against the paper configuration; alternative weights or thresholds need not reproduce paper reference values. Changing valid-observation definitions additionally requires updating the input audit, as described in the input contract.

## Verification

`verify` checks:

- Input/cohort counts and the 14/23 Pre-/Co-eruptive counts.
- Exact directional threshold-exceedance counts on every frequency-map pixel.
- GNSS pair membership and sign classifications for both observation windows.
- DI, Forward/Backward regional means, source-direction GNSS amplitudes, and formal uncertainties against independent stored reference tables, with absolute tolerance 1e-10.
- Full Forward/Backward arrays of all 127 original PGV maps, when attached, with absolute tolerance 1e-10.
- Fresh GNSS endpoint coverage and estimates against the frozen endpoint table, when recomputed, with absolute tolerance 1e-8 mm.

`output/verification.json` records which comparisons were performed and their largest numerical differences. The reference tables are independent baseline outputs, not values generated from the new package during its tests. `reference/numerical_provenance.json` records extraction sources.

## Figures and fonts

PDF and PNG exports include eruption-frequency maps, DI/GNSS sign comparison, and an actual/null/processed-map example. Default typography is DejaVu Sans. Use a locally installed Aptos directory to match the paper:

```bash
pgv-kilauea figures --font Aptos --font-dir /path/to/Aptos
```

The phase image uses cyclic hue, square-root coherence as saturation, and terrain hillshade as brightness. Green first/third quadrants in the GNSS plot mark sign agreement. Error bars show formal ±1 SE. Figure manifests record font and output checksums.
