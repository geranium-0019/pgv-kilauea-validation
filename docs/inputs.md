# Input contract

Paths in YAML resolve relative to `base_dir`, which resolves relative to the YAML file. The default configuration is `configs/kilauea.yaml`. Commands are intended to run from the repository root. Use `--config /absolute/path/to/config.yaml` from another working directory.

## Interferograms

Each manifest row requires `pair_id`, `track` (`ASC` or `DESC`), `quality_eligible`, `valid_fraction`, `analysis_class`, `null_seed`, master/slave image IDs and timezone-aware acquisition times, and relative phase/coherence paths with SHA256 hashes. Paths resolve below `data_dir`; absolute paths and parent traversal are rejected.

The two single-band GeoTIFFs must share shape, affine transform, and WGS84 geographic coordinates (EPSG:4326). Wrapped phase is in radians within [−π, π]; coherence is within [0, 1]. Raster masks and phase values below −9000 are treated as invalid. The pair manifest links to selected final interferograms; the tool does not regenerate, deramp, or atmospherically correct them.

The paper manifest carries independently recorded registration decisions and input hashes. Changing the phase inputs, coherence criterion, or gradient step requires an appropriately updated audit/manifest. A disagreement in the recomputed valid-gradient fraction raises an error rather than silently applying old eligibility to a different observation set.

## Small bundled inputs

- `acquisitions.csv`: image ID, track, and acquisition time.
- `eruption_episodes.csv`: UTC start/end intervals with catalog provenance.
- `image_estimates.csv`: previously computed GNSS endpoint positions, uncertainties, and coverage for both windows. These enable evaluation without raw GNSS files.
- `ASC_roi.npz`, `DESC_roi.npz`: Boolean ROI masks and matching `x_m`, `y_m` voting coordinates.
- `source_model.json`: fixed source/station geometry and the normalized three-component source-response vector at BYRL.
- `hmmr.geojson`: reference-area polygon for figures.

The reproduction uses these fixed geometry inputs. Predicting new LOS-specific ROIs or fitting a new source is outside this tool's scope.

## Large external inputs

`prepare-paper --source /path/to/graduation_thesis` attaches existing paper inputs as symlinks, or copies them with `--mode copy`:

```text
data/
  images/<pair_id>/phase_final.tif
  images/<pair_id>/coherence.tif
  dem.tif
  gnss/BYRL.2025.<day>.kenv.gz
  reference_maps/<pair_id>.npz
```

No data are downloaded. Source files are read only. `data/` and `output/` are ignored by Git. Licensed SAR imagery must be obtained separately.

The DEM supplies terrain context for figures and must cover the voting domain. The GNSS archive is required for fresh endpoint estimation; `evaluate` can use the bundled endpoints. Original PGV maps are optional for `verify`: when present they are compared in full; otherwise the report records zero map comparisons, while checking the bundled DI, GNSS, and frequency references.
