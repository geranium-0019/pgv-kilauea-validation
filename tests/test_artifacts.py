from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pgv_kilauea_validation.io import Config, digest
from pgv_kilauea_validation.pgv import (
    observations,
    run_pair,
    load_maps,
    export_geotiffs,
)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.data = root / "data"
        self.data.mkdir()
        gt = from_origin(-0.0003, 0.0003, 0.00005, 0.00005)
        rows, cols = np.indices((14, 14))
        phase = np.angle(np.exp(1j * (0.1 * cols + 0.05 * rows))).astype("float32")
        coh = np.full_like(phase, 0.8)
        for name, data in [("phase", phase), ("coherence", coh)]:
            with rasterio.open(
                self.data / (name + ".tif"),
                "w",
                driver="GTiff",
                width=14,
                height=14,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=gt,
                nodata=np.nan,
            ) as ds:
                ds.write(data, 1)
        values = {
            "reference": {"lat": 0.0, "lon": 0.0},
            "pgv": {
                "radius_m": 100,
                "coherence_threshold": 0.25,
                "gradient_step_px": 1,
                "vote_spacing_m": 10.0,
                "extent_m": 100.0,
                "ray_min_m": 5.0,
                "ray_max_m": 80.0,
                "ray_steps": 12,
                "smooth_sigma_m": 10.0,
                "target_vote_points": 1000,
                "weight": "coherence_squared",
            },
        }
        self.cfg = Config(
            root / "settings.yaml", root, values, root / "output", self.data
        )
        self.row = {
            "pair_id": "synthetic",
            "track": "ASC",
            "analysis_class": "not-selected",
            "null_seed": "19",
            "phase_path": "phase.tif",
            "coherence_path": "coherence.tif",
            "phase_sha256": digest(self.data / "phase.tif"),
            "coherence_sha256": digest(self.data / "coherence.tif"),
        }
        obs = observations(self.cfg, self.row)
        self.row["valid_fraction"] = str(obs["n_valid"] / obs["n_domain"])

    def tearDown(self):
        self.temp.cleanup()

    def test_output_reuse_checks_settings_and_file_integrity(self):
        record = run_pair(self.cfg, self.row)
        self.assertEqual(record, run_pair(self.cfg, self.row))
        self.cfg.values["pgv"]["weight"] = "uniform"
        with self.assertRaisesRegex(ValueError, "different settings"):
            run_pair(self.cfg, self.row)

    def test_modified_input_is_rejected(self):
        p = self.data / "phase.tif"
        with p.open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            run_pair(self.cfg, self.row)

    def test_modified_cached_result_is_rejected(self):
        run_pair(self.cfg, self.row)
        p = self.cfg.output / "pgv/synthetic.npz"
        with p.open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact was modified"):
            run_pair(self.cfg, self.row)

    def test_changed_observation_definition_requires_new_audit(self):
        self.row["valid_fraction"] = "0.01"
        with self.assertRaisesRegex(ValueError, "differs from audited manifest"):
            run_pair(self.cfg, self.row)

    def test_geotiff_grid_and_values_match_saved_map(self):
        run_pair(self.cfg, self.row)
        maps = load_maps(self.cfg, self.row)
        export_geotiffs(self.cfg, self.row)
        with rasterio.open(self.cfg.output / "geotiff/synthetic_fwd.tif") as ds:
            np.testing.assert_allclose(
                ds.read(1), maps["fwd"][::-1], rtol=1e-6, atol=1e-6
            )
            lon, lat = ds.xy(0, 0)
            self.assertAlmostEqual(lon * 111320, maps["x_m"][0], places=9)
            self.assertAlmostEqual(lat * 111320, maps["y_m"][-1], places=9)

    def test_manifest_path_cannot_escape_data_directory(self):
        self.row["phase_path"] = "../outside.tif"
        with self.assertRaises(ValueError):
            self.cfg.input_path(self.row, "phase")
