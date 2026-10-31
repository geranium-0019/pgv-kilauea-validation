# Numerical provenance

`_pgv_core.py` retains the gradient, coordinate conversion, circular-domain, and directional ray-voting functions from the author's PGV implementation. `_gnss_fit.py` retains the local robust linear position estimator. Their source-file hashes and retained objects are recorded in `reference/numerical_provenance.json`.

The numerical functions were extracted without changing their calculations. Dataset I/O, configuration, artifact checking, CLI orchestration, and plotting were implemented for this package. The source snapshot is the same one used by the Kilauea paper analysis. The related baseline implementation is https://github.com/geranium-0019/gradient_voting_DFC_2026 .

The legacy geotransform coordinate convention, floor-stride subsampling, seeded NumPy `RandomState`, and Gaussian-filter defaults are retained because they affect numerical reproducibility. Changing these conventions constitutes a separate experiment.

NumPy, SciPy, rasterio, Matplotlib, and PyYAML are installed dependencies, not vendored source. Their own licenses apply to those libraries. Aptos is an optional locally installed font and is not redistributed.
