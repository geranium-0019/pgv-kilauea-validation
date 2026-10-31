# Scientific methods

## Cohort and temporal classes

The input manifest records all 164 interferometric pairs. The paper cohort contains 127 pairs retained by the prior image-quality and registration audit. The stored valid-gradient fraction must exceed 3%; the quality audit may exclude additional pairs. Neither the GNSS sign nor the voting result determines eligibility. This package preserves that audited cohort rather than implementing a new registration-quality classifier.

Timing classes are recomputed from acquisition timestamps and the included eruption catalog. Pre-eruptive pairs do not overlap an eruption, and their second acquisition is within 24 hours before the next eruption. Co-eruptive pairs strictly contain one complete eruption and overlap no other eruption. Partial or multiple overlaps and remaining intervals are Other timing. The catalog coverage limits are retained from the original classification. These classes describe acquisition timing, not GNSS-derived deformation states.

There are 14 eligible Pre-eruptive pairs (ASC 9, DESC 5), 23 Co-eruptive pairs (ASC 14, DESC 9), and 90 Other timing pairs (ASC 46, DESC 44).

## Phase gradient and directional voting

Wrapped central differences use the argument of the complex phase product at offsets of three pixels on either side of the center. The center coherence must be at least 0.25; finite phase samples are required at the difference endpoints. Observations lie inside the prescribed 3 km radius circle. The audited `valid_fraction` therefore describes usable gradient observations, which is not always identical to counting coherence-qualified pixels alone.

Each observation casts a Forward ray along the gradient and a Backward ray in the opposite direction. The weight is coherence squared. Fifty samples per ray cover distances from 50 to 2500 m. The voting grid has 20 m spacing and spans −3500 to +3500 m in east and north coordinates relative to 19.409° N, 155.281° W.

A per-pair fixed seed generates uniformly random directions while retaining observation positions and weights. The resulting null votes are subtracted, and the difference is Gaussian-smoothed with 100 m standard deviation. The null correction reduces directional sampling bias; it does not establish that a peak is deformation or provide a false-alarm probability.

For numerical compatibility, the original corner-coordinate convention and floor-stride sampling are retained. `target_vote_points` sets the stride as `max(1, n_valid // target_vote_points)`; it is a sampling target, not a strict maximum. The same subsampled points are used for actual and null voting. Exact function origins are recorded in `reference/numerical_provenance.json`.

## Frequency maps and Deformation Index

Each direction is thresholded separately at twice the spatial population standard deviation (`ddof=0`) of that pair's null-corrected, smoothed map. The frequency is the number of maps exceeding this threshold at a grid location divided by the number of eligible maps in that class and orbit. It is a descriptive exceedance frequency, not a calibrated anomaly probability.

The orbit-specific fixed ROI masks were predicted using the paper's HMMR source model and LOS geometry. They are inputs, not fitted using the GNSS agreement. Within the appropriate ROI:

`DI = mean(Backward) − mean(Forward)`.

No temporal mean subtraction, temporal standardization, or GNSS displacement cutoff is applied. Positive DI is interpreted as inflation and negative DI as deflation under the paper's interferometric convention and viewing geometry. DI is not a displacement magnitude.

## GNSS endpoints and source-direction comparison

The raw input is the BYRL five-minute `.kenv.gz` archive. The loader uses the east/north/up reference-position columns and their formal uncertainties, converts meters to millimeters, and converts GPS timestamps to UTC using the configured 18 s offset valid for this 2025 dataset. It rejects inconsistent dates and duplicate observations.

At each acquisition time, the estimator fits a local linear trend independently to each component using measurement uncertainty and Huber reweighting. The fitted intercept at the acquisition time is the position estimate; it is not a plain arithmetic average. The primary observation window is ±30 minutes. The ±15 minute sensitivity comparison uses the same pair cohort by requiring complete ±30 minute windows as well. A missing expected five-minute observation excludes the pair; missing values are not interpolated.

The difference of the fitted positions yields three-component displacement `d`. With `g` the unit displacement direction predicted at BYRL by the fixed 1.5 km deep Mogi-type source, and diagonal `C` the sum of the two endpoint variance estimates:

`A = (gᵀ C⁻¹ d) / (gᵀ C⁻¹ g)`

`SE(A) = 1 / sqrt(gᵀ C⁻¹ g)`.

`A` is displacement amplitude in millimeters along the prescribed source-response direction at BYRL; it is not source volume change. Positive means inflation. Strict sign agreement is `DI × A > 0`. The model interpretation, diagonal errors, and formal standard errors are assumptions of this comparison. Shared observations and successive interferograms need not be statistically independent.

Of the 127 eligible pairs, seven are excluded for GNSS coverage, leaving 120. The primary agreement is 103/120 (85.8%); the ±15 minute comparison gives 102/120 (85.0%). These results measure consistency with the fixed-source GNSS interpretation, not general anomaly-detection accuracy.
