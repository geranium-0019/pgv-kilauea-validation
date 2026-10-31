import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
from pgv_kilauea_validation import _pgv_core as core
from pgv_kilauea_validation.gnss import fit_amplitude, expected_times, EPOCH
from pgv_kilauea_validation.analysis import classify_interval


class GradientTests(unittest.TestCase):
    def test_wrapped_plane_recovers_east_north_gradient(self):
        r, c = np.indices((40, 45))
        east = 0.17
        north = -0.11
        phase = np.angle(np.exp(1j * (east * c + north * (-r))))
        e, n, d = core.compute_wrapped_gradient(
            phase, np.ones_like(phase), 0.25, (0, 1, 0, 0, 0, -1), 1, 1, 3
        )
        np.testing.assert_allclose(e[3:-3, 3:-3], east, atol=1e-14)
        np.testing.assert_allclose(n[3:-3, 3:-3], north, atol=1e-14)
        np.testing.assert_allclose(d[3:-3, 3:-3], np.arctan2(north, east), atol=1e-13)

    def test_disconnected_pixels_can_vote_for_radial_center(self):
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        x, y = 200 * np.cos(angles), 200 * np.sin(angles)
        direction = angles + np.pi
        f, b = core.do_vote_directional(
            x,
            y,
            direction,
            vote_sp=10,
            xn=-300,
            xx=300,
            yn=-300,
            yx=300,
            vnx=61,
            vny=61,
            ray_min=10,
            ray_max=250,
            n_ray_steps=25,
            ray_spread=0,
            smooth_sigma_m=15,
            weights=np.ones(len(x)),
        )
        iy, ix = np.unravel_index(f.argmax(), f.shape)
        self.assertLess(np.hypot(ix * 10 - 300, iy * 10 - 300), 25)
        self.assertGreater(f.max(), b.max())

    def test_identical_null_directions_cancel_exactly(self):
        rng = np.random.RandomState(123)
        x = rng.uniform(-100, 100, 50)
        y = rng.uniform(-100, 100, 50)
        d = rng.uniform(-np.pi, np.pi, 50)
        kw = dict(
            vote_sp=10,
            xn=-200,
            xx=200,
            yn=-200,
            yx=200,
            vnx=41,
            vny=41,
            ray_min=10,
            ray_max=100,
            n_ray_steps=10,
            ray_spread=0,
            smooth_sigma_m=10,
        )
        a = np.array(core.do_vote_directional(x, y, d, **kw))
        b = np.array(core.do_vote_directional(x, y, d, **kw))
        np.testing.assert_array_equal(a - b, 0)


class GNSSTests(unittest.TestCase):
    def test_weighted_amplitude_recovers_signed_model_displacement(self):
        g = np.array([0.6, 0, 0.8])
        v = np.array([1.0, 9.0, 16.0])
        for amplitude in [-12, 0, 19]:
            estimate, se = fit_amplitude(amplitude * g, v, g)
            self.assertAlmostEqual(estimate, amplitude, places=12)
            self.assertAlmostEqual(se, 1 / np.sqrt(np.sum(g * g / v)), places=12)

    def test_uncertain_component_has_less_influence(self):
        g = np.array([1.0, 0, 1.0]) / np.sqrt(2)
        d = np.array([10.0, 0, 50.0])
        a, _ = fit_amplitude(d, [1, 1, 10000], g)
        self.assertLess(abs(a - 10 / g[0]), 0.02)

    def test_invalid_variance_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_amplitude([1, 2, 3], [1, 0, 1], [1, 0, 0])

    def test_coverage_grid_uses_gps_utc_offset_and_window_edges(self):
        t = datetime(2025, 1, 1, 0, 4, 42, tzinfo=timezone.utc)
        times = expected_times(t - timedelta(minutes=30), t + timedelta(minutes=30))
        self.assertEqual(len(times), 13)
        np.testing.assert_array_equal(np.mod(times + 18 - EPOCH.timestamp(), 300), 0)
        self.assertEqual(times[0], (t - timedelta(minutes=30)).timestamp())
        self.assertEqual(times[-1], (t + timedelta(minutes=30)).timestamp())


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.t = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.episodes = [
            (self.t, self.t + timedelta(hours=2)),
            (self.t + timedelta(days=3), self.t + timedelta(days=3, hours=2)),
            (self.t + timedelta(days=6), self.t + timedelta(days=6, hours=2)),
        ]

    def test_full_single_episode_is_co_eruptive(self):
        self.assertEqual(
            classify_interval(
                self.t + timedelta(days=2), self.t + timedelta(days=4), self.episodes
            ),
            "single-eruption",
        )

    def test_partial_episode_is_other(self):
        self.assertEqual(
            classify_interval(
                self.t + timedelta(days=3, hours=1),
                self.t + timedelta(days=4),
                self.episodes,
            ),
            "not-selected",
        )

    def test_pre_requires_24h_and_no_overlap(self):
        self.assertEqual(
            classify_interval(
                self.t + timedelta(days=1),
                self.t + timedelta(days=2, hours=12),
                self.episodes,
            ),
            "pre-eruptive-24h",
        )
        self.assertEqual(
            classify_interval(
                self.t + timedelta(hours=3), self.t + timedelta(days=1), self.episodes
            ),
            "not-selected",
        )

    def test_multiple_eruptions_are_other(self):
        self.assertEqual(
            classify_interval(
                self.t + timedelta(days=2),
                self.t + timedelta(days=6, hours=3),
                self.episodes,
            ),
            "not-selected",
        )

    def test_overlapping_catalog_is_rejected(self):
        with self.assertRaises(ValueError):
            classify_interval(
                self.t,
                self.t + timedelta(days=1),
                [
                    (self.t, self.t + timedelta(hours=2)),
                    (self.t, self.t + timedelta(hours=1)),
                ],
            )
