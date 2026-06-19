"""Tests for the measured-throughput ETA (core.ewma + core._update_eta).

Guarded import so it skips when the heavy stack is absent. Pure arithmetic — no
model, audio, or timing dependency (elapsed is passed in).
"""
import unittest
from types import SimpleNamespace

try:
    from audiblez.core import ewma, _update_eta, strfdelta
    _ERR = None
except Exception as e:
    _ERR = e


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class EwmaTest(unittest.TestCase):
    def test_none_prev_seeds_with_sample(self):
        self.assertEqual(ewma(None, 42), 42)

    def test_blends_prior_and_sample(self):
        # 0.3*200 + 0.7*100 = 130
        self.assertAlmostEqual(ewma(100, 200, alpha=0.3), 130.0)

    def test_steady_sample_is_fixed_point(self):
        self.assertAlmostEqual(ewma(50, 50), 50.0)


@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class UpdateEtaTest(unittest.TestCase):
    def _stats(self):
        return SimpleNamespace(total_chars=1000, processed_chars=0, chars_per_sec=50.0)

    def test_first_update_replaces_prior_then_blends(self):
        s = self._stats()
        _update_eta(s, measured_chars=100, elapsed=1.0)  # first measured sample: 100 cps
        # The flat 50 prior is only a guess, so the first real measurement REPLACES it
        # (otherwise a ~10x-off prior would dominate the ETA for the opening chapters).
        self.assertAlmostEqual(s.chars_per_sec, 100.0)
        self.assertEqual(s.processed_chars, 100)
        self.assertEqual(s.progress, 10)
        self.assertEqual(s.eta, strfdelta((1000 - 100) / 100.0))
        # A SECOND measurement now blends into the EWMA rather than replacing.
        _update_eta(s, measured_chars=50, elapsed=1.0)  # 50 cps
        self.assertAlmostEqual(s.chars_per_sec, 0.3 * 50 + 0.7 * 100)  # 85

    def test_converges_to_measured_rate(self):
        s = self._stats()
        s.total_chars = 10 ** 9  # keep it from finishing
        for _ in range(50):
            _update_eta(s, measured_chars=100, elapsed=1.0)  # steady 100 cps
        self.assertAlmostEqual(s.chars_per_sec, 100.0, places=1)

    def test_zero_elapsed_leaves_rate_unchanged(self):
        s = self._stats()
        _update_eta(s, measured_chars=100, elapsed=0.0)
        self.assertEqual(s.chars_per_sec, 50.0)
        self.assertEqual(s.processed_chars, 100)

    def test_overshoot_clamps_eta_to_zero(self):
        s = self._stats()
        _update_eta(s, measured_chars=5000, elapsed=1.0)  # processed > total
        self.assertEqual(s.eta, strfdelta(0))


if __name__ == '__main__':
    unittest.main()
