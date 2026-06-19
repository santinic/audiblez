"""Tests for GPU tuning knobs (audiblez/gpu.py).

The env/dispatch logic is pure stdlib+numpy, so it runs anywhere. The one path that
constructs a real ``torch.autocast`` is guarded with ``skipUnless`` so it runs only
where torch is installed (CI / a GPU box), never failing a numpy-only environment.
"""
import os
import contextlib
import importlib.util
import unittest
from unittest import mock
import numpy as np

from audiblez import gpu

_HAS_TORCH = importlib.util.find_spec('torch') is not None


class ConfigureTunableOpTest(unittest.TestCase):
    def test_noop_when_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gpu.configure_tunableop(False, 'rocm', out=lambda *_: None))
            self.assertNotIn('PYTORCH_TUNABLEOP_ENABLED', os.environ)

    def test_noop_on_cpu(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gpu.configure_tunableop(True, 'cpu', out=lambda *_: None))
            self.assertNotIn('PYTORCH_TUNABLEOP_ENABLED', os.environ)

    def test_noop_on_mlx(self):
        # mlx is GPU but not a torch backend -> TunableOp does not apply.
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(gpu.configure_tunableop(True, 'mlx', out=lambda *_: None))
            self.assertNotIn('PYTORCH_TUNABLEOP_ENABLED', os.environ)

    def test_enables_on_rocm_and_persists_csv(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmp:
                self.assertTrue(gpu.configure_tunableop(True, 'rocm', results_dir=tmp,
                                                        out=lambda *_: None))
                self.assertEqual(os.environ['PYTORCH_TUNABLEOP_ENABLED'], '1')
                self.assertEqual(os.environ['PYTORCH_TUNABLEOP_TUNING'], '1')
                self.assertTrue(os.environ['PYTORCH_TUNABLEOP_FILENAME'].startswith(tmp))

    def test_respects_existing_env(self):
        with mock.patch.dict(os.environ, {'PYTORCH_TUNABLEOP_ENABLED': '0'}, clear=True):
            gpu.configure_tunableop(True, 'cuda', out=lambda *_: None)
            self.assertEqual(os.environ['PYTORCH_TUNABLEOP_ENABLED'], '0')  # not clobbered


class AutocastContextTest(unittest.TestCase):
    def test_fp32_is_nullcontext(self):
        self.assertIsInstance(gpu.autocast_context('rocm', 'fp32'), contextlib.nullcontext)

    def test_cpu_is_nullcontext(self):
        self.assertIsInstance(gpu.autocast_context('cpu', 'fp16'), contextlib.nullcontext)

    def test_mlx_is_nullcontext(self):
        self.assertIsInstance(gpu.autocast_context('mlx', 'fp16'), contextlib.nullcontext)

    def test_invalid_precision_raises(self):
        with self.assertRaises(ValueError):
            gpu.autocast_context('rocm', 'int8')

    @unittest.skipUnless(_HAS_TORCH, 'torch not installed')
    def test_fp16_on_gpu_is_real_autocast(self):
        ctx = gpu.autocast_context('rocm', 'fp16')
        import torch
        self.assertIsInstance(ctx, torch.autocast)


class WaveformSimilarityTest(unittest.TestCase):
    def test_identical_is_one(self):
        a = np.sin(np.linspace(0, 10, 200)).astype(np.float32)
        self.assertAlmostEqual(gpu.waveform_similarity(a, a), 1.0, places=5)

    def test_anticorrelated_is_negative_one(self):
        a = np.sin(np.linspace(0, 10, 200))
        self.assertAlmostEqual(gpu.waveform_similarity(a, -a), -1.0, places=5)

    def test_length_tolerant(self):
        a = np.sin(np.linspace(0, 10, 200))
        # near-identical but a few samples longer (precision can shift durations)
        self.assertGreater(gpu.waveform_similarity(a, np.append(a, [0, 0, 0])), 0.99)

    def test_empty_is_zero(self):
        self.assertEqual(gpu.waveform_similarity(np.array([]), np.array([1.0])), 0.0)


if __name__ == '__main__':
    unittest.main()
