"""Hermetic tests for the backend registry + detection (audiblez/backends.py).

backends.py imports only stdlib at module scope (torch is imported lazily inside
available_backends), so these run anywhere — torch is faked via sys.modules.
"""
import unittest
from unittest import mock

from audiblez import backends


def _fake_torch(cuda=False, hip=None, cuda_ver=None, mps=False):
    t = mock.MagicMock()
    t.cuda.is_available.return_value = cuda
    t.version.hip = hip
    t.version.cuda = cuda_ver
    t.backends.mps.is_available.return_value = mps
    return t


class BackendTableTest(unittest.TestCase):
    def test_ids_and_order(self):
        self.assertEqual(backends.BACKEND_IDS, ('cpu', 'cuda', 'rocm', 'mps', 'mlx', 'moss'))

    def test_moss_is_llamacpp_engine_no_torch_device(self):
        info = backends.BACKENDS['moss']
        self.assertEqual(info.engine, 'llamacpp')
        self.assertIsNone(info.torch_device)

    def test_device_mapping(self):
        # cuda and rocm both ride the torch.cuda API -> same torch device
        self.assertEqual(backends.BACKENDS['cuda'].torch_device, 'cuda')
        self.assertEqual(backends.BACKENDS['rocm'].torch_device, 'cuda')
        self.assertEqual(backends.BACKENDS['mps'].torch_device, 'mps')
        self.assertEqual(backends.BACKENDS['cpu'].torch_device, 'cpu')
        # mlx is a separate engine, no torch device
        self.assertEqual(backends.BACKENDS['mlx'].engine, 'mlx')
        self.assertIsNone(backends.BACKENDS['mlx'].torch_device)

    def test_is_gpu(self):
        for gpu in ('cuda', 'rocm', 'mps', 'mlx', 'moss'):  # moss = Vulkan-on-GPU -> GPU ETA rate
            self.assertTrue(backends.is_gpu(gpu))
        self.assertFalse(backends.is_gpu('cpu'))


class DetectionTest(unittest.TestCase):
    def _avail(self, torch_obj, mlx=False, moss='absent'):
        # moss_status is patched too so these stay hermetic regardless of the host's MOSS install.
        with mock.patch.dict('sys.modules', {'torch': torch_obj}), \
             mock.patch.object(backends, '_mlx_importable', return_value=mlx), \
             mock.patch.object(backends, 'moss_status', return_value=moss):
            return backends.available_backends()

    def test_cpu_only(self):
        self.assertEqual(self._avail(_fake_torch()), ['cpu'])

    def test_nvidia(self):
        self.assertEqual(self._avail(_fake_torch(cuda=True, cuda_ver='12.1')), ['cpu', 'cuda'])

    def test_amd_rocm_never_cuda(self):
        # A HIP build reports rocm, never cuda.
        self.assertEqual(self._avail(_fake_torch(cuda=True, hip='6.2')), ['cpu', 'rocm'])

    def test_apple_silicon(self):
        self.assertEqual(self._avail(_fake_torch(mps=True), mlx=True), ['cpu', 'mps', 'mlx'])

    def test_torch_import_failure_degrades_to_cpu(self):
        boom = mock.MagicMock()
        boom.cuda.is_available.side_effect = RuntimeError('no torch')
        self.assertEqual(self._avail(boom), ['cpu'])

    def test_default_prefers_accelerator(self):
        cases = {
            ('cpu',): 'cpu',
            ('cpu', 'cuda'): 'cuda',
            ('cpu', 'rocm'): 'rocm',
            ('cpu', 'mps', 'mlx'): 'mlx',   # native MLX preferred over torch MPS (README: fastest on Mac)
            ('cpu', 'rocm', 'moss'): 'moss',  # MOSS is the quality default (ADR 0003) -> preferred over rocm
            ('cpu', 'cuda', 'moss'): 'moss',
        }
        for avail, expected in cases.items():
            with mock.patch.object(backends, 'available_backends', return_value=list(avail)):
                self.assertEqual(backends.default_backend(), expected)

    def test_moss_listed_only_when_present(self):
        # A fully-installed MOSS appears in availability; a partial/absent one does not, so the
        # auto-selector never half-picks a mis-installed MOSS.
        self.assertIn('moss', self._avail(_fake_torch(cuda=True, hip='6.2'), moss='present'))
        self.assertNotIn('moss', self._avail(_fake_torch(cuda=True, hip='6.2'), moss='partial'))
        self.assertNotIn('moss', self._avail(_fake_torch(cuda=True, hip='6.2'), moss='absent'))


class GpuWorksTest(unittest.TestCase):
    def test_cpu_and_unknown_backends_are_always_ok(self):
        self.assertTrue(backends.gpu_works('cpu'))
        self.assertTrue(backends.gpu_works('bogus'))

    def test_faulting_gpu_returns_false(self):
        boom = mock.MagicMock()
        boom.ones.side_effect = RuntimeError('HIP error: no usable kernel')
        with mock.patch.dict('sys.modules', {'torch': boom}):
            self.assertFalse(backends.gpu_works('rocm'))  # is_available() lies; kernel faults

    def test_working_gpu_returns_true(self):
        import numpy as np
        ok = mock.MagicMock()
        ok.ones.side_effect = lambda *a, **k: np.ones((8, 8))  # real array supports @/.sum().item()
        with mock.patch.dict('sys.modules', {'torch': ok}):
            self.assertTrue(backends.gpu_works('rocm'))


class MossDiscoveryTest(unittest.TestCase):
    """moss_paths/moss_repo_id/moss_status — pure path resolution, no spawn, hermetic."""

    def test_paths_from_env_when_files_exist(self):
        import os
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            binp = os.path.join(d, 'llama-moss-tts'); open(binp, 'w').close()
            bb = os.path.join(d, 'bb_Q5_K_M.gguf'); open(bb, 'w').close()
            dec = os.path.join(d, 'dec_f16.gguf'); open(dec, 'w').close()
            with mock.patch.dict('os.environ', {
                    'AUDIBLEZ_MOSS_BIN': binp, 'AUDIBLEZ_MOSS_BACKBONE': bb,
                    'AUDIBLEZ_MOSS_DECODER': dec, 'AUDIBLEZ_MOSS_ENCODER': '/no/such/enc.gguf'}):
                p = backends.moss_paths()
                self.assertEqual(str(p['binary']), binp)
                self.assertEqual(str(p['backbone']), bb)
                self.assertEqual(str(p['decoder']), dec)
                self.assertIsNone(p['encoder'])           # set but missing -> None
                self.assertEqual(backends.moss_status(), 'present')  # encoder not required

    def test_status_absent_when_nothing_found(self):
        # Force every resolved path to None: no binary on PATH, env points nowhere, defaults absent.
        with mock.patch.object(backends.shutil, 'which', return_value=None), \
             mock.patch.dict('os.environ', {
                 'AUDIBLEZ_MOSS_BIN': '/no/bin', 'AUDIBLEZ_MOSS_BACKBONE': '/no/bb',
                 'AUDIBLEZ_MOSS_DECODER': '/no/dec', 'AUDIBLEZ_MOSS_ENCODER': '/no/enc'}):
            self.assertEqual(backends.moss_status(), 'absent')

    def test_status_partial_when_binary_present_gguf_missing(self):
        import os
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            binp = os.path.join(d, 'llama-moss-tts'); open(binp, 'w').close()
            with mock.patch.dict('os.environ', {
                    'AUDIBLEZ_MOSS_BIN': binp, 'AUDIBLEZ_MOSS_BACKBONE': '/no/bb',
                    'AUDIBLEZ_MOSS_DECODER': '/no/dec', 'AUDIBLEZ_MOSS_ENCODER': '/no/enc'}):
                self.assertEqual(backends.moss_status(), 'partial')

    def test_repo_id_changes_with_gguf_identity(self):
        import os
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            bb = os.path.join(d, 'bb_Q5_K_M.gguf')
            with open(bb, 'w') as f:
                f.write('x')
            dec = os.path.join(d, 'dec_f16.gguf'); open(dec, 'w').close()
            env = {'AUDIBLEZ_MOSS_BACKBONE': bb, 'AUDIBLEZ_MOSS_DECODER': dec,
                   'AUDIBLEZ_MOSS_ENCODER': '/no/enc'}
            with mock.patch.dict('os.environ', env):
                rid1 = backends.moss_repo_id()
            # same names, different backbone SIZE -> different identity -> different repo_id
            with open(bb, 'w') as f:
                f.write('xxxxxxxx')
            with mock.patch.dict('os.environ', env):
                rid2 = backends.moss_repo_id()
            self.assertTrue(rid1.startswith('moss-gguf:'))
            self.assertNotEqual(rid1, rid2)

    def test_repo_id_stable_for_same_identities(self):
        with mock.patch.object(backends, 'moss_paths',
                               return_value={'binary': None, 'backbone': None,
                                             'decoder': None, 'encoder': None}):
            # all-absent still yields a deterministic id (basename:absent triples)
            self.assertEqual(backends.moss_repo_id(), backends.moss_repo_id())

    def test_clone_voice_id_same_ref_same_id_diff_ref_diff_id(self):
        import os
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            ref_a = os.path.join(d, 'a.wav')
            ref_b = os.path.join(d, 'b.wav')
            with open(ref_a, 'wb') as f:
                f.write(b'RIFF....voiceA....')
            with open(ref_b, 'wb') as f:
                f.write(b'RIFF....voiceB....')
            id_a = backends.clone_voice_id(ref_a)
            id_b = backends.clone_voice_id(ref_b)
            self.assertTrue(id_a.startswith('clone:'))
            self.assertEqual(id_a, backends.clone_voice_id(ref_a))   # same bytes -> same id
            self.assertNotEqual(id_a, id_b)                          # diff bytes -> diff id
            # content-addressed: a renamed copy of A's bytes hashes identically to A
            ref_a2 = os.path.join(d, 'renamed.wav')
            with open(ref_a2, 'wb') as f:
                f.write(b'RIFF....voiceA....')
            self.assertEqual(id_a, backends.clone_voice_id(ref_a2))

    def test_clone_voice_id_missing_ref_raises(self):
        # A clone run with no usable reference must fail loud, never silently use a default voice.
        with self.assertRaises((FileNotFoundError, OSError)):
            backends.clone_voice_id('/no/such/reference.wav')

    def test_sampling_sig_deterministic_and_override_sensitive(self):
        # The canonical default signature is stable run-to-run, includes all six params, and
        # changes when any one is overridden (so it can never silently drop a sampling change).
        base = backends.moss_sampling_sig()
        self.assertEqual(base, backends.moss_sampling_sig())
        self.assertEqual(base.count('='), 6)
        self.assertNotEqual(base, backends.moss_sampling_sig({'audio_top_p': 0.9}))
        # passing the pinned defaults explicitly equals passing nothing (order-independent)
        self.assertEqual(base, backends.moss_sampling_sig(dict(backends.MOSS_SAMPLING_DEFAULTS)))


class EngineChoicesTest(unittest.TestCase):
    """engine_choices() — the GUI's availability-aware engine list (PRD slice 1 / #7)."""

    _PATHS_PRESENT = {'binary': '/x/llama-moss-tts', 'backbone': '/x/b.gguf',
                      'decoder': '/x/d.gguf', 'encoder': '/x/e.gguf'}

    def _choices(self, avail, paths=None):
        with mock.patch.object(backends, 'available_backends', return_value=list(avail)), \
             mock.patch.object(backends, 'moss_paths', return_value=dict(paths or {})):
            return backends.engine_choices()

    def test_present_moss_is_available_no_reason(self):
        # MOSS in available_backends() -> selectable, empty reason, appears in display order.
        choices = self._choices(['cpu', 'rocm', 'moss'], self._PATHS_PRESENT)
        by_id = {c.id: c for c in choices}
        self.assertEqual([c.id for c in choices], ['cpu', 'rocm', 'moss'])
        self.assertTrue(by_id['moss'].available)
        self.assertEqual(by_id['moss'].reason, '')

    def test_kokoro_backends_always_available_and_ordered(self):
        # The torch/mlx backends are always returned available=True in available_backends order.
        choices = self._choices(['cpu', 'rocm'], {'binary': None, 'backbone': '/x/b.gguf',
                                                  'decoder': '/x/d.gguf', 'encoder': None})
        kokoro = [c for c in choices if c.id != 'moss']
        self.assertEqual([c.id for c in kokoro], ['cpu', 'rocm'])
        self.assertTrue(all(c.available and c.reason == '' for c in kokoro))

    def test_partial_missing_binary_disabled_with_reason(self):
        # binary absent, required GGUFs present -> MOSS present-but-disabled, reason names binary.
        choices = self._choices(['cpu', 'rocm'], {'binary': None, 'backbone': '/x/b.gguf',
                                                  'decoder': '/x/d.gguf', 'encoder': '/x/e.gguf'})
        moss = next(c for c in choices if c.id == 'moss')
        self.assertFalse(moss.available)
        self.assertIn('binary', moss.reason)
        self.assertNotIn('model', moss.reason)

    def test_partial_missing_model_disabled_with_reason(self):
        # binary present, a required GGUF absent -> disabled, reason names model (not binary).
        choices = self._choices(['cpu', 'rocm'], {'binary': '/x/llama-moss-tts', 'backbone': None,
                                                  'decoder': '/x/d.gguf', 'encoder': '/x/e.gguf'})
        moss = next(c for c in choices if c.id == 'moss')
        self.assertFalse(moss.available)
        self.assertIn('model', moss.reason)
        self.assertNotIn('binary', moss.reason)

    def test_absent_moss_disabled_with_nonempty_reason(self):
        # nothing installed -> still listed (present-but-disabled), reason non-empty (not hidden).
        choices = self._choices(['cpu'], {'binary': None, 'backbone': None,
                                          'decoder': None, 'encoder': None})
        moss = next(c for c in choices if c.id == 'moss')
        self.assertFalse(moss.available)
        self.assertTrue(moss.reason)
        self.assertIn('binary', moss.reason)
        self.assertIn('model', moss.reason)


if __name__ == '__main__':
    unittest.main()
