"""Hermetic tests for voice quality metadata and blend resolution.

audiblez.voices is stdlib-only (no torch/kokoro), so these run anywhere with no
heavy deps, model, or network.
"""
import unittest

from audiblez import voices as v


class VoiceQualityTest(unittest.TestCase):
    def test_default_voice_is_top_graded(self):
        self.assertEqual(v.DEFAULT_VOICE, 'af_heart')
        self.assertEqual(v.VOICE_QUALITY['af_heart'], 'A')

    def test_old_default_was_low_grade(self):
        # af_sky (the previous default) is one of the weakest US voices.
        self.assertEqual(v.VOICE_QUALITY['af_sky'], 'C-')
        self.assertGreater(v.grade_rank('C-'), v.grade_rank('A'))

    def test_every_quality_key_is_a_real_voice(self):
        self.assertTrue(set(v.VOICE_QUALITY).issubset(v.ALL_VOICES))

    def test_recommended_and_presets_reference_real_voices(self):
        self.assertTrue(set(v.RECOMMENDED_VOICES).issubset(v.ALL_VOICES))
        for recipe in v.PRESET_BLENDS.values():
            for vid, weight in recipe:
                self.assertIn(vid, v.ALL_VOICES)
                self.assertIsInstance(weight, int)
                self.assertGreater(weight, 0)

    def test_preset_info_matches_blends(self):
        self.assertEqual(set(v.PRESET_BLEND_INFO), set(v.PRESET_BLENDS))


class ParseVoiceSpecTest(unittest.TestCase):
    def test_single_voice(self):
        self.assertEqual(v.parse_voice_spec('af_heart'), [('af_heart', 1)])

    def test_preset_blend(self):
        self.assertEqual(v.parse_voice_spec('af_warm'), [('af_heart', 1), ('af_bella', 1)])

    def test_inline_equal_blend(self):
        self.assertEqual(v.parse_voice_spec('af_bella,af_heart'), [('af_bella', 1), ('af_heart', 1)])

    def test_inline_weighted_blend_is_reduced(self):
        # 60/40 -> minimal integer ratio 3:2
        self.assertEqual(v.parse_voice_spec('af_bella:60,af_heart:40'), [('af_bella', 3), ('af_heart', 2)])

    def test_inline_fractional_weights(self):
        self.assertEqual(v.parse_voice_spec('af_bella:0.6,af_heart:0.4'), [('af_bella', 3), ('af_heart', 2)])

    def test_unknown_voice_raises(self):
        with self.assertRaises(ValueError):
            v.parse_voice_spec('af_nonexistent')

    def test_empty_raises(self):
        for bad in ('', '   ', None):
            with self.assertRaises(ValueError):
                v.parse_voice_spec(bad)

    def test_bad_weight_raises(self):
        with self.assertRaises(ValueError):
            v.parse_voice_spec('af_bella:abc,af_heart:40')

    def test_non_finite_weight_raises_valueerror(self):
        # inf/-inf/1e400 parse as floats but used to escape as an uncaught OverflowError in Fraction().
        for bad in ('af_bella:inf,af_heart:1', 'af_bella:-inf,af_heart:1', 'af_bella:1e400,af_heart:1'):
            with self.assertRaises(ValueError):
                v.parse_voice_spec(bad)


class KokoroVoiceStringTest(unittest.TestCase):
    def test_single_voice_is_bare_id(self):
        self.assertEqual(v.kokoro_voice_string('af_heart'), 'af_heart')

    def test_equal_blend_is_comma_join(self):
        self.assertEqual(v.kokoro_voice_string('af_warm'), 'af_heart,af_bella')

    def test_weighted_blend_uses_repetition(self):
        # 60/40 -> 3:2 -> three bella + two heart, which both engines average to 0.6/0.4.
        self.assertEqual(v.kokoro_voice_string('af_bella:60,af_heart:40'),
                         'af_bella,af_bella,af_bella,af_heart,af_heart')

    def test_deep_preset_repetition(self):
        self.assertEqual(v.kokoro_voice_string('am_deep'),
                         'am_puck,am_puck,am_onyx')


class VoiceLangCodeTest(unittest.TestCase):
    def test_single(self):
        self.assertEqual(v.voice_lang_code('af_heart'), 'a')

    def test_preset_name(self):
        self.assertEqual(v.voice_lang_code('ab_storyteller'), 'a')

    def test_british_first(self):
        self.assertEqual(v.voice_lang_code('bf_emma,af_heart'), 'b')


class VoiceLabelTest(unittest.TestCase):
    def test_plain_voice(self):
        self.assertEqual(v.voice_label('af_heart'), 'af_heart')

    def test_preset_name(self):
        self.assertEqual(v.voice_label('af_warm'), 'af_warm')

    def test_inline_blend_is_filesystem_safe(self):
        label = v.voice_label('af_bella:60,af_heart:40')
        self.assertNotIn(':', label)
        self.assertNotIn(',', label)
        self.assertEqual(label, 'blend-af_bellax3-af_heartx2')


class IsBlendTest(unittest.TestCase):
    def test_single_is_not_blend(self):
        self.assertFalse(v.is_blend('af_heart'))

    def test_preset_is_blend(self):
        self.assertTrue(v.is_blend('af_warm'))


class BlendWeightSafetyTest(unittest.TestCase):
    """Regression tests pinning the blend-weight hardening (no non-positive
    weights, bounded comma expansion, defined duplicate semantics)."""

    def test_zero_weight_raises(self):
        with self.assertRaises(ValueError):
            v.parse_voice_spec('af_bella:0,af_heart:40')

    def test_negative_weight_raises(self):
        with self.assertRaises(ValueError):
            v.parse_voice_spec('af_bella:-5,af_heart:40')

    def test_extreme_numerator_is_capped_not_unbounded(self):
        # 'af_bella:1000000,af_heart:1' would naively expand to ~1M comma parts
        # (~500GB of voicepacks) and OOM. The rescale clamps each voice's count
        # to MAX_BLEND_PARTS (a positive weight is floored at 1, never dropped),
        # so the comma string can never explode — its part count is bounded by
        # the cap plus at most one extra part per distinct voice.
        spec = 'af_bella:1000000,af_heart:1'
        comps = v.parse_voice_spec(spec)
        for _, weight in comps:
            self.assertLessEqual(weight, v.MAX_BLEND_PARTS)
        parts = v.kokoro_voice_string(spec).split(',')
        self.assertLessEqual(len(parts), v.MAX_BLEND_PARTS + len(comps))
        # Both voices survive the rescale (no positive weight dropped to zero).
        self.assertIn('af_bella', parts)
        self.assertIn('af_heart', parts)

    def test_near_equal_coprime_weights_stay_bounded(self):
        # 33:33:34 are pairwise near-equal coprime integers; a naive LCM/GCD
        # reduction keeps them at ~100 parts. They must collapse to ~1:1:1.
        comps = v.parse_voice_spec('af_heart:33,af_bella:33,af_nicole:34')
        total_weight = sum(w for _, w in comps)
        self.assertLess(total_weight, 100)
        self.assertLessEqual(total_weight, v.MAX_BLEND_PARTS)
        self.assertEqual(len(v.kokoro_voice_string('af_heart:33,af_bella:33,af_nicole:34').split(',')),
                         total_weight)

    def test_duplicate_voice_id_sums_weights(self):
        # A voice repeated in a blend sums its weights into one component rather
        # than double-counting it as two separate parts; first-seen order kept.
        self.assertEqual(v.parse_voice_spec('af_heart:60,af_heart:40'), [('af_heart', 1)])
        self.assertEqual(
            v.parse_voice_spec('af_bella:30,af_heart:40,af_bella:30'),
            [('af_bella', 3), ('af_heart', 2)])


class GradeRankUngradedTest(unittest.TestCase):
    def test_ungraded_sorts_last(self):
        self.assertEqual(v.grade_rank(''), 99)
        self.assertGreater(v.grade_rank(''), v.grade_rank('F'))


if __name__ == '__main__':
    unittest.main()
