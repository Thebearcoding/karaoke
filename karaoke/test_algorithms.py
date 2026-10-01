import unittest
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from align import activity, align_lines
from dsp import apply_masks, istft, masks_for, stft
from exact_stems import exact_split, reconstruction_report
from phonetic_align import FormantSynth, alignment_features, align_phonetic, templates
from subtitles import countdown_events, countdown_state, highlight_coverage_errors, lyric_cues, presentation_cues, read_ahead_state, timestamp, write_captions
from match_video_lyrics import caption_similarity, strongest_cluster
from anchors_from_asr import anchors_from_words
from three_stems import check_three
from lyric_progress import map_words, build as build_progress
from align_lyric_phonemes import guided_scores, protect_line_order, protect_units
from phoneme_progress import reading_parts, japanese_phone_groups, spoken_units
from align_repeated_lyrics import agree_onset, audible, reference_pairs
from align_lyrics_whisper import alignment_blocks, recover_changed_text, recover_skipped_repeats, stitch_blocks
from transcribe_lyrics import filter_silent_transcription
from word_onset_metrics import score_onsets, word_events


class SignalIntegrity(unittest.TestCase):
    def test_nested_three_stems_are_exact_after_storage(self):
        rng = np.random.default_rng(53)
        source = rng.normal(0, .15, (1024, 2)).astype('float32')
        vocal_estimate = rng.normal(0, .2, source.shape).astype('float32')
        source[0] = [0., -0.]
        vocal_estimate[0] = [0., -0.]
        vocals, music, _ = exact_split(source, vocal_estimate)
        lead_estimate = rng.normal(0, .1, source.shape).astype('float32')
        lead_estimate[0] = [1e-12, -1e-12]
        lead, backing, _ = exact_split(vocals, lead_estimate)
        with tempfile.TemporaryDirectory() as directory:
            restored = []
            for name, wave in [('lead', lead), ('backing', backing), ('music', music)]:
                path = Path(directory)/f'{name}.wav'
                sf.write(path, wave, 48000, subtype='DOUBLE')
                restored.append(sf.read(path, dtype='float64', always_2d=True)[0])
            checks = check_three(source, *restored)
        self.assertTrue(checks['vocal_first']['bitwise_equal'])
        np.testing.assert_array_equal(restored[2].view('uint64'), music.view('uint64'))

    def test_three_stem_report_distinguishes_addition_orders(self):
        source = np.array([[1e-30]], dtype='float64')
        checks = check_three(source, np.ones_like(source), -np.ones_like(source), source)
        self.assertTrue(checks['vocal_first']['bitwise_equal'])
        self.assertFalse(checks['music_plus_backing_first']['bitwise_equal'])

    def test_complementary_masters_are_bit_exact_after_storage(self):
        rng = np.random.default_rng(821)
        source = rng.normal(0, .15, (1024, 2)).astype('float32')
        estimate = rng.normal(0, .2, source.shape).astype('float32')
        source[0] = [0., -0.]
        estimate[0] = [.1, 0.]
        source[1] = [np.nextafter(np.float32(0), np.float32(1)), -1e-30]
        # Real model tails around -130 dBFS previously aborted the whole package.
        estimate[1] = [3.28574e-7, -1e-10]
        vocals, backing, repair = exact_split(source, estimate)
        self.assertGreater(repair['reassigned_sample_values'], 0)
        self.assertLessEqual(repair['maximum_vocal_change'], 1e-6)
        np.testing.assert_array_equal(vocals[2:], estimate[2:].astype('float64'))
        with tempfile.TemporaryDirectory() as directory:
            restored = []
            for name, wave in [('vocals', vocals), ('backing', backing)]:
                path = Path(directory)/f'{name}.wav'
                sf.write(path, wave, 48000, subtype='DOUBLE')
                restored.append(sf.read(path, dtype='float64', always_2d=True)[0])
            report = reconstruction_report(source, *restored)
        self.assertTrue(report['bitwise_equal'])
        self.assertEqual(report['different_sample_values'], 0)
        self.assertEqual(report['source_pcm_sha256'], report['recombined_pcm_sha256'])

    def test_exact_sum_does_not_silently_discard_audible_estimate(self):
        source = np.array([[1e-30]], dtype='float32')
        with self.assertRaisesRegex(ValueError, 'above the .* repair limit'):
            exact_split(source, np.array([[.1]], dtype='float32'))

    def test_exact_tail_repair_stays_bounded_and_respects_explicit_limit(self):
        source = np.array([[1e-30, -1e-30]], dtype='float32')
        estimate = np.array([[4.2e-6, -4.2e-6]], dtype='float32')
        vocals, backing, repair = exact_split(source, estimate)
        self.assertTrue(reconstruction_report(source, vocals, backing)['bitwise_equal'])
        self.assertLessEqual(repair['maximum_vocal_change'], 1e-5)
        with self.assertRaisesRegex(ValueError, 'above the .* repair limit'):
            exact_split(source, estimate*3)
        with self.assertRaisesRegex(ValueError, 'above the .* repair limit'):
            exact_split(source, estimate/10, repair_limit=1e-8)

    def test_exact_tail_repair_does_not_expand_budget_for_non_silent_source(self):
        source = np.array([[np.nextafter(1e-7, np.inf)]], dtype='float64')
        estimate = np.array([[4.2e-6]], dtype='float64')
        # A non-silent double-precision source can also suffer rounding. Its
        # normal repair cap must not become an excuse to discard real content.
        with self.assertRaisesRegex(ValueError, 'above the .* repair limit'):
            exact_split(source, estimate)

    def test_reconstruction_check_detects_single_value_change(self):
        source = np.array([[.25, -.5], [0., .125]], dtype='float32')
        vocals, backing, _ = exact_split(source, np.array([[.125, -.25], [.5, 0.]]))
        backing[1, 1] = np.nextafter(backing[1, 1], np.inf)
        report = reconstruction_report(source, vocals, backing)
        self.assertFalse(report['bitwise_equal'])
        self.assertEqual(report['different_sample_values'], 1)
        self.assertNotEqual(report['source_pcm_sha256'], report['recombined_pcm_sha256'])

    def test_roundtrip_preserves_length_and_samples(self):
        for length in [2048, 10001, 44123]:
            x=np.random.default_rng(length).normal(0,.03,(length,2)).astype('float32')
            y=istft(stft(x),len(x))
            self.assertEqual(y.shape,x.shape)
            np.testing.assert_allclose(y,x,atol=1e-7)

    def test_fixed_mixture_masks_preserve_attribution(self):
        rng=np.random.default_rng(7)
        v=rng.normal(0,.05,(10001,2)).astype('float32')
        b=rng.normal(0,.03,(10001,2)).astype('float32')
        z=stft(v+b)
        for config in [
            {'method':'spatial','strength':.99,'protection':.5},
            {'method':'spatial','strength':.99,'protection':.2,'coherence_power':1.5,
             'gain_power':2,'side_strength':1},
            {'method':'regression','seconds':.8,'bass_hz':180,'floor':.12},
        ]:
            mask=masks_for(z,config)
            out=istft(apply_masks(z,mask),len(v))
            parts=istft(apply_masks(stft(v),mask),len(v))+istft(apply_masks(stft(b),mask),len(b))
            np.testing.assert_allclose(out,parts,atol=1e-7)

    def test_center_cancel_removes_center_and_preserves_difference(self):
        x=np.random.default_rng(11).normal(0,.05,(10001,1)).astype('float32')
        center=np.repeat(x,2,axis=1)
        side=np.concatenate([x,-x],axis=1)
        np.testing.assert_allclose(istft(apply_masks(stft(center),(0,1)),len(x)),0,atol=1e-8)
        np.testing.assert_allclose(istft(apply_masks(stft(side),(0,1)),len(x)),side,atol=1e-7)


class SubtitleIntegrity(unittest.TestCase):
    def test_countdown_follows_sung_ends_including_gaps_inside_a_line(self):
        cues = [dict(start=0., end=20., units=[dict(start=0., end=1.)]),
                dict(start=11., end=30., units=[dict(start=12., end=13.), dict(start=24., end=25.)])]
        events = countdown_events(cues)
        self.assertEqual([(e['line'], e['unit'], e['start'], e['onset']) for e in events],
                         [(1, 0, 8., 12.), (1, 1, 20., 24.)])
        # A long note is not a long silence even when its onset was long ago.
        self.assertEqual(countdown_events([dict(units=[dict(start=0., end=9.),
                                                       dict(start=10., end=11.)])]), [])

    def test_countdown_prepares_first_entry_and_tracks_seeking(self):
        cues = [dict(units=[dict(start=10., end=12.)])]
        events = countdown_events(cues)
        for instant, count in [(5.99, 0), (6., 3), (7., 2), (8., 1),
                               (9., 0), (9.999, 0), (10., 0), (30., 0), (7.5, 2)]:
            self.assertEqual(countdown_state(events, instant)[1], count)
        self.assertIsNotNone(countdown_state(events, 9.)[0])
        self.assertIsNotNone(countdown_state(events, 9.999)[0])
        self.assertIsNone(countdown_state(events, 10.)[0])
        self.assertEqual(cues[0]['units'][0], dict(start=10., end=12.))

    def test_countdown_has_four_complete_frame_intervals_before_singing(self):
        events = countdown_events([dict(units=[dict(start=10., end=12.)])])
        frames = [countdown_state(events, 6.+i/30) for i in range(121)]
        for block, expected in enumerate((3, 2, 1, 0)):
            self.assertTrue(all(event is not None and count == expected
                                for event, count in frames[block*30:(block+1)*30]))
        self.assertEqual(frames[120], (None, 0))

    def test_japanese_context_accepts_tokenizer_space_normalization(self):
        for space in (' ', '　'):
            units = spoken_units('明日'+space+'晴れる', 'Japanese')
            self.assertEqual(''.join(u['text'] for u in units), '明日晴れる')
            self.assertEqual(next(u['first'] for u in units if u['text'].startswith('晴')), 3)

    def test_repeat_recovery_requires_an_exact_stable_reference(self):
        def cue(text, failed):
            return {'text': text, 'units': [{'text': text, 'timing_source':
                'prior_word_fallback' if failed else 'singing_phonemes'}]}
        cues = [cue('你會在哪裡', False), cue('你会在哪里', True),
                cue('你會在哪一天', True), cue('一起唱这首歌', True), cue('一起唱这首歌', True)]
        self.assertEqual(reference_pairs(cues, 'Chinese'), [(1, 0)])

    def test_repeat_recovery_does_not_accept_silence_or_disagreement(self):
        self.assertFalse(audible(np.zeros(200), .05))
        self.assertFalse(audible(np.full(200, 1e-6), .05))
        self.assertTrue(audible(np.full(200, .02), .05))
        for estimates, reference_spread in [([2.1], .01), ([2.1, 3.0], .01),
                                            ([2.1, float('nan')], .01), ([2.1, 2.12], .5)]:
            self.assertIsNone(agree_onset(estimates, reference_spread, 1., 4.))

    def test_repeat_recovery_can_correct_a_large_previous_error(self):
        self.assertEqual(agree_onset([3.1, 3.18], .04, 1., 4.), 3.14)
        self.assertIsNone(agree_onset([4.1, 4.18], .04, 1., 4.))

    def test_repeat_recovery_keeps_prior_timing_when_no_audio_path_exists(self):
        import json
        from unittest.mock import patch
        from align_repeated_lyrics import main

        cues = [dict(text='甲乙丙丁戊己', aligned_start=start, aligned_end=end,
                     units=[dict(text='甲乙丙丁戊己', first=0, stop=6, start=start, end=end,
                                 timing_source=source)])
                for start, end, source in [(1., 2., 'singing_phonemes'),
                                           (10., 20., 'prior_word_fallback')]]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'cues.json').write_text(json.dumps(cues))
            (folder/'jobs.json').write_text(json.dumps([dict(song='test', language='Chinese',
                audio='unused.wav', cues=str(folder/'cues.json'), output=str(folder/'output'))]))
            # Even identical features cannot fit a 12-second target into a
            # 3-second reference with the configured DTW step limits.
            with patch('sys.argv', ['align_repeated_lyrics', '--jobs', str(folder/'jobs.json'), '--device', 'cpu']), \
                 patch('align_repeated_lyrics.sf.read', return_value=(np.ones((400, 1)), 10)), \
                 patch('align_repeated_lyrics.audio_features', return_value={'pitch': np.ones((2, 4000))}), \
                 patch('align_repeated_lyrics.SingingAligner'), \
                 patch('align_repeated_lyrics.reference_units', return_value={}):
                main()
            self.assertEqual(json.loads((folder/'output/cues.json').read_text()), cues)
            attempt = json.loads((folder/'output/run.json').read_text())['attempts'][0]
            self.assertEqual(attempt['accepted_units'], 0)
            self.assertIn('No valid sub-sequence warping path', attempt['error'])

    def test_repeated_phonemes_keep_independent_time_windows(self):
        anchors = [{'first': 0, 'stop': 1, 'start': 0., 'end': .2},
                   {'first': 1, 'stop': 2, 'start': 1.2, 'end': 1.5}]
        ids, scores = guided_scores(np.array([0, 1, 0, 1, 0]), np.zeros((4, 2)),
            [-1, 0, -1, 1, -1], anchors, anchors, np.array([0., .15, 1.25, 1.5]))
        self.assertNotEqual(ids[1], ids[3])
        self.assertGreater(scores[0, ids[1]], scores[0, ids[3]])
        self.assertGreater(scores[-1, ids[3]], scores[-1, ids[1]])
        np.testing.assert_array_equal(scores[:, 0], 0.)

    def test_collapsed_character_retains_prior_timing_and_text(self):
        cue = {'text': '甲乙丙', 'aligned_start': 1., 'aligned_end': 2.4,
               'units': [dict(first=i, stop=i+1, text=c, start=t, end=t+.3)
                         for i,c,t in [(0,'甲',1.),(1,'乙',1.4),(2,'丙',1.8)]]}
        candidate = [dict(first=i, stop=i+1, text=c, start=a, end=b)
                     for i,c,a,b in [(0,'甲',.9,1.3),(1,'乙',1.35,1.35),(2,'丙',1.9,2.3)]]
        units, reasons = protect_units(cue, candidate, 'Chinese')
        self.assertEqual(''.join(u['text'] for u in units), '甲乙丙')
        self.assertEqual([u['start'] for u in units], [.9,1.4,1.9])
        self.assertEqual(units[1]['timing_source'], 'prior_word_fallback')
        self.assertTrue(all(u['end']>u['start'] for u in units))
        self.assertTrue(reasons)
        cue['aligned_end']=4.
        candidate[-1].update(start=3.,end=3.5)
        guarded,_=protect_units(cue,candidate,'Chinese')
        self.assertEqual(guarded[-1]['start'],1.8)
        self.assertEqual(guarded[-1]['timing_source'],'prior_word_fallback')

    def test_context_retry_splits_merged_characters_only_with_usable_audio_boundaries(self):
        import copy
        import importlib.util
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location('cjk_context', Path(__file__).parent/'experiments/cjk-onsets-v003/run.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        protocol = dict(context_padding_seconds=[.35, .6, 1.], maximum_spread_seconds=.08,
                        maximum_change_seconds=.4, minimum_relative_rms=.03)
        before = [dict(text='前', aligned_start=.8, aligned_end=.98,
                       units=[dict(text='前', first=0, stop=1, start=.8, end=.98)]),
                  dict(text='甲乙丙', aligned_start=1., aligned_end=2.7,
                       units=[dict(text='甲乙', first=0, stop=2, start=1., end=1.9),
                              dict(text='丙', first=2, stop=3, start=2., end=2.6)]),
                  dict(text='后', aligned_start=3., aligned_end=4.,
                       units=[dict(text='后', first=0, stop=1, start=3., end=4.)])]
        for mode in ('valid', 'silent', 'disagree', 'collapsed', 'outside', 'large_change',
                     'leading_punctuation_large_change', 'crossing'):
            with self.subTest(mode=mode):
                calls = []
                baseline = copy.deepcopy(before)
                if mode == 'leading_punctuation_large_change':
                    baseline[1]['text'] = '，甲乙丙'
                    baseline[1]['units'][0].update(text='，甲乙', stop=3)
                    baseline[1]['units'][1].update(first=3, stop=4)

                class Model:
                    def infer(self, audio, begin, units, context):
                        calls.append(context['text'])
                        times = dict(前=(.8, .98), 甲=(1.1, 1.4), 乙=(1.5, 1.9), 丙=(2., 2.6), 后=(3., 4.))
                        if mode == 'disagree' and len(calls) == 1:
                            times['乙'] = (1.7, 1.9)
                        if mode == 'collapsed':
                            times['乙'] = (1.5, 1.51)
                        if mode == 'outside':
                            times['乙'] = (2.8, 3.)
                        if mode in ('large_change', 'leading_punctuation_large_change'):
                            times['甲'] = (1.45, 1.49)
                        if mode == 'crossing':
                            times['甲'] = (.7, 1.4)
                        return dict(units=[dict(u, start=times[u['text']][0], end=times[u['text']][1]) for u in units])

                wave = np.zeros(44100*5) if mode == 'silent' else np.ones(44100*5)*.1
                parsed = lambda text, language: [dict(text=c, first=i, stop=i+1, phones=['a'])
                                                 for i, c in enumerate(text) if c.isalnum()]
                with patch.object(module, 'spoken_units', side_effect=parsed):
                    result, report = module.repair_character_groups(copy.deepcopy(baseline), baseline, wave, Model(), protocol)
                self.assertEqual(calls, ['前 '+baseline[1]['text']+' 后']*3)
                self.assertEqual(before[1]['units'][0]['text'], '甲乙')
                self.assertEqual([c['text'] for c in result], [c['text'] for c in baseline])
                if mode == 'valid':
                    self.assertEqual([u['start'] for u in result[1]['units']], [1.1, 1.5, 2.])
                    self.assertFalse(highlight_coverage_errors(result, 'Chinese'))
                    self.assertTrue(report[0]['groups'][0]['accepted'])
                else:
                    self.assertEqual([u['start'] for u in result[1]['units']], [1., 2.])
                    self.assertTrue(highlight_coverage_errors(result, 'Chinese'))
                    self.assertFalse(report[0]['groups'][0]['accepted'])

    def test_first_phoneme_keeps_its_onset_on_either_side_of_coarse_start(self):
        cue = {'text': '甲乙', 'aligned_start': 1., 'aligned_end': 2.,
               'units': [dict(first=0, stop=1, text='甲', start=1., end=1.4),
                         dict(first=1, stop=2, text='乙', start=1.5, end=2.)]}
        for onset in (.8, 1.2):
            candidate = [dict(cue['units'][0], start=onset), dict(cue['units'][1])]
            units, _ = protect_units(cue, candidate, 'Chinese')
            self.assertEqual(units[0]['start'], onset)

    def test_cross_model_recovery_preserves_foreign_words_and_rejects_unverified_changes(self):
        import copy
        from unittest.mock import patch
        from cjk_recovery import recover_groups
        protocol = dict(maximum_spread_seconds=.08, minimum_relative_rms=.03)
        original = [dict(text='go', aligned_start=.1, aligned_end=.4,
                         units=[dict(text='go', first=0, stop=2, start=.1, end=.4)]),
                    dict(text='甲乙 no', aligned_start=1.2, aligned_end=2.2,
                         units=[dict(text='甲乙 ', first=0, stop=3, start=1.2, end=1.8),
                                dict(text='no', first=3, stop=5, start=1.8, end=2.2)])]
        # The old group is 600 ms late, but the foreign word already has a
        # usable boundary and must neither be romanized as Chinese nor moved.
        for mode in ('valid', 'one_witness', 'different_text', 'disagree', 'collapsed',
                     'context_disagree', 'silent', 'crossing', 'invalid'):
            with self.subTest(mode=mode):
                calls = []

                class Model:
                    def infer(self, audio, begin, units, cue):
                        calls.append((begin, len(audio)))
                        self_units = [dict(u, start=start, end=end) for u, start, end in
                                      zip(units, [.6, .9], [.9, 1.3])]
                        if mode == 'collapsed':
                            self_units[0]['end'] = .61
                        if mode == 'context_disagree' and len(calls) == 1:
                            self_units[0]['start'] = .75
                        return dict(units=self_units)

                def witness(cue, wave):
                    values = [[dict(first=i, stop=i+1, text=text, start=start, end=end)
                               for i, text, start, end in [(0, '甲', .61, .88), (1, '乙', .91, 1.3)]]
                              for _ in range(2)]
                    if mode == 'one_witness':
                        values.pop()
                    elif mode == 'different_text':
                        values[1][1]['text'] = '丙'
                    elif mode == 'disagree':
                        values[1][1]['start'] = 1.2
                    elif mode == 'invalid':
                        values[0][0]['end'] = float('nan')
                    return values

                before = copy.deepcopy(original)
                if mode == 'crossing':
                    before[0]['units'][0]['end'] = .7
                wave = np.full(44100*5, 0. if mode == 'silent' else .1)

                def parsed(text, language):
                    self.assertEqual(text, '甲乙 ')
                    return [dict(first=i, stop=i+1, text=c, phones=['a']) for i, c in enumerate('甲乙')]

                with patch('cjk_recovery.spoken_units', side_effect=parsed):
                    actual, report = recover_groups(before, wave, Model(), witness, protocol)
                self.assertEqual(len(set(calls)), 3)
                self.assertEqual(actual[0], before[0])
                self.assertEqual(actual[1]['units'][-1], before[1]['units'][-1])
                if mode == 'valid':
                    self.assertEqual([u['start'] for u in actual[1]['units']], [.6, .9, 1.8])
                    self.assertFalse(highlight_coverage_errors(actual, 'Chinese'))
                    self.assertTrue(report[0]['accepted'])
                else:
                    self.assertEqual(actual, before)
                    self.assertFalse(report[0]['accepted'])

    def test_ctc_onsets_follow_phone_events_not_emission_width_or_equal_spacing(self):
        import torch
        from align_lyric_phonemes import ctc_word_units
        units = [dict(text=c, first=i, stop=i+1, phones=phones)
                 for i, (c, phones) in enumerate(zip('甲乙丙', [['a', 'b'], ['c'], ['d']]))]
        vocab = dict(SP=0, a=1, b=2, c=3, d=4)
        for middle in (40, 46):
            logits = torch.full((1, 100, 5), -12.)
            logits[:, :, 0] = 12.
            for token, frame in enumerate((6, 12, middle, 80), 1):
                logits[0, frame, 0] = -12.
                logits[0, frame, token] = 12.
            result = ctc_word_units(logits, units, vocab, 10., .01)
            self.assertEqual([u['start'] for u in result], [10.06, 10.+middle*.01, 10.8])
            self.assertEqual(result[0]['end'], result[1]['start'])
            self.assertEqual(result[1]['end'], result[2]['start'])
            self.assertAlmostEqual(result[1]['emission_end']-result[1]['start'], .01)
            self.assertEqual([u['text'] for u in result], list('甲乙丙'))
        logits[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid CTC'):
            ctc_word_units(logits, units, vocab, 10., .01)

    def test_internal_ctc_recovery_still_requires_independent_audio_agreement(self):
        from unittest.mock import patch
        from cjk_recovery import recover_groups
        original = [dict(text='甲乙丙', aligned_start=.5, aligned_end=2., units=[
            dict(text='甲乙', first=0, stop=2, start=.5, end=1.5),
            dict(text='丙', first=2, stop=3, start=1.5, end=2.)])]
        parsed = [dict(text=c, first=i, stop=i+1, phones=['a']) for i, c in enumerate('甲乙丙')]
        protocol = dict(maximum_spread_seconds=.08, minimum_relative_rms=.03,
                        context_padding_seconds=[.35, .6, 1.])
        for mode in ('agrees', 'disagrees', 'silent', 'missing_onset_audio',
                     'short_before_next', 'pulse_crosses_next'):
            with self.subTest(mode=mode):
                calls = []
                class Model:
                    def infer(self, audio, begin, units, cue, *, decoder='phonemes'):
                        calls.append(decoder)
                        result = [dict(u, start=s, end=e, emission_end=s+.01)
                                  for u, s, e in zip(units, [.5, 1., 1.5], [1., 1.5, 2.])]
                        if decoder == 'phonemes':
                            result[0]['end'] = .51
                        else:
                            for unit in result:
                                unit['timing_evidence'] = 'ctc_event'
                            if mode == 'short_before_next':
                                result[0]['end'] = 1.48
                                result[1].update(start=1.48, end=1.55, emission_end=1.49)
                            if mode == 'pulse_crosses_next':
                                result[1].update(end=1.55, emission_end=1.52)
                        return dict(units=result)
                def witness(cue, wave):
                    middle = 1.49 if mode == 'short_before_next' else 1.01
                    return [[dict(u, start=s, end=s+.1) for u, s in zip(parsed, [.51, middle, 1.51])],
                            [dict(u, start=s, end=s+.1) for u, s in zip(parsed, [.51, 1.3 if mode == 'disagrees' else middle, 1.51])]]
                wave = np.full(5*44100, 0. if mode == 'silent' else .1)
                if mode == 'missing_onset_audio':
                    wave[round(.5*44100):round(.535*44100)] = 0
                with patch('cjk_recovery.spoken_units', return_value=parsed):
                    result, report = recover_groups(original, wave, Model(), witness, protocol)
                self.assertEqual(calls, ['phonemes']*3+['ctc']*3)
                if mode == 'agrees':
                    self.assertEqual([u['start'] for u in result[0]['units']], [.5, 1., 1.5])
                    self.assertTrue(report[0]['accepted'])
                    self.assertEqual(result[0]['units'][0]['timing_source'], 'cross_model_ctc')
                    self.assertFalse(highlight_coverage_errors(result, 'Chinese'))
                else:
                    self.assertEqual(result, original)
                    self.assertFalse(report[0]['accepted'])

    def test_pronunciation_mapping_preserves_written_glyphs(self):
        self.assertEqual(reading_parts('感じ', 'かんじ'), [(0,1,'かん'),(1,2,'じ')])
        self.assertEqual(japanese_phone_groups('ー', previous_vowel='o'), [['o']])
        units = spoken_units('妳還好嗎', 'Chinese')
        self.assertEqual(''.join(u['text'] for u in units), '妳還好嗎')
        self.assertEqual(units[1]['reading'], 'hai')

    def test_progress_preserves_display_text_and_groups_zero_length_tokens(self):
        cue = {'text': '颳風，這天', 'start': 3.8, 'end': 6.2, 'aligned_start': 4., 'aligned_end': 6.}
        words = [{'word': '刮', 'start': 1., 'end': 1.}, {'word': '风，', 'start': 1., 'end': 1.5},
                 {'word': '这天', 'start': 1.7, 'end': 3.}]
        units = map_words(cue, words, 3., 'Chinese')
        self.assertEqual(units[0]['text'], '颳風，')
        self.assertEqual((units[0]['start'], units[0]['end']), (4., 4.5))
        self.assertEqual(cue['text'][units[1]['first']:units[1]['stop']], '這天')
        self.assertEqual(units[1]['start'], 4.7)

    def test_progress_keeps_contractions_when_model_splits_around_apostrophe(self):
        from subtitles import normalized
        for language in ('Chinese', 'English'):
            with self.subTest(language=language):
                cue = dict(text="Hello, I'm here.", start=1., end=5.)
                words = [dict(word=text, start=i+1., end=i+1.8)
                         for i, text in enumerate(['Hello', 'I', 'm', 'here'])]
                units = map_words(cue, words, 0., language)
                self.assertEqual(''.join(u['text'] for u in units), cue['text'])
                self.assertEqual(normalized(''.join(u['text'] for u in units)), normalized(cue['text']))
                self.assertEqual([u['start'] for u in units], [1., 2., 4.])
                self.assertEqual(units[1]['end'], 3.8)
                for unit in units:
                    self.assertEqual(unit['text'], cue['text'][unit['first']:unit['stop']])
                with self.assertRaisesRegex(ValueError, 'differs'):
                    map_words(cue, words[:-1], 0., language)

    def test_progress_ignores_alignment_context_and_falls_back_on_mismatch(self):
        cues = [{'text': '甲', 'start': 0., 'end': 1.}, {'text': '乙', 'start': 1., 'end': 2.}]
        first = {'words': [{'word': '甲', 'start': .1, 'end': .9}]}
        wrong = {'words': [{'word': '丙', 'start': 1.1, 'end': 1.9}]}
        blocks = [{'range': [0, 1], 'context': [0, 2], 'source_range': [0., 2.],
                   'result': {'segments': [first, wrong]}},
                  {'range': [1, 2], 'context': [0, 2], 'source_range': [0., 2.],
                   'result': {'segments': [first, wrong]}}]
        result, report = build_progress(cues, blocks, 'Chinese')
        self.assertEqual(len(result[0]['units']), 1)
        self.assertEqual(result[1]['units'], [])
        self.assertEqual(report['timed_lines'], 1)
        self.assertEqual(report['fallback_lines'][0]['line'], 2)

    def test_word_progress_coalesces_subwords_without_splitting_merged_onsets(self):
        cue = dict(text='唱 Moderato yo,no', start=0., end=6.)
        words = [dict(word=text, start=float(i), end=i+.7)
                 for i, text in enumerate(['唱', 'Moder', 'ato', 'yo', 'no'])]
        units = map_words(cue, words, 0., 'Chinese')
        self.assertEqual([u['text'] for u in units], ['唱 ', 'Moderato ', 'yo,', 'no'])
        self.assertEqual([u['start'] for u in units], [0., 1., 3., 4.])
        self.assertEqual(units[1]['end'], 2.7)
        self.assertFalse(highlight_coverage_errors([dict(cue, units=units)], 'Chinese'))
        cue = dict(text='two words', start=0., end=1.)
        units = map_words(cue, [dict(word='two words', start=0., end=.7)], 0., 'English')
        self.assertEqual(len(units), 1)
        self.assertTrue(highlight_coverage_errors([dict(cue, units=units)], 'English'))

    def test_traditional_chinese_pronunciation_uses_phrase_context(self):
        from unittest.mock import patch
        from phoneme_progress import chinese_units
        cases = {
            '透著光與著作': ['tou', 'zhe', 'guang', 'yu', 'zhu', 'zuo'],
            '怎么那么这么什么': ['zen', 'me', 'na', 'me', 'zhe', 'me', 'shen', 'me'],
            '怎麼那么這麼什么': ['zen', 'me', 'na', 'me', 'zhe', 'me', 'shen', 'me'],
            '幺妹看著幺雞': ['yao', 'mei', 'kan', 'zhe', 'yao', 'ji'],
        }
        dictionary = {word: [word] for words in cases.values() for word in words}
        for text, readings in cases.items():
            with self.subTest(text=text), patch('phoneme_progress.mandarin_dictionary', return_value=dictionary):
                units = chinese_units(text)
                self.assertEqual([u['reading'] for u in units], readings)
                self.assertEqual(''.join(u['text'] for u in units), text)
                self.assertEqual([text[u['first']:u['stop']] for u in units], list(text))

    def test_highlights_preserve_words_characters_and_japanese_reading_groups(self):
        for language, parts in [
            ('Chinese', ['你', 'Hello', '世', '界']),
            ('English', ["'café' ", "don't ", 'Ooh-ooh']),
            ('English', ['Cafe\u0301—', 'world']),
            ('Japanese', ['東京', 'で', 'Hello']),
        ]:
            with self.subTest(language=language, parts=parts):
                units, cursor = [], 0
                for part in parts:
                    units.append(dict(text=part, first=cursor, stop=cursor+len(part)))
                    cursor += len(part)
                cue = dict(text=''.join(parts), units=units)
                self.assertEqual(highlight_coverage_errors([cue], language), [])
                # The same complete text cannot be published as one highlight.
                cue['units'] = [dict(text=cue['text'], first=0, stop=len(cue['text']))]
                self.assertTrue(highlight_coverage_errors([cue], language))

    def test_highlight_validation_rejects_collapsed_split_and_invalid_spans(self):
        text = 'feel so,I feel so,so'
        cue = dict(text=text, units=[dict(text=text, first=0, stop=len(text),
                                         start=216.12, end=216.14)])
        for language in ('Chinese', 'English', 'Japanese'):
            with self.subTest(language=language):
                errors = highlight_coverage_errors([cue], language)
                self.assertEqual([e['text'] for e in errors], ['feel', 'so', 'I', 'feel', 'so', 'so'])
        cue = dict(text="don't", units=[dict(text="don'", first=0, stop=4),
                                       dict(text='t', first=4, stop=5)])
        self.assertTrue(highlight_coverage_errors([cue], 'English'))
        cue['units'] = []
        self.assertTrue(highlight_coverage_errors([cue], 'English'))
        for unit in (dict(text="don't", first=0, stop=6), dict(text='other', first=0, stop=5)):
            cue['units'] = [unit]
            self.assertEqual(highlight_coverage_errors([cue], 'English')[0]['reason'], 'invalid display span')

    def test_display_holds_preserve_acoustic_times_and_instrumental_gaps(self):
        cues = [{'text': '甲', 'start': 1., 'end': 2., 'status': 'candidate'},
                {'text': '乙', 'start': 1.8, 'end': 3., 'status': 'candidate'},
                {'text': '丙', 'start': 8., 'end': 9., 'status': 'candidate'}]
        display, report = presentation_cues(cues, 10.)
        self.assertEqual(cues[0]['end'], 2.)
        self.assertEqual(display[0]['aligned_end'], 2.)
        self.assertEqual(display[0]['end'], display[1]['start'])
        self.assertLess(display[1]['end'], display[2]['start'])
        self.assertTrue(report['overlap_adjustments'])
        self.assertTrue(all(0 <= c['start'] < c['end'] <= 10. for c in display))

    def test_display_policy_does_not_hide_large_alignment_errors(self):
        cues = [{'text': '甲', 'start': 1., 'end': 3., 'status': 'candidate'},
                {'text': '乙', 'start': 1.8, 'end': 4., 'status': 'candidate'}]
        with self.assertRaisesRegex(ValueError, 'needs review'):
            presentation_cues(cues, 10.)

    def test_read_ahead_keeps_current_line_while_previewing_next(self):
        cues = [{'start': 2.8, 'end': 6.3, 'aligned_start': 3., 'aligned_end': 6.},
                {'start': 6.3, 'end': 9.4, 'aligned_start': 6.5, 'aligned_end': 9.}]
        self.assertEqual(read_ahead_state(cues, 0.), (0, 1))
        self.assertEqual(read_ahead_state(cues, 3.), (0, 1))
        self.assertEqual(read_ahead_state(cues, 4.99), (0, 1))
        self.assertEqual(read_ahead_state(cues, 5.), (0, 1))
        self.assertEqual(read_ahead_state(cues, 6.49), (0, 1))
        self.assertEqual(read_ahead_state(cues, 6.5), (1, None))
        self.assertEqual(cues[0]['aligned_end'], 6.)

    def test_read_ahead_waits_through_gaps_and_supports_seeking(self):
        cues = [{'start': 2.8, 'end': 6.4, 'aligned_start': 3., 'aligned_end': 6.},
                {'start': 14.8, 'end': 18.4, 'aligned_start': 15., 'aligned_end': 18.}]
        self.assertEqual(read_ahead_state(cues, 12.), (0, 1))
        self.assertEqual(read_ahead_state(cues, 14.99), (0, 1))
        self.assertEqual(read_ahead_state(cues, 15.), (1, None))
        self.assertEqual(read_ahead_state(cues, 19.), (1, None))
        self.assertEqual(read_ahead_state(cues, 3.5), (0, 1))
        self.assertEqual(read_ahead_state([], 0.), (None, None))

    def test_read_ahead_advances_at_first_word_not_old_line_anchor(self):
        cues = [{'start': 2.8, 'aligned_start': 3., 'units': [{'start': 3.2}]},
                {'start': 5.8, 'aligned_start': 6., 'units': [{'start': 6.5}]}]
        self.assertEqual(read_ahead_state(cues, 6.2), (0, 1))
        self.assertEqual(read_ahead_state(cues, 6.5), (1, None))

    def test_asr_substitution_does_not_move_first_chorus_to_second(self):
        words = [{'word': letter, 'start': float(i), 'end': float(i+1)}
                 for i, letter in enumerate('abcdegabcdef')]
        cues = anchors_from_words(['abc', 'def', 'abc', 'def'], words, 'Japanese')
        self.assertEqual([(c['start'], c['end']) for c in cues],
                         [(0., 3.), (3., 5.), (6., 9.), (9., 12.)])

    def test_lyric_anchors_do_not_collect_scattered_letters_from_dialogue(self):
        for prefix, suffix in [('axbxcxdxexfx', ''), ('', 'xaxbxcxdxexf'),
                               ('axbxcxdxexfx', 'xaxbxcxdxexf')]:
            with self.subTest(prefix=prefix, suffix=suffix):
                words = [dict(word=char, start=float(i), end=float(i+1))
                         for i, char in enumerate(prefix+'abcdef'+suffix)]
                cues = anchors_from_words(['abc', 'def'], words, 'English')
                start = len(prefix)
                self.assertEqual([(c['start'], c['end']) for c in cues],
                                 [(start, start+3), (start+3, start+6)])
                self.assertTrue(all(c['matched_fraction'] == 1. for c in cues))

    def test_missing_asr_evidence_stays_unresolved(self):
        for words in ([], [dict(word='xyz', start=1., end=2.)]):
            with self.subTest(words=words):
                cues = anchors_from_words(['abc'], words, 'English', allow_missing=True)
                self.assertEqual(cues[0]['status'], 'unresolved_text')
                self.assertIsNone(cues[0]['start'])
                with self.assertRaisesRegex(ValueError, 'Insufficient ASR support'):
                    anchors_from_words(['abc'], words, 'English')

    def test_split_video_cards_match_one_lyric_line(self):
        self.assertEqual(caption_similarity('星光落在心里面', '星光落在'), 1.)
        self.assertEqual(caption_similarity('星光落在心里面', '心里面'), 1.)
        self.assertLess(caption_similarity('星光落在心里面', '广告文字'), .67)

    def test_video_cluster_ignores_a_distant_false_match(self):
        false = {'time': 10., 'similarity': 1.}
        real = [{'time': t, 'similarity': .8} for t in [23., 23.5, 24., 24.5]]
        self.assertEqual(strongest_cluster([false]+real), real)

    def test_repeated_lyrics_keep_their_recording_order(self):
        units = [{'text': '你好', 'start': .5, 'end': 1.2},
                 {'text': '世界', 'start': 1.3, 'end': 2.},
                 {'text': '你好', 'start': 4., 'end': 4.6},
                 {'text': '世界', 'start': 4.7, 'end': 5.5}]
        cues = lyric_cues(['你好，世界。', '你好，世界。'], units, 6.)
        self.assertEqual([(c['start'], c['end']) for c in cues], [(.5, 2.), (4., 5.5)])

    def test_missing_lyric_text_is_not_silently_skipped(self):
        with self.assertRaises(ValueError):
            lyric_cues(['你好世界'], [{'text': '你好', 'start': 0., 'end': 1.}], 2.)

    def test_caption_export_reports_invalid_timing_and_carries_milliseconds(self):
        cues = lyric_cues(['甲', '乙'], [{'text': '甲', 'start': 59.9996, 'end': 61.},
                                         {'text': '乙', 'start': 62., 'end': 62.}], 70.)
        self.assertEqual(timestamp(59.9996), '00:01:00,000')
        with tempfile.TemporaryDirectory() as directory:
            result = write_captions(cues, directory)
            self.assertEqual(result['unresolved_lines'], [2])
            self.assertIn('00:01:00,000 --> 00:01:01,000', (Path(directory)/'lyrics.srt').read_text())
            self.assertNotIn('乙', (Path(directory)/'lyrics.srt').read_text())


class WordOnsetScoring(unittest.TestCase):
    def test_risk_checker_keeps_weak_missing_words_and_measures_its_own_errors(self):
        from copy import deepcopy
        from word_onset_metrics import onset_risks, score_risks

        reference = [dict(id=i, text=t, start=float(i+1)) for i, t in enumerate(['a', 'b', 'c', 'd'])]
        predictions, verifier = deepcopy(reference), deepcopy(reference)
        predictions[0]['start'] += .4  # An independently detected real error.
        verifier[1]['start'] += .4     # The verifier itself is wrong: false alert.
        predictions[2]['start'] = verifier[2]['start'] = 8.  # Shared error is missed.
        verifier[3]['start'] = float('nan')   # Quiet word not recognized: retain and review it.
        before = deepcopy(predictions)
        risks = onset_risks(predictions, verifier)
        measured = score_risks(score_onsets(reference, predictions), risks)
        self.assertEqual(predictions, before)
        self.assertEqual([r['text'] for r in risks], ['a', 'b', 'c', 'd'])
        self.assertEqual(risks[-1]['reason'], 'missing_acoustic_evidence')
        self.assertIsNone(risks[-1]['verifier'])
        self.assertEqual(measured['true_positive'], 1)
        self.assertEqual(measured['false_negative'], 1)
        self.assertEqual(measured['false_positive'], 2)
        self.assertEqual(measured['recall'], .5)
        self.assertAlmostEqual(measured['precision'], 1/3)
        with self.assertRaises(ValueError):
            onset_risks(predictions, verifier[:-1])

    def test_chinese_merged_highlight_preserves_both_expected_characters(self):
        from word_onset_metrics import onset_risks

        events = word_events([dict(text='轻唱', units=[dict(first=0, stop=2, start=1.)])], 'Chinese')
        self.assertEqual([e['text'] for e in events], ['轻', '唱'])
        self.assertTrue(all(r['review'] for r in onset_risks(events, events)))
        with self.assertRaises(ValueError):
            word_events([dict(text='日本語', units=[])], 'Japanese')

    def test_missing_and_late_words_stay_in_accuracy_denominator(self):
        reference = [dict(id=i, text=str(i), start=float(i+1)) for i in range(4)]
        predictions = [dict(r) for r in reference]
        predictions[0]['start'] -= .05
        predictions[1]['start'] += .05
        predictions[2]['start'] += .3
        predictions[3]['start'] = None
        score = score_onsets(reference, predictions)['summary']
        self.assertEqual(score['pco']['50'], .5)
        self.assertEqual(score['missing'], 1)
        self.assertEqual(score['catastrophic_or_missing'], .5)
        self.assertFalse(score['passed'])

    def test_one_highlight_for_two_words_is_scored_twice(self):
        reference = [dict(id=0, text='hello', start=1.), dict(id=1, text='world', start=1.8)]
        for text in ('hello world', 'hello,world'):
            with self.subTest(text=text):
                events = word_events([dict(text=text, units=[dict(first=0, stop=11, start=1.)])])
                score = score_onsets(reference, events)['summary']
                self.assertEqual(score['units'], 2)
                self.assertEqual(score['pco']['50'], .5)
                self.assertEqual(score['merged_units'], 2)

    def test_frame_quantization_can_change_a_boundary_hit(self):
        reference = [dict(id=0, text='hello', start=1.)]
        predictions = [dict(id=0, text='hello', start=1.049)]
        self.assertEqual(score_onsets(reference, predictions)['summary']['pco']['50'], 1.)
        self.assertEqual(score_onsets(reference, predictions, fps=30)['summary']['pco']['50'], 0.)
        with self.assertRaises(ValueError):
            score_onsets(reference, [])


class StemQualityScoring(unittest.TestCase):
    def test_reconstruction_cannot_hide_a_lost_quiet_word_or_swapped_roles(self):
        from stem_metrics import score_stems

        rng = np.random.default_rng(51)
        refs = {name: rng.normal(0, .05, (3000, 2)) for name in ('lead', 'backing', 'music')}
        refs['lead'][1000:2000] *= .01  # A genuinely quiet reference is still scored.
        refs['lead'][2000:] = 0
        damaged = {name: wave.copy() for name, wave in refs.items()}
        damaged['music'][1000:2000] += damaged['lead'][1000:2000]
        damaged['lead'][1000:2000] = 0
        np.testing.assert_allclose(sum(damaged.values()), sum(refs.values()), atol=1e-15)
        scores = score_stems(refs, damaged, 1000)
        self.assertEqual(scores['summary']['roles']['lead']['active_windows'], 2)
        self.assertEqual(scores['windows'][1]['roles']['lead']['fixed_gain_db'], 0.)
        self.assertEqual(scores['windows'][1]['roles']['lead']['si_sdr_db'], -300.)
        self.assertLess(scores['summary']['roles']['lead']['active_window_p10']['fixed_gain_db'], 31.)
        swapped = dict(refs, lead=refs['backing'], backing=refs['lead'])
        scores = score_stems(refs, swapped, 1000)
        self.assertLess(scores['summary']['roles']['lead']['fixed_gain_db'], 0.)
        self.assertLess(scores['summary']['roles']['lead']['si_sir_db'], 0.)
        self.assertGreater(scores['summary']['roles']['lead']['silent_window_peak_output_dbfs'], -30.)

    def test_gain_loss_and_correlated_stems_are_not_hidden_by_scale_invariance(self):
        from stem_metrics import score_stems

        rng = np.random.default_rng(72)
        lead = rng.normal(0, .05, (1000, 2))
        refs = dict(lead=lead, backing=lead.copy())
        scores = score_stems(refs, dict(lead=lead*.5, backing=lead*1.5), 1000)
        row = scores['summary']['roles']['lead']
        self.assertAlmostEqual(row['fixed_gain_db'], 6.020599913)
        self.assertAlmostEqual(row['target_gain'], .5)
        self.assertGreater(row['si_sdr_db'], 120.)
        self.assertIsNone(row['si_sir_db'])
        self.assertFalse(scores['summary']['interference_identifiable'])
        with self.assertRaises(ValueError):
            score_stems(refs, dict(lead=lead[:-1], backing=lead), 1000)

    def test_quality_report_requires_references_for_accuracy_and_hashes_inputs(self):
        import json
        import tempfile
        from pathlib import Path
        from karaoke.tools.quality import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = [dict(id=0, text='quiet', start=1.)]
            for name in ('candidate', 'verifier', 'reference'):
                (root/f'{name}.json').write_text(json.dumps(events))
            case = dict(id='fixture', prediction='candidate.json', verifier='verifier.json')
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(dict(onsets=[case])))
            report = evaluate(manifest)
            self.assertIsNone(report['cases'][0]['accuracy'])
            self.assertFalse(report['cases'][0]['risks'][0]['review'])
            case.update(reference='reference.json', reference_provenance='independent fixture')
            manifest.write_text(json.dumps(dict(onsets=[case])))
            report = evaluate(manifest)
            self.assertEqual(report['cases'][0]['accuracy']['summary']['pco']['50'], 1.)
            self.assertEqual(len(report['input_sha256']), 4)
            wave = np.linspace(-.1, .1, 320).reshape(-1, 1)
            sf.write(root/'truth.wav', wave, 16000, subtype='DOUBLE')
            sf.write(root/'half.wav', wave*.5, 16000, subtype='DOUBLE')
            stem = dict(id='stem', reference_provenance='independent fixture',
                        references=dict(vocals='truth.wav'), predictions=dict(vocals='half.wav'))
            manifest.write_text(json.dumps(dict(stems=[stem])))
            report = evaluate(manifest)
            self.assertAlmostEqual(report['cases'][0]['scores']['summary']['roles']['vocals']['target_gain'], .5)
            sf.write(root/'half.wav', wave*.5, 8000, subtype='DOUBLE')
            with self.assertRaises(ValueError):
                evaluate(manifest)


class AlignmentIntegrity(unittest.TestCase):
    def test_untimed_line_recovery_retains_measured_times_and_rejects_bad_candidates(self):
        from copy import deepcopy
        from unittest.mock import Mock
        from align_lyrics_whisper import recover_unresolved_block

        lines = ['opening', 'context', 'long-vocalization']
        anchors = [dict(status='candidate', start=1., end=2.),
                   dict(status='candidate', start=8., end=9.), dict(status='unresolved_text')]
        def segment(text, start, end):
            return dict(text=text, start=start, end=end, words=[dict(word=text, start=start, end=end)])
        original = dict(range=[2, 3], context=[1, 3], source_range=[7.4, 20.], result=dict(segments=[
            segment(lines[1], .7, 1.5), segment(lines[2], 12.6, 12.6)]))
        candidate = dict(segments=[segment(lines[0], .7, 1.7), segment(lines[1], 7.7, 8.5),
                                  segment(lines[2], 10.9, 18.7)])
        for failure in (None, 'text', 'collapsed', 'nan', 'outside', 'order'):
            with self.subTest(failure=failure):
                prediction = deepcopy(candidate)
                word = prediction['segments'][-1]['words'][0]
                if failure == 'text': word['word'] = 'rewritten'
                if failure == 'collapsed': word['end'] = word['start']
                if failure == 'nan': word['start'] = float('nan')
                if failure == 'outside': word['end'] = 30.
                if failure == 'order': word['start'] = .1
                align = Mock(return_value=prediction)
                result = recover_unresolved_block(original, lines, anchors, 20., align)
                self.assertEqual(align.call_count, 1)
                self.assertEqual(align.call_args.args, (.4, 20., '\n'.join(lines)))
                report = result['unresolved_recovery']
                self.assertEqual(report['original_result'], original['result'])
                self.assertEqual(report['accepted'], failure is None)
                if failure is None:
                    self.assertEqual(result['context'], [0, 3])
                    self.assertEqual(result['source_range'], [.4, 20.])
                    self.assertEqual(result['result'], prediction)  # No clamping of the long word.
                else:
                    self.assertEqual(result['result'], original['result'])
                    self.assertEqual(result['source_range'], original['source_range'])
        # A short zero-length consonant in an otherwise timed line is handled
        # by the later phoneme stage; do not rerun every valid phrase.
        valid = deepcopy(original)
        valid['result']['segments'][-1]['words'] = [dict(word='long', start=8., end=8.),
                                                   dict(word='vocalization', start=8., end=9.)]
        align = Mock()
        self.assertEqual(recover_unresolved_block(valid, lines, anchors, 20., align), valid)
        align.assert_not_called()

    def test_singing_phonemes_keep_hyphenated_syllables_separate(self):
        import os
        from unittest.mock import patch
        import espeakng_loader
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        from phonemizer.separator import Separator
        from align_english_onsets import EnglishOnsetAligner

        with patch.dict(os.environ, ESPEAK_DATA_PATH=espeakng_loader.get_data_path()):
            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            aligner = object.__new__(EnglishOnsetAligner)
            aligner.method = 'singing'
            aligner.phonemizer = EspeakBackend('en-us')
            aligner.separator = Separator(phone=';', word=None)
            aligner.vocab, aligner.space = {'uː': 1, 'h': 2, 'm': 4}, 3
            targets, spans = aligner.tokenize(['Ooh-ooh-ooh-ooh', 'Ooh', 'Mmm'])
        # The written h is silent: deleting the hyphens invents consonants
        # between the four vowels and changes the forced-alignment target.
        self.assertEqual(targets, [1, 1, 1, 1, 3, 1, 3, 4])
        self.assertEqual(spans, [(0, 4), (5, 6), (7, 8)])

    def test_global_english_alignment_retries_only_out_of_context_blocks(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock
        from align_english_onsets import align_job
        from onset_consensus import context_scope_violations

        lines = ['opening line', 'second line', 'later line']
        anchors = [dict(text=text, status='candidate', start=start, end=start+1.4)
                   for text, start in zip(lines, (10., 12., 25.))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root/'audio.wav', np.zeros(30*16000), 16000)
            (root/'lyrics.json').write_text(json.dumps(dict(language='English', lines=lines)))
            (root/'anchors.json').write_text(json.dumps(anchors))
            for mode in ('repair', 'failed_retry', 'already_valid'):
                with self.subTest(mode=mode):
                    def align(audio, words):
                        if len(audio) == 30*16000:
                            first = 10.17 if mode == 'already_valid' else 1.
                            starts = [first, first+.5, 12.17, 12.67, 25.17, 25.67]
                        else:
                            # These are measured offsets in the supplied crop,
                            # deliberately different from the coarse anchors.
                            starts = [.77, 1.27, 2.77, 3.27, 15.77, 16.27]
                            if mode == 'failed_retry':
                                starts[0] = -2.
                        return [dict(text=w, start=t, end=t+.25, confidence=.9)
                                for w, t in zip(words, starts)], None, .02

                    model = SimpleNamespace(method='singing', device='cpu', onset_ratio=.02,
                                            model_identity={}, align=Mock(side_effect=align))
                    output = root/mode
                    report = align_job(model, dict(audio=str(root/'audio.wav'), lyrics=str(root/'lyrics.json'),
                        anchors=str(root/'anchors.json'), output=str(output)), full_song=True)
                    cues = json.loads((output/'cues.json').read_text())
                    self.assertEqual(model.align.call_count, 1 if mode == 'already_valid' else 2)
                    self.assertAlmostEqual(cues[2]['start'], 25.17)
                    if mode == 'already_valid':
                        self.assertEqual(report['context_retries'], [])
                    else:
                        retry = report['context_retries'][0]
                        self.assertEqual(retry['core'], [0, 2])
                        self.assertEqual(retry['original_cues'][0]['start'], 1.)
                        self.assertEqual(retry['accepted'], mode == 'repair')
                    if mode == 'failed_retry':
                        self.assertTrue(report['failures'])
                        self.assertTrue(context_scope_violations(cues, report['scope_blocks']))
                    else:
                        self.assertFalse(report['failures'])
                        self.assertFalse(context_scope_violations(cues, report['scope_blocks']))
                        self.assertAlmostEqual(cues[0]['start'], 10.17)

    def test_late_coarse_boundary_does_not_cut_off_a_sustained_syllable(self):
        import json
        from types import SimpleNamespace
        from align_english_onsets import align_job

        def align(audio, words):
            if len(audio) == 20*16000:
                starts = [1.1, 6.8, 12.1]
            elif words[0] == 'first':
                starts = [.7, 6.4, 11.7]
            else:  # The short crop has already lost the held syllable's onset.
                starts = [.1, 4.7]
            return [dict(text=w, start=t, end=t+.7, confidence=.9)
                    for w, t in zip(words, starts)], None, .02

        model = SimpleNamespace(method='singing', device='cpu', onset_ratio=.02, model_identity={}, align=align)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lines = ['first', 'held-vowel', 'last']
            sf.write(root/'audio.wav', np.zeros(20*16000), 16000)
            (root/'lyrics.json').write_text(json.dumps(dict(language='English', lines=lines)))
            (root/'anchors.json').write_text(json.dumps([dict(text=text, status='candidate', start=t, end=t+1.)
                                                       for text, t in zip(lines, [1., 8., 12.])]))
            report = align_job(model, dict(audio=str(root/'audio.wav'), lyrics=str(root/'lyrics.json'),
                anchors=str(root/'anchors.json'), output=str(root/'output')), full_song=True)
            self.assertFalse(report['failures'])
            cues = json.loads((root/'output/cues.json').read_text())
            np.testing.assert_allclose([c['start'] for c in cues], [1.1, 6.8, 12.1])

    def test_english_context_crossings_expand_without_moving_acceptance_bounds(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from align_english_onsets import align_job

        lines = ['first', 'second', 'third', 'fourth', 'fifth', 'last']
        anchors = [dict(text=s, status='candidate', start=t, end=t+1)
                   for s, t in zip(lines, (1., 4., 7., 10., 13., 16.))]
        scope = [(i, i+1, i, i+1, 0., 30.) for i in range(6)]
        scope[3] = (3, 4, 3, 4, 9.4, 11.4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root/'audio.wav', np.zeros(30*16000), 16000)
            (root/'lyrics.json').write_text(json.dumps(dict(language='English', lines=lines)))
            (root/'anchors.json').write_text(json.dumps(anchors))
            for mode in ('repair', 'outside_scope', 'new_conflict'):
                with self.subTest(mode=mode):
                    def align(audio, words):
                        times = {6: [1., 4., 9., 18., 20., 23.],
                                 3: [.1, 1.1, 6.6], 4: [.1, .4, 6.6, 9.6],
                                 5: [.6, 3.6, 6.6, 9.6, 12.6]}[len(words)].copy()
                        if len(words) == 5:
                            if mode == 'outside_scope': times[3] = 18.
                            if mode == 'new_conflict': times[2] = 2.
                        return [dict(text=w, start=t, end=t+.2, confidence=.9)
                                for w, t in zip(words, times)], None, .02
                    model = SimpleNamespace(method='singing', device='cpu', onset_ratio=.02,
                                            model_identity={}, align=Mock(side_effect=align))
                    with patch('align_english_onsets.context_blocks', return_value=list(scope)):
                        report = align_job(model, dict(audio=str(root/'audio.wav'), lyrics=str(root/'lyrics.json'),
                            anchors=str(root/'anchors.json'), output=str(root/mode)), full_song=True)
                    result = json.loads((root/mode/'cues.json').read_text())
                    self.assertEqual(model.align.call_count, 4)
                    self.assertEqual(len(report['boundary_retries']), 2)
                    self.assertFalse(report['boundary_retries'][0]['accepted'])
                    self.assertEqual(report['boundary_retries'][1]['accepted'], mode == 'repair')
                    self.assertEqual(report['scope_blocks'][3], (3, 4, 2, 5, 6.4, 14.6))
                    expected = [1., 4., 7., 10., 20., 23.] if mode == 'repair' else [1., 4., 9., 7.5, 20., 23.]
                    np.testing.assert_allclose([c['start'] for c in result], expected)
                    self.assertEqual(bool(report['failures']), mode != 'repair')

    def test_english_phoneme_alignment_keeps_punctuation_out_of_spoken_words(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock
        from align_english_onsets import align_job
        from onset_consensus import blend_cues

        lines = ['“ Come … back ! ”', "Don't go…", 'Feel so,so-sorry!']
        expected = ['Come', 'back', "Don't", 'go', 'Feel', 'so', 'so-sorry']
        aligned = [dict(text=word, start=float(i+1), end=i+1.5, confidence=.9)
                   for i, word in enumerate(expected)]
        aligner = SimpleNamespace(method='singing', device='cpu', onset_ratio=.02,
            model_identity={}, align=Mock(return_value=(aligned, None, .02)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root/'audio.wav', np.zeros(9*16000), 16000)
            (root/'lyrics.json').write_text(json.dumps(dict(language='English', lines=lines)))
            job = dict(audio=str(root/'audio.wav'), lyrics=str(root/'lyrics.json'), output=str(root/'onsets'))
            report = align_job(aligner, job, full_song=True)
            self.assertFalse(report['failures'])
            self.assertEqual(aligner.align.call_args.args[1], expected)
            cues = json.loads((root/'onsets/cues.json').read_text())
            self.assertEqual([''.join(u['text'] for u in c['units']) for c in cues], lines)
            self.assertEqual([u['start'] for c in cues for u in c['units']], [1., 2., 3., 4., 5., 6., 7.])
            self.assertFalse(highlight_coverage_errors(cues, 'English'))
            blend_cues(cues, cues, max_disagreement=.2, primary_weight=.75)

            aligner.align.reset_mock()
            (root/'lyrics.json').write_text(json.dumps(dict(language='English', lines=['…'])))
            report = align_job(aligner, dict(job, output=str(root/'invalid')), full_song=True)
            self.assertIn('alignable text', report['failures'][0]['error'])
            aligner.align.assert_not_called()

    def test_punctuation_only_row_cannot_shift_following_line_ownership(self):
        lines = ['first', '”', 'last']
        raw = [dict(range=[0, 3], context=[0, 3], source_range=[0., 5.], result=dict(segments=[
            dict(text=text, words=[dict(word=text, start=float(i), end=i+.5)])
            for i, text in enumerate(lines)]))]
        with self.assertRaisesRegex(ValueError, 'alignable text'):
            stitch_blocks(lines, raw, 5., 'English')

    def test_changed_alignment_text_requires_a_complete_acoustic_retry(self):
        from copy import deepcopy
        from unittest.mock import Mock

        words = [dict(word=word, start=float(i), end=i+.5)
                 for i, word in enumerate(['keep', 'these', 'words'])]
        candidate = dict(segments=[dict(words=words, text='keep these words')])
        original = deepcopy(candidate)
        original['segments'][0]['words'].insert(2, dict(word='these', start=4., end=4.))
        original['segments'][0]['words'][-1].update(start=4., end=4.)
        realign = Mock(return_value=candidate)
        recovered, report = recover_changed_text(original, ['keep these words'], 4., realign)
        self.assertEqual(recovered, candidate)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['original_result'], original)
        realign.assert_called_once_with()
        realign.reset_mock()
        self.assertEqual(recover_changed_text(candidate, ['keep these words'], 4., realign), (candidate, None))
        realign.assert_not_called()

        for problem in ('missing', 'collapsed', 'outside', 'unordered', 'nonfinite', 'wrong_line', 'empty'):
            with self.subTest(problem=problem):
                retry = deepcopy(candidate)
                row = retry['segments'][0]['words']
                if problem == 'missing':
                    row.pop()
                elif problem == 'collapsed':
                    row[-1].update(start=4., end=4.)
                elif problem == 'outside':
                    row[-1]['end'] = 5.
                elif problem == 'unordered':
                    row[-1]['start'] = 0.
                elif problem == 'nonfinite':
                    row[-1]['end'] = float('nan')
                elif problem == 'wrong_line':
                    retry['segments'] = [dict(words=row[:1]), dict(words=row[1:])]
                else:
                    retry = None
                recovered, report = recover_changed_text(original, ['keep these words'], 4., lambda: retry)
                self.assertEqual(recovered, original)
                self.assertFalse(report['accepted'])

    def test_rejected_alignment_retains_raw_evidence_without_publishing_captions(self):
        import json
        from unittest.mock import Mock, patch
        from align_lyrics_whisper import main

        result = dict(segments=[dict(text='extra word', start=1., end=2., words=[
            dict(word='extra word', start=1., end=2., probability=.8)])])
        model = Mock()
        model._buffers = {'alignment_heads': None}
        model.align.return_value.to_dict.return_value = result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root/'audio.wav', np.zeros(3*16000), 16000)
            (root/'lyrics.json').write_text(json.dumps(dict(language='English', lines=['word'])))
            (root/'jobs.json').write_text(json.dumps([dict(audio=str(root/'audio.wav'),
                lyrics=str(root/'lyrics.json'), output=str(root/'alignment'))]))
            with patch('stable_whisper.load_model', return_value=model), patch('sys.argv',
                    ['align', '--jobs', str(root/'jobs.json'), '--device', 'cpu']):
                with self.assertRaisesRegex(ValueError, 'exactly cover'):
                    main()
            self.assertEqual(json.loads((root/'alignment/raw.json').read_text())[0]['result'], result)
            self.assertEqual(json.loads((root/'alignment/units.json').read_text())[0]['text'], 'extra word')
            self.assertFalse((root/'alignment/cues.json').exists())

    def test_skipped_refrain_requires_two_acoustic_windows_inside_the_gap(self):
        from copy import deepcopy

        text = 'run into the light'

        def segment(start):
            return dict(text=text, start=start, end=start+1.4, words=[
                dict(word=word, start=start+i*.4, end=start+i*.4+.2)
                for i, word in enumerate(text.split())])

        missing = segment(10.)
        for word in missing['words']:
            word.update(start=10., end=10.)
        missing.update(start=10., end=10.)
        original = dict(segments=[segment(1.), segment(6.), missing])
        saved = deepcopy(original)
        calls = []

        def align(begin, end, requested):
            calls.append((begin, end, requested))
            absolute = [segment(1.), segment(3.5), segment(6.)] if '\n' in requested else [segment(3.5)]
            for s in absolute:
                s['start'] -= begin
                s['end'] -= begin
                for word in s['words']:
                    word['start'] -= begin
                    word['end'] -= begin
            return dict(segments=absolute)

        repaired, reports = recover_skipped_repeats(original, 10., align)
        self.assertTrue(reports[0]['accepted'])
        self.assertEqual(original, saved)
        self.assertEqual(len(calls), 2)
        self.assertEqual(repaired['segments'][1], segment(3.5))
        self.assertEqual(repaired['segments'][2], original['segments'][1])
        raw = [dict(range=[0, 3], context=[0, 3], source_range=[0., 10.], result=repaired)]
        cues, stitched, _ = stitch_blocks([text]*3, raw, 10., 'English')
        presentation_cues(cues, 10., lead=0, tail=0, bridge_gap=0)
        _, mapping = build_progress(cues, stitched, 'English')
        self.assertFalse(mapping['fallback_lines'])
        calls.clear()
        self.assertEqual(recover_skipped_repeats(repaired, 10., align)[1], [])
        self.assertFalse(calls)

        # Timing disagreement, matching only a clipped first word, an empty
        # prediction, and a phrase outside the gap must all leave the failure.
        for problem in ('disagreement', 'clipped', 'empty', 'outside'):
            with self.subTest(problem=problem):
                def unreliable(begin, end, requested):
                    prediction = align(begin, end, requested)
                    if '\n' not in requested:
                        words = prediction['segments'][0]['words']
                        if problem == 'empty':
                            prediction['segments'][0]['words'] = []
                        else:
                            for i, word in enumerate(words):
                                if problem == 'clipped' and i == 0:
                                    word['start'] = 0.
                                else:
                                    word['start'] += .5
                                    word['end'] += .5
                    elif problem == 'outside':
                        prediction['segments'][1]['words'][0]['start'] = 0.
                    return prediction

                unchanged, rejected = recover_skipped_repeats(original, 10., unreliable)
                self.assertEqual(unchanged, original)
                self.assertFalse(rejected[0]['accepted'])

    def test_crossing_can_use_context_from_the_third_neighboring_window(self):
        lines = ['first', 'middle', 'last']

        def segment(text, start, end):
            return dict(text=text, words=[dict(word=text, start=start, end=end)])

        raw = [dict(range=[0, 1], context=[0, 2], source_range=[0., 6.], result=dict(segments=[
                   segment('first', 0., 1.), segment('middle', 1.2, 2.)])),
               dict(range=[1, 2], context=[1, 2], source_range=[0., 6.], result=dict(segments=[
                   segment('middle', 1.2, 5.)])),
               dict(range=[2, 3], context=[1, 3], source_range=[0., 6.], result=dict(segments=[
                   segment('middle', .2, 2.), segment('last', 2.5, 3.5)]))]
        cues, stitched, repairs = stitch_blocks(lines, raw, 6., 'English')
        presentation_cues(cues, 6., lead=0, tail=0, bridge_gap=0)
        words, report = build_progress(cues, stitched, 'English')
        self.assertFalse(report['fallback_lines'])
        self.assertEqual([(c['start'], c['end']) for c in cues], [(0., 1.), (1.2, 2.), (2.5, 3.5)])
        self.assertEqual([c['units'][0]['start'] for c in words], [0., 1.2, 2.5])
        self.assertEqual(repairs[0]['selected_block'], 0)
        self.assertEqual(repairs[0]['lines'], [2, 2])
        self.assertEqual(raw[1]['result']['segments'][0]['words'][0]['end'], 5.)

    def test_crossing_block_boundary_uses_one_complete_joint_prediction(self):
        lines = ['first tail', 'second line', 'last phrase']

        def segment(text, start, middle, end):
            first, last = text.split()
            return dict(text=text, words=[dict(word=first, start=start, end=middle),
                                         dict(word=last, start=middle, end=end)])

        raw = [dict(range=[0, 1], context=[0, 2], source_range=[2., 7.], result=dict(segments=[
                   segment(lines[0], 0., 1., 1.2), segment(lines[1], 1.2, 1.6, 2.)])),
               dict(range=[1, 3], context=[1, 3], source_range=[3., 8.], result=dict(segments=[
                   segment(lines[1], -.01, .4, .9), segment(lines[2], 2., 3., 4.)]))]
        cues, stitched, repairs = stitch_blocks(lines, raw, 10., 'English')
        words, report = build_progress(cues, stitched, 'English')
        self.assertFalse(report['fallback_lines'])
        self.assertEqual([[u['start'] for u in c['units']] for c in words],
                         [[2., 3.], [3.2, 3.6], [5., 6.]])
        self.assertEqual([b['range'] for b in stitched], [[0, 2], [2, 3]])
        self.assertEqual([b['range'] for b in raw], [[0, 1], [1, 3]])
        self.assertEqual(repairs[0]['lines'], [1, 2])
        self.assertFalse(protect_line_order(words, words)[1])
        self.assertEqual(stitch_blocks(lines, stitched, 10., 'English')[2], [])

        # Reject new conflicts in either word onsets or line ends, even when
        # the replacement resolves the original crossing.
        for middle, end in [(3.5, 4.), (1.6, 5.)]:
            with self.subTest(middle=middle, end=end):
                raw[0]['result']['segments'][1] = segment(lines[1], 1.2, middle, end)
                cues, unchanged, repairs = stitch_blocks(lines, raw, 10., 'English')
                self.assertEqual(repairs, [])
                words, _ = build_progress(cues, unchanged, 'English')
                with self.assertRaisesRegex(ValueError, 'Coarse lyric onsets'):
                    protect_line_order(words, words)

    def test_unresolved_crossing_realigns_with_context_but_changes_only_core_lines(self):
        from copy import deepcopy
        from unittest.mock import Mock

        lines = ['left', 'context', 'first', 'second', 'after', 'tail']
        old_times = [(1., 1.5), (2., 2.5), (3., 5.4), (4.7, 6.4), (7., 7.5), (8., 8.5)]
        def result(times, offset=0.):
            return dict(segments=[dict(text=text, words=[dict(word=text, start=a-offset, end=b-offset)])
                                  for text, (a, b) in zip(lines, times)])
        raw = [dict(range=[i, i+1], context=[i, i+1], source_range=[0., 10.],
                    result=dict(segments=[s])) for i, s in enumerate(result(old_times)['segments'])]
        original = deepcopy(raw)
        new_times = [(1.1, 1.6), (2.2, 2.7), (3.2, 4.8), (4.9, 6.2), (7.1, 7.6), (8.1, 8.6)]
        for failure in (None, 'text', 'crossing', 'neighbour', 'outside_audio', 'missing'):
            with self.subTest(failure=failure):
                candidate = result(new_times, .4)
                if failure == 'text': candidate['segments'][2]['words'][0]['word'] = 'changed'
                if failure == 'crossing': candidate['segments'][3]['words'][0]['start'] = 3.6
                if failure == 'neighbour': candidate['segments'][2]['words'][0]['start'] = 1.6
                if failure == 'outside_audio': candidate['segments'][-1]['words'][0]['end'] = 9.
                if failure == 'missing': candidate = None
                align = Mock(return_value=candidate)
                cues, stitched, repairs = stitch_blocks(lines, raw, 10., 'English', align)
                align.assert_called_once()
                self.assertEqual(align.call_args.args, (.4, 9.1, '\n'.join(lines)))
                self.assertEqual(raw, original)
                self.assertEqual(repairs[-1]['accepted'], failure is None)
                self.assertEqual(repairs[-1]['result'], candidate)
                if failure is None:
                    expected = old_times[:2]+new_times[2:4]+old_times[4:]
                    self.assertEqual([(c['start'], c['end']) for c in cues], expected)
                    words, report = build_progress(cues, stitched, 'English')
                    self.assertFalse(report['fallback_lines'])
                    self.assertEqual([c['units'][0]['start'] for c in words], [a for a, _ in expected])
                    presentation_cues(cues, 10., lead=0, tail=0, bridge_gap=0)
                    again = Mock()
                    stitch_blocks(lines, stitched, 10., 'English', again)
                    again.assert_not_called()
                else:
                    self.assertEqual([(c['start'], c['end']) for c in cues], old_times)
                    with self.assertRaisesRegex(ValueError, 'overlap'):
                        presentation_cues(cues, 10., lead=0, tail=0, bridge_gap=0)

    def test_asr_silence_filter_keeps_quiet_speech_and_nearby_collapsed_words(self):
        audio = np.zeros(5*16000, dtype='float32')
        audio[16000:2*16000] = 1e-7
        spoken = dict(start=1., end=2., text='Soft singing', word='soft')
        nearby = dict(start=2.1, end=2.1, text='', word='ending')
        hallucinated = dict(start=4., end=4.1, text='Invented', word='invented')
        outside = dict(start=8., end=9., text='Outside', word='outside')
        data = {kind: [spoken, nearby, hallucinated, outside] for kind in ('segments', 'words')}
        filtered, rejected = filter_silent_transcription(data, audio, 16000)
        for kind in data:
            self.assertEqual(filtered[kind], [spoken, nearby])
            self.assertEqual(len(data[kind]), 4)
        self.assertEqual(filtered['text'], 'Soft singing')
        self.assertEqual([r['reason'] for r in rejected],
                         ['digital_silence', 'outside_audio', 'digital_silence', 'outside_audio'])

    def test_quiet_opening_and_closing_lines_keep_their_audio(self):
        audio = np.zeros(10*16000, dtype='float32')
        audio[16000:2*16000] = .001
        audio[4*16000:6*16000] = .1
        audio[8*16000:9*16000] = .001
        anchors = [dict(start=1., end=2., status='candidate'),
                   dict(start=8., end=9., status='candidate')]
        blocks = alignment_blocks(anchors, audio)
        self.assertEqual(len(blocks), 2)
        for block, anchor in zip(blocks, anchors):
            begin, end = block[-2:]
            self.assertLess(begin, anchor['start'])
            self.assertGreaterEqual(end, anchor['end'])
            self.assertGreater(len(audio[round(begin*16000):round(end*16000)]), 0)

    def test_out_of_audio_anchor_fails_before_model_inference(self):
        with self.assertRaisesRegex(ValueError, 'Empty alignment audio.*1-1'):
            alignment_blocks([dict(start=20., end=21., status='candidate')], np.zeros(16000))

    def test_silence_does_not_invent_word_times(self):
        audio=np.zeros((32000,2),dtype='float32')
        times,cue=activity(audio,16000)
        result=align_lines(times,cue,['a silent recording'])
        self.assertIsNone(result[0]['start'])
        model=templates(['a silent recording'])
        result=align_phonetic(*alignment_features(audio,16000),model,['a silent recording'])
        self.assertIsNone(result[0]['start'])

    def test_phonetic_path_tracks_known_inserted_pause(self):
        lines=['we will sing together','and start again']
        synth=FormantSynth()
        try:
            a,_=synth.speak(lines[0]);b,_=synth.speak(lines[1]);rate=synth.rate
        finally:
            synth.close()
        audio=np.concatenate([np.zeros(rate),a,np.zeros(2*rate),b,np.zeros(rate)]).astype('float32')
        audio=np.repeat(audio[:,None],2,axis=1)
        result=align_phonetic(*alignment_features(audio,rate),templates(lines),lines,
                             activity_weight=1.2,duration_weight=.1)
        expected=[1,3+len(a)/rate]
        for row,start in zip(result,expected):
            self.assertLess(abs(row['start']-start),.15)
            self.assertGreater(row['end'],row['start'])
        self.assertLess(result[0]['end'],result[1]['start'])


if __name__=='__main__':unittest.main()
