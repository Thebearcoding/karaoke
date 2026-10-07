import unittest
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from align import activity, align_lines
from dsp import apply_masks, istft, masks_for, stft
from exact_stems import complementary_split, reconstruction_report
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
    def test_nested_three_stems_preserve_estimates_after_storage(self):
        rng = np.random.default_rng(53)
        source = rng.normal(0, .15, (1024, 2)).astype('float32')
        vocal_estimate = rng.normal(0, .2, source.shape).astype('float32')
        source[0] = [1e-30, -1e-30]
        vocal_estimate[1] = [1e-30, -1e-30]
        vocals, music = complementary_split(source, vocal_estimate)
        lead_estimate = rng.normal(0, .1, source.shape).astype('float32')
        lead, backing = complementary_split(vocals, lead_estimate)
        with tempfile.TemporaryDirectory() as directory:
            restored = []
            for name, wave in [('lead', lead), ('backing', backing), ('music', music)]:
                path = Path(directory)/f'{name}.wav'
                sf.write(path, wave, 48000, subtype='DOUBLE')
                restored.append(sf.read(path, dtype='float64', always_2d=True)[0])
            checks = check_three(source, *restored)
        np.testing.assert_array_equal(vocals, vocal_estimate)
        np.testing.assert_array_equal(restored[0], lead_estimate)
        np.testing.assert_array_equal(restored[2], music)
        self.assertLess(checks['vocal_first']['max_absolute_error'], 1e-15)

    def test_complement_keeps_quiet_tails_instead_of_reassigning_them(self):
        # Cancellation used to abort these tails or erase the vocal estimate.
        source = np.array([[1e-30, -1e-30], [0., -0.],
                           [np.nextafter(1e-7, np.inf), -1e-30]])
        estimate = np.array([[.000288816, -.000112535], [.1, -.1],
                             [4.2e-6, -9.1246e-5]])
        vocals, backing = complementary_split(source, estimate)
        np.testing.assert_array_equal(vocals, estimate)
        np.testing.assert_allclose(vocals+backing, source, atol=1e-15, rtol=0)
        self.assertGreater(reconstruction_report(source, vocals, backing)['max_absolute_error'], 0)

    def test_complement_rejects_invalid_audio(self):
        for source, estimate in [
                (np.zeros((0, 2)), np.zeros((0, 2))),
                (np.zeros((3, 2)), np.zeros((4, 2))),
                (np.zeros(3), np.zeros(3)),
                (np.full((3, 2), np.nan), np.zeros((3, 2))),
                (np.zeros((3, 2)), np.full((3, 2), np.inf)),
                (np.full((3, 2), 1e308), np.full((3, 2), -1e308))]:
            with self.subTest(source=source, estimate=estimate), np.errstate(over='ignore'):
                with self.assertRaises(ValueError):
                    complementary_split(source, estimate)

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
                         [(0, 0, -4., 0.), (1, 0, 8., 12.), (1, 1, 20., 24.)])
        # A long note is not a long silence even when its onset was long ago.
        sustained = countdown_events([dict(units=[dict(start=0., end=9.), dict(start=10., end=11.)])])
        self.assertEqual([event['onset'] for event in sustained], [0.])

    def test_countdown_covers_short_openings_and_four_second_gaps_without_moving_lyrics(self):
        from copy import deepcopy
        for onset in (0., .631, 3.9, 4., 7.9):
            with self.subTest(onset=onset):
                cues = [dict(units=[dict(start=onset, end=onset+1.),
                                   dict(start=onset+5., end=onset+6.),
                                   dict(start=onset+9.99, end=onset+11.)])]
                original = deepcopy(cues)
                events = countdown_events(cues)
                self.assertEqual([event['onset'] for event in events], [onset, onset+5.])
                for event in events:
                    for i, count in enumerate((3, 2, 1, 0)):
                        self.assertEqual(countdown_state(events, event['start']+i+.5)[1], count)
                self.assertEqual(cues, original)

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

    def test_timing_contract_relocates_the_minimum_span_and_never_resurrects_old_times(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, timing_violations, validate_timeline
        protocol = dict(maximum_spread_seconds=.08, maximum_change_seconds=.4, minimum_relative_rms=.03)
        text = 'あいうえお'
        old = [dict(first=i, stop=i+1, text=c, start=a, end=b, timing_source='prior_word_fallback')
               for i,(c,a,b) in enumerate(zip(text,[1.,1.2,1.4,4.5,5.],[1.2,1.4,2.,4.9,5.5]))]
        original = [dict(text=text,start=1.,end=5.5,aligned_start=1.,aligned_end=5.5,units=old)]
        for mode in ('valid','suffix_disagreement','one_view','text_mismatch','prefix_disagreement',
                     'window_disagreement','silent','silent_neighbor','invalid','held_note','quiet_voice'):
            with self.subTest(mode=mode):
                times = [1.,1.2,1.4,4.5,5.] if mode=='held_note' else [3.,3.2,3.6,4.5,5.]
                measured = [dict(u,start=t,end=t+.15) for u,t in zip(old,times)]
                views = [deepcopy(measured),deepcopy(measured)]
                if mode=='one_view': views.pop()
                if mode=='text_mismatch': views[1][0]['text']='ん'
                if mode=='prefix_disagreement':
                    views[1][1]['start']+=.2
                    views[1][1]['end']+=.2
                if mode=='suffix_disagreement': views[1][-1]['start']-=.2
                if mode=='invalid': views[1][0]['start']=float('nan')
                predictions = [[dict(u,timing_evidence='ctc_event',emission_end=u['start']+.04)
                                for u in measured] for _ in range(3)]
                if mode=='window_disagreement':
                    for u in predictions[0]:u.update(start=u['start']+.12,end=u['end']+.12,emission_end=u['emission_end']+.12)
                wave=np.full(44100*6,0. if mode=='silent' else 1e-6 if mode=='quiet_voice' else .1)
                wave[44100:44100*3]=0. if mode!='held_note' else .1
                if mode=='silent_neighbor':wave[round(3.2*44100):round(3.4*44100)]=0.
                result,report=reconcile_timing(original,wave,lambda *a,**k:views,
                    lambda *a:predictions,protocol,'Japanese')
                self.assertEqual(original[0]['units'],old)
                if mode in ('valid','suffix_disagreement'):
                    self.assertEqual([u['start'] for u in result[0]['units']],times)
                    self.assertEqual([u['start'] for u in result[0]['units'][3:]],[4.5,5.])
                    self.assertFalse(report['unresolved'])
                    self.assertTrue(timing_violations(original,report['constraints']))
                    # Even a later model agreeing with the stale prior cannot
                    # make the preserved contradiction disappear.
                    again,checked=reconcile_timing(original,wave,lambda *a,**k:[old,old],
                        lambda *a:[],protocol,'Japanese',report['constraints'])
                    self.assertTrue(checked['unresolved'])
                    with self.assertRaisesRegex(ValueError,'contradicted'):
                        validate_timeline(again,[text],6,checked['constraints'])
                elif mode in ('prefix_disagreement','silent_neighbor'):
                    self.assertTrue(report['unresolved'])
                else:
                    self.assertEqual([u['start'] for u in result[0]['units']],[u['start'] for u in old])
                    self.assertFalse(report['constraints'])

    def test_timing_contract_checks_middle_words_and_all_languages(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing
        protocol=dict(maximum_spread_seconds=.08,maximum_change_seconds=.4,minimum_relative_rms=.03)
        for language,text in [('Chinese','甲乙丙'),('Japanese','あいう'),('English','one two three')]:
            with self.subTest(language=language):
                spans=[(0,1),(1,2),(2,3)] if language!='English' else [(0,3),(4,7),(8,13)]
                units=[dict(first=a,stop=b,text=text[a:b],start=t,end=t+.2,timing_source='singing_phonemes')
                       for (a,b),t in zip(spans,[.2,1.,4.])]
                cues=[dict(text=text,start=.2,end=4.2,aligned_start=.2,aligned_end=4.2,units=units)]
                observed=deepcopy(units);observed[1].update(start=3.,end=3.3)
                result,report=reconcile_timing(cues,np.ones(44100*5)*.1,lambda *a,**k:[observed,observed],
                    lambda *a:[observed]*3,protocol,language)
                self.assertEqual([u['start'] for u in result[0]['units']],[.2,3.,4.])
                self.assertFalse(report['unresolved'])
                self.assertEqual(len(report['constraints']),1)

    def test_timing_context_retry_uses_confirmed_crop_without_relaxing_evidence(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, validate_timeline
        protocol = dict(maximum_spread_seconds=.08, maximum_change_seconds=.4, minimum_relative_rms=.03)
        units = [dict(first=i, stop=i+1, text=text, start=start, end=start+.2,
                      timing_source='prior_word_fallback') for i, (text, start) in enumerate([('甲', 1.), ('乙', 1.5)])]
        cue = dict(text='甲乙', start=1., end=4., aligned_start=1., aligned_end=4., units=units)
        neighbour = dict(text='丙', start=5., end=5.5,
            units=[dict(first=0, stop=1, text='丙', start=5., end=5.5, timing_source='singing_phonemes')])
        measured = [dict(unit, start=start, end=start+.2) for unit, start in zip(units, [3., 3.4])]
        predictions = [deepcopy(measured) for _ in range(3)]
        for mode in ('recovered', 'unchanged', 'wrong_text', 'unavailable', 'oom'):
            with self.subTest(mode=mode):
                calls = []
                def witness(context, wave, **kwargs):
                    calls.append(context['aligned_start'])
                    # Cropping must not become an assigned word onset.
                    self.assertEqual(context['units'][0]['start'], 1.)
                    views = [deepcopy(measured), deepcopy(measured)]
                    if len(calls) == 1 or mode == 'unchanged':
                        views[1][1].update(start=3.7, end=3.9)
                    elif mode == 'wrong_text':
                        views[0][1]['text'] = '丁'
                    elif mode in ('unavailable', 'oom'):
                        raise RuntimeError('CUDA out of memory' if mode == 'oom' else 'Witness unavailable')
                    return views
                if mode == 'oom':
                    with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                        reconcile_timing([cue, neighbour], np.ones(44100*6)*.1, witness,
                            lambda *a: predictions, protocol, 'Chinese')
                    continue
                result, report = reconcile_timing([cue, neighbour], np.ones(44100*6)*.1, witness,
                    lambda *a: predictions, protocol, 'Chinese')
                self.assertEqual(calls, [1., 3.])
                self.assertEqual(report['lines'][-1]['context_retry'], [3., 4.])
                self.assertEqual(result[1]['units'][0]['start'], 5.)
                self.assertEqual([u['start'] for u in cue['units']], [1., 1.5])
                if mode == 'recovered':
                    self.assertEqual([u['start'] for u in result[0]['units']], [3., 3.4])
                    self.assertFalse(report['unresolved'])
                    validate_timeline(result, ['甲乙', '丙'], 6, report['constraints'], 'Chinese')
                else:
                    self.assertEqual([u['start'] for u in result[0]['units']], [1., 1.5])
                    self.assertTrue(report['unresolved'])
                    with self.assertRaisesRegex(ValueError, 'contradicted'):
                        validate_timeline(result, ['甲乙', '丙'], 6, report['constraints'], 'Chinese')

    def test_internal_merged_word_reobserves_a_stale_phrase_window_once(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, validate_timeline
        protocol=dict(maximum_spread_seconds=.08,maximum_change_seconds=.4,minimum_relative_rms=.03)
        text='甲乙丙丁'
        units=[dict(first=0,stop=1,text='甲',start=1.,end=3.1),
               dict(first=1,stop=2,text='乙',start=3.12,end=3.3),
               dict(first=2,stop=4,text='丙丁',start=7.,end=8.,timing_source='linear_fallback')]
        cue=dict(text=text,start=1.,end=8.,aligned_start=1.,aligned_end=8.,units=units)
        measured=[dict(first=i,stop=i+1,text=c,start=t,end=t+.1)
                  for i,(c,t) in enumerate(zip(text,[2.,2.9,3.1,3.3]))]
        predictions=[deepcopy(measured) for _ in range(3)]
        # The start of the first word remains unconfirmed. Only the interior
        # has a proved conflict, so the old first-word-only retry missed it.
        predictions[0][0].update(start=1.6,end=1.7)
        for recovers in (True,False):
            with self.subTest(recovers=recovers):
                calls=[]
                def witness(context,*args,**kwargs):
                    calls.append((context['aligned_start'],context['aligned_end']))
                    self.assertEqual(context['units'],[dict(u,acoustic_end=None if u.get('timing_source')=='linear_fallback' else u['end']) for u in units])
                    views=[deepcopy(measured),deepcopy(measured)]
                    if len(calls)==1 or not recovers:
                        views[1][-1].update(start=3.6,end=3.7)
                    return views
                result,report=reconcile_timing([cue],np.ones(9*44100)*.1,witness,
                    lambda *args:predictions,protocol,'Chinese')
                self.assertEqual(calls,[(1.,8.),(2.,3.7)])
                self.assertEqual(report['lines'][-1]['context_retry'],[2.,3.7])
                self.assertEqual(cue['units'],units)
                if recovers:
                    self.assertEqual([u['start'] for u in result[0]['units']],[1.,2.9,3.1,3.3])
                    self.assertFalse(report['unresolved'])
                    validate_timeline(result,[text],9.,report['constraints'],'Chinese')
                else:
                    self.assertTrue(report['unresolved'])

    def test_countdown_uses_singing_offsets_and_does_not_invent_them_from_ctc(self):
        # The first word stays visible until the next word, but its singing
        # has ended. Extending the display must not erase a real vocal gap.
        units=[dict(start=1.,end=9.,acoustic_end=2.),dict(start=9.,end=10.,acoustic_end=10.)]
        self.assertEqual([e['onset'] for e in countdown_events([dict(units=units)])],[1.,9.])
        units[0]=dict(start=1.,end=9.,emission_end=1.05,timing_evidence='ctc_event')
        self.assertEqual([e['onset'] for e in countdown_events([dict(units=units)])],[1.])

    def test_playback_timing_groups_only_uncertain_words_between_measured_boundaries(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, validate_timeline
        protocol = dict(maximum_spread_seconds=.08, maximum_change_seconds=.4, minimum_relative_rms=.03)
        units = [dict(first=i, stop=i+1, text=char, start=start, end=start+.2,
                      timing_source='prior_word_fallback')
                 for i,(char,start) in enumerate(zip('甲乙丙丁', [1.,1.5,4.1,5.]))]
        cue = dict(text='甲乙丙丁', start=1., end=5.2, units=units)
        measured = [dict(u,start=start,end=start+.2) for u,start in zip(units,[3.,3.4,4.2,5.])]
        for mode in ('playback','strict','missing_right_anchor','contradicted_interior',
                     'interior_outside_bounds','silent_anchor','text_mismatch'):
            with self.subTest(mode=mode):
                views = [deepcopy(measured),deepcopy(measured)]
                views[1][1].update(start=3.7,end=3.9)
                constraints = []
                wave = np.ones(44100*6)*.1
                if mode=='missing_right_anchor':
                    for i in (2,3):
                        views[1][i]['start']+=.2
                        views[1][i]['end']+=.2
                if mode=='contradicted_interior':
                    constraints=[dict(line=1,text=cue['text'],first=1,stop=2,start_min=3.2,start_max=3.6,onset=3.4)]
                if mode=='interior_outside_bounds':views[1][1]['start']=2.8
                if mode=='silent_anchor':wave[3*44100:round(3.25*44100)]=0.
                if mode=='text_mismatch':views[1][2]['text']='錯'
                result, report = reconcile_timing([cue],wave,lambda *a,**k:views,
                    lambda *a:[measured]*3,protocol,'Chinese',constraints,
                    playback=mode!='strict',_retry_context=False)
                groups=[u for u in result[0]['units'] if u.get('timing_source')=='linear_fallback']
                if mode=='playback':
                    self.assertEqual([(u['text'],u['start'],u['end']) for u in groups],[('甲乙',3.,4.2)])
                    self.assertEqual(result[0]['units'][-1],dict(units[-1],acoustic_end=units[-1]['end']))
                    self.assertEqual(result[0]['units'][1]['start'],4.2)
                    self.assertFalse(report['unresolved'])
                    self.assertIsNone(report['accuracy'])
                    self.assertTrue(highlight_coverage_errors(result,'Chinese'))
                    validate_timeline(result,[cue['text']],6,report['constraints'],'Chinese')
                else:
                    self.assertFalse(groups)
                self.assertEqual([u['start'] for u in cue['units']],[1.,1.5,4.1,5.])

    def test_playback_timing_repairs_existing_groups_without_losing_guard_words(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, validate_timeline
        text = 'We are here for you'
        protocol = dict(maximum_spread_seconds=.08, maximum_change_seconds=.4, minimum_relative_rms=.03)
        old = [dict(first=0, stop=12, text=text[:12], start=1., end=3.5, timing_source='linear_fallback'),
               dict(first=12, stop=19, text=text[12:], start=3.5, end=4.5, timing_source='linear_fallback')]
        cue = dict(text=text, start=1., end=4.5, units=old)
        measured = [dict(first=a, stop=b, text=text[a:b], start=t, end=t+.2)
                    for a,b,t in [(0,2,.5),(3,6,1.5),(7,11,2.5),(12,15,3.5),(16,19,4.)]]
        for mode in ('grouped', 'partial_guard', 'partial_guard_constraint', 'missing_guard_word', 'contradicted_interior'):
            with self.subTest(mode=mode):
                views = [deepcopy(measured),deepcopy(measured)]
                views[1][2]['start']+=.15
                constraints = []
                if mode == 'missing_guard_word': views[1].pop()
                if mode.startswith('partial_guard'):
                    views[1][-1].update(start=4.2,end=4.4)
                    if mode=='partial_guard_constraint':
                        constraints=[dict(line=1,text=text,first=16,stop=19,start_min=3.9,start_max=4.3,onset=4.1)]
                if mode == 'contradicted_interior':
                    constraints = [dict(line=1,text=text,first=7,stop=11,start_min=2.1,start_max=2.9,onset=2.5)]
                result, report = reconcile_timing([cue], np.ones(44100*6)*.1,
                    lambda *a,**k: views, lambda *a: [measured]*3, protocol, 'English', constraints,
                    playback=True, _retry_context=False)
                if mode == 'grouped':
                    self.assertEqual([(u['text'],u['start'],u['end']) for u in result[0]['units']],
                                     [('We are here ',.5,3.5),('for',3.5,3.7),('you',4.,4.2)])
                    self.assertFalse(report['unresolved'])
                    validate_timeline(result,[text],6,report['constraints'])
                elif mode=='partial_guard':
                    self.assertEqual([(u['text'],u['start'],u['end']) for u in result[0]['units']],
                                     [('We are here ',.5,3.5),('for you',3.5,4.5)])
                    self.assertIsNone(result[0]['units'][-1]['acoustic_end'])
                    self.assertFalse(report['unresolved'])
                    validate_timeline(result,[text],6,report['constraints'])
                else:
                    self.assertEqual(result[0]['units'][0]['start'],1.)
                self.assertEqual(cue['units'],old)

    def test_playback_timing_closes_left_conflict_at_confirmed_onsets_without_splitting_guard(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, validate_timeline
        protocol=dict(maximum_spread_seconds=.08,maximum_change_seconds=.4,minimum_relative_rms=.03)
        text='甲乙丙丁戊己庚'
        old=[dict(first=a,stop=b,text=text[a:b],start=t,end=t+.15,timing_source='prior_word_fallback')
             for a,b,t in [(0,1,1.),(1,2,1.3),(2,3,2.1),(3,4,2.5),(4,6,2.7),(6,7,3.4)]]
        cue=dict(text=text,start=1.,end=3.55,units=old)
        measured=[dict(first=i,stop=i+1,text=c,start=t,end=t+.05)
                  for i,(c,t) in enumerate(zip(text,[1.,1.3,1.65,1.9,2.,2.3,3.4]))]
        for mode in ('playback','strict','missing_prefix','guard_constraint'):
            with self.subTest(mode=mode):
                views=[deepcopy(measured),deepcopy(measured)]
                for i,shift in ((2,-.15),(3,-.15),(5,.2)):
                    views[1][i].update(start=measured[i]['start']+shift,end=measured[i]['end']+shift)
                constraints=[]
                if mode=='missing_prefix':
                    for i in (0,1):views[1][i].update(start=measured[i]['start']+.11,end=measured[i]['end']+.11)
                if mode=='guard_constraint':
                    constraints=[dict(line=1,text=text,first=5,stop=6,start_min=2.2,start_max=2.6,onset=2.4)]
                result,report=reconcile_timing([cue],np.ones(4*44100)*.1,lambda *a,**k:views,
                    lambda *a:[measured]*3,protocol,'Chinese',constraints,playback=mode!='strict',_retry_context=False)
                self.assertEqual(cue['units'],old)
                self.assertEqual(result[0]['units'][0]['start'],old[0]['start'])
                self.assertEqual(result[0]['units'][-1]['start'],old[-1]['start'])
                if mode=='playback':
                    self.assertEqual([(u['text'],u['start'],u['end']) for u in result[0]['units'][1:-1]],
                                     [('乙丙丁',1.3,2.),('戊己',2.,2.85)])
                    self.assertFalse(report['unresolved'])
                    self.assertIsNone(report['accuracy'])
                    validate_timeline(result,[text],4.,report['constraints'],'Chinese')
                else:
                    self.assertTrue(report['unresolved'])
                    with self.assertRaisesRegex(ValueError,'contradicted'):
                        validate_timeline(result,[text],4.,report['constraints'],'Chinese')

    def test_timing_repair_closes_over_different_token_boundaries(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing
        text = 'あいうえおか'
        protocol = dict(maximum_spread_seconds=.08, maximum_change_seconds=.4, minimum_relative_rms=.03)
        for old_spans, new_spans in [
                ([(0,1),(1,2),(2,4),(4,6)], [(0,2),(2,4),(4,6)]),
                ([(0,1),(1,3),(3,4),(4,6)], [(0,2),(2,4),(4,6)])]:
            for disagreement in (False, True):
                with self.subTest(old_spans=old_spans, disagreement=disagreement):
                    old = [dict(first=a, stop=b, text=text[a:b], start=t, end=t+.2,
                                timing_source='prior_word_fallback')
                           for (a,b),t in zip(old_spans,[.1,.4,2.5,4.5])]
                    measured = [dict(first=a, stop=b, text=text[a:b], start=t, end=t+.2)
                                for (a,b),t in zip(new_spans,[1.5,2.5,4.5])]
                    views = [deepcopy(measured),deepcopy(measured)]
                    if disagreement:
                        views[1][0]['start'] += .2
                    cue = dict(text=text,start=.1,end=4.7,units=old)
                    result,report = reconcile_timing([cue],np.ones(44100*6)*.1,
                        lambda *a,**k:views,lambda *a:[measured]*3,protocol,'Japanese')
                    if disagreement:
                        self.assertEqual([u['start'] for u in result[0]['units']],[u['start'] for u in old])
                    else:
                        self.assertFalse(report['unresolved'])
                        self.assertEqual([(u['first'],u['stop']) for u in result[0]['units']],new_spans)
                        self.assertEqual([u['start'] for u in result[0]['units']],[1.5,2.5,4.5])
                    self.assertEqual(result[0]['units'][-1]['start'],old[-1]['start'])

    def test_local_timing_repair_distinguishes_a_closure_from_missing_voice(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, validate_timeline
        protocol = dict(maximum_spread_seconds=.08, maximum_change_seconds=.4, minimum_relative_rms=.03)
        cue = dict(text='あがっこう', start=.2, end=5., units=[
            dict(first=0, stop=1, text='あ', start=.2, end=.4),
            dict(first=1, stop=5, text='がっこう', start=4.5, end=5., timing_source='prior_word_fallback')])
        measured = [dict(first=i, stop=i+1, text=c, start=t, end=t+.1)
                    for i, (c,t) in enumerate(zip(cue['text'], [.2,2.,2.3,2.5,2.7]))]
        for mode in ('closure', 'vowel', 'mixed_phone', 'missing_prediction', 'silent_neighbor',
                     'adjacent_closure', 'edge_closure'):
            with self.subTest(mode=mode):
                wave = np.ones(44100*6)*.1
                wave[round(2.3*44100):round(2.4*44100)] = 0.
                predictions = [deepcopy(measured) for _ in range(3)]
                for view in predictions:
                    view[2]['phones'] = ['cl']
                if mode == 'vowel':
                    for view in predictions:
                        view[2]['phones'] = ['a']
                if mode == 'mixed_phone': predictions[1][2]['phones'] = ['cl', 'a']
                if mode == 'missing_prediction': predictions[1].pop(2)
                if mode == 'silent_neighbor': wave[round(2.5*44100):round(2.6*44100)] = 0.
                if mode in ('adjacent_closure', 'edge_closure'):
                    index = 3 if mode == 'adjacent_closure' else 4
                    for view in predictions:
                        view[index]['phones'] = ['cl']
                    unit = measured[index]
                    wave[round(unit['start']*44100):round(unit['end']*44100)] = 0.
                result, report = reconcile_timing([cue], wave,
                    lambda *a, **k: [measured, measured], lambda *a: predictions,
                    protocol, 'Japanese', _retry_context=False)
                if mode == 'closure':
                    self.assertEqual([u['start'] for u in result[0]['units']], [.2,2.,2.3,2.5,2.7])
                    self.assertEqual(result[0]['units'][2]['support'], 'phonetic_closure_between_audible_units')
                    self.assertFalse(report['unresolved'])
                    validate_timeline(result, [cue['text']], 6., report['constraints'], 'Japanese')
                else:
                    self.assertEqual(result[0]['units'][1]['start'], 4.5)
                    self.assertTrue(report['unresolved'])
                self.assertEqual(cue['units'][1]['start'], 4.5)

    def test_timing_repair_keeps_the_evidence_support_contract(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, validate_timeline
        protocol = dict(maximum_spread_seconds=.08, maximum_change_seconds=.4, minimum_relative_rms=.03)
        cue = dict(text='but hey', start=1., end=3.3, units=[
            dict(first=0, stop=4, text='but ', start=1., end=1.3),
            dict(first=4, stop=7, text='hey', start=3., end=3.3)])
        measured = [dict(first=0, stop=3, text='but', start=2., end=2.2),
                    dict(first=4, stop=7, text='hey', start=3., end=3.3)]
        for mode in ('phonetic', 'ctc', 'silent'):
            with self.subTest(mode=mode):
                wave = np.ones(44100*4)*.1
                wave[2*44100:round((2.2 if mode=='silent' else 2.05)*44100)] = 0.
                predictions = [deepcopy(measured) for _ in range(3)]
                if mode == 'ctc':
                    for view in predictions:
                        for u in view:
                            u.update(timing_evidence='ctc_event', emission_end=u['start']+.02)
                result, report = reconcile_timing([cue], wave,
                    lambda *a, **k: [measured, measured], lambda *a: predictions,
                    protocol, 'English', _retry_context=False)
                self.assertEqual(result[0]['units'][-1], dict(cue['units'][-1], acoustic_end=3.3))
                if mode == 'phonetic':
                    self.assertEqual(result[0]['units'][0]['start'], 2.)
                    self.assertFalse(report['unresolved'])
                    validate_timeline(result, [cue['text']], 4., report['constraints'], 'English')
                else:
                    self.assertEqual(result[0]['units'][0]['start'], 1.)
                self.assertEqual(cue['units'][0]['start'], 1.)

    def test_merged_timing_repair_reuses_corroborated_interior_onsets(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, validate_timeline
        protocol = dict(maximum_spread_seconds=.08, maximum_change_seconds=.4, minimum_relative_rms=.03)
        cue = dict(text='甲乙丙丁', start=.2, end=5., units=[
            dict(first=0, stop=1, text='甲', start=.2, end=.4),
            dict(first=1, stop=4, text='乙丙丁', start=4.5, end=5., timing_source='prior_word_fallback')])
        measured = [dict(first=i, stop=i+1, text=c, start=t, end=t+.2)
                    for i, (c,t) in enumerate(zip(cue['text'], [.2,2.,3.04,4.]))]
        measured[2]['end'] = 3.06
        for mode in ('corroborated', 'silent', 'context_disagreement'):
            with self.subTest(mode=mode):
                wave = np.ones(44100*6)*.1
                wave[round(3.04*44100):round(3.06*44100)] = 0.
                predictions = [deepcopy(measured) for _ in range(3)]
                for view in predictions:
                    view[2].update(start=3., end=3.2)
                if mode == 'silent':
                    wave[3*44100:round(3.2*44100)] = 0.
                if mode == 'context_disagreement':
                    predictions[0][2].update(start=3.15, end=3.3)
                result, report = reconcile_timing([cue], wave,
                    lambda *a, **k: [measured, measured], lambda *a: predictions,
                    protocol, 'Chinese', _retry_context=False)
                if mode == 'corroborated':
                    self.assertEqual([u['start'] for u in result[0]['units']], [.2,2.,3.,4.])
                    self.assertFalse(report['unresolved'])
                    self.assertEqual([p['first'] for p in report['constraints']], [1])
                    validate_timeline(result, [cue['text']], 6., report['constraints'], 'Chinese')
                else:
                    self.assertEqual(result[0]['units'][1]['start'], 4.5)
                    self.assertTrue(report['unresolved'])
                self.assertEqual(len(cue['units']), 2)

    def test_playback_timing_can_close_a_tail_group_without_inventing_a_next_onset(self):
        from copy import deepcopy
        from timing_validation import reconcile_timing, validate_timeline
        protocol = dict(maximum_spread_seconds=.08, maximum_change_seconds=.4, minimum_relative_rms=.03)
        units = [dict(first=i,stop=i+1,text=c,start=t,end=t+.2,timing_source='prior_word_fallback')
                 for i,(c,t) in enumerate(zip('甲乙丙丁',[.2,1.,1.5,2.]))]
        cue = dict(text='甲乙丙丁',start=.2,end=3.9,units=units)
        measured = [dict(u,start=t,end=t+.2) for u,t in zip(units,[.2,3.,3.3,3.6])]
        for mode in ('valid','short_word_event','display_hold','strict','end_disagreement','context_end_disagreement',
                     'context_start_disagreement','disjoint_word_events',
                     'missing_emission_end','wrong_tail_text','interior_constraint','neighbor_overlap'):
            with self.subTest(mode=mode):
                views = [deepcopy(measured),deepcopy(measured)]
                views[0][-1]['end']=3.85
                views[1][2].update(start=3.6,end=3.75)
                views[1][3].update(start=3.75,end=3.9)
                predictions = [[dict(u,timing_evidence='ctc_event',emission_end=u['start']+.1)
                                for u in measured] for _ in range(3)]
                for v in predictions:v[-1].update(end=3.85,emission_end=3.85)
                constraints=[]
                neighbour=dict(text='戊',start=5.,end=5.5,
                    units=[dict(first=0,stop=1,text='戊',start=5.,end=5.5,timing_source='singing_phonemes')])
                if mode=='display_hold':
                    for v in predictions:v[-1]['end']=5.
                if mode=='end_disagreement':views[1][-1]['end']=4.2
                if mode=='short_word_event':
                    for v in predictions:v[-1].update(end=4.05,emission_end=4.05)
                if mode=='context_end_disagreement':predictions[0][-1].update(end=4.2,emission_end=4.2)
                if mode=='context_start_disagreement':predictions[0][-1]['start']=3.3
                if mode=='disjoint_word_events':
                    for v in predictions:v[-1].update(start=4.2,end=4.45,emission_end=4.45)
                if mode=='missing_emission_end':predictions[0][-1]['emission_end']=None
                if mode=='wrong_tail_text':predictions[0][-1]['text']='錯'
                if mode=='interior_constraint':
                    constraints=[dict(line=1,text=cue['text'],first=2,stop=3,start_min=3.2,start_max=3.4,onset=3.3)]
                if mode=='neighbor_overlap':neighbour['units'][0]['start']=3.88
                result,report=reconcile_timing([cue,neighbour],np.ones(44100*6)*.1,
                    lambda c,*a,**k:views,lambda *a:predictions,protocol,'Chinese',constraints,
                    playback=mode!='strict',_retry_context=False)
                groups=[u for u in result[0]['units'] if u.get('timing_source')=='linear_fallback']
                if mode in ('valid','short_word_event','display_hold'):
                    self.assertEqual([(g['text'],g['start'],g['end']) for g in groups],
                                     [('乙丙丁',3.,4.05 if mode=='short_word_event' else 3.9)])
                    self.assertIsNone(groups[0]['acoustic_end'])
                    self.assertEqual(groups[0]['bounds_source'],'cross_model_span_end')
                    self.assertFalse(report['unresolved'])
                    validate_timeline(result,['甲乙丙丁','戊'],6,report['constraints'],'Chinese')
                else:
                    self.assertFalse(groups)
                self.assertEqual(result[0]['units'][0]['start'],.2)
                self.assertEqual(result[1]['units'],[dict(neighbour['units'][0],acoustic_end=5.5)])
                self.assertEqual([u['start'] for u in units],[.2,1.,1.5,2.])

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

    def test_english_groups_in_cjk_songs_require_independent_acoustic_agreement(self):
        from copy import deepcopy
        from cjk_recovery import recover_latin_groups
        original = [dict(text='前', aligned_start=.2, aligned_end=.8,
                         units=[dict(first=0, stop=1, text='前', start=.2, end=.8)]),
                    dict(text='go home now', aligned_start=2., aligned_end=3.8, units=[
                        dict(first=0, stop=8, text='go home ', start=2., end=3.4),
                        dict(first=8, stop=11, text='now', start=3.4, end=3.8)]),
                    dict(text='後', aligned_start=5.2, aligned_end=5.8,
                         units=[dict(first=0, stop=1, text='後', start=5.2, end=5.8)])]
        protocol = dict(context_padding_seconds=[.35, .6, 1.],
                        maximum_spread_seconds=.08, minimum_relative_rms=.03)
        for mixed, fragmented in ((False, False), (True, False), (False, True), (True, True)):
            for mode in ('valid', 'one_witness', 'disagree', 'context_disagree', 'changed_words', 'crossing'):
                with self.subTest(mode=mode, mixed=mixed, fragmented=fragmented):
                    calls = []
                    class Model:
                        model_identity = {'model': 'independent phoneme model'}
                        def align(_, audio, words):
                            self.assertEqual(words, ['go', 'home', 'now'])
                            begin = float(audio[0])-.1
                            calls.append(begin)
                            result = [dict(text=w, start=a-begin, end=b-begin) for w, a, b in
                                      zip(words, [2.05, 2.55, 3.4], [2.45, 3.15, 3.8])]
                            if mode == 'context_disagree' and len(calls) == 1:
                                result[1]['start'] += .2
                            if mode == 'changed_words':
                                result[0]['text'] = 'different'
                            return result, None, .01
                    def witness(cue, wave):
                        self.assertEqual(cue['text'], 'go home now')
                        result = [[dict(first=a, stop=b, text=cue['text'][a:b], start=start, end=end)
                                   for a, b, start, end in [(0, 2, 2.06, 2.44), (3, 7, 2.56, 3.14), (8, 11, 3.4, 3.8)]]
                                  for _ in range(2)]
                        if mode == 'one_witness':
                            result.pop()
                        if mode == 'disagree':
                            result[1][1]['start'] += .2
                        return result
                    before = deepcopy(original)
                    if fragmented:
                        before[1]['units'][:1] = [dict(first=0, stop=1, text='g', start=2., end=2.1),
                                                 dict(first=1, stop=8, text='o home ', start=2.1, end=3.4)]
                    if mode == 'crossing':
                        before[0]['units'][0]['end'] = 2.2
                    if mixed:
                        text, units = '', []
                        for cue in before:
                            units.extend(dict(u, first=u['first']+len(text), stop=u['stop']+len(text))
                                         for u in cue['units'])
                            text += cue['text']
                        before = [dict(text=text, aligned_start=.2, aligned_end=5.8, units=units)]
                    unchanged = deepcopy(before)
                    wave = np.arange(44100*6)/44100+.1
                    actual, report = recover_latin_groups(before, wave, witness, protocol, Model())
                    self.assertEqual(before, unchanged)
                    old_units = [u for c in before for u in c['units']]
                    units = [u for c in actual for u in c['units']]
                    self.assertEqual([units[0], units[-2], units[-1]],
                                     [old_units[0], old_units[-2], old_units[-1]])
                    if mode == 'valid':
                        np.testing.assert_allclose(calls, [1.65, 1.4, 1.])
                        self.assertEqual([u['start'] for u in units[1:-1]], [2.05, 2.55, 3.4])
                        self.assertFalse(highlight_coverage_errors(actual, 'Japanese'))
                        self.assertTrue(report[0]['groups'][0]['accepted'])
                    else:
                        self.assertEqual(actual, before)
                        self.assertFalse(any(g['accepted'] for g in report[0]['groups']))

    def test_split_english_word_requires_new_evidence_and_known_language_boundary(self):
        from cjk_recovery import recover_latin_groups
        protocol = dict(context_padding_seconds=[.35, .6, 1.],
                        maximum_spread_seconds=.08, minimum_relative_rms=.03)
        for mode in ('english', 'foreign', 'repeated'):
            with self.subTest(mode=mode):
                calls = []
                class Model:
                    model_identity = {'model': 'independent phoneme model'}
                    def align(_, audio, words):
                        self.assertEqual(words, ['home'])
                        begin = float(audio[0])-.1
                        calls.append(begin)
                        return [dict(text='home', start=2.05-begin, end=2.55-begin)], None, .01
                def witness(cue, wave):
                    return [[dict(first=0, stop=4, text='home', start=2.06, end=2.54)] for _ in range(2)]
                cue = dict(text='home', aligned_start=2., aligned_end=2.8, units=[
                    dict(first=0, stop=2, text='ho', start=2., end=2.4),
                    dict(first=2, stop=4, text='me', start=2.4, end=2.8)])
                if mode == 'foreign':
                    cue.update(text='あ home', units=[
                        dict(first=0, stop=4, text='あ ho', start=2., end=2.4),
                        dict(first=4, stop=6, text='me', start=2.4, end=2.8)])
                if mode == 'repeated':
                    cue['units'][0]['timing_source'] = 'repeated_lyric_audio'
                actual, report = recover_latin_groups([cue], np.arange(44100*6)/44100+.1,
                                                       witness, protocol, Model())
                if mode == 'english':
                    self.assertEqual(len(calls), 3)
                    self.assertEqual(actual[0]['units'], [dict(first=0, stop=4, text='home',
                        start=2.05, end=2.55, timing_source='cross_model_phonemes')])
                    self.assertFalse(highlight_coverage_errors(actual, 'Japanese'))
                else:
                    self.assertEqual(actual, [cue])
                    self.assertFalse(calls)

    def test_independent_witness_keeps_both_audio_views_inside_foreign_word_bounds(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from cjk_recovery import AcousticWitness

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root/'models/alignment/mms/model.pt'
            weights.parent.mkdir(parents=True)
            weights.touch()
            original = np.full(48000*6, .8, dtype='float32')
            original[round(1.8*48000):4*48000] = .1
            sf.write(root/'original.wav', original, 48000, subtype='FLOAT')
            voice = np.full(44100*6, .8, dtype='float32')
            voice[round(1.8*44100):4*44100] = .1
            observed = []
            def model(audio):
                observed.append((audio.shape[1], audio.mean().item()))
                return torch.zeros(1, 100, 3), None
            witness = AcousticWitness(root/'original.wav', 'cpu')
            witness.model = model
            witness.bundle = SimpleNamespace(get_tokenizer=lambda: lambda words: words,
                get_aligner=lambda: lambda emissions, words:
                    [[SimpleNamespace(start=i*5, end=i*5+2)] for i in range(len(words))])
            cue = dict(text='go home', aligned_start=2., aligned_end=3.8)
            with patch('cjk_recovery.ROOT', root):
                bounded = witness(dict(cue, audio_bounds=[1.8, 4.]), voice)
                self.assertEqual([n for n, _ in observed], [35200, 35200])
                self.assertTrue(all(abs(mean-.1) < .001 for _, mean in observed))
                self.assertEqual([len(view) for view in bounded], [2, 2])
                observed.clear()
                witness(cue, voice)
                self.assertEqual([n for n, _ in observed], [48000, 48000])
                self.assertTrue(all(mean > .2 for _, mean in observed))
                tokens = []
                witness.bundle.get_tokenizer = lambda: lambda words: tokens.append(words) or words
                japanese = witness(dict(cue, text='君が笑う'), voice, language='Japanese')
                self.assertEqual(tokens, [['*', 'kimi', 'ga', 'wara', 'u', '*']]*2)
                self.assertEqual([(u['first'], u['stop'], u['text']) for u in japanese[0]],
                                 [(0, 1, '君'), (1, 2, 'が'), (2, 3, '笑'), (3, 4, 'う')])
                witness.model = lambda audio: (torch.zeros(1, 1, 3), None)
                with self.assertRaisesRegex(ValueError, 'Insufficient acoustic frames'):
                    witness(cue, voice)
                with patch.object(witness, 'model', side_effect=RuntimeError('CUDA out of memory')):
                    with self.assertRaisesRegex(RuntimeError, 'CUDA out of memory'):
                        witness(cue, voice)

    def test_joint_recovery_rechecks_stale_neighbor_without_moving_foreign_words(self):
        from cjk_recovery import recover_context_groups
        from phoneme_progress import spoken_units
        original = [dict(text='go', aligned_start=.1, aligned_end=.4,
                         units=[dict(text='go', first=0, stop=2, start=.1, end=.4)]),
                    dict(text='甲乙丙丁', aligned_start=1., aligned_end=3., units=[
                        dict(text='甲', first=0, stop=1, start=1., end=1.8),
                        dict(text='乙丙', first=1, stop=3, start=1.8, end=2.3),
                        dict(text='丁', first=3, stop=4, start=2.3, end=3.)])]
        protocol = dict(maximum_spread_seconds=.08, minimum_relative_rms=.03)
        times = {'甲': (1., 1.5), '乙': (1.5, 2.), '丙': (2., 2.3), '丁': (2.4, 3.)}
        for mode in ('valid', 'unverified_neighbor', 'unstable', 'silent', 'collapsed'):
            with self.subTest(mode=mode):
                calls = []

                class Model:
                    def infer(self, audio, begin, units, cue):
                        calls.append((begin, len(audio)))
                        values = [dict(u, start=times[u['text']][0], end=times[u['text']][1]) for u in units]
                        if mode == 'unstable' and len(calls) == 1:
                            values[1]['start'] += .2
                        if mode == 'collapsed':
                            values[1]['end'] = values[1]['start']+.01
                        return dict(units=values)

                def witness(cue, wave):
                    self.assertNotIn('go', cue['text'])
                    parsed = spoken_units(cue['text'], 'Chinese')
                    values = [[dict(u, start=times[u['text']][0]+.02, end=times[u['text']][1])
                               for u in parsed] for _ in range(2)]
                    if mode == 'unverified_neighbor':
                        values[0][0]['start'] += .3
                    return values

                wave = np.full(44100*6, 0. if mode == 'silent' else .1)
                result, report = recover_context_groups(original, wave, Model(), witness, protocol)
                self.assertEqual(len(set(calls)), 3)
                self.assertTrue(all(begin >= .4 for begin, _ in calls))
                self.assertEqual(result[0], original[0])
                if mode == 'valid':
                    self.assertEqual([u['start'] for u in result[1]['units']], [1., 1.5, 2., 2.4])
                    self.assertEqual(result[1]['units'][0]['end'], 1.5)
                    self.assertFalse(highlight_coverage_errors(result[1:], 'Chinese'))
                    self.assertTrue(report[0]['accepted'])
                else:
                    self.assertEqual(result, original)
                    self.assertFalse(report[0]['accepted'])

    def test_repeated_characters_need_verified_reference_and_target_audio_agreement(self):
        from unittest.mock import patch
        from cjk_recovery import recover_repeated_groups
        import copy
        cues = [dict(text='甲乙丙丁戊', aligned_start=1., aligned_end=2.5, units=[
                    dict(text='甲乙', first=0, stop=2, start=1., end=1.9),
                    *[dict(text=c, first=i+2, stop=i+3, start=s, end=e)
                      for i, (c, s, e) in enumerate(zip('丙丁戊', [1.9, 2.1, 2.3], [2.1, 2.3, 2.5]))]]),
                dict(text='甲乙丙丁戊', aligned_start=3.1, aligned_end=4.5, units=[
                    dict(text=c, first=i, stop=i+1, start=s, end=e, timing_source='cross_model_ctc')
                    for i, (c, s, e) in enumerate(zip('甲乙丙丁戊', [3.1, 3.7, 3.9, 4.1, 4.3], [3.7, 3.9, 4.1, 4.3, 4.5]))])]
        for mode in ('valid', 'stale_neighbor', 'unverified_neighbor',
                     'unverified_reference', 'poor_match', 'divergent_paths',
                     'divergent_witness', 'missing_witness', 'silent'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                original = Path(folder)/'original.wav'
                wave = np.full(44100*6, 0. if mode == 'silent' else .1)
                sf.write(original, wave, 44100)
                before = copy.deepcopy(cues)
                if mode == 'unverified_reference':
                    before[1]['units'][0]['timing_source'] = 'prior_word_fallback'
                if mode in ('stale_neighbor', 'unverified_neighbor'):
                    before[0]['units'][0]['end'] = before[0]['units'][1]['start'] = 1.71
                if mode == 'unverified_neighbor':
                    before[1]['units'][2]['timing_source'] = 'prior_word_fallback'
                class Witness:
                    original_audio = original

                    def __call__(self, cue, audio):
                        starts = [1.11, 1.71, 1.91, 2.11, 2.31]
                        values = [[dict(text=c, first=i, stop=i+1, start=s, end=s+.1)
                                   for i, (c, s) in enumerate(zip('甲乙丙丁戊', starts))]
                                  for _ in range(2)]
                        if mode == 'divergent_witness':
                            values[1][0]['start'] += .3
                        if mode == 'missing_witness':
                            values.pop()
                        return values

                calls = []
                def path(features, source, target):
                    calls.append((source, target))
                    shift = .8 if mode == 'divergent_paths' and len(calls) == 1 else 0.
                    return np.array([3., 4., 5.]), np.array([1., 2., 3.])+shift, .99 if mode == 'poor_match' else .01

                with patch('align_repeated_lyrics.audio_features', return_value={'pitch': np.ones((12, 600)), 'spectrum': None}), \
                     patch('align_repeated_lyrics.audio_path', side_effect=path):
                    result, reports = recover_repeated_groups(before, wave, Witness(), dict(maximum_spread_seconds=.08))
                self.assertEqual(result[1], before[1])
                if mode in ('valid', 'stale_neighbor'):
                    self.assertEqual(len(calls), 4)
                    self.assertTrue(reports[0]['accepted'])
                    self.assertEqual([u['start'] for u in result[0]['units']], [1.1, 1.7, 1.9, 2.1, 2.3])
                    self.assertFalse(highlight_coverage_errors(result, 'Chinese'))
                else:
                    self.assertEqual(result, before)
                    self.assertFalse(any(r['accepted'] for r in reports))

    def test_repeated_alignment_handles_a_key_change_without_moving_onsets(self):
        from align_repeated_lyrics import audio_path, repeated_paths
        # The same changing phrase returns three seconds later, two semitones
        # higher. Timbral evidence is unchanged and supplies the key estimate.
        rng = np.random.default_rng(53)
        pitch = np.full((12, 300), .01)
        for block, note in enumerate((0, 4, 7, 2, 5, 9, 0, 7, 4, 11)):
            pitch[note, block*30:(block+1)*30] = 1.
        pitch /= np.linalg.norm(pitch, axis=0)
        spectrum = rng.normal(size=(12, 300))
        features = dict(pitch=np.concatenate([pitch, np.roll(pitch, 2, axis=0)], axis=1),
                        spectrum=np.concatenate([spectrum, spectrum], axis=1))
        source = dict(aligned_start=1., aligned_end=2.)
        target = dict(aligned_start=4., aligned_end=5.)
        paths, shift = repeated_paths(features, source, target)
        self.assertEqual(shift, 10)
        self.assertGreater(audio_path(features['pitch'], source, target)[2], .35)
        for _, (x, y, cost) in paths:
            self.assertLess(cost, .01)
            np.testing.assert_allclose(np.interp([1.1, 1.6, 1.9], x, y), [4.1, 4.6, 4.9], atol=.01)

    def test_joint_recovery_can_use_one_line_when_neighboring_context_is_ambiguous(self):
        from cjk_recovery import recover_context_groups
        from phoneme_progress import spoken_units
        times = {'前': (.1, .4), '甲': (1., 1.5), '乙': (1.5, 2.),
                 '丙': (2., 2.3), '丁': (2.4, 3.), '后': (4., 4.5)}
        def units(text):
            return [dict(u, start=times[u['text']][0], end=times[u['text']][1])
                    for u in spoken_units(text, 'Chinese')]
        original = [dict(text=t, aligned_start=times[t[0]][0], aligned_end=times[t[-1]][1], units=units(t))
                    for t in ('前', '甲乙丙丁', '后')]
        original[1]['units'] = [dict(units('甲')[0], end=1.8),
            dict(text='乙丙', first=1, stop=3, start=1.8, end=2.3), units('甲乙丙丁')[-1]]
        for disagree in (False, True):
            with self.subTest(disagree=disagree):
                class Model:
                    def infer(self, audio, begin, parsed, cue):
                        values = units(cue['text'])
                        if len(values) > 4 or disagree:
                            values = [dict(u, start=u['start']+.2) for u in values]
                        return dict(units=values)
                result, reports = recover_context_groups(original, np.full(44100*6, .1), Model(),
                    lambda cue, wave: [units(cue['text']), units(cue['text'])],
                    dict(maximum_spread_seconds=.08, minimum_relative_rms=.03))
                self.assertEqual(result[0], original[0])
                self.assertEqual(result[2], original[2])
                if disagree:
                    self.assertEqual(result, original)
                else:
                    self.assertFalse(highlight_coverage_errors(result, 'Chinese'))
                    self.assertTrue(reports[0]['accepted'])
                    self.assertEqual(reports[0]['attempts'][-1]['context_radius'], 0)

    def test_alternate_witness_keeps_raw_invalid_timestamps_and_checks_word_identity(self):
        from types import SimpleNamespace
        from qwen_witness import raw_units
        cue = dict(text="甲乙 let's go")
        values = [SimpleNamespace(text=text, start_time=start, end_time=end)
                  for text, start, end in [('甲', .5, .8), ('乙', .5, .5), ("let's", 1., 1.4), ('go', 1.4, 1.7)]]
        units = raw_units(cue, values, 10.)
        self.assertEqual([u['start'] for u in units], [10.5, 10.5, 11., 11.4])
        self.assertEqual(units[1]['end'], units[1]['start'])
        self.assertEqual(units[2]['text'], "let's")
        for bad in (values[:-1], [SimpleNamespace(text='丙', start_time=.5, end_time=.8), *values[1:]]):
            with self.assertRaisesRegex(ValueError, 'displayed units'):
                raw_units(cue, bad, 10.)

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
        with self.assertRaisesRegex(ValueError, 'Insufficient acoustic frames'):
            ctc_word_units(logits[:, :3], units, vocab, 10., .01)
        repeats = [dict(text=c, first=i, stop=i+1, phones=['a']) for i,c in enumerate('甲乙')]
        with self.assertRaisesRegex(ValueError, 'Insufficient acoustic frames'):
            ctc_word_units(logits[:, :2], repeats, vocab, 10., .01)
        from align_english_onsets import EnglishOnsetAligner
        aligner = EnglishOnsetAligner.__new__(EnglishOnsetAligner)
        aligner.method, aligner.device = 'wav2vec2', 'cpu'
        aligner.tokenize = lambda words: ([1, 1], [(0, 2)])
        aligner.model = lambda audio: (torch.zeros(1, 1, 3), None)
        with self.assertRaisesRegex(ValueError, 'Insufficient acoustic frames'):
            aligner.align(np.zeros(400, dtype='float32'), ['keep'])
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

    def test_playback_fallback_limits_estimation_to_broken_word_groups(self):
        from subtitles import playback_units
        for language, text, spans in [('Chinese', '今晚一起唱', [(0, 1), (1, 4), (4, 5)]),
                                      ('English', 'We sing this now', [(0, 3), (3, 13), (13, 16)])]:
            with self.subTest(language=language):
                units = [dict(text=text[a:b], first=a, stop=b, start=i+1., end=i+1.5)
                         for i, (a, b) in enumerate(spans)]
                cue = dict(text=text, start=0., end=5., units=units)
                result = playback_units(cue, language)
                self.assertEqual(result[0], units[0])
                self.assertEqual(result[2], units[2])
                self.assertEqual(result[1]['start'], units[1]['start'])
                self.assertEqual(result[1]['end'], units[1]['end'])
                self.assertEqual(result[1]['timing_source'], 'linear_fallback')
                self.assertTrue(highlight_coverage_errors([dict(cue, units=result)], language))
                self.assertNotIn('timing_source', units[1])

    def test_playback_missing_words_use_neighbor_bounds_not_a_whole_line_reset(self):
        from subtitles import playback_units
        first = dict(text='甲', first=0, stop=1, start=1., end=2.)
        last = dict(text='丁', first=3, stop=4, start=4., end=5.)
        cue = dict(text='甲乙丙丁', start=0., end=6., units=[first, last])
        result = playback_units(cue, 'Chinese')
        self.assertEqual(result[0], first)
        self.assertEqual(result[-1], last)
        self.assertEqual((result[1]['text'], result[1]['start'], result[1]['end']), ('乙丙', 2., 4.))
        # One word with a fabricated interval still cannot pass strict acceptance.
        missing = dict(text='hello', start=1., end=3., units=[])
        estimate = playback_units(missing, 'English')
        self.assertTrue(highlight_coverage_errors([dict(missing, units=estimate)], 'English'))
        # Overlapping tails expand the local fallback only as far as necessary.
        overlap = playback_units(dict(cue, units=[dict(first, end=4.2), last]), 'Chinese')
        self.assertEqual(overlap[-1], last)
        self.assertEqual((overlap[0]['text'], overlap[0]['start'], overlap[0]['end']), ('甲乙丙', 1., 4.))

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

    def test_display_holds_allow_overlapping_tails_but_never_crossed_word_onsets(self):
        from copy import deepcopy
        for tail in (3., 6.):
            cues = [{'text': '甲', 'start': 1., 'end': tail, 'status': 'candidate',
                     'units': [dict(text='甲', start=1., end=tail)]},
                    {'text': '乙', 'start': 1.8, 'end': 4., 'status': 'candidate',
                     'units': [dict(text='乙', start=1.8, end=4.)]}]
            original = deepcopy(cues)
            display, _ = presentation_cues(cues, 10., lead=0)
            self.assertEqual(cues, original)
            self.assertEqual(display[0]['end'], display[1]['start'])
            self.assertEqual(display[1]['start'], 1.8)
            self.assertEqual(display[0]['aligned_end'], tail)
            self.assertEqual([c['units'] for c in display], [c['units'] for c in cues])
        cues[0]['units'].append(dict(text='丙', start=2., end=2.2))
        with self.assertRaisesRegex(ValueError, 'Unordered'):
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

    def test_japanese_spelling_recovery_keeps_existing_anchors_and_missing_vocals(self):
        from karaoke.song_sources import lyric_coverage
        lines = ['赤い月', '駄目', '白い雲', '静かな朝']
        words = [dict(word=text, start=start, end=start+1.) for text, start in
                 [('赤い月', 1.), ('ダメ', 3.), ('白い雲', 5.), ('静かな朝', 16.)]]
        words[3:3] = [dict(word='らららら', start=float(i), end=i+.5) for i in range(7, 15)]
        cues = anchors_from_words(lines, words, 'Japanese')
        self.assertEqual([c['text'] for c in cues], lines)
        self.assertEqual([(c['start'], c['end']) for c in cues],
                         [(1., 2.), (3., 4.), (5., 6.), (16., 17.)])
        self.assertEqual(cues[1]['match_basis'], 'pronunciation_recovery')
        self.assertTrue(all(cues[i]['matched_fraction'] == 1. for i in (0, 2, 3)))
        coverage = lyric_coverage(cues, words, 'Japanese', 18.)
        self.assertEqual(coverage['status'], 'incomplete')
        self.assertEqual(coverage['uncovered_passages'],
                         [dict(start=6., end=16., recognized_units=32)])

    def test_japanese_spelling_recovery_cannot_borrow_words_across_resolved_lines(self):
        words = [dict(word=text, start=float(i*2), end=float(i*2+1)) for i, text in
                 enumerate(['ダメ', '赤い月', 'ねこ', '白い雲', 'ダメ'])]
        cues = anchors_from_words(['赤い月', '駄目', '白い雲'], words, 'Japanese', allow_missing=True)
        self.assertEqual([(c['start'], c['end']) for c in cues],
                         [(2., 3.), (None, None), (6., 7.)])
        self.assertEqual(cues[1]['status'], 'unresolved_text')
        with self.assertRaisesRegex(ValueError, 'Insufficient ASR support for lyric line 2'):
            anchors_from_words(['赤い月', '駄目', '白い雲'], words, 'Japanese')

    def test_japanese_spelling_recovery_handles_recording_edges(self):
        lines = ['駄目', '赤い月', '駄目']
        words = [dict(word=text, start=float(i*2), end=float(i*2+1)) for i, text in
                 enumerate(['ダメ', '赤い月', 'ダメ'])]
        cues = anchors_from_words(lines, words, 'Japanese')
        self.assertEqual([(c['start'], c['end']) for c in cues],
                         [(0., 1.), (2., 3.), (4., 5.)])

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

    def test_omitted_lyric_cannot_be_located_by_scattered_dialogue_matches(self):
        lines = ['star light', 'clear sky', 'the closing phrase']
        dialogue = 'qzxqzx'.join('starlightclearsky')
        words = [dict(word=char, start=float(i), end=float(i+1)) for i,char in enumerate(dialogue)]
        words.append(dict(word=lines[-1],start=150.,end=151.))
        cues = anchors_from_words(lines,words,'English',allow_missing=True)
        self.assertTrue(all(c['matched_fraction']==1. and c['matched_density']<.5 for c in cues[:2]))
        self.assertEqual([c['status'] for c in cues],['unresolved_text','unresolved_text','candidate'])
        self.assertTrue(all(c['start'] is None and 'search_start' not in c for c in cues[:2]))
        self.assertEqual((cues[-1]['start'],cues[-1]['end']),(150.,151.))

    def test_text_density_preserves_partial_recognition_and_long_held_notes(self):
        words = [dict(word=text,start=t,end=t+10.) for text,t in [('star',1.),('light',20.)]]
        cue = anchors_from_words(['star lights'],words,'English')[0]
        self.assertGreater(cue['matched_fraction'],.5)
        self.assertEqual(cue['matched_density'],1.)
        self.assertEqual((cue['start'],cue['end']),(1.,30.))

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
        from word_onset_metrics import onset_risks, score_risks

        events = word_events([dict(text='轻唱', units=[dict(first=0, stop=2, start=1.)])], 'Chinese')
        self.assertEqual([e['text'] for e in events], ['轻', '唱'])
        self.assertTrue(all(r['review'] for r in onset_risks(events, events)))
        # Even close onsets cannot make one shared highlight two independent ones.
        reference = [dict(id=0, text='轻', start=1.), dict(id=1, text='唱', start=1.04)]
        accuracy = score_onsets(reference, events)
        score = accuracy['summary']
        self.assertEqual(score['pco']['50'], 1.)
        self.assertFalse(score['passed'])
        detected = score_risks(accuracy, onset_risks(events, events))
        self.assertEqual(detected['true_positive'], 2)
        self.assertEqual(detected['false_positive'], 0)
        independent = [dict(r) for r in reference]
        self.assertTrue(score_onsets(reference, independent)['summary']['passed'])
        independent[-1]['incomplete_highlight'] = True
        self.assertFalse(score_onsets(reference, independent)['summary']['passed'])
        detected = score_risks(score_onsets(reference, independent), onset_risks(independent, reference))
        self.assertEqual(detected['true_positive'], 1)
        self.assertEqual(detected['true_negative'], 1)
        # A verifier defect alone must not turn a correct prediction into an error.
        detected = score_risks(score_onsets(reference, reference), onset_risks(reference, independent))
        self.assertEqual(detected['true_positive'], 0)
        self.assertEqual(detected['false_positive'], 1)
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'lyrics.json').write_text(json.dumps(dict(language='English', lines=lines)))
            for scale in (1, 4):
                anchors = [dict(text=s, status='candidate', start=t*scale, end=(t+1)*scale)
                           for s, t in zip(lines, (1., 4., 7., 10., 13., 16.))]
                scope = [(i, i+1, i, i+1, 0., 30.*scale) for i in range(6)]
                scope[3] = (3, 4, 3, 4, 9.4*scale, 11.4*scale)
                sf.write(root/'audio.wav', np.zeros(30*scale*16000), 16000)
                (root/'anchors.json').write_text(json.dumps(anchors))
                for mode in ('repair', 'outside_scope', 'new_conflict'):
                    with self.subTest(mode=mode, scale=scale):
                        def align(audio, words):
                            times = {6: [t*scale for t in (1., 4., 9., 18., 20., 23.)],
                                     3: [.1, .5*scale+.6, 6*scale+.6],
                                     4: [.1, .4, 6*scale+.6, 9*scale+.6],
                                     5: [t*scale+.6 for t in (0, 3, 6, 9, 12)]}[len(words)]
                            if len(words) == 5:
                                if mode == 'outside_scope': times[3] = 18.*scale
                                if mode == 'new_conflict': times[2] = 2.*scale
                            return [dict(text=w, start=t, end=t+.2, confidence=.9)
                                    for w, t in zip(words, times)], None, .02
                        model = SimpleNamespace(method='singing', device='cpu', onset_ratio=.02,
                                                model_identity={}, align=Mock(side_effect=align))
                        output = root/f'{mode}-{scale}'
                        with patch('align_english_onsets.context_blocks', return_value=list(scope)):
                            report = align_job(model, dict(audio=str(root/'audio.wav'), lyrics=str(root/'lyrics.json'),
                                anchors=str(root/'anchors.json'), output=str(output)), full_song=True)
                        result = json.loads((output/'cues.json').read_text())
                        self.assertEqual(model.align.call_count, 4)
                        self.assertEqual(len(report['boundary_retries']), 2)
                        self.assertFalse(report['boundary_retries'][0]['accepted'])
                        self.assertEqual(report['boundary_retries'][1]['accepted'], mode == 'repair')
                        self.assertEqual(report['scope_blocks'][3], (3, 4, 2, 5, 7*scale-.6, 14*scale+.6))
                        expected = [1., 4., 7., 10., 20., 23.] if mode == 'repair' else [1., 4., 9., 7.5, 20., 23.]
                        np.testing.assert_allclose([c['start'] for c in result], np.array(expected)*scale)
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

    def test_changed_line_uses_local_audio_before_a_whole_block_retry(self):
        from copy import deepcopy
        from unittest.mock import Mock
        lines = ['one line', 'two lines', 'three lines', 'four lines', 'five lines']
        good = dict(segments=[dict(text=text, start=i*3+1., end=i*3+2., words=[
            dict(word=word, start=i*3+1.+j*.5, end=i*3+1.5+j*.5)
            for j, word in enumerate(text.split())]) for i, text in enumerate(lines)])
        original = deepcopy(good)
        original['segments'][2]['words'][0]['word'] = 'wrong'
        calls = []
        def align(begin, end, text, **options):
            calls.append((begin, end, text))
            candidate = deepcopy(good)
            candidate['segments'] = candidate['segments'][1:4]
            for segment in candidate['segments']:
                for word in segment['words']:
                    word['start'] -= begin
                    word['end'] -= begin
            return candidate
        full = Mock(return_value=good)
        repaired, report = recover_changed_text(original, lines, 16., full, align=align)
        self.assertEqual(repaired, good)
        self.assertEqual(calls, [(3.4, 11.6, '\n'.join(lines[1:4]))])
        full.assert_not_called()
        self.assertTrue(report['accepted'])
        self.assertEqual(report['original_result'], original)
        full.reset_mock()
        repaired, report = recover_changed_text(original, lines, 16., full, align=lambda *a, **k: None)
        self.assertEqual(repaired, good)
        full.assert_called_once_with()

    def test_foreign_lyrics_retry_alignment_in_clip_language_before_asr(self):
        import json
        import shutil
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from align_lyrics_whisper import main

        lines, starts = ['Guten Morgen', 'Komm zurück', 'あしたも歌おう'], [1., 4., 8.]
        recognized = False
        def align(samples, text, **options):
            begin = float(samples[0])
            segments = []
            for line in text.splitlines():
                start = starts[lines.index(line)]-begin
                words = [dict(word=word, start=start+i*.4, end=start+(i+1)*.4, probability=.9)
                         for i, word in enumerate(line.split())]
                if line == lines[1] and not (recognized or options.get('language') == 'de'):
                    for word in words:
                        word.update(start=start, end=start)
                segments.append(dict(text=line, start=words[0]['start'], end=words[-1]['end'], words=words))
            return SimpleNamespace(to_dict=lambda: dict(segments=segments))
        def transcribe(samples, language, **options):
            nonlocal recognized
            recognized = language in (None, 'de')
            begin = float(samples[0])
            words = [dict(word=line if recognized else 'コム', start=start-begin, end=start+.8-begin)
                     for line, start in zip(lines, starts)]
            return SimpleNamespace(to_dict=lambda: dict(language='de' if recognized else 'ja',
                segments=[dict(words=words)]))
        model = Mock()
        model._buffers = {'alignment_heads': None}
        model.dims.n_mels, model.device = 80, 'cpu'
        model.detect_language.return_value = (None, {'de': .9, 'ja': .1})
        model.align.side_effect, model.transcribe.side_effect = align, transcribe
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root/'audio.wav', np.arange(12*16000)/16000, 16000, subtype='FLOAT')
            (root/'lyrics.json').write_text(json.dumps(dict(language='Japanese', lines=lines)))
            (root/'jobs.json').write_text(json.dumps([dict(audio=str(root/'audio.wav'),
                lyrics=str(root/'lyrics.json'), output=str(root/'alignment'))]))
            with patch('stable_whisper.load_model', return_value=model), patch(
                    'align_lyrics_whisper.sha', return_value='test-model-hash'), patch('sys.argv',
                    ['align', '--jobs', str(root/'jobs.json'), '--device', 'cpu']):
                main()
            cues = json.loads((root/'alignment/cues.json').read_text())
            self.assertEqual([c['text'] for c in cues], lines)
            np.testing.assert_allclose([c['start'] for c in cues], starts, atol=.001)

            model.transcribe.assert_not_called()
            self.assertEqual(json.loads((root/'alignment/run.json').read_text())['local_alignment_languages'][0]['language'], 'de')
            # If direct retries still fail, recognition detects the crop language
            # independently rather than forcing the song's dominant language.
            shutil.rmtree(root/'alignment')
            model._buffers = {'alignment_heads': None}
            model.detect_language.return_value = (None, {'ja': 1.})
            with patch('stable_whisper.load_model', return_value=model), patch(
                    'align_lyrics_whisper.sha', return_value='test-model-hash'), patch('sys.argv',
                    ['align', '--jobs', str(root/'jobs.json'), '--device', 'cpu']):
                main()
            self.assertTrue(recognized)
            cues = json.loads((root/'alignment/cues.json').read_text())
            self.assertEqual([c['text'] for c in cues], lines)
            np.testing.assert_allclose([c['start'] for c in cues], starts, atol=.001)

            # A long localization must not inherit an intro's language. The
            # existing local-ASR policy votes across interior audio windows.
            starts[:] = [31., 65., 100.]
            recognized = False
            sf.write(root/'audio.wav', np.arange(120*16000)/16000, 16000, subtype='FLOAT')
            shutil.rmtree(root/'alignment')
            model._buffers = {'alignment_heads': None}
            model.transcribe.reset_mock()
            model.detect_language.side_effect = [(None, {code: 1.}) for code in ('en', 'de', 'de')]
            with patch('stable_whisper.load_model', return_value=model), patch(
                    'align_lyrics_whisper.sha', return_value='test-model-hash'), patch('sys.argv',
                    ['align', '--jobs', str(root/'jobs.json'), '--device', 'cpu']):
                main()
            self.assertEqual(model.transcribe.call_args.kwargs['language'], 'de')
            cues = json.loads((root/'alignment/cues.json').read_text())
            np.testing.assert_allclose([c['start'] for c in cues], starts, atol=.001)
            self.assertEqual([c['text'] for c in cues], lines)

    def test_known_text_localizer_falls_back_when_specialized_vocabulary_cannot_read_the_lyrics(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from align_lyrics_whisper import main
        from subtitles import highlight_spans

        for middle, oom in [('还是想你', False), ('唱到 Open Sky', False), ('唱到 Open Sky', True)]:
            with self.subTest(middle=middle, oom=oom), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lines, starts = ['你好', middle, '再见'], [1., 4., 8.]
                def align(samples, text, **options):
                    begin = float(samples[0])
                    segments = []
                    for line in text.splitlines():
                        start = starts[lines.index(line)]-begin
                        words = [dict(word=line[a:b], start=start+i*.3, end=start+i*.3+.2, probability=.9)
                                 for i,(a,b) in enumerate(highlight_spans(line, 'Chinese'))]
                        if line == middle:
                            for word in words:
                                word.update(start=start, end=start)
                        segments.append(dict(text=line, words=words, start=start, end=words[-1]['end']))
                    return SimpleNamespace(to_dict=lambda: dict(segments=segments))
                measured = [dict(first=a, stop=b, text=middle[a:b], start=4.+i*.3, end=4.2+i*.3)
                            for i,(a,b) in enumerate(highlight_spans(middle, 'Chinese'))]
                def witness(cue, wave, **options):
                    self.assertEqual(cue['text'], middle)
                    self.assertLess(cue['audio_bounds'][0], 4.)
                    self.assertEqual(cue['audio_bounds'][1], 8.)
                    if oom:
                        raise RuntimeError('CUDA out of memory')
                    return [measured, measured]
                model = Mock()
                model._buffers = {'alignment_heads': None}
                model.dims.n_mels, model.device = 80, 'cpu'
                model.detect_language.return_value = (None, {'zh': 1.})
                model.align.side_effect = align
                singing = Mock(checkpoint=root/'model')
                singing.infer.return_value = dict(units=measured)
                sf.write(root/'audio.wav', np.arange(12*16000)/16000, 16000, subtype='FLOAT')
                sf.write(root/'original.wav', np.arange(12*44100)/44100, 44100, subtype='FLOAT')
                (root/'lyrics.json').write_text(json.dumps(dict(language='Chinese', lines=lines)))
                (root/'jobs.json').write_text(json.dumps([dict(audio=str(root/'audio.wav'),
                    original_audio=str(root/'original.wav'), lyrics=str(root/'lyrics.json'),
                    localize_with_asr=False, output=str(root/'alignment'))]))
                with patch('stable_whisper.load_model', return_value=model), patch(
                        'align_lyrics_whisper.sha', return_value='test-model-hash'), patch(
                        'cjk_recovery.AcousticWitness', return_value=witness) as multilingual, patch(
                        'align_lyric_phonemes.SingingAligner', return_value=singing) as specialized, patch(
                        'sys.argv', ['align', '--jobs', str(root/'jobs.json'), '--device', 'cpu']):
                    if oom:
                        with self.assertRaisesRegex(RuntimeError, 'CUDA out of memory'):
                            main()
                        continue
                    main()
                model.transcribe.assert_not_called()
                cues = json.loads((root/'alignment/cues.json').read_text())
                self.assertEqual([c['text'] for c in cues], lines)
                np.testing.assert_allclose([c['start'] for c in cues], starts, atol=.001)
                report = json.loads((root/'alignment/run.json').read_text())
                recovery = next(r for r in report['line_recoveries'] if r.get('isolated') and r['accepted'])
                evidence = recovery['acoustic_fallback']
                if 'Open' in middle:
                    specialized.assert_not_called()
                    multilingual.assert_called_once()
                    self.assertEqual(evidence['model'], 'MMS_FA')
                    self.assertIn('Unsupported Chinese pronunciation', evidence['specialized_pronunciation_error'])
                else:
                    specialized.assert_called_once()
                    multilingual.assert_not_called()
                    self.assertEqual(evidence['model'], 'SOFA_CTC')

    def test_rejected_alignment_retains_raw_evidence_without_publishing_captions(self):
        import json
        import shutil
        from unittest.mock import Mock, patch
        from align_lyrics_whisper import main
        from timing_validation import validate_timeline

        result = dict(segments=[dict(text='extra word', start=1., end=2., words=[
            dict(word='extra word', start=1., end=2., probability=.8)])])
        model = Mock()
        model._buffers = {'alignment_heads': None}
        model.align.return_value.to_dict.return_value = result
        model.transcribe.return_value.to_dict.return_value = dict(segments=[])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root/'audio.wav', np.zeros(3*16000), 16000)
            (root/'lyrics.json').write_text(json.dumps(dict(language='English', lines=['word'])))
            (root/'jobs.json').write_text(json.dumps([dict(audio=str(root/'audio.wav'),
                lyrics=str(root/'lyrics.json'), output=str(root/'alignment'))]))
            with patch('stable_whisper.load_model', return_value=model), patch(
                    'align_lyrics_whisper.sha', return_value='test-model-hash') as hash_file, patch('sys.argv',
                    ['align', '--jobs', str(root/'jobs.json'), '--device', 'cpu']):
                main()
                shutil.rmtree(root/'alignment')
                model._buffers = {'alignment_heads': None}
                main()
            self.assertEqual(model.transcribe.call_count, 1)
            self.assertEqual(sum(Path(call.args[0]).suffix == '.pt' for call in hash_file.call_args_list), 2)
            self.assertEqual(json.loads((root/'alignment/raw.json').read_text())[0]['result'], result)
            self.assertEqual(json.loads((root/'alignment/units.json').read_text())[0]['text'], 'extra word')
            cues = json.loads((root/'alignment/cues.json').read_text())
            self.assertEqual(cues[0]['text'], 'word')
            self.assertNotEqual(cues[0]['status'], 'candidate')
            with self.assertRaises(ValueError):
                validate_timeline(cues, ['word'], 3.)
            jobs = json.loads((root/'jobs.json').read_text())
            jobs[0]['localize_with_asr'] = False
            (root/'jobs.json').write_text(json.dumps(jobs))
            shutil.rmtree(root/'alignment')
            model._buffers = {'alignment_heads': None}
            with patch('stable_whisper.load_model', return_value=model), patch(
                    'align_lyrics_whisper.sha', return_value='test-model-hash'), patch('sys.argv',
                    ['align', '--jobs', str(root/'jobs.json'), '--device', 'cpu']):
                main()
            self.assertEqual(model.transcribe.call_count, 1)
            self.assertFalse(json.loads((root/'alignment/raw.json').read_text())[0].get('asr_localizations'))
            # Deliberately rejected word timings are not a text mutation. Keep
            # the unresolved candidate for source fallback, but never publish it.
            result['segments'] = [dict(text='word', words=[])]
            for language in ('English', 'Chinese'):
                (root/'lyrics.json').write_text(json.dumps(dict(language=language, lines=['word'])))
                shutil.rmtree(root/'alignment')
                model._buffers = {'alignment_heads': None}
                with patch('stable_whisper.load_model', return_value=model), patch(
                        'align_lyrics_whisper.sha', return_value='test-model-hash'), patch('sys.argv',
                        ['align', '--jobs', str(root/'jobs.json'), '--device', 'cpu']):
                    main()
                cues = json.loads((root/'alignment/cues.json').read_text())
                self.assertEqual(cues[0]['text'], 'word')
                self.assertNotEqual(cues[0]['status'], 'candidate')
                with self.assertRaises(ValueError):
                    validate_timeline(cues, ['word'], 3.)

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
                   segment('middle', 2.6, 5.)])),
               dict(range=[2, 3], context=[1, 3], source_range=[0., 6.], result=dict(segments=[
                   segment('middle', 0., 2.), segment('last', 2.5, 3.5)]))]
        cues, stitched, repairs = stitch_blocks(lines, raw, 6., 'English')
        presentation_cues(cues, 6., lead=0, tail=0, bridge_gap=0)
        words, report = build_progress(cues, stitched, 'English')
        self.assertFalse(report['fallback_lines'])
        self.assertEqual([(c['start'], c['end']) for c in cues], [(0., 1.), (1.2, 2.), (2.5, 3.5)])
        self.assertEqual([c['units'][0]['start'] for c in words], [0., 1.2, 2.5])
        self.assertEqual(repairs[0]['selected_block'], 0)
        self.assertEqual(repairs[0]['lines'], [2, 2])
        self.assertEqual(raw[1]['result']['segments'][0]['words'][0]['end'], 5.)

    def test_missing_line_does_not_prevent_reusing_a_complete_context_prediction(self):
        from copy import deepcopy
        from karaoke.pipeline import alignment_recovery_summary
        lines = ['first', 'middle', 'last']
        def segment(text, start):
            return dict(text=text, words=[dict(word=text,start=start,end=start+.5)])
        missing = dict(range=[0,3],context=[0,3],source_range=[0.,6.],result=dict(segments=[
            segment('first',1.),dict(text='middle',words=[]),segment('last',5.)]))
        missing['asr_localizations'] = [dict(contradicted_lines=[2], accepted_lines=[])]
        context = dict(range=[0,0],context=[0,3],source_range=[0.,6.],result=dict(segments=[
            segment(text,start) for text,start in zip(lines,[1.,3.,5.])]))
        original = deepcopy(missing)
        cues, stitched, repairs = stitch_blocks(lines,[missing,context],6.,'English')
        self.assertEqual([c['start'] for c in cues],[1.,3.,5.])
        self.assertEqual(missing,original)
        self.assertTrue(repairs)
        words, report = build_progress(cues,stitched,'English')
        self.assertFalse(report['fallback_lines'])
        self.assertEqual([c['units'][0]['text'] for c in words],lines)
        recovered = alignment_recovery_summary(stitched)
        self.assertEqual(recovered['unresolved_localization_lines'], [])
        self.assertIn(2, recovered['recovered_alignment_lines'])
        self.assertEqual(alignment_recovery_summary([missing])['unresolved_localization_lines'], [2])
        presentation_cues(cues,6.,lead=0,tail=0,bridge_gap=0)
        context['result']['segments'][1]['words'][0]['word'] = 'different text'
        _, rejected, _ = stitch_blocks(lines,[missing,context],6.,'English')
        self.assertEqual(alignment_recovery_summary(rejected)['unresolved_localization_lines'], [2])

    def test_missing_line_can_be_recovered_inside_measured_context(self):
        lines = ['alpha','bravo','charlie','delta','echo','foxtrot']
        good = [dict(text=text,words=[dict(word=text,start=float(i+1),end=i+1.5)]) for i,text in enumerate(lines)]
        raw = dict(range=[0,6],context=[0,6],source_range=[0.,8.],result=dict(segments=[
            dict(text=s['text'],words=[]) if i==2 else s for i,s in enumerate(good)]))
        for succeeds in (True,False):
            with self.subTest(succeeds=succeeds):
                def align(begin,end,text,**options):
                    if not succeeds:return None
                    return dict(segments=[dict(text=s['text'],words=[dict(w,start=w['start']-begin,end=w['end']-begin)
                        for w in s['words']]) for s in good if s['text'] in text.splitlines()])
                cues, stitched, repairs = stitch_blocks(lines,[raw],8.,'English',align)
                self.assertEqual(cues[2]['status'],'candidate' if succeeds else 'unresolved_timing')
                if succeeds:
                    self.assertEqual([c['start'] for c in cues],list(map(float,range(1,7))))
                else:
                    self.assertEqual(stitched[0]['result'],raw['result'])

    def test_joint_repair_expands_only_across_connected_edge_conflicts(self):
        from copy import deepcopy
        from unittest.mock import Mock
        lines = ['alpha', 'bravo', 'charlie', 'delta', 'omega']
        old, new = [1.,7.,9.,4.,13.], [1.1,3.,5.,7.,13.1]
        def result(starts, offset=0.):
            return dict(segments=[dict(text=t,words=[dict(word=t,start=s-offset,end=s+.8-offset)])
                                  for t,s in zip(lines,starts)])
        raw = dict(range=[0,5],context=[0,5],source_range=[0.,15.],result=result(old))
        original = deepcopy(raw)
        for cached in (True, False):
            with self.subTest(cached=cached):
                predictions = [raw]
                align = Mock(side_effect=lambda begin,end,text,**kw: result(new,begin))
                if cached:
                    predictions.append(dict(range=[0,0],context=[0,5],source_range=[0.,15.],result=result(new)))
                cues, stitched, repairs = stitch_blocks(lines,predictions,15.,'English',align)
                self.assertEqual([c['start'] for c in cues],[1.,3.,5.,7.,13.])
                self.assertEqual(raw,original)
                self.assertEqual(align.call_count,0 if cached else 1)
                presentation_cues(cues,15.,lead=0,tail=0,bridge_gap=0)
                self.assertEqual([b['range'] for b in stitched],[[0,1],[1,4],[4,5]])

    def test_joint_repair_extends_crop_when_crossing_reaches_context_edge(self):
        from copy import deepcopy
        lines = ['alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel']
        old = [(1.,1.8),(4.5,5.1),(5.,5.8),(7.,7.8),(9.,9.8),(6.,6.8),(13.,13.8),(15.,15.8)]
        new = [(1.,1.8),(4.,4.2),(4.4,5.),(5.,5.6),(5.6,6.2),(6.2,6.8),(13.,13.8),(15.,15.8)]
        def segment(text,times,offset=0.):
            return dict(text=text,words=[dict(word=text,start=times[0]-offset,end=times[1]-offset)])
        raw = dict(range=[0,8],context=[0,8],source_range=[0.,18.],result=dict(segments=[
            segment(t,s) for t,s in zip(lines,old)]))
        original, calls = deepcopy(raw), []
        def align(begin,end,text,**options):
            calls.append(text.splitlines())
            return dict(segments=[segment(t,new[lines.index(t)],begin) for t in text.splitlines()])
        cues, _, repairs = stitch_blocks(lines,[raw],18.,'English',align)
        self.assertEqual(calls,[lines[2:],lines[1:]])
        self.assertEqual(raw,original)
        self.assertEqual(repairs[0]['required_context'],[1,8])
        np.testing.assert_allclose([(c['start'],c['end']) for c in cues],new,rtol=0,atol=1e-12)
        presentation_cues(cues,18.,lead=0,tail=0,bridge_gap=0)

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

        # A new word crossing remains invalid. A long overlapping tail only
        # shortens the display hold; it does not invalidate measured onsets.
        for middle, end in [(3.5, 4.), (1.6, 5.)]:
            with self.subTest(middle=middle, end=end):
                raw[0]['result']['segments'][1] = segment(lines[1], 1.2, middle, end)
                cues, unchanged, repairs = stitch_blocks(lines, raw, 10., 'English')
                words, _ = build_progress(cues, unchanged, 'English')
                if middle == 3.5:
                    self.assertEqual(repairs, [])
                    with self.assertRaisesRegex(ValueError, 'Coarse lyric onsets'):
                        protect_line_order(words, words)
                else:
                    self.assertTrue(repairs)
                    self.assertEqual([[u['start'] for u in c['units']] for c in words],
                                     [[2., 3.], [3.2, 3.6], [5., 6.]])

    def test_unresolved_crossing_realigns_with_context_but_changes_only_core_lines(self):
        from copy import deepcopy
        from unittest.mock import Mock

        lines = ['left', 'context', 'first', 'second', 'after', 'tail']
        old_times = [(1., 1.5), (2., 2.5), (4.9, 5.4), (4.7, 6.4), (7., 7.5), (8., 8.5)]
        def result(times, offset=0.):
            return dict(segments=[dict(text=text, words=[dict(word=text, start=a-offset, end=b-offset)])
                                  for text, (a, b) in zip(lines, times)])
        raw = [dict(range=[i, i+1], context=[i, i+1], source_range=[0., 10.],
                    result=dict(segments=[s])) for i, s in enumerate(result(old_times)['segments'])]
        original = deepcopy(raw)
        new_times = [(1.1, 1.6), (2.2, 2.7), (3.2, 4.8), (4.9, 6.2), (7.1, 7.6), (8.1, 8.6)]
        for failure in (None, 'text', 'crossing', 'neighbour', 'outside_audio', 'missing', 'empty_attention', 'oom'):
            with self.subTest(failure=failure):
                candidate = result(new_times, .4)
                if failure == 'text': candidate['segments'][2]['words'][0]['word'] = 'changed'
                if failure == 'crossing': candidate['segments'][3]['words'][0]['start'] = 2.7
                if failure == 'neighbour': candidate['segments'][2]['words'][0]['start'] = 1.6
                if failure == 'outside_audio': candidate['segments'][-1]['words'][0]['end'] = 9.
                if failure == 'missing': candidate = None
                align = Mock(return_value=candidate)
                if failure in ('empty_attention', 'oom'):
                    align.side_effect = RuntimeError('selected index k out of range' if failure == 'empty_attention' else 'CUDA out of memory')
                if failure == 'oom':
                    with self.assertRaisesRegex(RuntimeError, 'CUDA out of memory'):
                        stitch_blocks(lines, raw, 10., 'English', align)
                    continue
                cues, stitched, repairs = stitch_blocks(lines, raw, 10., 'English', align)
                align.assert_called_once()
                self.assertEqual(align.call_args.args, (.4, 9.1, '\n'.join(lines)))
                self.assertEqual(raw, original)
                self.assertEqual(repairs[-1]['accepted'], failure is None)
                if failure == 'empty_attention':
                    self.assertIn('selected index k out of range', repairs[-1]['error'])
                else:
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
                    with self.assertRaisesRegex(ValueError, 'Unordered'):
                        presentation_cues(cues, 10., lead=0, tail=0, bridge_gap=0)

    def test_distant_context_does_not_block_a_bounded_boundary_repair(self):
        from copy import deepcopy
        lines = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot']
        for gap in ('before', 'after', 'inside_core'):
            with self.subTest(gap=gap):
                times = ([(1.,2.), (10.,11.), (93.,93.4), (92.7,94.4), (96.,97.), (110.,111.)]
                         if gap == 'before' else
                         [(1.,2.), (10.,11.), (23.,23.4), (22.7,24.4), (96.,97.), (110.,111.)])
                if gap == 'inside_core':
                    times[2:4] = [(93.,93.4), (12.,94.4)]
                segments = [dict(text=t,words=[dict(word=t,start=a,end=b)]) for t,(a,b) in zip(lines,times)]
                raw = dict(range=[0,6],context=[0,6],source_range=[0.,120.],result=dict(segments=segments))
                original, calls = deepcopy(raw), []
                def align(begin,end,text,**options):
                    calls.append((begin,end,text.splitlines()))
                    self.assertLessEqual(end-begin,45.)
                    selected = deepcopy([segments[lines.index(t)] for t in text.splitlines()])
                    for segment in selected:
                        word = segment['words'][0]
                        if word['word'] == 'charlie':word['start'] -= .6
                        if word['word'] == 'delta':word['start'] += .2
                        word['start'] -= begin
                        word['end'] -= begin
                    return dict(segments=selected)
                cues, stitched, repairs = stitch_blocks(lines,[raw],120.,'English',align)
                self.assertEqual(raw,original)
                if gap == 'inside_core':
                    self.assertEqual(calls,[])
                    self.assertEqual(stitched,[raw])
                    continue
                self.assertEqual(len(calls),1)
                self.assertEqual(calls[0][2],lines[2:] if gap == 'before' else lines[:4])
                self.assertEqual(repairs[0]['range'],[2,4])
                self.assertEqual([(cues[i]['start'],cues[i]['end']) for i in (0,1,4,5)],
                                 [times[i] for i in (0,1,4,5)])
                presentation_cues(cues,120.,lead=0,tail=0,bridge_gap=0)
                self.assertFalse(build_progress(cues,stitched,'English')[1]['fallback_lines'])

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
