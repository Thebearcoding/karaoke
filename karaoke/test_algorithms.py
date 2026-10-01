import unittest
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from align import activity, align_lines
from dsp import apply_masks, istft, masks_for, stft
from exact_stems import exact_split, reconstruction_report
from phonetic_align import FormantSynth, alignment_features, align_phonetic, templates
from subtitles import countdown_events, countdown_state, lyric_cues, presentation_cues, read_ahead_state, timestamp, write_captions
from match_video_lyrics import caption_similarity, strongest_cluster
from anchors_from_asr import anchors_from_words
from three_stems import check_three
from lyric_progress import map_words, build as build_progress
from align_lyric_phonemes import guided_scores, protect_units
from phoneme_progress import reading_parts, japanese_phone_groups, spoken_units
from align_repeated_lyrics import agree_onset, audible, reference_pairs
from align_lyrics_whisper import alignment_blocks
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
        estimate = np.array([[2e-6, -2e-6]], dtype='float32')
        with self.assertRaisesRegex(ValueError, 'above the .* repair limit'):
            exact_split(source, estimate)
        with self.assertRaisesRegex(ValueError, 'above the .* repair limit'):
            exact_split(source, estimate/10, repair_limit=1e-8)

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

    def test_first_phoneme_keeps_its_onset_on_either_side_of_coarse_start(self):
        cue = {'text': '甲乙', 'aligned_start': 1., 'aligned_end': 2.,
               'units': [dict(first=0, stop=1, text='甲', start=1., end=1.4),
                         dict(first=1, stop=2, text='乙', start=1.5, end=2.)]}
        for onset in (.8, 1.2):
            candidate = [dict(cue['units'][0], start=onset), dict(cue['units'][1])]
            units, _ = protect_units(cue, candidate, 'Chinese')
            self.assertEqual(units[0]['start'], onset)

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
        self.assertEqual(units[0]['text'], '颳風')
        self.assertEqual((units[0]['start'], units[0]['end']), (4., 4.5))
        self.assertEqual(cue['text'][units[1]['first']:units[1]['stop']], '這天')
        self.assertEqual(units[1]['start'], 4.7)

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
        events = word_events([dict(text='hello world', units=[
            dict(first=0, stop=11, start=1.)])])
        reference = [dict(id=0, text='hello', start=1.), dict(id=1, text='world', start=1.8)]
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


class AlignmentIntegrity(unittest.TestCase):
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
