"""Map verified lyric lines to rough ASR word times, preserving recording order."""

import argparse
from array import array
import json
from pathlib import Path
import re

from Bio.Align import PairwiseAligner

if __package__:
    from .subtitles import normalized, write_captions
else:
    from subtitles import normalized, write_captions


def text_units(text, language):
    value = normalized(text).lower()
    if language == 'Chinese':
        from pypinyin import lazy_pinyin
        return lazy_pinyin(value, errors=lambda s: list(s))
    return list(value)


def _match_words(lines, words, units):
    keys = [units(line) for line in lines]
    expected = [unit for key in keys for unit in key]
    observed, owners = [], []
    for i, word in enumerate(words):
        value = units(word['word'])
        observed.extend(value)
        owners.extend([i]*len(value))
    correspondence = {}
    if expected and observed:
        alphabet = {unit: i for i, unit in enumerate(dict.fromkeys(expected+observed))}
        # Minimize ordinary edit distance first, then internal gap runs.
        # Otherwise scattered letters in a spoken intro can tie a contiguous
        # lyric match and falsely give a minute-long line 100% text coverage.
        base = len(expected)+len(observed)+1
        aligner = PairwiseAligner(mode='global', match_score=0, mismatch_score=-base,
                                  open_gap_score=-base-1, extend_gap_score=-base)
        aligner.open_end_gap_score = -base  # Intro/outro gaps do not fragment lyrics.
        alignment = aligner.align(array('i', (alphabet[u] for u in expected)),
                                  array('i', (alphabet[u] for u in observed)))[0]
        for (first, stop), (begin, end) in zip(*alignment.aligned):
            for i, j in zip(range(first, stop), range(begin, end)):
                if expected[i] == observed[j]:
                    correspondence[i] = owners[j]
    position = 0
    cues = []
    for line, key in zip(lines, keys):
        matched = [correspondence[i] for i in range(position, position+len(key)) if i in correspondence]
        coverage = len(matched)/len(key) if key else 0.
        if coverage < .5:
            cue = {'text': line, 'start': None, 'end': None,
                   'matched_fraction': coverage, 'status': 'unresolved_text'}
            prefix = []
            for i in range(position, position+len(key)):
                owner = correspondence.get(i)
                if owner is None or (prefix and owner > prefix[-1]+1):
                    break
                prefix.append(owner)
            # A recognized prefix can narrow a later acoustic search even
            # when the rest (often a language switch) is not recognized.
            # It is not a timing for the complete line.
            if len(set(prefix)) >= 3:
                cue['search_start'] = round(words[prefix[0]]['start'], 3)
            cues.append(cue)
            position += len(key)
            continue
        start, end = words[min(matched)]['start'], words[max(matched)]['end']
        cues.append({'text': line, 'start': round(start, 3), 'end': round(end, 3),
                     'matched_fraction': coverage, 'status': 'candidate' if end > start else 'unresolved_timing',
                     'source': 'asr_rough_anchor'})
        position += len(key)
    return cues


def japanese_reading_units(text):
    if re.search(r'[\u3040-\u30ff\u3400-\u9fff]', text):
        import pyopenjtalk
        text = pyopenjtalk.g2p(text, kana=True)
    return text_units(text, 'Japanese')


def anchors_from_words(lines, words, language, allow_missing=False):
    cues = _match_words(lines, words, lambda text: text_units(text, language))
    if language == 'Japanese':
        # Recover kana/kanji spelling differences only where written matching
        # failed. Phonetic matches must not move existing anchors or absorb
        # unlisted backing vocals into an otherwise resolved lyric line.
        original, first = list(cues), 0
        while first < len(cues):
            if original[first]['status'] == 'candidate':
                first += 1
                continue
            stop = first+1
            while stop < len(cues) and original[stop]['status'] != 'candidate':
                stop += 1
            begin = original[first-1]['end'] if first else 0.
            end = original[stop]['start'] if stop < len(cues) else max((w['end'] for w in words), default=0.)
            inside = [w for w in words if begin-.001 <= w['start'] < w['end'] <= end+.001]
            if inside and begin < end:
                for offset, cue in enumerate(_match_words(lines[first:stop], inside, japanese_reading_units)):
                    if cue['status'] == 'candidate':
                        cues[first+offset] = dict(cue, match_basis='pronunciation_recovery')
            first = stop
    if not allow_missing:
        for i, cue in enumerate(cues, 1):
            if cue['status'] == 'unresolved_text':
                raise ValueError(f"Insufficient ASR support for lyric line {i}: {cue['matched_fraction']:.1%}")
            if cue['status'] == 'unresolved_timing':
                raise ValueError(f'Collapsed ASR interval for lyric line {i}')
    return cues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lyrics', type=Path, required=True)
    parser.add_argument('--asr', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lyrics = json.loads(args.lyrics.read_text())
    asr = json.loads(args.asr.read_text())
    cues = anchors_from_words(lyrics['lines'], asr['words'], lyrics['language'], args.allow_missing)
    result = write_captions(cues, args.output)
    result['minimum_matched_fraction'] = min(c['matched_fraction'] for c in cues)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
