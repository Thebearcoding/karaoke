"""Map verified lyric lines to rough ASR word times, preserving recording order."""

import argparse
from array import array
import json
from pathlib import Path

from Bio.Align import PairwiseAligner

from subtitles import normalized, write_captions


def text_units(text, language):
    value = normalized(text).lower()
    if language == 'Chinese':
        from pypinyin import lazy_pinyin
        return lazy_pinyin(value, errors=lambda s: list(s))
    return list(value)


def anchors_from_words(lines, words, language, allow_missing=False):
    keys = [text_units(line, language) for line in lines]
    expected = [unit for key in keys for unit in key]
    observed, owners = [], []
    for i, word in enumerate(words):
        units = text_units(word['word'], language)
        observed.extend(units)
        owners.extend([i]*len(units))
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
            if not allow_missing:
                raise ValueError(f'Insufficient ASR support for lyric line {len(cues)+1}: {coverage:.1%}')
            cues.append({'text': line, 'start': None, 'end': None,
                         'matched_fraction': coverage, 'status': 'unresolved_text'})
            position += len(key)
            continue
        start, end = words[min(matched)]['start'], words[max(matched)]['end']
        if end <= start and not allow_missing:
            raise ValueError(f'Collapsed ASR interval for lyric line {len(cues)+1}')
        cues.append({'text': line, 'start': round(start, 3), 'end': round(end, 3),
                     'matched_fraction': coverage, 'status': 'candidate' if end > start else 'unresolved_timing',
                     'source': 'asr_rough_anchor'})
        position += len(key)
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
