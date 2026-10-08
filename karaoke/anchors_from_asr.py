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
    correspondence, boundaries = {}, []
    if expected and observed:
        # An omitted verse must not consume words belonging to a later,
        # completely observed phrase. Establish those phrase boundaries before
        # asking edit distance to match the incomplete material between them.
        # A repeated phrase is only unambiguous when every occurrence was heard.
        positions, offset = {}, 0
        for key in keys:
            if key:
                positions.setdefault(tuple(key), []).append(offset)
            offset += len(key)
        candidates = []
        for key, starts in positions.items():
            # A short refrain can also occur inside a longer lyric line.
            # Those occurrences count too, even with different line breaks.
            occurrences = [i for i in range(len(expected)-len(key)+1)
                           if tuple(expected[i:i+len(key)]) == key]
            if occurrences != starts:
                continue
            hits = [i for i in range(len(observed)-len(key)+1)
                    if tuple(observed[i:i+len(key)]) == key]
            if len(hits) == len(starts):
                candidates.extend((start, start+len(key), hit, hit+len(key))
                                  for start, hit in zip(starts, hits))
        # Conflicting exact matches can come from dialogue or a changed verse
        # order. Neither is an independent boundary for the other.
        boundaries = [a for a in sorted(candidates) if all(a == b or
                      (a[1] <= b[0] and a[3] <= b[2]) or
                      (b[1] <= a[0] and b[3] <= a[2]) for b in candidates)]
        alphabet = {unit: i for i, unit in enumerate(dict.fromkeys(expected+observed))}
        # Minimize ordinary edit distance first, then internal gap runs.
        # Otherwise scattered letters in a spoken intro can tie a contiguous
        # lyric match and falsely give a minute-long line 100% text coverage.
        base = len(expected)+len(observed)+1
        aligner = PairwiseAligner(mode='global', match_score=0, mismatch_score=-base,
                                  open_gap_score=-base-1, extend_gap_score=-base)
        aligner.open_end_gap_score = -base  # Intro/outro gaps do not fragment lyrics.
        first = begin = 0
        for stop, following, end, next_observed in [*boundaries, (len(expected), len(expected), len(observed), len(observed))]:
            if stop > first and end > begin:
                alignment = aligner.align(array('i', (alphabet[u] for u in expected[first:stop])),
                                          array('i', (alphabet[u] for u in observed[begin:end])))[0]
                for (a, b), (c, d) in zip(*alignment.aligned):
                    for i, j in zip(range(a+first, b+first), range(c+begin, d+begin)):
                        if expected[i] == observed[j]:
                            correspondence[i] = j
            correspondence.update(zip(range(stop, following), range(end, next_observed)))
            first, begin = following, next_observed
    position = 0
    cues = []
    for line, key in zip(lines, keys):
        matched = [correspondence[i] for i in range(position, position+len(key)) if i in correspondence]
        coverage = len(matched)/len(key) if key else 0.
        density = len(matched)/(max(matched)-min(matched)+1) if matched else 0.
        # When ASR omits a lyric, global edit distance can collect its letters
        # across unrelated dialogue. Coverage alone does not locate a phrase.
        # Measure text density, not duration: a genuinely held note is valid.
        if coverage < .5 or density < .5:
            cue = {'text': line, 'start': None, 'end': None,
                   'matched_fraction': coverage, 'matched_density': density, 'status': 'unresolved_text'}
            prefix = []
            for i in range(position, position+len(key)):
                observed_index = correspondence.get(i)
                owner = owners[observed_index] if observed_index is not None else None
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
        start, end = words[owners[min(matched)]]['start'], words[owners[max(matched)]]['end']
        cues.append({'text': line, 'start': round(start, 3), 'end': round(end, 3),
                     'matched_fraction': coverage, 'matched_density': density,
                     'status': 'candidate' if end > start else 'unresolved_timing',
                     'source': 'asr_rough_anchor'})
        if any(a == position and b == position+len(key) for a, b, _, _ in boundaries):
            cues[-1]['match_basis'] = 'complete_phrase'
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
                raise ValueError(f"Insufficient ASR support for lyric line {i}: coverage "
                                 f"{cue['matched_fraction']:.1%}, text density {cue['matched_density']:.1%}")
            if cue['status'] == 'unresolved_timing':
                raise ValueError(f'Collapsed ASR interval for lyric line {i}')
    return cues


def timeline_support(cues, words, language):
    """Check coarse phrase location against already observed audio text.

    This is not an onset-accuracy measurement. Silence and held syllables do
    not fail a duration heuristic; only independently located lyrics can
    contradict a caption, and ambiguous/missing recognition stays unknown.
    """
    anchors = anchors_from_words([c['text'] for c in cues], words, language, allow_missing=True)
    checked, conflicts = [], []
    for i, (cue, anchor) in enumerate(zip(cues, anchors), 1):
        if anchor.get('match_basis') != 'complete_phrase' or anchor['status'] != 'candidate':
            continue
        checked.append(i)
        units = cue.get('units') or []
        begin = min((u['start'] for u in units), default=cue['start'])
        end = max((u.get('acoustic_end') or u['end'] for u in units), default=cue['end'])
        # Rough recognition can truncate a held tail or stretch a first word.
        # A two-second separation still establishes a different passage.
        if end+2 < anchor['start'] or begin-2 > anchor['end']:
            conflicts.append(dict(line=i, caption_range=[begin, end],
                                  observed_range=[anchor['start'], anchor['end']]))
    return dict(checked_lines=checked, conflicts=conflicts)


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
