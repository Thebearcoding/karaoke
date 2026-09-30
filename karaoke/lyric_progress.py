"""Map existing audio-aligned word groups onto the original displayed lyrics."""
import argparse
import json
from pathlib import Path
import unicodedata

from opencc import OpenCC

CHINESE = OpenCC('t2s')


def canonical_positions(text, language):
    chars, positions = [], []
    for i, char in enumerate(text):
        if language == 'Chinese':
            char = CHINESE.convert(char).replace('妳', '你')
        for value in unicodedata.normalize('NFKC', char).casefold():
            if value.isalnum():
                chars.append(value)
                positions.append(i)
    return ''.join(chars), positions


def map_words(cue, words, offset, language):
    target, positions = canonical_positions(cue['text'], language)
    normalized = [canonical_positions(word['word'], language)[0] for word in words]
    if ''.join(normalized) != target:
        raise ValueError('Aligned text differs from displayed lyric')
    start = cue.get('aligned_start', cue['start'])
    end = cue.get('aligned_end', cue['end'])
    units, cursor, pending = [], 0, None
    for word, text in zip(words, normalized):
        if not text:
            continue
        first, stop = positions[cursor], positions[cursor+len(text)-1]+1
        cursor += len(text)
        begin = min(end, max(start, offset+word['start']))
        finish = min(end, max(begin, offset+word['end']))
        if pending is not None:
            first = pending
        if finish-begin < .015:
            pending = first
            continue
        pending = None
        units.append({'start': round(begin, 3), 'end': round(finish, 3),
                      'first': first, 'stop': stop, 'text': cue['text'][first:stop]})
    if pending is not None and units:
        units[-1]['stop'] = len(cue['text'])
        units[-1]['text'] = cue['text'][units[-1]['first']:]
    if not units or any(b['start'] < a['start'] for a, b in zip(units, units[1:])):
        raise ValueError('No usable monotonic word timing')
    units[-1]['stop'] = len(cue['text'])
    return units


def build(display_cues, raw_blocks, language):
    result = [dict(cue, units=[], timing_status='unavailable') for cue in display_cues]
    errors = []
    seen = set()
    for block in raw_blocks:
        for i in range(*block['range']):
            if i in seen:
                raise ValueError(f'Duplicate core lyric line: {i}')
            seen.add(i)
            segment = block['result']['segments'][i-block['context'][0]]
            try:
                result[i]['units'] = map_words(result[i], segment['words'], block['source_range'][0], language)
                result[i]['timing_status'] = 'audio_aligned_word_groups'
            except ValueError as error:
                errors.append({'line': i+1, 'reason': str(error)})
    if seen != set(range(len(result))):
        raise ValueError('Raw alignment does not cover exactly the displayed lyrics')
    return result, {'lines': len(result), 'timed_lines': sum(bool(c['units']) for c in result),
        'fallback_lines': errors, 'unit': 'Existing audio-aligned word group; no equal character interpolation',
        'zero_duration_policy': 'Group with the next timed token, or the preceding final token'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cues', required=True, type=Path)
    parser.add_argument('--raw', required=True, type=Path)
    parser.add_argument('--language', required=True, choices=['Chinese', 'Japanese', 'Latin'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cues, report = build(json.loads(args.cues.read_text()), json.loads(args.raw.read_text()), args.language)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cues, ensure_ascii=False, indent=2)+'\n')
    args.output.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))
