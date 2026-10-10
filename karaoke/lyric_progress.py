"""Map existing audio-aligned word groups onto the original displayed lyrics."""
import argparse
import json
from pathlib import Path
import unicodedata

from opencc import OpenCC
from subtitles import highlight_spans, latin_word_text

CHINESE = OpenCC('t2s')


def canonical_positions(text, language):
    chars, positions = [], []
    for i, char in enumerate(latin_word_text(text)):
        if language in ('Chinese', 'Cantonese'):
            char = CHINESE.convert(char).replace('妳', '你')
        for value in unicodedata.normalize('NFKC', char).casefold():
            if value.isalnum():
                chars.append(value)
                positions.append(i)
    return ''.join(chars), positions


def japanese_tokenization(text):
    """Keep original spans when Open JTalk rewrites width, marks or numbers."""
    import re
    import pyopenjtalk
    from num2words import num2words

    def number_spelling(match):
        value = match[0]
        return num2words(int(value), lang='ja') if len(value) < 16 and (value == '0' or not value.startswith('0')) else value

    normalized, positions = '', []
    for cluster in re.finditer(r'.[\u0300-\u036f\u3099\u309a\uff9e\uff9f]*', text, re.S):
        value = unicodedata.normalize('NFKC', cluster[0])
        normalized += value
        positions.extend([(cluster.start(), cluster.end())]*len(value))
    pieces, mapped, cursor = [], [], 0
    for match in re.finditer(r'[0-9]+', normalized):
        pieces.extend([normalized[cursor:match.start()], number_spelling(match)])
        mapped.extend(positions[cursor:match.start()])
        mapped.extend([(positions[match.start()][0], positions[match.end()-1][1])]*len(pieces[-1]))
        cursor = match.end()
    pieces.append(normalized[cursor:]); mapped.extend(positions[cursor:])
    normalized = ''.join(pieces)
    words = [dict(word, string=re.sub(r'[0-9]+', number_spelling,
                  unicodedata.normalize('NFKC', word['string'])))
             for word in pyopenjtalk.run_frontend(normalized)]
    return normalized, mapped, words


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
    # Whisper tokens may split a Latin word (e.g. "Moder" + "ato"). Keep
    # its measured first onset and last end as one displayed word. A token
    # spanning multiple words cannot be split without new acoustic evidence.
    for first, stop in highlight_spans(cue['text'], language):
        indexes = [i for i, u in enumerate(units) if u['first'] < stop and u['stop'] > first]
        if len(indexes) > 1:
            left, right = indexes[0], indexes[-1]
            if units[left]['first'] == first and units[right]['stop'] == stop:
                units[left:right+1] = [dict(units[left], stop=stop, end=units[right]['end'],
                                          text=cue['text'][first:stop])]
    # Keep display punctuation between model tokens, including the apostrophe
    # in split contractions. This changes only character spans, never onsets.
    units[0]['first'] = 0
    for index, unit in enumerate(units):
        unit['stop'] = units[index+1]['first'] if index+1 < len(units) else len(cue['text'])
        unit['text'] = cue['text'][unit['first']:unit['stop']]
    return units


def partial_alignment_cues(lines, raw_blocks, duration, language):
    """Keep independently usable lines when a different line prevents stitching."""
    from subtitles import lyric_cues, collapsed_phrase
    cues = [dict(text=text, status='unresolved_timing') for text in lines]
    for block in raw_blocks:
        first, stop = block['context']
        segments = block['result']['segments']
        if not 0 <= first < stop <= len(lines) or len(segments) != stop-first:
            continue
        offset = block['source_range'][0]
        for index in range(*block['range']):
            if not first <= index < stop:
                continue
            try:
                words = segments[index-first]['words']
                units = [dict(text=w['word'], start=offset+w['start'], end=offset+w['end']) for w in words]
                cue = lyric_cues([lines[index]], units, duration)[0]
                if cue['status'] != 'candidate':
                    continue
                cue['units'] = map_words(cue, words, offset, language)
                if collapsed_phrase(cue, language):
                    cue['status'] = 'collapsed_timing'
                cues[index] = cue
            except (ValueError, KeyError, IndexError, TypeError):
                continue
    return cues


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
    parser.add_argument('--language', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cues, report = build(json.loads(args.cues.read_text()), json.loads(args.raw.read_text()), args.language)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cues, ensure_ascii=False, indent=2)+'\n')
    args.output.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))
