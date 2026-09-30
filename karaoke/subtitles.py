"""Map aligned text units back to source lyric lines and write timed captions."""

import json
from bisect import bisect_right
import math
from pathlib import Path
import unicodedata


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text)
                   if unicodedata.category(c)[0] in ('L', 'N') or c == "'")


def lyric_cues(lines, units, duration):
    expected = ''.join(normalized(line) for line in lines)
    observed = ''.join(normalized(unit['text']) for unit in units)
    if expected != observed:
        raise ValueError('Aligned units do not exactly cover the supplied lyric text')
    spans = []
    position = 0
    for unit in units:
        length = len(normalized(unit['text']))
        if length:
            spans.append((position, position+length, unit))
        position += length
    cues = []
    position = 0
    for line in lines:
        length = len(normalized(line))
        if not length:
            continue
        matched = [unit for begin, end, unit in spans if begin < position+length and end > position]
        start = min(u['start'] for u in matched)
        end = max(u['end'] for u in matched)
        valid = 0 <= start < end <= duration+.001
        cues.append({'text': line, 'start': start, 'end': end,
                     'status': 'candidate' if valid else 'unresolved_timing'})
        position += length
    return cues


def presentation_cues(cues, duration, lead=.2, tail=.4, bridge_gap=1.5):
    """Make readable line holds without changing the recorded acoustic timings."""
    adjusted = [dict(cue) for cue in cues]
    changes = []
    for i, cue in enumerate(adjusted):
        if cue['status'] != 'candidate' or not 0 <= cue['start'] < cue['end'] <= duration+.001:
            raise ValueError(f'Unresolved alignment on line {i+1}')
        cue['aligned_start'], cue['aligned_end'] = cue['start'], cue['end']
    for i, (left, right) in enumerate(zip(adjusted, adjusted[1:])):
        if not left['start'] < right['start'] or not left['end'] < right['end']:
            raise ValueError(f'Unordered alignment at line {i+1}')
        overlap = left['end']-right['start']
        if overlap > .5:
            raise ValueError(f'Alignment overlap needs review at line {i+1}: {overlap:.3f}s')
        if overlap > 0:
            boundary = round((left['end']+right['start'])/2, 3)
            left['end'] = right['start'] = boundary
            changes.append({'lines': [i+1, i+2], 'overlap_seconds': round(overlap, 3), 'boundary': boundary})
    starts = [max(0., cue['start']-lead) for cue in adjusted]
    for i, cue in enumerate(adjusted):
        end = min(duration, cue['end']+tail)
        if i+1 < len(adjusted):
            next_start = starts[i+1]
            end = next_start if next_start-cue['end'] <= bridge_gap else min(end, next_start)
        cue['start'], cue['end'] = round(starts[i], 3), round(end, 3)
        if cue['end'] <= cue['start']:
            raise ValueError(f'No display interval for line {i+1}')
    return adjusted, {'lead_seconds': lead, 'tail_seconds': tail,
                      'bridge_gap_seconds': bridge_gap, 'overlap_adjustments': changes}


def read_ahead_state(cues, instant):
    """Keep the current and next lines visible; advance only at sung onsets."""
    if not cues:
        return None, None
    starts = [c['units'][0]['start'] if c.get('units')
              else c.get('aligned_start', c['start']) for c in cues]
    current = max(0, bisect_right(starts, instant)-1)
    following = current+1
    return current, following if following < len(cues) else None


def countdown_events(cues, minimum_silence=8., dots=3, step_seconds=1., settle_seconds=1.):
    """Find vocal entries after long silence, using sung ends rather than line holds."""
    duration = dots*step_seconds+settle_seconds
    if dots < 1 or step_seconds <= 0 or settle_seconds < 0 or minimum_silence < duration:
        raise ValueError('Countdown must fit inside the minimum vocal gap')
    events = []
    previous_end = 0.
    for line, cue in enumerate(cues):
        units = cue.get('units') or [{'start': cue.get('aligned_start', cue['start']),
                                     'end': cue.get('aligned_end', cue['end'])}]
        for index, unit in enumerate(units):
            onset, end = unit['start'], unit['end']
            if not (math.isfinite(onset) and math.isfinite(end) and 0 <= onset < end):
                raise ValueError('Countdown requires valid acoustic intervals')
            if onset-previous_end >= minimum_silence:
                events.append(dict(line=line, unit=index, start=onset-duration,
                                   onset=onset, previous_end=previous_end,
                                   dots=dots, step_seconds=step_seconds, settle_seconds=settle_seconds))
            previous_end = max(previous_end, end)
    return events


def countdown_state(events, instant):
    """Stateless countdown: seeking and pausing always preserve the audio clock."""
    index = bisect_right([event['start'] for event in events], instant)-1
    if index < 0 or instant >= events[index]['onset']:
        return None, 0
    event = events[index]
    remaining = math.ceil((event['onset']-event.get('settle_seconds', 0.)-instant)/event['step_seconds']-1e-9)
    return event, max(0, min(event['dots'], remaining))


def timestamp(seconds, decimal=','):
    value = round(seconds*1000)
    hours, value = divmod(value, 3600000)
    minutes, value = divmod(value, 60000)
    seconds, milliseconds = divmod(value, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}{decimal}{milliseconds:03}'


def write_captions(cues, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'cues.json').write_text(json.dumps(cues, ensure_ascii=False, indent=2)+'\n')
    valid = [cue for cue in cues if cue['status'] == 'candidate']
    for extension, decimal, header in [('srt', ',', ''), ('vtt', '.', 'WEBVTT\n\n')]:
        body = '\n\n'.join(f"{i}\n{timestamp(c['start'], decimal)} --> "
                           f"{timestamp(c['end'], decimal)}\n{c['text']}"
                           for i, c in enumerate(valid, 1))
        (directory/f'lyrics.{extension}').write_text(header+body+'\n')
    return {'total_lines': len(cues), 'exported_lines': len(valid),
            'unresolved_lines': [i+1 for i, c in enumerate(cues) if c['status'] != 'candidate'],
            'overlap_pairs': [[i, i+1] for i in range(1, len(valid))
                              if valid[i-1]['end'] > valid[i]['start']]}
