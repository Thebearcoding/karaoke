"""Map verified lyric lines to rough ASR word times, preserving recording order."""

import argparse
from array import array
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from Bio.Align import PairwiseAligner

if __package__:
    from .subtitles import highlight_spans, normalized, write_captions
else:
    from subtitles import highlight_spans, normalized, write_captions


def text_units(text, language):
    value = normalized(text).lower()
    if language == 'Chinese':
        from pypinyin import lazy_pinyin
        return lazy_pinyin(value, errors=lambda s: list(s))
    return list(value)


def matching_spans(text):
    """Keep word boundaries, including script characters omitted by display grouping."""
    spans, cursor = [], 0
    for first, stop in [*highlight_spans(text), (len(text), len(text))]:
        spans.extend((i, i+1) for i in range(cursor, first) if text[i].isalnum())
        if first < stop:
            spans.append((first, stop))
        cursor = stop
    return spans


def _match_words(lines, words, units, *, word_evidence=True, other_word_views=()):
    keys = [units(line) for line in lines]
    expected = [unit for key in keys for unit in key]
    observed, owners = [], []
    for i, word in enumerate(words):
        value = units(word['word'])
        observed.extend(value)
        owners.extend([i]*len(value))
    other_observed = [[unit for word in view for unit in units(word['word'])]
                      for view in other_word_views if view is not words]
    correspondence, boundaries, source_ranges, ambiguous = {}, [], {}, []
    if expected and observed:
        observed_words = [tuple(units(word['word'][a:b])) for word in (words if word_evidence else [])
                          for a,b in matching_spans(word['word'])]
        fragment_matcher = SequenceMatcher(None, [], observed_words, autojunk=False)
        # An omitted verse must not consume words belonging to a later,
        # completely observed phrase. Establish those phrase boundaries before
        # asking edit distance to match the incomplete material between them.
        # A repeated phrase is only unambiguous when every occurrence was heard.
        positions, clause_positions, fragments, offset = {}, {}, set(), 0
        for line, key in zip(lines, keys):
            if key:
                positions.setdefault(tuple(key), []).append(offset)
                source_ranges[offset, offset+len(key)] = (0, len(line))
            # Parentheses commonly contain a simultaneous second voice. A
            # recognizer may hear one complete clause without hearing both.
            # Keep the full text, but allow independently observed clauses to
            # bound the ordinary edit-distance match around them.
            if re.search(r'[（(][^()（）]+[)）]', line):
                for part in re.finditer(r'[^()（）]+', line):
                    begin = len(units(line[:part.start()]))
                    end = len(units(line[:part.end()]))
                    clause = tuple(key[begin:end])
                    if clause and clause != tuple(key):
                        clause_positions.setdefault(clause, []).append(offset+begin)
                        source_ranges[offset+begin, offset+end] = (part.start(), part.end())
            # A misheard word must not let scattered dialogue consume an
            # otherwise heard phrase. Keep exact runs of whole words as
            # boundaries, subject to the same occurrence/order checks below.
            spans = matching_spans(line) if word_evidence else []
            tokens = [tuple(units(line[a:b])) for a,b in spans]
            fragment_matcher.set_seq1(tokens)
            for block in fragment_matcher.get_matching_blocks():
                if block.size < 3:
                    continue
                begin = len(units(line[:spans[block.a][0]]))
                end = len(units(line[:spans[block.a+block.size-1][1]]))
                clause = tuple(key[begin:end])
                if len(key)/2 <= len(clause) < len(key):
                    starts = clause_positions.setdefault(clause, [])
                    if offset+begin not in starts:
                        starts.append(offset+begin)
                        fragments.add((offset+begin, offset+end))
                    source_ranges[offset+begin, offset+end] = (spans[block.a][0], spans[block.a+block.size-1][1])
            offset += len(key)
        candidates = []
        # A refrain occurring both as a whole line and inside another voice's
        # parentheses cannot acquire an occurrence number from clause matching.
        for key, starts in [*positions.items(), *clause_positions.items()]:
            # A short refrain can also occur inside a longer lyric line.
            # Those occurrences count too, even with different line breaks.
            occurrences = [i for i in range(len(expected)-len(key)+1)
                           if tuple(expected[i:i+len(key)]) == key]
            if occurrences != starts:
                continue
            hits = [i for i in range(len(observed)-len(key)+1)
                    if tuple(observed[i:i+len(key)]) == key]
            if len(hits) == len(starts):
                # Equal counts in one partial recognition do not prove that
                # every occurrence was heard. Another view may contain more.
                if any(sum(tuple(view[i:i+len(key)]) == key for i in range(len(view)-len(key)+1))
                       > len(starts) for view in other_observed):
                    ambiguous.extend((start, start+len(key)) for start in starts)
                    continue
                candidates.extend((start, start+len(key), hit, hit+len(key))
                                  for start, hit in zip(starts, hits))
        for refinement in range(2):
            # A fully heard line already includes its clauses. Remove only nested
            # copies of that same correspondence before checking real conflicts.
            candidates = [a for a in candidates if not any(a != b and
                b[0] <= a[0] and a[1] <= b[1] and a[0]-b[0] == a[2]-b[2] and
                a[1]-b[0] == a[3]-b[2] for b in candidates)]
            # Conflicting exact matches can come from dialogue or a changed verse
            # order. Neither is an independent boundary for the other.
            boundaries = [a for a in sorted(candidates) if all(a == b or
                          (a[1] <= b[0] and a[3] <= b[2]) or
                          (b[1] <= a[0] and b[3] <= a[2]) for b in candidates)]
            if refinement:
                break
            # A missing copy elsewhere need not erase an unambiguous refrain
            # bracketed by two independently heard phrases. Count occurrences
            # within both text intervals; song edges alone supply no evidence.
            for left, right in zip(boundaries, boundaries[1:]):
                first, stop, begin, end = left[1], right[0], left[3], right[2]
                lo, hi = words[owners[begin-1]]['end'], words[owners[end]]['start']
                other_within = [[unit for word in view if lo <= (word['start']+word['end'])/2 <= hi
                                 for unit in units(word['word'])]
                                for view in other_word_views if view is not words]
                for key, starts in [*positions.items(), *clause_positions.items()]:
                    within = [s for s in starts if first <= s and s+len(key) <= stop]
                    if not within or within != [i for i in range(first, stop-len(key)+1)
                                                if tuple(expected[i:i+len(key)]) == key]:
                        continue
                    hits = [i for i in range(begin, end-len(key)+1)
                            if tuple(observed[i:i+len(key)]) == key]
                    if len(hits) == len(within):
                        if any(sum(tuple(view[i:i+len(key)]) == key for i in range(len(view)-len(key)+1))
                               > len(within) for view in other_within):
                            ambiguous.extend((start, start+len(key)) for start in within)
                            continue
                        candidates.extend((s, s+len(key), h, h+len(key)) for s,h in zip(within,hits))
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
        if any(position <= a < b <= position+len(key) for a,b in ambiguous):
            cues[-1]['ambiguous_occurrences'] = True
        clauses = [(a,b,c,d) for a,b,c,d in boundaries
                   if position <= a < b <= position+len(key) and b-a >= len(key)/2]
        if any(a == position and b == position+len(key) for a, b, _, _ in clauses):
            cues[-1]['match_basis'] = 'complete_phrase'
        else:
            if clauses:
                cues[-1].update(match_basis='complete_clause' if any((a,b) not in fragments
                    for a,b,c,d in clauses) else 'exact_fragment', observed_range=[
                    min(words[owners[c]]['start'] for a,b,c,d in clauses),
                    max(words[owners[d-1]]['end'] for a,b,c,d in clauses)],
                    matched_prefix=any(a == position for a,b,c,d in clauses),
                    matched_suffix=any(b == position+len(key) for a,b,c,d in clauses))
        if clauses and word_evidence:
            edges = []
            for a,b,c,d in clauses:
                first, stop = source_ranges[a,b]
                inside = matching_spans(line[first:stop])
                # ASR may spread a pause across its first or last tokens.
                # Locate consecutive words, not an isolated token boundary.
                for group in (inside[:3], inside[-3:]) if inside else []:
                    lo, hi = first+group[0][0], first+group[-1][1]
                    begin, end = position+len(units(line[:lo])), position+len(units(line[:hi]))
                    edge = dict(first=lo, stop=hi,
                        start=words[owners[c+begin-a]]['start'], end=words[owners[c+end-a-1]]['end'])
                    if edge not in edges:
                        edges.append(edge)
            cues[-1]['observed_edges'] = edges
        position += len(key)
    return cues


def japanese_reading_units(text):
    if re.search(r'[\u3040-\u30ff\u3400-\u9fff]', text):
        import pyopenjtalk
        text = pyopenjtalk.g2p(text, kana=True)
    return text_units(text, 'Japanese')


def anchors_from_words(lines, words, language, allow_missing=False, *, other_word_views=()):
    cues = _match_words(lines, words, lambda text: text_units(text, language), other_word_views=other_word_views)
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
                # Whole-word readings can differ from isolated kanji readings;
                # this fallback supplies phrase location, not lexical edges.
                for offset, cue in enumerate(_match_words(lines[first:stop], inside, japanese_reading_units,
                                                         word_evidence=False)):
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


def observed_context(lines, words, language, first, stop):
    """Include complete phrases heard outside a stale predicted lyric slice."""
    anchors = anchors_from_words(lines, words, language, allow_missing=True)
    heard = [i for i,a in enumerate(anchors) if a.get('match_basis') in ('complete_phrase', 'complete_clause')
             and a['status'] == 'candidate']
    return min([first,*heard]), max([stop,*(i+1 for i in heard)])


def combine_anchor_views(views):
    """Keep the strongest ordered phrase evidence across recording views."""
    if not views:
        return []
    texts = [a['text'] for a in views[0]]
    if any([a['text'] for a in view] != texts for view in views[1:]):
        raise ValueError('Recognition views must describe the same lyric text')
    nodes = []
    for index, alternatives in enumerate(zip(*views)):
        for view, anchor in enumerate(alternatives):
            if anchor['status'] != 'candidate':
                continue
            previous = max((n for n in nodes if n['line'] < index and
                n['anchor']['start'] < anchor['start'] and n['anchor']['end'] <= anchor['end']),
                key=lambda n:n['score'], default=None)
            score = len(normalized(anchor['text']))*anchor['matched_fraction']*anchor['matched_density']
            nodes.append(dict(line=index, view=view, anchor=anchor,
                score=score+(previous['score'] if previous else 0), previous=previous))
    selected = {}
    node = max(nodes, key=lambda n:n['score'], default=None)
    while node is not None:
        selected[node['line']] = dict(node['anchor'], recognition_view=node['view'])
        node = node['previous']
    # A conflict stays unresolved. This chooses existing observations; it never
    # shifts their times or treats a missing observation as missing lyrics.
    return [selected.get(i, dict(anchor, status='unresolved_text', start=None, end=None))
            for i,anchor in enumerate(views[0])]


def phrase_conflict(cue, anchor, *, fragments=True):
    """Compare heard text with its caption, including independently heard edges."""
    basis = anchor.get('match_basis')
    if (anchor['status'] != 'candidate' or basis not in ('complete_phrase', 'complete_clause', 'exact_fragment')
            or basis == 'exact_fragment' and not fragments):
        return False
    units = [u for u in cue.get('units', []) if normalized(u['text'])]
    intervals = [(min((u['start'] for u in units), default=cue['start']),
                  max((u.get('acoustic_end') or u['end'] for u in units), default=cue['end']))]
    for unit, edge in ((units[:1], 'matched_prefix'), (units[-1:], 'matched_suffix')):
        if unit and (anchor['match_basis'] == 'complete_phrase' or anchor.get(edge)):
            intervals.append((unit[0]['start'], unit[0].get('acoustic_end') or unit[0]['end']))
    begin, end = anchor.get('observed_range', [anchor['start'], anchor['end']])
    if any(stop+2 < begin or start-2 > end for start,stop in intervals):
        return True
    for edge in anchor.get('observed_edges', []):
        covering = [u for u in units if 'first' in u and 'stop' in u
                    and u['first'] < edge['stop'] and edge['first'] < u['stop']]
        if covering and (max(u.get('acoustic_end') or u['end'] for u in covering)+2 < edge['start']
                         or min(u['start'] for u in covering)-2 > edge['end']):
            return True
    return False


def timeline_support(cues, words, language, *, other_word_views=(), combine=True):
    """Check coarse phrase location against already observed audio text.

    This is not an onset-accuracy measurement. Silence and held syllables do
    not fail a duration heuristic; only independently located lyrics can
    contradict a caption, and ambiguous/missing recognition stays unknown.
    """
    views = [words, *other_word_views]
    matches = [anchors_from_words([c['text'] for c in cues], view, language, allow_missing=True,
                                  other_word_views=views) for view in (views if combine else [words])]
    anchors = combine_anchor_views(matches) if len(matches) > 1 else matches[0]
    checked, conflicts, supported, ambiguous = [], [], [], []
    for i, (cue, anchor) in enumerate(zip(cues, anchors), 1):
        # A rough match with unresolved occurrence cannot dismiss a conflict
        # established by an independently matched phrase in another view.
        if (anchor['status'] != 'candidate' or
                anchor.get('match_basis') not in ('complete_phrase', 'complete_clause', 'exact_fragment')):
            if any(view[i-1].get('ambiguous_occurrences') for view in matches):
                ambiguous.append(i)
            continue
        units = cue.get('units') or []
        begin = min((u['start'] for u in units), default=cue['start'])
        end = max((u.get('acoustic_end') or u['end'] for u in units), default=cue['end'])
        # Rough recognition can truncate a held tail or stretch a first word.
        # A two-second separation still establishes a different passage.
        observed_begin, observed_end = anchor.get('observed_range', [anchor['start'], anchor['end']])
        conflict = phrase_conflict(cue, anchor)
        if not (conflict or end+2 < observed_begin or begin-2 > observed_end):
            supported.append(i)
        checked.append(i)
        if conflict:
            conflicts.append(dict(line=i, caption_range=[begin, end],
                                  observed_range=[observed_begin, observed_end]))
    return dict(checked_lines=checked, conflicts=conflicts, supported_lines=supported, ambiguous_lines=ambiguous)


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
