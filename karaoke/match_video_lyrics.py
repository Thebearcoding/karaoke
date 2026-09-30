"""Extract existing caption timing, using supplied lyric text and coarse audio anchors."""

import argparse
import difflib
import json
from pathlib import Path

from subtitles import normalized, write_captions


def caption_similarity(lyric, caption):
    # A video may reveal one source line in several consecutive caption cards.
    if len(caption) >= 3 and caption in lyric:
        return 1.
    return difflib.SequenceMatcher(None, lyric, caption).ratio()


def strongest_cluster(observations, max_gap=1.1):
    groups = []
    for row in observations:
        if not groups or row['time']-groups[-1][-1]['time'] > max_gap:
            groups.append([])
        groups[-1].append(row)
    return max(groups, key=lambda group: sum(r['similarity'] for r in group)) if groups else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ocr', required=True, nargs='+', type=Path)
    parser.add_argument('--lyrics', required=True, type=Path)
    parser.add_argument('--anchors', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = [json.loads(p.read_text()) for p in args.ocr]
    by_time = {round(f['requested_time'], 5): f for source in data for f in source['frames']}
    raw = sorted(by_time.values(), key=lambda f: f['time'])
    lyrics = json.loads(args.lyrics.read_text())['lines']
    anchors = json.loads(args.anchors.read_text())
    assert [a['text'] for a in anchors] == lyrics
    keys = [normalized(line) for line in lyrics]
    matched = []
    for frame in raw:
        variants = [normalized(t['text']) for t in frame['texts']]
        if len(variants) > 1:
            variants.append(''.join(variants))
        candidates = []
        for i, key in enumerate(keys):
            similarity = max([caption_similarity(key, text) for text in variants] or [0.])
            distance = abs(frame['time']-(anchors[i]['start']+anchors[i]['end'])/2)
            if similarity >= .67 and distance < 15:
                score = similarity-.008*max(0., distance-2.)
                candidates.append((score, similarity, -distance, i))
        best = max(candidates) if candidates else None
        matched.append({'time': frame['time'], 'requested_time': frame['requested_time'],
                        'line': best[3] if best else None, 'similarity': best[1] if best else 0.,
                        'texts': variants})
    cues, evidence, missing, refinement = [], [], [], set()
    for i, text in enumerate(lyrics):
        cluster = strongest_cluster([f for f in matched if f['line'] == i])
        if not cluster:
            missing.append(i+1)
            continue
        first, last = matched.index(cluster[0]), matched.index(cluster[-1])
        previous, following = max(0, first-1), min(len(matched)-1, last+1)
        start = (matched[previous]['time']+cluster[0]['time'])/2
        end = (cluster[-1]['time']+matched[following]['time'])/2
        valid = 0 <= start < end <= data[0]['duration_seconds']+.001
        cues.append({'text': text, 'start': round(start, 3), 'end': round(end, 3),
                     'lyric_line': i+1,
                     'status': 'candidate' if valid else 'unresolved_timing',
                     'source': 'existing_video_caption'})
        covered = set()
        for frame in cluster:
            for caption in frame['texts']:
                for block in difflib.SequenceMatcher(None, keys[i], caption).get_matching_blocks():
                    covered.update(range(block.a, block.a+block.size))
        evidence.append({'line': i+1, 'frames': len(cluster),
            'minimum_similarity': min(f['similarity'] for f in cluster),
            'text_coverage': len(covered)/len(keys[i]),
            'start_bracket': [matched[previous]['time'], cluster[0]['time']],
            'end_bracket': [cluster[-1]['time'], matched[following]['time']]})
        for left, right in [(previous, first), (last, following)]:
            a, b = matched[left]['requested_time'], matched[right]['requested_time']
            if b-a > .15:
                for k in range(1, 5):
                    refinement.add(round(a+(b-a)*k/5, 4))
    summary = write_captions(cues, args.output)
    summary.update({'expected_lines': len(lyrics), 'missing_lines': missing,
        'incomplete_text_evidence': [e['line'] for e in evidence if e['text_coverage'] < 1.],
        'caption_time_reference': 'Visible text in the supplied video; not a vocal-onset ground truth.'})
    (args.output/'evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    (args.output/'matched-frames.json').write_text(json.dumps(matched, indent=2)+'\n')
    (args.output/'refinement-times.json').write_text(json.dumps(sorted(refinement))+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
