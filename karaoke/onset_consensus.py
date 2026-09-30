"""Combine close estimates of the same English word onset without reordering lyrics."""
import copy

from word_onset_metrics import canonical, word_events


def blend_cues(primary, auxiliary, *, max_disagreement, primary_weight):
    if not 0 <= primary_weight <= 1 or max_disagreement < 0:
        raise ValueError('Invalid consensus parameters')
    cues = copy.deepcopy(primary)
    units = [u for c in cues for u in c['units']]
    prior = word_events(auxiliary)
    if len(units) != len(prior) or any(canonical(u['text']) != canonical(p['text'])
                                      for u,p in zip(units,prior)):
        raise ValueError('Consensus requires matching word identities in both timelines')
    original = [u['start'] for u in units]
    if any(a >= b for a,b in zip(original,original[1:])):
        raise ValueError('Primary word onsets must be strictly ordered')
    proposed = [primary_weight*u['start']+(1-primary_weight)*p['start']
                if p['start'] is not None and abs(u['start']-p['start']) <= max_disagreement
                else u['start'] for u,p in zip(units,prior)]
    while True:
        bad = {j for i in range(len(proposed)-1) if proposed[i] >= proposed[i+1] for j in (i,i+1)}
        changed = [i for i in bad if proposed[i] != original[i]]
        if not changed:
            break
        for i in changed:
            proposed[i] = original[i]
    for i,(unit,start) in enumerate(zip(units,proposed)):
        unit['start'] = start
        unit['end'] = max(unit['end'], start+.001)
        if i+1 < len(units):
            unit['end'] = min(unit['end'], proposed[i+1])
        unit['timing_source'] = 'agreement_gated_onset_estimates' if start!=original[i] else 'singing_onset_edge'
    for cue in cues:
        cue['start'] = cue['aligned_start'] = cue['units'][0]['start']
        cue['end'] = cue['aligned_end'] = cue['units'][-1]['end']
    return cues
