"""Offline quality scoring. Does not download, run models, or alter job status.

Usage: python -m karaoke.tools.quality --manifest INPUT.json --output REPORT.json
Each onset case provides event JSON files (id, text, start) in prediction,
optional reference, and optional verifier. Each stem case provides named WAV
paths in references and predictions. References require reference_provenance.
Paths are relative to the manifest. See docs/automatic-quality.md.
"""
import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf

from ..stem_metrics import score_stems
from ..word_onset_metrics import onset_risks, score_onsets, score_risks


def evaluate(manifest):
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    hashes = {}

    def path(name):
        file = (manifest.parent/name).resolve(strict=True)
        with file.open('rb') as stream:
            hashes[str(file)] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return file

    def events(name):
        return json.loads(path(name).read_text())

    rows = []
    for case in data.get('onsets', []):
        predicted = events(case['prediction'])
        row = dict(id=case['id'], kind='onsets', accuracy=None)
        if case.get('reference'):
            if not case.get('reference_provenance'):
                raise ValueError('Independent reference provenance is required')
            row['reference_provenance'] = case['reference_provenance']
            row['accuracy'] = score_onsets(events(case['reference']), predicted)
        if case.get('verifier'):
            row['risks'] = onset_risks(predicted, events(case['verifier']))
            if row['accuracy']:
                row['detector'] = {str(t): score_risks(row['accuracy'], row['risks'], t)
                                   for t in (50, 100, 250)}
        rows.append(row)
    for case in data.get('stems', []):
        if not case.get('reference_provenance'):
            raise ValueError('Independent reference provenance is required')
        rates = set()
        waves = {}
        for group in ('references', 'predictions'):
            waves[group] = {}
            for role, name in case[group].items():
                wave, rate = sf.read(path(name), dtype='float64', always_2d=True)
                waves[group][role] = wave
                rates.add(rate)
        if len(rates) != 1:
            raise ValueError('Reference and output sample rates differ; no resampling during scoring')
        rows.append(dict(id=case['id'], kind='stems', reference_provenance=case['reference_provenance'],
                         scores=score_stems(waves['references'], waves['predictions'], rates.pop(),
                                            window_seconds=data.get('window_seconds', 1.))))
    if not rows:
        raise ValueError('No quality cases in manifest')
    path(manifest.name)
    return dict(cases=rows, input_sha256=hashes,
                note='Reference accuracy and model disagreement are separate. No automatic job acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new report path to preserve previous evidence')
    result = evaluate(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(result['cases']), report=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
