"""Refine coarse English lyric lines with a dense singing phoneme boundary model."""
import argparse
from functools import lru_cache
import importlib.util
import json
import re
import sys
import unicodedata

import numpy as np
import torch
import torchaudio
import yaml

from align_english_onsets import ROOT, align_job
from align_lyrics_whisper import sha
from subtitles import latin_word_text

ASSETS = ROOT / 'models/alignment/sofa/english/tgm_en_v100'


class EnglishSofaAligner:
    method = 'sofa'
    onset_ratio = None
    sample_rate = 44100

    def __init__(self, device):
        self.device = device
        sys.path.insert(0, str(ROOT / 'research/subtitles-2026-09-18/sofa'))
        from modules.task.forced_alignment import LitForcedAlignmentTask
        from modules.rmvpe.spec import MelSpectrogram
        checkpoint = torch.load(ASSETS / 'model.ckpt', map_location='cpu', weights_only=True)
        self.model = LitForcedAlignmentTask(**checkpoint['hyper_parameters'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval().to(device)
        c = self.model.melspec_config
        self.extract = MelSpectrogram(n_mel_channels=c['n_mels'], sampling_rate=c['sample_rate'],
            win_length=c['win_length'], hop_length=c['hop_length'], n_fft=c['n_fft'],
            mel_fmin=c['fmin'], mel_fmax=c['fmax'], clamp=c['clamp'])
        self.lexicon = {}
        for line in (ASSETS / 'dict.txt').read_text().splitlines():
            fields = line.split()
            if fields:
                self.lexicon.setdefault(fields[0].lower(), fields[1:])
        source = ROOT / 'research/word-onset-2026-09-18/labelmakr/models/g2p_model.py'
        spec = importlib.util.spec_from_file_location('labelmakr_g2p', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = yaml.safe_load((ASSETS / 'g2p/cfg.yaml').read_text())
        encoder = module.Encoder(**{k:v for k,v in config['encoder'].items() if k!='_target_'})
        decoder = module.Decoder(**{k:v for k,v in config['decoder'].items() if k!='_target_'})
        self.g2p = module.G2p(config['max_len'], encoder, decoder)
        self.g2p.load_state_dict(torch.load(ASSETS / 'g2p/model.ptsd', map_location='cpu', weights_only=True))
        self.g2p.eval()
        self.model_identity = dict(model='tgm_en_v100', model_sha256=sha(ASSETS / 'model.ckpt'),
            dictionary_sha256=sha(ASSETS / 'dict.txt'), g2p_sha256=sha(ASSETS / 'g2p/model.ptsd'),
            g2p_source_sha256=sha(source), runner_sha256=sha(__file__),
            pronunciation='Release dictionary first; bundled G2P for unseen words; no per-song changes',
            window='Coarse singing-model line interval +/- 0.8s; no word-time priors or first-word pinning')

    @lru_cache(maxsize=4096)
    def pronunciation(self, word):
        text = unicodedata.normalize('NFKD', latin_word_text(word).casefold()).replace('’', "'")
        key = re.sub(r"[^a-z'\-]", '', text).strip("'-")
        phones = self.lexicon.get(key)
        if phones is None:
            phones = [p for p in self.g2p.predict_str(key) if not p.startswith('<')] if key else []
        if not phones or any(p not in self.model.vocab for p in phones):
            raise ValueError(f'Unsupported pronunciation for {word}: {phones}')
        return phones

    @torch.inference_mode()
    def align(self, audio, words):
        if not len(audio):
            raise ValueError('Empty coarse window')
        c = self.model.melspec_config
        waveform = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
        if self.sample_rate != c['sample_rate']:
            waveform = torchaudio.functional.resample(waveform, self.sample_rate, c['sample_rate'])
        waveform = waveform * min(1., .97/max(float(waveform.abs().max()), 1e-10))
        mel = self.extract(waveform)
        mel = (mel-mel.mean())/mel.std().clamp_min(1e-6)
        mel = mel.repeat_interleave(c['scale_factor'], dim=-1).to(self.device)
        phones, owners = ['SP'], [-1]
        for i,word in enumerate(words):
            pronunciation = self.pronunciation(word)
            phones.extend(pronunciation)
            owners.extend([i]*len(pronunciation))
            phones.append('SP')
            owners.append(-1)
        ids = [str(i) for i in range(len(words))]
        _, _, decoded, intervals, confidence, _, _ = self.model._infer_once(
            mel, len(audio)/self.sample_rate, phones, ids, owners)
        if list(decoded) != ids:
            raise ValueError('Dense phoneme alignment omitted one or more required words')
        result = [dict(text=w, start=float(t[0]), end=float(t[1]), confidence=float(confidence))
                  for w,t in zip(words, intervals)]
        return result, None, c['hop_length']/(c['sample_rate']*c['scale_factor'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = EnglishSofaAligner(args.device)
    for job in json.loads(open(args.jobs).read()):
        print(json.dumps(align_job(model, job)), flush=True)
