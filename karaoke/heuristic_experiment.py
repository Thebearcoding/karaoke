"""Non-neural center-vocal suppression experiment, not a finished separator.

Run with the existing music/.venv/bin/python. No learned weights or song-specific
settings. A shared mask is applied to known synthetic components only to audit
the attenuation/distortion tradeoff; real songs have no ground-truth scores.
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal


ROOT = Path(__file__).resolve().parent
RATE = 48000
FFT = 2048
HOP = 512


def spectrum(audio):
    return signal.stft(audio.T, fs=RATE, nperseg=FFT,
                       noverlap=FFT-HOP, boundary="zeros")[2]


def reconstruct(z, length):
    return signal.istft(z, fs=RATE, nperseg=FFT,
                        noverlap=FFT-HOP, boundary=True)[1].T[:length]


def center_gain(z, strength, protection):
    mid = (z[0] + z[1]) * 0.5
    side = (z[0] - z[1]) * 0.5
    mid_power, side_power = np.abs(mid)**2, np.abs(side)**2
    # Past-only smoothing; no full-song statistics or repeated-section search.
    def past(x):
        return signal.lfilter([0.25], [1, -0.75], x, axis=-1)
    dominance = past(mid_power) / (past(mid_power + side_power) + 1e-12)
    cross = past(np.real(z[0] * np.conj(z[1])))
    coherence = np.clip(cross / np.sqrt(
        past(np.abs(z[0])**2) * past(np.abs(z[1])**2) + 1e-24), 0, 1)
    freq = np.fft.rfftfreq(FFT, 1 / RATE)
    band = np.clip((freq - 100) / 120, 0, 1) * np.clip((10000-freq)/2500, 0, 1)
    suppression = strength * dominance * coherence**2 * band[:, None]
    if protection:
        # Broadband vertical energy is a percussion cue, not proof of drums.
        mag = np.abs(mid)
        broadband = ndimage.median_filter(mag, size=(17, 1), mode="nearest")
        percussive = np.clip(broadband / (past(mag) + 1e-8), 0, 1)
        suppression *= 1 - protection * percussive**2
    return np.asarray(1 - suppression, dtype=np.float32)


def apply_gain(z, gain):
    mid = (z[0] + z[1]) * (0.5 * gain)
    side = (z[0] - z[1]) * 0.5
    return np.stack([mid + side, mid - side])


def db_ratio(a, b):
    return float(10 * np.log10((np.mean(a*a) + 1e-20) / (np.mean(b*b) + 1e-20)))


def synthetic(seed, vocal_pan=0, reverb=0):
    """Known artificial components: mechanism check, not singing realism."""
    rng = np.random.default_rng(seed)
    t = np.arange(RATE * 6) / RATE
    f0 = 205 + 22 * np.sin(2*np.pi*0.7*t) + 3*np.sin(2*np.pi*5*t)
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    envelope = np.clip(np.sin(2*np.pi*1.3*t), 0, 1)**0.5
    voice = sum(np.sin(k*phase)/k for k in range(1, 9)) * envelope * 0.06
    vocals = np.stack([voice*(1-vocal_pan), voice*(1+vocal_pan)], axis=1)
    if reverb:
        for channel in range(2):
            delay = int(RATE * (0.037 + channel*0.016))
            vocals[delay:, channel] += reverb * voice[:-delay]
    bass = 0.12*np.sin(2*np.pi*82.4*t)
    keys = sum(np.sin(2*np.pi*f*t) for f in [329.6, 415.3, 493.9]) * 0.022
    backing = np.stack([bass+keys, bass+keys], axis=1)
    for channel, frequency in enumerate([587.3, 698.5]):
        backing[:, channel] += 0.025*np.sin(2*np.pi*frequency*t)
    drum = np.zeros_like(t)
    for onset in np.arange(0.25, 6, 0.5):
        start = int(onset*RATE)
        count = min(int(0.14*RATE), len(t)-start)
        drum[start:start+count] += 0.18*rng.normal(size=count)*np.exp(-np.arange(count)/(RATE*0.024))
    backing += drum[:, None]
    return vocals.astype(np.float32), backing.astype(np.float32)


def audit(vocals, backing, strength, protection, cancel=False):
    z = spectrum(vocals + backing)
    gain = 0 if cancel else center_gain(z, strength, protection)
    remaining_voice = reconstruct(apply_gain(spectrum(vocals), gain), len(vocals))
    remaining_backing = reconstruct(apply_gain(spectrum(backing), gain), len(backing))
    return {
        "vocal_attenuation_db_higher_is_more": -db_ratio(remaining_voice, vocals),
        "backing_error_db_lower_is_less_damage": db_ratio(remaining_backing-backing, backing),
    }


def main():
    out = ROOT / "experiments" / "heuristic-v001"
    out.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Local audio suppression only; excludes search, download, lyrics, UI and fresh Python startup.",
              "limitations": "Synthetic tests are mechanism probes, not estimates of real-song quality. Shared masks are derived from mixtures. No neural inference, neural outputs, learned weights, or lyrics used.",
              "sample_rate": RATE, "fft_samples": FFT, "hop_samples": HOP}
    # Choose a percussion-protection setting on one known mixture. Keep an
    # explicit suppression floor, otherwise doing nothing wins distortion.
    vocals, backing = synthetic(17)
    sweep = []
    for strength in [0.90, 0.96, 0.99]:
        for protection in [0, 0.25, 0.5, 0.75]:
            scores = audit(vocals, backing, strength, protection)
            sweep.append({"strength": strength, "protection": protection, **scores})
    feasible = [row for row in sweep if row["vocal_attenuation_db_higher_is_more"] >= 10]
    chosen = min(feasible or sweep, key=lambda row: row["backing_error_db_lower_is_less_damage"])
    report["synthetic_calibration_sweep"] = sweep
    report["chosen_parameters"] = chosen
    methods = {"cancel-center": None, "adaptive-v1": (0.96, 0),
               "adaptive-v2": (chosen["strength"], chosen["protection"])}
    report["synthetic_holdout"] = {}
    for name, pan, reverb in [("centered", 0, 0), ("panned-reverberant", 0.18, 0.25)]:
        vocals, backing = synthetic(93, pan, reverb)
        report["synthetic_holdout"][name] = {
            method: audit(vocals, backing, *(params or (0, 0)), cancel=params is None)
            for method, params in methods.items()}
    manifest = json.loads((ROOT / "samples" / "manifest.json").read_text())
    report["real_song_excerpts"] = []
    # Same mechanical selection for all songs, fixed before evaluating: 25
    # seconds beginning at 30% of source duration. Not a manually labelled chorus.
    for song in manifest["songs"]:
        folder = out / song["id"]
        folder.mkdir()
        start = round(song["duration_seconds"] * 0.30, 3)
        begin = time.perf_counter()
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-nostdin", "-ss", str(start),
            "-i", str(ROOT / "samples" / song["file"]), "-t", "25", "-vn",
            "-ac", "2", "-ar", str(RATE), "-f", "f32le", "pipe:1"])
        audio = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).copy()
        decode_s = time.perf_counter() - begin
        assert len(audio) == RATE*25 and np.isfinite(audio).all()
        assert np.max(np.abs(reconstruct(spectrum(audio), len(audio))-audio)) < 1e-5
        versions = {"original": audio}
        timings = {}
        for method, params in methods.items():
            begin = time.perf_counter()
            if params is None:
                side = (audio[:, 0]-audio[:, 1])*0.5
                processed = np.stack([side, -side], axis=1)
            else:
                z = spectrum(audio)
                processed = reconstruct(apply_gain(z, center_gain(z, *params)), len(audio))
            timings[method] = time.perf_counter() - begin
            assert processed.shape == audio.shape and np.isfinite(processed).all()
            versions[method] = processed
        # One shared gain for all exports of this song: no separate loudness
        # normalization that would hide vocal reduction or amplify artifacts.
        gain = min(1.0, 0.95/max(float(np.max(np.abs(x))) for x in versions.values()))
        for name, version in versions.items():
            sf.write(folder / f"{name}.wav", version*gain, RATE, subtype="PCM_24")
        # Compare the same 8 seconds in each version, with short silence gaps.
        pieces = []
        for version in versions.values():
            cut = (version[8*RATE:16*RATE]*gain).copy()
            fade = np.linspace(0, 1, 480)[:, None]
            cut[:480] *= fade
            cut[-480:] *= fade[::-1]
            pieces.extend([cut, np.zeros((RATE//2, 2), dtype=np.float32)])
        sf.write(folder / "comparison.wav", np.concatenate(pieces), RATE, subtype="PCM_24")
        item = {"id": song["id"], "source_sha256": song["sha256"], "start_seconds": start,
                "duration_seconds": len(audio)/RATE, "decode_seconds": decode_s,
                "processing_seconds": timings, "shared_export_gain": gain,
                "comparison_order": list(versions), "quality": "not rated; listening required"}
        report["real_song_excerpts"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"chosen": chosen, "holdout": report["synthetic_holdout"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
