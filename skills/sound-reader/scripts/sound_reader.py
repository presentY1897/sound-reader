#!/usr/bin/env python3
"""sound-reader: turn audio into images + numbers that a vision-capable LLM can read.

Usage:
  sound_reader.py analyze FILE [--out DIR] [--start S] [--end S] [--music] [--linear-freq]
  sound_reader.py compare FILE_A FILE_B [--out DIR] [--start S] [--end S]

Outputs (in --out, default ./sound-reader-out/<name>/):
  overview.png  waveform + loudness + spectrogram (+ f0 track, + chroma with --music)
  spectrum.png  average spectrum with labelled peaks + per-band energy
  detail.png    sample-level zooms: loudest moment, first attack, first clip
  summary.json  all measured numbers
  compare.png / compare.json for the compare command

Dependencies: numpy, scipy, matplotlib. ffmpeg (optional) for mp3/m4a/ogg/flac/etc.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from scipy import signal as sps
    from scipy.io import wavfile
except ImportError:  # pragma: no cover
    sys.exit("sound-reader needs scipy:  pip install numpy scipy matplotlib")

import warnings

warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", message="Mean of empty slice")  # all-NaN columns are expected (masked areas)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

EPS = 1e-12
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BANDS = [
    ("sub", 20, 60),
    ("bass", 60, 250),
    ("low_mid", 250, 500),
    ("mid", 500, 2000),
    ("high_mid", 2000, 4000),
    ("presence", 4000, 6000),
    ("brilliance", 6000, 20000),
]


# ----------------------------------------------------------------------------- loading


def load_audio(path: str, start: float | None = None, end: float | None = None):
    """Return (samples float32 [n, ch] in -1..1, sr, info dict)."""
    info: dict = {"file": os.path.abspath(path), "file_size_bytes": os.path.getsize(path)}
    data, sr = None, None
    if path.lower().endswith((".wav", ".wave")):
        try:
            with warnings.catch_warnings():  # LIST/INFO metadata chunks are harmless
                warnings.simplefilter("ignore", wavfile.WavFileWarning)
                sr, raw = wavfile.read(path)
            info["source_dtype"] = str(raw.dtype)
            data = _to_float(raw)
        except Exception as e:  # odd wav subtypes -> fall through to ffmpeg
            info["wav_read_error"] = str(e)
    if data is None:
        try:
            import soundfile as sf  # optional

            raw, sr = sf.read(path, dtype="float32", always_2d=True)
            data = raw
            info["source_dtype"] = sf.info(path).subtype
        except Exception:
            data, sr = _ffmpeg_decode(path, info)
    if data.ndim == 1:
        data = data[:, None]
    info["lossy_codec"] = detect_lossy_codec(path, info.get("source_dtype", ""))
    info["sample_rate"] = int(sr)
    info["channels"] = int(data.shape[1])
    info["total_duration_s"] = round(data.shape[0] / sr, 4)
    if start is not None or end is not None:
        s = int(max(0.0, start or 0.0) * sr)
        e = int(min(data.shape[0], (end if end is not None else data.shape[0] / sr) * sr))
        data = data[s:e]
        info["window_s"] = [round(s / sr, 4), round(e / sr, 4)]
    info["offset_s"] = float(start or 0.0)
    return data.astype(np.float32, copy=False), int(sr), info


def _to_float(raw: np.ndarray) -> np.ndarray:
    if raw.dtype == np.uint8:
        return (raw.astype(np.float32) - 128.0) / 128.0
    if np.issubdtype(raw.dtype, np.integer):
        return raw.astype(np.float32) / float(-np.iinfo(raw.dtype).min)
    return raw.astype(np.float32)


def _ffmpeg_decode(path: str, info: dict):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit(
            f"Cannot decode {path}: not a plain WAV and neither soundfile nor ffmpeg is available.\n"
            "Install ffmpeg (apt install ffmpeg / brew install ffmpeg) or pip install soundfile."
        )
    sr = _ffprobe_sr(path) or 48000
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "decoded.wav")
        cmd = [ffmpeg, "-v", "error", "-y", "-i", path, "-vn", "-c:a", "pcm_f32le", "-ar", str(sr), out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"ffmpeg failed to decode {path}:\n{r.stderr.strip()}")
        sr, raw = wavfile.read(out)
    info["decoded_with"] = "ffmpeg"
    info["source_dtype"] = "compressed/other (decoded to float32)"
    return raw.astype(np.float32), sr


LOSSY_EXT = {".mp3": "MP3", ".ogg": "Vorbis", ".oga": "Vorbis", ".opus": "Opus", ".m4a": "AAC", ".aac": "AAC",
             ".wma": "WMA", ".webm": "Opus/Vorbis", ".mp4": "AAC", ".mkv": "lossy?", ".amr": "AMR"}
LOSSY_CODECS = ("mp3", "mpeg_layer", "vorbis", "opus", "aac", "wma", "amr", "ac3", "eac3", "dts")


def detect_lossy_codec(path: str, subtype: str) -> str | None:
    """Name of the lossy codec the audio went through, or None for PCM/FLAC/etc."""
    st = subtype.lower()
    for c in LOSSY_CODECS:
        if c in st:
            return subtype
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
                            "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
        name = r.stdout.strip().lower()
        if name:
            return name if any(c in name for c in LOSSY_CODECS) else None
    return LOSSY_EXT.get(Path(path).suffix.lower())


def _ffprobe_sr(path: str) -> int | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


# ----------------------------------------------------------------------------- helpers


def db(x, floor=-120.0):
    return np.maximum(20 * np.log10(np.maximum(np.abs(x), EPS)), floor)


def pdb(p, floor=-150.0):
    return np.maximum(10 * np.log10(np.maximum(p, EPS)), floor)


def hz_to_note(f: float) -> str:
    if not f or f <= 0 or not np.isfinite(f):
        return "-"
    midi = 69 + 12 * math.log2(f / 440.0)
    m = int(round(midi))
    cents = int(round((midi - m) * 100))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}{'+' if cents >= 0 else ''}{cents}c"


def fmt_hz(f, _pos=None):
    return f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"


def r(x, n=2):
    if x is None:
        return None
    x = float(x)
    return round(x, n) if np.isfinite(x) else None


def frame_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        return np.array([np.sqrt(np.mean(x ** 2) + EPS)])
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(0, len(x) - win + 1, hop)
    return np.sqrt((c[starts + win] - c[starts]) / win + EPS)


def clip_mask(x: np.ndarray, min_run: int = 3):
    """Clipping = runs of >= min_run consecutive samples sitting flat on the signal's ceiling.

    A clean full-scale sine touches 1.0 for a single sample per cycle, so a plain threshold would
    flag it; flat tops are what clipping leaves behind (also when the file was normalised down
    after clipping). Float samples > 1.0 always count.
    """
    mask = np.zeros(x.shape, dtype=bool)
    events = 0
    for c in range(x.shape[1]):
        y = x[:, c]
        a = np.abs(y)
        ceiling = float(a.max()) if a.size else 0.0
        if ceiling < 0.1:
            continue
        hot = a >= 0.999 * ceiling
        # flat top = no curvature. A sine near its peak still curves (2nd diff ~ A*w^2), a clipped
        # plateau does not, even after small later changes (hum, dither, gain) are added on top.
        curv = np.full(len(y), np.inf)
        curv[1:-1] = np.abs(y[2:] - 2 * y[1:-1] + y[:-2])
        flat = curv < 5e-6 * max(ceiling, 0.1)
        cand = hot & (flat | np.roll(flat, 1) | np.roll(flat, -1))
        d = np.diff(np.concatenate([[0], cand.astype(np.int8), [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= min_run:
                mask[s:e, c] = True
        mask[:, c] |= a > 1.0
        d = np.diff(np.concatenate([[0], mask[:, c].astype(np.int8)]))
        events += int(np.sum(d == 1))
    return mask, events


def regions_from_mask(mask: np.ndarray, t: np.ndarray, min_len: float):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            t0, t1 = t[i], t[j]
            if t1 - t0 >= min_len:
                out.append([t0, t1])
            i = j + 1
        else:
            i += 1
    return out


# ----------------------------------------------------------------------------- analysis


def stft_mag(x: np.ndarray, sr: int):
    n_fft = 4096 if sr > 64000 else 2048 if sr > 24000 else 1024
    hop = n_fft // 4
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    f, t, Z = sps.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=True)
    return f, t, np.abs(Z).astype(np.float32), n_fft, hop


def loudness_lufs(x: np.ndarray, sr: int):
    """ITU-R BS.1770-4 integrated loudness, momentary (400 ms) and short-term (3 s) max."""
    # K-weighting coefficients for arbitrary sample rates (same derivation as libebur128)
    K = np.tan(np.pi * 1681.974450955533 / sr)
    Q = 0.7071752369554196
    Vh = 10 ** (3.999843853973347 / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b1 = np.array([Vh + Vb * K / Q + K * K, 2 * (K * K - Vh), Vh - Vb * K / Q + K * K]) / a0
    a1 = np.array([a0, 2 * (K * K - 1), 1 - K / Q + K * K]) / a0
    K = np.tan(np.pi * 38.13547087613982 / sr)
    Q = 0.5003270373253953
    a0 = 1 + K / Q + K * K
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([a0, 2 * (K * K - 1), 1 - K / Q + K * K]) / a0
    y = sps.lfilter(b2, a2, sps.lfilter(b1, a1, x, axis=0), axis=0)
    ch = y.shape[1]
    g = np.array([1.0, 1.0, 1.0, 1.41, 1.41][:ch] + [1.0] * max(0, ch - 5))

    def blocks(win_s, hop_s):
        win, hop = int(win_s * sr), int(hop_s * sr)
        if len(y) < win:
            return np.array([np.sum(g * np.mean(y ** 2, axis=0))]), np.array([0.0])
        ms = np.stack([frame_rms(y[:, c], win, hop) ** 2 for c in range(ch)], axis=1)
        z = np.sum(g * ms, axis=1)
        return z, np.arange(len(z)) * hop_s + win_s / 2

    z, tz = blocks(0.4, 0.1)
    lk = -0.691 + 10 * np.log10(z + EPS)
    gated = z[lk > -70]
    integrated = None
    if len(gated):
        rel = -0.691 + 10 * np.log10(np.mean(gated)) - 10
        g2 = z[(lk > -70) & (lk > rel)]
        if len(g2):
            integrated = -0.691 + 10 * np.log10(np.mean(g2))
    zs, _ = blocks(3.0, 1.0)
    st = -0.691 + 10 * np.log10(zs + EPS)
    lra = None
    st_g = st[st > -70]
    if len(st_g) > 2:
        rel = -0.691 + 10 * np.log10(np.mean(10 ** ((st_g + 0.691) / 10))) - 20
        st_g = st_g[st_g > rel]
        if len(st_g) > 2:
            lra = np.percentile(st_g, 95) - np.percentile(st_g, 10)
    return {
        "integrated_lufs": r(integrated, 1),
        "momentary_max_lufs": r(lk.max(), 1),
        "short_term_max_lufs": r(st.max(), 1),
        "loudness_range_lu": r(lra, 1),
    }, tz, lk


def true_peak_dbfs(x: np.ndarray) -> float:
    n = min(len(x), 48000 * 600)
    up = sps.resample_poly(x[:n], 4, 1, axis=0)
    return float(db(np.max(np.abs(up))))


def detect_f0(mono: np.ndarray, sr: int, gate_db: float):
    """Frame-wise f0 via normalised autocorrelation (YIN-like). Returns times, f0 (nan=unvoiced), clarity."""
    target = 16000
    x = sps.resample_poly(mono, target, sr) if sr != target else mono
    sr2 = target
    win, hop = 1024, 160  # 64 ms window, 10 ms hop
    fmin, fmax = 50.0, 1600.0
    if len(x) < win:
        return np.array([]), np.array([]), np.array([])
    n_frames = 1 + (len(x) - win) // hop
    lag_min, lag_max = int(sr2 / fmax), int(sr2 / fmin)
    f0 = np.full(n_frames, np.nan)
    clar = np.zeros(n_frames)
    window = np.hanning(win)
    wac = np.fft.irfft(np.abs(np.fft.rfft(window, n=2 * win)) ** 2)[: lag_max + 2]
    wac = np.maximum(wac / wac[0], 0.05)
    for s in range(0, n_frames, 2000):
        idx = np.arange(s, min(n_frames, s + 2000))
        fr = np.lib.stride_tricks.as_strided(
            x[idx[0] * hop:], shape=(len(idx), win), strides=(x.strides[0] * hop, x.strides[0])
        ).astype(np.float64)
        fr = fr - fr.mean(axis=1, keepdims=True)
        rms = db(np.sqrt(np.mean(fr ** 2, axis=1)))
        spec = np.fft.rfft(fr * window, n=2 * win, axis=1)
        ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, : lag_max + 2]
        ac = ac / (ac[:, :1] + EPS) / wac  # undo the Hann taper so long lags are not penalised
        seg = ac[:, lag_min: lag_max + 1]
        # first local maximum that reaches 90% of the global one -> avoids octave-down errors
        lag = np.zeros(len(idx), dtype=int)
        val = np.zeros(len(idx))
        is_pk = np.zeros_like(seg, dtype=bool)
        is_pk[:, 1:-1] = (seg[:, 1:-1] >= seg[:, :-2]) & (seg[:, 1:-1] > seg[:, 2:])
        gmax = seg.max(axis=1)
        for k in range(len(idx)):
            cand = np.nonzero(is_pk[k] & (seg[k] >= 0.9 * gmax[k]))[0]
            j = int(cand[0]) if len(cand) else int(np.argmax(seg[k]))
            lag[k] = j + lag_min
            val[k] = seg[k, j]
        # parabolic interpolation
        lagf = lag.astype(float)
        for k, lg in enumerate(lag):
            if 1 <= lg < ac.shape[1] - 1:
                a, b, c = ac[k, lg - 1], ac[k, lg], ac[k, lg + 1]
                d = a - 2 * b + c
                if abs(d) > EPS:
                    lagf[k] = lg + 0.5 * (a - c) / d
        voiced = (val > 0.6) & (rms > gate_db) & (lagf >= lag_min - 1) & (lagf <= lag_max + 1)
        f0[idx[voiced]] = sr2 / lagf[voiced]
        clar[idx] = val
    t = (np.arange(n_frames) * hop + win / 2) / sr2
    return t, f0, clar


def estimate_tempo(flux: np.ndarray, frame_rate: float):
    if len(flux) < frame_rate * 4:
        return None, None
    o = flux - flux.mean()
    ac = np.correlate(o, o, mode="full")[len(o) - 1:]
    ac /= ac[0] + EPS
    bpms = np.arange(40, 241)
    lags = frame_rate * 60.0 / bpms
    vals = np.interp(lags, np.arange(len(ac)), ac)
    # weight toward 120 BPM to reduce octave ambiguity
    w = np.exp(-0.5 * (np.log2(bpms / 120.0) / 1.0) ** 2)
    score = vals * w
    i = int(np.argmax(score))
    return float(bpms[i]), float(vals[i])


def beat_grid(flux, ts, f, S, bpm, beats_per_bar=4, t_start=0.0, frame_db=None):
    """Place a beat grid on the tempo estimate: the phase that best lines up with onset strength, then the
    downbeat = the beat (of beats_per_bar) where big changes (music start, section changes) and low-frequency
    (kick/bass) onsets land. Kicks alone can't separate beat 1 from beat 3; section changes can."""
    if not bpm or len(ts) < 8:
        return None
    period = 60.0 / bpm
    hop = float(np.median(np.diff(ts)))
    low = S[(f >= 30) & (f <= 150)].sum(axis=0)
    lowflux = np.concatenate([[0], np.maximum(0, np.diff(np.log1p(100 * low / (low.max() + EPS))))])
    lowflux /= lowflux.max() + EPS
    # beats are where kicks/bass land; full-band flux alone locks onto off-beat hi-hats half a beat away
    fl = flux / (flux.max() + EPS) + 2.0 * lowflux
    best, phase = -1.0, 0.0
    for ph in np.arange(0, period, max(hop / 2, 0.005)):
        grid = np.arange(ts[0] + ph, ts[-1], period)
        sc = float(np.mean(np.interp(grid, ts, fl))) if len(grid) else 0
        if sc > best:
            best, phase = sc, ph
    b0 = ts[0] + phase
    b0 -= np.floor((b0 - t_start + hop) / period) * period  # extend the grid back to the start of the file
    beats = np.arange(b0, ts[-1], period)
    # change points: where the level steps up by >= 6 dB within ~0.5 s (music start, drops, section entries)
    changes = []
    if frame_db is not None and len(frame_db) > 4:
        k5 = max(1, int(0.5 / hop))
        sm = np.convolve(frame_db, np.ones(k5) / k5, mode="same")
        rise = np.zeros_like(sm)
        rise[k5:] = sm[k5:] - sm[:-k5]
        pk, _ = sps.find_peaks(rise, height=6, distance=max(1, int(4 / hop)))
        # the step's onset is about half a window before the peak of the smoothed rise; bigger steps (music
        # start, a drop) count more than a mid-phrase fill
        changes = [(float(ts[i]) - 0.25, min(float(rise[i]), 20.0) - 5.0) for i in pk]
    scores = []
    for k in range(beats_per_bar):
        bars = beats[k::beats_per_bar]
        if not len(bars):
            scores.append(0.0)
            continue
        kick = float(np.mean(np.interp(bars, ts, lowflux)))
        if changes:
            w = np.array([c[1] for c in changes])
            hit = np.array([np.min(np.abs(bars - c[0])) < period / 2 for c in changes], dtype=float)
            kick += 2.0 * float(np.sum(w * hit) / np.sum(w))
        scores.append(kick)
    k = int(np.argmax(scores))
    first_down = float(beats[k]) if len(beats) > k else float(beats[0])
    rand = float(np.mean(fl))
    return {"beat_period_s": period, "first_beat_s": float(beats[0]) if len(beats) else None,
            "bar_period_s": period * beats_per_bar, "first_downbeat_s": first_down,
            "beats_per_bar_assumed": beats_per_bar,
            "grid_strength": best / (rand + EPS),  # onset strength on the grid vs. average (1 = random)
            "downbeat_confidence": (max(scores) - sorted(scores)[-2]) / (max(scores) + EPS) if len(scores) > 1 else 0}


def estimate_key(chroma_mean: np.ndarray):
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = (-2.0, None)
    scores = []
    for mode, prof in (("major", major), ("minor", minor)):
        for k in range(12):
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.nan_to_num(np.corrcoef(np.roll(prof, k), chroma_mean)[0, 1], nan=-1.0)
            scores.append(c)
            if c > best[0]:
                best = (c, f"{NOTE_NAMES[k]} {mode}")
    scores.sort(reverse=True)
    return best[1], r(best[0], 3), r(scores[0] - scores[1], 3)


def chroma_from_stft(f, S):
    valid = (f >= 55) & (f <= 5000)
    pcs = np.round(12 * np.log2(f[valid] / 440.0) + 69).astype(int) % 12
    P = S[valid] ** 2
    C = np.zeros((12, S.shape[1]), dtype=np.float32)
    for pc in range(12):
        C[pc] = P[pcs == pc].sum(axis=0)
    C /= C.max(axis=0, keepdims=True) + EPS
    return C


def find_peaks_spectrum(freqs, pow_db, n=8, min_hz=20):
    m = freqs >= min_hz
    fz, pz = freqs[m], pow_db[m]
    # prominence relative to a smoothed floor
    k = max(3, len(pz) // 200)
    floor = sps.medfilt(pz, kernel_size=k * 2 + 1)
    idx, props = sps.find_peaks(pz, prominence=6, distance=max(1, len(pz) // 400))
    if not len(idx):
        return []
    order = np.argsort(pz[idx])[::-1]
    out = []
    for j in order[:n]:
        i = idx[j]
        hz = fz[i]
        if 0 < i < len(pz) - 1:  # parabolic interpolation for sub-bin accuracy
            a, b, c = pz[i - 1], pz[i], pz[i + 1]
            d = a - 2 * b + c
            if abs(d) > EPS:
                hz = fz[i] + 0.5 * (a - c) / d * (fz[1] - fz[0])
        out.append({"hz": r(hz, 1), "note": hz_to_note(hz), "db": r(pz[i], 1),
                    "above_floor_db": r(pz[i] - floor[i], 1)})
    return out


def detect_hum(freqs, pow_db):
    """Look for 50/60 Hz mains hum (fundamental + harmonics standing above the local floor)."""
    res = {}
    for base in (50, 60):
        hits = []
        for h in range(1, 6):
            f0 = base * h
            if f0 >= freqs[-1]:
                break
            near = (freqs > f0 - 2.5) & (freqs < f0 + 2.5)
            ring = ((freqs > f0 - 15) & (freqs < f0 - 5)) | ((freqs > f0 + 5) & (freqs < f0 + 15))
            if near.any() and ring.any():
                hits.append(float(pow_db[near].max() - np.median(pow_db[ring])))
        res[base] = hits
    best = max(res, key=lambda b: (sum(h > 10 for h in res[b]), res[b][0] if res[b] else -99))
    strong = sum(h > 10 for h in res[best])
    # harmonics present, or a pure tone sitting right on 50/60 Hz
    sus = strong >= 2 or (bool(res[best]) and res[best][0] > 8)
    return {"suspected": sus, "mains_hz": best if sus else None,
            "harmonic_excess_db": [r(h, 1) for h in res[best]]}


def analyze(x: np.ndarray, sr: int, info: dict, music: bool = False):
    n, ch = x.shape
    mono = x.mean(axis=1)
    off = info.get("offset_s", 0.0)
    res: dict = {"file": info}
    if ch > 1 and np.sqrt(np.mean(mono ** 2)) < 0.5 * np.sqrt(np.mean(x[:, 0] ** 2)):
        mono = x[:, 0].copy()  # L+R cancels (antiphase): analyse the left channel instead
        info["analysis_channel"] = "left only (mono sum cancels)"
    res["duration_s"] = r(n / sr, 3)

    # --- level / integrity
    peak = float(np.max(np.abs(x))) if n else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if n else 0.0
    clipped, clip_runs = clip_mask(x)
    over = int(np.sum(np.abs(x) > 1.0))
    first_clip = None
    if clipped.any():
        first_clip = off + int(np.argmax(clipped.any(axis=1))) / sr
    dc = x.mean(axis=0)
    lvl = {
        "peak_dbfs": r(db(peak), 2),
        "true_peak_dbtp": r(true_peak_dbfs(x), 2) if n > 16 else None,
        "rms_dbfs": r(db(rms), 2),
        "crest_factor_db": r(db(peak) - db(rms), 2),
        "dc_offset": [r(d, 5) for d in dc],
        "clipped_samples": int(clipped.sum()),
        "clip_events": clip_runs,
        "samples_over_full_scale": over,
        "first_clip_s": r(first_clip, 4),
        "first_sample_dbfs": r(db(np.max(np.abs(x[0]))), 1) if n else None,
        "last_sample_dbfs": r(db(np.max(np.abs(x[-1]))), 1) if n else None,
    }
    lufs, tz, lk = loudness_lufs(x.astype(np.float64), sr)
    lvl.update(lufs)
    res["level"] = lvl

    # --- envelope, silence, noise floor
    win, hop = int(0.02 * sr), int(0.01 * sr)
    env = db(frame_rms(mono, max(win, 1), max(hop, 1)))
    tenv = off + (np.arange(len(env)) * hop + win / 2) / sr
    noise_floor = float(np.percentile(env, 10))
    active_level = max(float(np.percentile(env, 95)), float(env.max()) - 20)  # sparse hits barely reach p95
    sil_thr = max(-60.0, min(noise_floor + 6, active_level - 30))
    silences = regions_from_mask(env < sil_thr, tenv, 0.1)
    act = env > sil_thr
    res["dynamics"] = {
        "noise_floor_dbfs_p10": r(noise_floor, 1),
        "active_level_dbfs_p95": r(active_level, 1),
        "snr_estimate_db": r(active_level - noise_floor, 1),
        "silence_threshold_dbfs": r(sil_thr, 1),
        "silence_regions_s": [[r(a, 3), r(b, 3)] for a, b in silences][:50],
        "silence_total_s": r(sum(b - a for a, b in silences), 3),
        "leading_silence_s": r((tenv[np.argmax(act)] - off) if act.any() else n / sr, 3),
        "trailing_silence_s": r((off + n / sr - tenv[len(act) - 1 - np.argmax(act[::-1])]) if act.any() else n / sr, 3),
    }
    sil_total = sum(b - a for a, b in silences)
    res["dynamics"]["noise_floor_meaningful"] = bool(sil_total >= 0.02 * n / sr or noise_floor < -80)
    if not res["dynamics"]["noise_floor_meaningful"]:
        res["dynamics"]["note"] = ("no pauses: the 'floor' is the quietest 10% of the programme itself (music, "
                                   "ambience), not background noise; SNR is really a dynamic-range figure")

    # --- spectral
    f, ts, S, n_fft, shop = stft_mag(mono, sr)
    ts = ts + off
    P = S.astype(np.float64) ** 2
    pt = P.sum(axis=0) + EPS
    centroid = (f[:, None] * P).sum(axis=0) / pt
    cum = np.cumsum(P, axis=0) / pt
    roll85 = f[np.argmax(cum >= 0.85, axis=0)]
    gm = np.exp(np.mean(np.log(P[1:] + EPS), axis=0))
    flat = gm / (P[1:].mean(axis=0) + EPS)
    frame_db = pdb(pt)
    loud = frame_db > (frame_db.max() - 40)
    flat_med = float(np.median(flat[loud])) if loud.any() else float(np.median(flat))
    wf, wp = sps.welch(mono, fs=sr, nperseg=min(len(mono), 8192 if sr > 24000 else 4096))
    wdb = pdb(wp)
    band_e = {}
    tot = wp[(wf >= 20)].sum() + EPS
    for name, lo, hi in BANDS:
        m = (wf >= lo) & (wf < min(hi, sr / 2))
        band_e[name] = r(100 * wp[m].sum() / tot, 1) if m.any() else 0.0
    above = wf[wdb > wdb.max() - 60]
    # are the quietest frames noise-like (hiss/room) or just quiet signal?
    quiet = frame_db <= np.percentile(frame_db, 20)
    quiet &= frame_db > -100  # digital silence is not "noise"
    quiet_flat = float(np.median(flat[quiet])) if quiet.any() else 0.0
    broadband_bands = sum(1 for v in band_e.values() if v and v > 0.5)
    res["spectral"] = {
        "stft": {"n_fft": n_fft, "hop": shop, "freq_resolution_hz": r(sr / n_fft, 2)},
        "centroid_hz_median": r(np.median(centroid[loud]) if loud.any() else np.median(centroid), 0),
        "rolloff85_hz_median": r(np.median(roll85[loud]) if loud.any() else np.median(roll85), 0),
        "flatness_median": r(np.median(flat[loud]) if loud.any() else np.median(flat), 3),
        "bandwidth_hz_upper_-60db": r(above.max() if len(above) else 0, 0),
        "band_energy_percent": band_e,
        "broadband_band_count": broadband_bands,
        "quiet_frame_flatness": r(quiet_flat, 3),
        "dominant_peaks": find_peaks_spectrum(wf, wdb),
        "mains_hum": detect_hum(wf, wdb),
    }

    # --- onsets / tempo (spectral flux)
    logS = np.log1p(100 * S)
    flux = np.maximum(0, np.diff(logS, axis=1)).sum(axis=0)
    flux = np.concatenate([[0], flux])
    frame_rate = sr / shop
    if len(flux) > 3:
        k = min(int(frame_rate * 0.5), len(flux) - 1) | 1  # medfilt warns if the kernel outgrows the signal
        thr = sps.medfilt(flux, kernel_size=k) + 0.5 * np.std(flux)
        pk, _ = sps.find_peaks(flux, height=thr, distance=max(1, int(frame_rate * 0.05)))
    else:
        pk = np.array([], dtype=int)
    onsets = ts[pk]
    bpm, bpm_conf = estimate_tempo(flux, frame_rate)
    if len(onsets) < 4 or (bpm_conf or 0) < 0.1:
        bpm = None  # no rhythmic content to speak of
    ioi = np.diff(onsets)
    grid = beat_grid(flux, ts, f, S, bpm, t_start=off, frame_db=frame_db) if bpm else None
    res["rhythm"] = {
        "onset_count": int(len(onsets)),
        "onsets_s": [r(o, 3) for o in onsets[:200]],
        "onset_rate_per_s": r(len(onsets) / max(n / sr, EPS), 2),
        "median_inter_onset_s": r(np.median(ioi), 3) if len(ioi) else None,
        "tempo_bpm_estimate": r(bpm, 1),
        "tempo_confidence": r(bpm_conf, 2),
        "grid": None if not grid else {k: (r(v, 4) if isinstance(v, float) else v) for k, v in grid.items()},
        "grid_note": "bar n starts at first_downbeat_s + n * bar_period_s (4/4 assumed). Tempo can be off by x2; "
                     "check the grid against onset ticks / the structure view before quoting bar numbers.",
    }

    # --- pitch
    tf0, f0, clar = detect_f0(mono, sr, gate_db=sil_thr)
    tf0 = tf0 + off
    # drop "virtual" f0 frames: in chords the periodicity is a missing fundamental with no energy at that
    # frequency (e.g. C3 under a C-E-G triad), which reads as a wrong note on the plot and in the CSV
    vm = np.isfinite(f0)
    virtual = 0
    if vm.any() and len(ts):
        col = np.clip(np.searchsorted(ts, tf0[vm]), 0, len(ts) - 1)
        row = np.clip(np.searchsorted(f, f0[vm]), 1, len(f) - 2)
        here = np.maximum.reduce([S[row - 1, col], S[row, col], S[row + 1, col]])
        rel = 20 * np.log10((here + EPS) / (S[:, col].max(axis=0) + EPS))
        bad = np.nonzero(vm)[0][rel < -25]
        virtual = len(bad)
        f0[bad] = np.nan
    v = f0[np.isfinite(f0)]
    both = np.isfinite(f0[1:]) & np.isfinite(f0[:-1])
    steps = np.abs(12 * np.log2(f0[1:][both] / f0[:-1][both]))
    hop_frac = float(np.mean(steps > 1.0)) if both.sum() > 10 else 0.0
    # a real melody line moves smoothly frame to frame (10 ms); in chords / full mixes the
    # tracker keeps hopping between instruments and octaves
    reliable = len(v) > 0 and hop_frac < 0.08
    res["pitch"] = {
        "reliable": bool(reliable),
        "frame_jump_fraction": r(hop_frac, 3),
        "voiced_fraction": r(len(v) / max(1, len(f0)), 3),
        "virtual_f0_frames_removed": int(virtual),
        "f0_median_hz": r(np.median(v), 1) if len(v) else None,
        "f0_median_note": hz_to_note(np.median(v)) if len(v) else None,
        "f0_p10_p90_hz": [r(np.percentile(v, 10), 1), r(np.percentile(v, 90), 1)] if len(v) else None,
        "note": "f0 is only meaningful for monophonic/periodic sounds (voice, single instrument, tones); "
                "reliable=false means the track jumps around (chords / full mix) - ignore f0 then.",
    }

    # --- tonality / key
    C = chroma_from_stft(f, S)
    cm = (C[:, loud] if loud.any() else C).mean(axis=1)
    key, kc, kmargin = estimate_key(cm)
    if (kc or 0) < 0.6 or np.sum(cm > 0.25 * cm.max()) < 3 or (res["pitch"]["voiced_fraction"] == 0 and flat_med > 0.3):
        key = None  # single tone or noise: a key is meaningless
    res["tonal"] = {
        "chroma_mean": {NOTE_NAMES[i]: r(cm[i], 2) for i in range(12)},
        "key_estimate": key, "key_correlation": kc, "key_margin": kmargin,
        "note": "Krumhansl-profile guess; null when the material is not clearly tonal (correlation < 0.6).",
    }

    # --- stereo
    if ch >= 2:
        L, R = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
        corr = float(np.sum(L * R) / (np.sqrt(np.sum(L ** 2) * np.sum(R ** 2)) + EPS))
        mid, side = (L + R) / 2, (L - R) / 2
        s2m = db(np.sqrt(np.mean(side ** 2))) - db(np.sqrt(np.mean(mid ** 2)))
        res["stereo"] = {
            "correlation": r(corr, 3),
            "side_to_mid_db": r(db(np.sqrt(np.mean(side ** 2))) - db(np.sqrt(np.mean(mid ** 2))), 1),
            "balance_L_minus_R_db": r(db(np.sqrt(np.mean(L ** 2))) - db(np.sqrt(np.mean(R ** 2))), 2),
            # lossy stereo coding leaves tiny L/R differences, so also accept a negligible side signal
            "effectively_mono": bool(np.max(np.abs(L - R)) < 1e-4 or (s2m < -30 and corr > 0.99)),
            "phase_issue_suspected": bool(corr < 0),
        }

    # --- events (separate sounds) with envelope + pitch movement
    peak_track = spectral_peak_track(f, S)
    t5, env5, events = detect_events(mono, sr, off, sil_thr, n)
    for i, ev in enumerate(events, 1):
        ev["index"] = i
        describe_event(ev, t5, env5, tf0, f0, ts, peak_track, frame_db, centroid, onsets, off + n / sr, f, S, flat,
                       x, mono, sr, off)
    res["events"] = {"count": len(events), "list": [compact_event(e) for e in events[:40]]}

    res["_flags"] = make_flags(res)
    res["_series"] = dict(tenv=tenv, env=env, tz=tz + off, lk=lk, f=f, ts=ts, S=S, tf0=tf0, f0=f0,
                          C=C, onsets=onsets, wf=wf, wdb=wdb, silences=silences, sil_thr=sil_thr,
                          centroid=centroid, clip=clipped.any(axis=1), flat=flat, peak_track=peak_track,
                          t5=t5, env5=env5, events=events, mono=mono)
    return res


# ----------------------------------------------------------------------------- events


def spectral_peak_track(f, S, fmin=50.0):
    """Frequency of the strongest spectral peak per frame, refined by parabolic interpolation."""
    lo = int(np.searchsorted(f, fmin))
    if lo >= len(f) - 2:
        return np.zeros(S.shape[1])
    L = np.log(S[lo:].astype(np.float64) + EPS)
    i = np.argmax(L, axis=0)
    i = np.clip(i, 1, L.shape[0] - 2)
    cols = np.arange(L.shape[1])
    a, b, c = L[i - 1, cols], L[i, cols], L[i + 1, cols]
    d = a - 2 * b + c
    shift = np.where(np.abs(d) > EPS, 0.5 * (a - c) / np.where(np.abs(d) > EPS, d, 1), 0.0)
    return f[lo + i] + np.clip(shift, -0.5, 0.5) * (f[1] - f[0])


def note_sequence(times, hz, min_dur=0.04, max_notes=10):
    """Collapse a pitch track into the sequence of held notes, e.g. 'C5 E5 G5 C6'."""
    seq, cur, t0 = [], None, None
    for t, h in zip(times, hz):
        m = int(round(69 + 12 * math.log2(h / 440))) if h and h > 0 and np.isfinite(h) else None
        if m != cur:
            if cur is not None and t - t0 >= min_dur and (not seq or seq[-1] != cur):
                seq.append(cur)
            cur, t0 = m, t
    if cur is not None and times is not None and len(times) and times[-1] - t0 >= min_dur and (not seq or seq[-1] != cur):
        seq.append(cur)
    names = [f"{NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in seq]
    return " ".join(names[:max_notes]) + (" ..." if len(names) > max_notes else "")


def detect_events(mono, sr, off, sil_thr, n, merge_gap=0.08, min_len=0.01, max_events=500):
    """Split the signal into separate sounds: active regions above the silence threshold,
    merging gaps shorter than merge_gap. Returns a fine (5 ms) envelope too."""
    win, hop = max(1, int(0.01 * sr)), max(1, int(0.005 * sr))
    env5 = db(frame_rms(mono, win, hop))
    t5 = off + (np.arange(len(env5)) * hop + win / 2) / sr
    regions = regions_from_mask(env5 > sil_thr, t5, 0.0)
    merged = []
    for a, b in regions:
        if merged and a - merged[-1][1] < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    events = []
    for a, b in merged:
        a, b = max(off, a - win / sr / 2), min(off + n / sr, b + win / sr / 2)  # frame centres -> frame edges
        if b - a >= min_len - 1e-9:
            events.append({"start_s": a, "end_s": b})
    return t5, env5, events[:max_events]


def _semitones(a, b):
    return 12 * math.log2(b / a) if a and b and a > 0 and b > 0 else None


def describe_event(ev, t5, env5, tf0, f0, ts, peak_track, frame_db, centroid, onsets, file_end, f, S, flat,
                   x, mono, sr, off):
    a, b = ev["start_s"], ev["end_s"]
    ev["duration_s"] = b - a
    m = (t5 >= a) & (t5 <= b)
    e, te = env5[m], t5[m]
    if not len(e):
        return
    i0, i1 = max(0, int((a - off) * sr)), min(len(x), int((b - off) * sr))
    seg = x[i0:i1]
    pk = int(np.argmax(e))
    peak_db = float(e[pk])
    ev["peak_dbfs"] = float(db(np.max(np.abs(seg)))) if len(seg) else peak_db  # true sample peak
    ev["peak_rms10ms_dbfs"] = peak_db
    ev["peak_time_s"] = float(te[pk])
    ev["rms_dbfs"] = float(db(np.sqrt(np.mean(10 ** (e / 10)))))
    ev["freq_res_hz"] = float(f[1] - f[0])
    ev["onsets_inside"] = int(np.sum((onsets >= a) & (onsets <= b)))
    ev["ends_at_file_end"] = bool(file_end - b < 0.01)
    if ev["duration_s"] > 10:
        # a long continuous stretch (music, ambience): attack/decay/start-vs-end pitch say nothing - use the views
        return
    # envelope (ADSR-style): attack = -20 dB point -> peak, decay = peak -> -20 dB below peak
    # attack = from the -20 dB point to the FIRST local maximum near the top (a multi-hit event has several
    # peaks; its attack is how fast the first hit lands, not when the loudest one arrives)
    rise = np.nonzero(e[: pk + 1] >= peak_db - 20)[0]
    first_top = pk
    if len(rise):
        for i in range(rise[0], pk):
            if e[i] >= peak_db - 10 and e[i] >= e[i + 1]:
                first_top = i
                break
    ev["attack_ms"] = 0.0
    if len(rise):
        # refine on a 0.5 ms grid (the 5 ms envelope can only say 5 or 10 ms)
        j0 = max(0, int((te[rise[0]] - 0.012 - off) * sr))
        j1 = min(len(mono), int((te[first_top] + 0.008 - off) * sr))
        fine = db(frame_rms(mono[j0:j1], max(1, int(0.002 * sr)), max(1, int(0.0005 * sr)))) if j1 > j0 else []
        if len(fine) > 2:
            top_i = int(np.argmax(fine))
            up = np.nonzero(fine[: top_i + 1] >= fine[top_i] - 20)[0]
            ev["attack_ms"] = float((top_i - up[0]) * 0.5) if len(up) else 0.0
        else:
            ev["attack_ms"] = float((te[first_top] - te[rise[0]]) * 1000)
    fall = np.nonzero(e[pk:] < peak_db - 20)[0]
    ev["decay_to_-20db_ms"] = float((te[pk + fall[0]] - te[pk]) * 1000) if len(fall) else None
    body = e[pk:]
    ev["body_level_db_rel_peak"] = float(np.median(body) - peak_db) if len(body) > 2 else 0.0
    ev["end_level_db_rel_peak"] = float(e[-1] - peak_db)
    # pitch movement: f0 track if it is stable inside the event, otherwise the strongest spectral peak
    mf = (tf0 >= a) & (tf0 <= b) & np.isfinite(f0)
    fv = f0[mf]
    f0_ok = False
    if len(fv) >= 3:
        steps = np.abs(12 * np.log2(fv[1:] / fv[:-1]))
        # f0 must sit on real energy: chords/arpeggios give a "missing fundamental" with nothing there
        col = np.clip(np.searchsorted(ts, tf0[mf]), 0, len(ts) - 1)
        row = np.clip(np.searchsorted(f, fv), 1, len(f) - 2)
        here = np.maximum.reduce([S[row - 1, col], S[row, col], S[row + 1, col]])
        rel = 20 * np.log10((here + EPS) / (S[:, col].max(axis=0) + EPS))
        fund_present = float(np.median(rel)) > -25
        # count real jumps only: a sung/spoken glide can move ~1 semitone per 10 ms frame
        coverage = mf.sum() / max(1, ((tf0 >= a) & (tf0 <= b)).sum())
        f0_ok = fund_present and float(np.mean(steps > 3.0)) < 0.1 and coverage >= 0.5
        k = max(1, min(3, len(fv) // 3))
        ev["f0_start_hz"], ev["f0_end_hz"] = float(np.median(fv[:k])), float(np.median(fv[-k:]))
        ev["f0_median_hz"] = float(np.median(fv))
    ms = (ts >= a) & (ts <= b)
    ms &= frame_db > (frame_db[ms].max() - 30 if ms.any() else -999)
    if (ms & (flat < 0.2)).sum() >= 3:
        ms &= flat < 0.2  # tonal frames only: hiss/noise frames have no meaningful peak
    pt = peak_track[ms]
    if len(pt):
        k = max(1, len(pt) // 5)
        ev["peak_start_hz"], ev["peak_end_hz"] = float(np.median(pt[:k])), float(np.median(pt[-k:]))
        midi = np.round(69 + 12 * np.log2(np.maximum(pt, 1) / 440))
        vals, counts = np.unique(midi, return_counts=True)
        top = vals[np.argmax(counts)]
        ev["dominant_hz"] = float(np.median(pt[midi == top]))
        ev["centroid_hz"] = float(np.median(centroid[ms]))
    track = fv if f0_ok else pt if len(pt) else None
    if track is not None and len(track) >= 3:
        ev["held_notes"] = note_sequence(tf0[mf] if f0_ok else ts[ms], track)
        st = 12 * np.log2(track / track[0])
        k = max(1, len(st) // 5)
        s0, s1 = float(np.median(st[:k])), float(np.median(st[-k:]))
        hi, lo = float(np.percentile(st, 95)), float(np.percentile(st, 5))
        ev["pitch_range_st"] = hi - lo
        if hi - max(s0, s1) > 2:
            ev["pitch_contour"] = "rise-fall"
        elif min(s0, s1) - lo > 2:
            ev["pitch_contour"] = "fall-rise"
        else:
            ev["pitch_contour"] = "rising" if s1 - s0 > 1 else "falling" if s1 - s0 < -1 else "flat"
    if f0_ok:
        ch, src = _semitones(ev["f0_start_hz"], ev["f0_end_hz"]), "f0"
    elif "peak_start_hz" in ev:
        ch, src = _semitones(ev["peak_start_hz"], ev["peak_end_hz"]), "spectral_peak"
    else:
        ch, src = None, None
    ev["pitch_change_st"] = ch
    ev["pitch_source"] = src
    ev["pitch_direction"] = (None if ch is None else "rising" if ch > 1 else "falling" if ch < -1 else "flat")


EVENT_COLUMNS = ["index", "start_s", "end_s", "duration_s", "peak_time_s", "peak_dbfs", "peak_rms10ms_dbfs", "rms_dbfs",
                 "attack_ms",
                 "decay_to_-20db_ms", "body_level_db_rel_peak", "end_level_db_rel_peak", "ends_at_file_end",
                 "onsets_inside", "dominant_hz", "dominant_note", "centroid_hz", "f0_start_hz", "f0_end_hz",
                 "f0_median_hz", "peak_start_hz", "peak_end_hz", "pitch_change_st", "pitch_direction", "pitch_contour",
                 "pitch_range_st", "pitch_source", "held_notes"]


def compact_event(ev):
    d = {"index": ev.get("index")}
    for k in EVENT_COLUMNS[1:]:
        v = ev.get(k)
        if isinstance(v, float):
            v = r(v, 1 if k.endswith(("_ms", "_hz", "_db", "_dbfs", "_st", "_peak")) else 3)
        d[k] = v
    if ev.get("dominant_hz"):
        hz, res = ev["dominant_hz"], ev.get("freq_res_hz", 0)
        if res and res / hz > 0.15:
            # a bin spans >= ~2.5 semitones here (bass): even with peak interpolation the cents are not trustworthy
            note = hz_to_note(hz)
            d["dominant_note"] = "~" + note[: next((i for i, c in enumerate(note) if c in "+-" and i > 1), len(note))]
        else:
            d["dominant_note"] = hz_to_note(hz)
    return d


def write_csvs(res, out: Path):
    """events.csv (one row per sound) and frames.csv (time series) for exact values."""
    import csv

    s = res["_series"]
    rows = [compact_event(e) for e in s["events"]]
    with open(out / "events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EVENT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    dur = res["duration_s"]
    off = res["file"].get("offset_s", 0.0)
    step = max(0.02, dur / 2000)  # at most ~2000 rows
    grid = np.arange(off, off + dur, step)

    def at(t_src, v_src, how="interp"):
        if len(t_src) == 0:
            return np.full(len(grid), np.nan)
        if how == "interp":
            return np.interp(grid, t_src, v_src)
        idx = np.clip(np.searchsorted(t_src, grid), 0, len(t_src) - 1)
        return v_src[idx]

    cols = {
        "t_s": grid,
        "rms_dbfs": at(s["tenv"], s["env"]),
        "loudness_lufs_m": at(s["tz"], s["lk"]),
        "f0_hz": at(s["tf0"], s["f0"], "nearest"),
        "peak_hz": at(s["ts"], s["peak_track"], "nearest"),
        "centroid_hz": at(s["ts"], s["centroid"], "nearest"),
        "flatness": at(s["ts"], s["flat"], "nearest"),
    }
    with open(out / "frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(cols) + ["f0_note"])
        for i in range(len(grid)):
            row = []
            for k, v in cols.items():
                x = v[i]
                row.append("" if not np.isfinite(x) else f"{x:.3f}" if k in ("t_s", "flatness") else f"{x:.1f}")
            row.append(hz_to_note(cols["f0_hz"][i]) if np.isfinite(cols["f0_hz"][i]) else "")
            w.writerow(row)


def make_flags(res):
    fl = []
    L = res["level"]
    if L["clipped_samples"] > 0:
        fl.append(f"CLIPPING: {L['clipped_samples']} samples at full scale in {L['clip_events']} events "
                  f"(first at {L['first_clip_s']}s)")
    if L["samples_over_full_scale"]:
        fl.append(f"{L['samples_over_full_scale']} FLOAT SAMPLES ABOVE 1.0 (will hard-clip when converted to fixed point / played)")
    if L["last_sample_dbfs"] is not None and L["last_sample_dbfs"] > -40:
        fl.append(f"ABRUPT END: last sample at {L['last_sample_dbfs']} dBFS (no fade-out -> likely click)")
    if L["first_sample_dbfs"] is not None and L["first_sample_dbfs"] > -40:
        fl.append(f"ABRUPT START: first sample at {L['first_sample_dbfs']} dBFS (no fade-in -> possible click)")
    if L["true_peak_dbtp"] is not None and L["true_peak_dbtp"] > -1.0:
        fl.append(f"TRUE PEAK {L['true_peak_dbtp']} dBTP > -1 dBTP (risk of inter-sample clipping after encoding)")
    if any(abs(d) > 0.01 for d in L["dc_offset"] if d is not None):
        fl.append(f"DC OFFSET {L['dc_offset']}")
    if L["peak_dbfs"] < -20:
        fl.append(f"VERY QUIET: peak {L['peak_dbfs']} dBFS")
    if L["integrated_lufs"] is None:
        fl.append("NO MEASURABLE LOUDNESS (silent or nearly silent)")
    D = res["dynamics"]
    Sp = res["spectral"]
    if D["snr_estimate_db"] is not None and D["snr_estimate_db"] < 20 and Sp["quiet_frame_flatness"] > 0.2:
        fl.append(f"HIGH NOISE FLOOR: quietest parts are noise-like, only {D['snr_estimate_db']} dB below active level")
    if D["leading_silence_s"] and D["leading_silence_s"] > 0.05:
        fl.append(f"LEADING SILENCE {D['leading_silence_s']}s (adds latency if this is a triggered sound)")
    if D["trailing_silence_s"] and D["trailing_silence_s"] > 0.5:
        fl.append(f"TRAILING SILENCE {D['trailing_silence_s']}s")
    if res["spectral"]["mains_hum"]["suspected"]:
        fl.append(f"MAINS HUM suspected at {res['spectral']['mains_hum']['mains_hz']} Hz + harmonics")
    sr = res["file"]["sample_rate"]
    bw = res["spectral"]["bandwidth_hz_upper_-60db"]
    if bw and sr >= 32000 and bw < 0.35 * sr and Sp["broadband_band_count"] >= 4:
        codec = res["file"].get("lossy_codec")
        if codec:
            fl.append(f"BAND-LIMITED (expected): no energy above ~{int(bw)} Hz - the {codec} encoder's low-pass, "
                      "normal for lossy files at this bitrate, not a defect of the source")
        else:
            fl.append(f"BAND-LIMITED: little energy above ~{int(bw)} Hz (sr {sr}) in an uncompressed file - "
                      "it was likely low-passed, upsampled, or decoded from a lossy source earlier")
    st = res.get("stereo")
    if st:
        if st["effectively_mono"]:
            fl.append(f"STEREO FILE IS EFFECTIVELY MONO (L ~= R, side {st["side_to_mid_db"]} dB below mid) - "
                      "could be saved as mono to halve the size")
        if st["phase_issue_suspected"]:
            fl.append(f"NEGATIVE L/R CORRELATION {st['correlation']} (phase problem, mono-collapse cancellation)")
        if abs(st["balance_L_minus_R_db"]) > 3:
            fl.append(f"CHANNEL IMBALANCE {st['balance_L_minus_R_db']} dB (L-R)")
    return fl


# ----------------------------------------------------------------------------- plotting

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "axes.grid": True,
    "grid.alpha": 0.35, "figure.dpi": 100, "savefig.dpi": 100,
})


def add_note_axis(ax, fmin, fmax, every="C"):
    """Right-hand axis with note names (C1..C9) and faint guide lines, so pitches can be read off the image."""
    notes = []
    for octave in range(0, 10):
        for name in (["C"] if every == "C" else NOTE_NAMES):
            hz = 440.0 * 2 ** ((NOTE_NAMES.index(name) + 12 * (octave + 1) - 69) / 12)
            if fmin <= hz <= fmax:
                notes.append((hz, f"{name}{octave}"))
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(ax.get_ylim() if ax.get_ylim()[0] > 0 else (fmin, fmax))
    ax2.set_yticks([h for h, _ in notes])
    ax2.set_yticklabels([n for _, n in notes], fontsize=8)
    ax2.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax2.tick_params(axis="y", which="minor", length=0)
    ax2.grid(False)
    for h, nm in notes:
        ax.axhline(h, color="white", lw=0.4, alpha=0.25 if nm[0] != "C" or every != "C" else 0.35, ls=":")
    return ax2


def _logfreq_image(f, S, fmin=20.0, n_bins=320):
    fmax = f[-1]
    edges = np.geomspace(max(fmin, f[1]), fmax, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    Sdb = pdb(S.astype(np.float64) ** 2)
    img = np.empty((n_bins, S.shape[1]), dtype=np.float32)
    idx = np.searchsorted(f, edges)
    for i in range(n_bins):
        a, b = idx[i], max(idx[i + 1], idx[i] + 1)
        if b - a <= 1:  # bin narrower than STFT resolution: take nearest row
            img[i] = Sdb[int(np.interp(centers[i], f, np.arange(len(f))))]
        else:
            img[i] = Sdb[a:b].max(axis=0)
    return edges, img


def _pool_time(t, img, max_cols=1800):
    if img.shape[1] <= max_cols:
        return t, img
    k = int(math.ceil(img.shape[1] / max_cols))
    m = img.shape[1] // k * k
    return t[:m:k], img[:, :m].reshape(img.shape[0], -1, k).max(axis=2)


def plot_overview(res, x, sr, out: Path, music=False, linear_freq=False, title=""):
    s = res["_series"]
    n, ch = x.shape
    off = res["file"].get("offset_s", 0.0)
    rows = [1.2, 1.0, 2.6] + ([1.0] if music else [])
    fig, axes = plt.subplots(len(rows), 1, figsize=(16, sum(rows) * 2.3 + 0.8), sharex=True,
                             gridspec_kw={"height_ratios": rows})
    t_end = off + n / sr

    # waveform min/max envelope
    ax = axes[0]
    cols = 2000
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for c in range(min(ch, 4)):
        y = x[:, c]
        if n > cols * 2:
            k = int(math.ceil(n / cols))
            m = int(math.ceil(n / k))
            yy = np.pad(y, (0, m * k - n), mode="edge").reshape(m, k)
            tt = off + np.minimum((np.arange(m) * k + k / 2) / sr, n / sr)
            ax.fill_between(tt, yy.min(1), yy.max(1), color=colors[c], alpha=0.55 if ch > 1 else 0.8,
                            lw=0, label=f"ch{c}" + (" (L)" if c == 0 and ch == 2 else " (R)" if c == 1 else ""))
        else:
            ax.plot(off + np.arange(n) / sr, y, color=colors[c], lw=0.8, label=f"ch{c}")
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.axhline(-1, color="k", ls="--", lw=0.8)
    clipped = s["clip"]
    if clipped.any():
        ct = off + np.nonzero(clipped)[0] / sr
        ax.plot(ct, np.ones_like(ct) * 1.04, "|", color="red", ms=10, label="clip")
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("amplitude")
    ax.set_title("Waveform (min/max per column; dashed = full scale)")
    if ch > 1 or clipped.any():
        ax.legend(loc="upper right", fontsize=9, ncol=4)

    # loudness
    ax = axes[1]
    ax.plot(s["tenv"], s["env"], color="#444", lw=1, label="RMS 20 ms (dBFS)")
    ax.plot(s["tz"], s["lk"], color="#ff7f0e", lw=1.4, label="momentary loudness 400 ms (LUFS)")
    ax.axhline(s["sil_thr"], color="#888", ls=":", lw=1, label=f"silence thr {s['sil_thr']:.0f} dB")
    for a, b in s["silences"]:
        ax.axvspan(a, b, color="#bbb", alpha=0.35, lw=0)
    if len(s["onsets"]) <= 150:  # beyond that the ticks merge into a solid block
        for o in s["onsets"]:
            ax.axvline(o, color="#2ca02c", lw=0.6, alpha=0.6)
    lo = max(-90, np.percentile(s["env"], 1) - 5)
    ax.set_ylim(lo, 3)
    ax.set_ylabel("dB")
    ax.set_title("Level over time (grey = silence, green ticks = onsets"
                 + (")" if len(s["onsets"]) <= 150 else f"; {len(s['onsets'])} onsets not drawn - zoom in)"))
    ax.legend(loc="lower right", fontsize=9, ncol=3)

    # spectrogram
    ax = axes[2]
    S, f, ts = s["S"], s["f"], s["ts"]
    if linear_freq:
        img = pdb(S.astype(np.float64) ** 2)
        tt, img = _pool_time(ts, img)
        edges = np.concatenate([f, [f[-1] + (f[1] - f[0])]])
    else:
        edges, img = _logfreq_image(f, S)
        tt, img = _pool_time(ts, img)
    vmax = float(np.percentile(img, 99.8))
    vmin = vmax - 90
    dt = (tt[1] - tt[0]) if len(tt) > 1 else 0.01
    te = np.concatenate([tt - dt / 2, [tt[-1] + dt / 2]])
    mesh = ax.pcolormesh(te, edges, img, cmap="magma", vmin=vmin, vmax=vmax, shading="flat",
                         rasterized=True)
    if not linear_freq:
        ax.set_yscale("log")
        ticks = [t for t in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000] if edges[0] <= t <= edges[-1]]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_hz))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    vf = np.isfinite(s["f0"])
    if vf.any() and res["pitch"]["voiced_fraction"] > 0.05 and res["pitch"]["reliable"]:
        ax.plot(s["tf0"], s["f0"], color="cyan", lw=1.6, label="f0 estimate")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(edges[0], edges[-1])
    if not linear_freq:
        add_note_axis(ax, edges[0], edges[-1])
    ax.set_ylabel("frequency (Hz)")
    ax.grid(alpha=0.15)
    ax.set_title(f"Spectrogram ({'linear' if linear_freq else 'log'} freq, dB, colour range {vmin:.0f}..{vmax:.0f})")
    cax = fig.add_axes([0.945, 0.12 if not music else 0.28, 0.008, 0.3])
    fig.colorbar(mesh, cax=cax, label="dB")

    if music:
        ax = axes[3]
        tt, C = _pool_time(ts, s["C"])
        ax.pcolormesh(np.concatenate([tt - dt / 2, [tt[-1] + dt / 2]]), np.arange(13) - 0.5, C,
                      cmap="Greys", shading="flat", rasterized=True)
        ax.set_yticks(range(12))
        ax.set_yticklabels(NOTE_NAMES, fontsize=8)
        ax.set_ylabel("pitch class")
        ax.set_title("Chromagram (dark = strong)")
        ax.grid(False)

    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(off, t_end)
    for a in axes:
        a.tick_params(labelbottom=True)
    L = res["level"]
    fig.suptitle(
        f"{title}  |  {res['duration_s']} s  {sr} Hz  {ch} ch  |  peak {L['peak_dbfs']} dBFS  "
        f"RMS {L['rms_dbfs']} dBFS  {L['integrated_lufs']} LUFS",
        fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.905, top=0.94, bottom=0.05, hspace=0.35)
    fig.savefig(out)
    plt.close(fig)


def plot_spectrum(res, out: Path, title=""):
    s = res["_series"]
    wf, wdb = s["wf"], s["wdb"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [3, 1]})
    m = wf >= 20
    ax.semilogx(wf[m], wdb[m], color="#1f77b4", lw=1.2)
    labelled = []
    for p in res["spectral"]["dominant_peaks"]:
        ax.plot(p["hz"], p["db"], "o", color="red", ms=5)
        if any(abs(math.log2(p["hz"] / q)) < 0.35 for q in labelled):
            continue  # too close to an existing label; the dot is enough (values are in the JSON)
        labelled.append(p["hz"])
        ax.annotate(f"{p['hz']:g} Hz\n{p['note']}", (p["hz"], p["db"]), textcoords="offset points",
                    xytext=(4, 6), fontsize=8.5, color="darkred")
    for _, lo, _hi in BANDS[1:]:
        ax.axvline(lo, color="#999", lw=0.6, ls=":")
    ticks = [t for t in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000] if t <= wf[-1]]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_hz))
    ax.set_xlim(20, wf[-1])
    top = float(wdb[m].max())
    ax.set_ylim(top - 100, top + 8)
    ax.set_xlabel("frequency (Hz, log)")
    ax.set_ylabel("power (dB)")
    ax.set_title("Average spectrum (Welch) with labelled peaks")
    be = res["spectral"]["band_energy_percent"]
    names = list(be.keys())
    vals = [be[k] or 0 for k in names]
    bars = bx.barh(names[::-1], vals[::-1], color="#ff7f0e")
    for b_, v in zip(bars, vals[::-1]):
        bx.text(b_.get_width() + 0.5, b_.get_y() + b_.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
    bx.set_xlim(0, max(vals + [1]) * 1.25)
    bx.set_xlabel("% of energy")
    bx.set_title("Energy per band")
    labels = [f"{n} {lo}-{hi}" for n, lo, hi in BANDS][::-1]
    bx.set_yticks(range(len(labels)))
    bx.set_yticklabels(labels, fontsize=9)
    fig.suptitle(f"{title}  |  centroid {res['spectral']['centroid_hz_median']} Hz  "
                 f"rolloff85 {res['spectral']['rolloff85_hz_median']} Hz", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_detail(res, x, sr, out: Path, title=""):
    n = x.shape[0]
    off = res["file"].get("offset_s", 0.0)
    views = []
    w = int(0.03 * sr)
    if n > w:
        e = frame_rms(x.mean(axis=1), w, max(1, w // 2))
        i = int(np.argmax(e)) * (w // 2)
        views.append(("Loudest 30 ms (cycle shape)", i, i + w))
    on = res["_series"]["onsets"]
    if len(on):
        i = int((on[0] - off) * sr)
        views.append((f"First attack @ {on[0]:.3f}s (-10..+100 ms)", i - int(0.01 * sr), i + int(0.1 * sr)))
    clipped = np.nonzero(res["_series"]["clip"])[0]
    if len(clipped):
        i = int(clipped[0])
        views.append((f"First clip @ {off + i / sr:.4f}s (±5 ms)", i - int(0.005 * sr), i + int(0.005 * sr)))
    if not views:
        views.append(("Whole signal", 0, n))
    fig, axes = plt.subplots(len(views), 1, figsize=(16, 3.2 * len(views)), squeeze=False)
    for ax, (name, a, b) in zip(axes[:, 0], views):
        a, b = max(0, a), min(n, b)
        t = (off + np.arange(a, b) / sr) * 1000
        for c in range(min(x.shape[1], 2)):
            ax.plot(t, x[a:b, c], lw=0.9, marker="." if b - a < 400 else None, ms=2,
                    label=["L/ch0", "R/ch1"][c] if x.shape[1] > 1 else None)
        ax.axhline(1, color="k", ls="--", lw=0.7)
        ax.axhline(-1, color="k", ls="--", lw=0.7)
        ax.axhline(0, color="#888", lw=0.5)
        ax.set_title(name)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("amplitude")
        if x.shape[1] > 1:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{title}  |  sample-level detail", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ----------------------------------------------------------------------------- output


def public(res):
    return {k: v for k, v in res.items() if k != "_series"}


def print_summary(res, paths):
    L, D, Sp, Rh, Pi = res["level"], res["dynamics"], res["spectral"], res["rhythm"], res["pitch"]
    f = res["file"]
    lines = [
        f"FILE  {f['file']}",
        f"      {res['duration_s']} s | {f['sample_rate']} Hz | {f['channels']} ch | {f.get('source_dtype', '')}"
        + (f" | window {f['window_s']} of {f['total_duration_s']} s" if "window_s" in f else ""),
        f"LEVEL peak {L['peak_dbfs']} dBFS | true-peak {L['true_peak_dbtp']} dBTP | RMS {L['rms_dbfs']} dBFS | "
        f"crest {L['crest_factor_db']} dB | {L['integrated_lufs']} LUFS-I | LRA {L['loudness_range_lu']} LU",
        (f"NOISE floor {D['noise_floor_dbfs_p10']} dBFS | SNR~{D['snr_estimate_db']} dB | "
         if D.get("noise_floor_meaningful", True) else
         f"NOISE n/a (no pauses; quietest 10% of the programme is {D['noise_floor_dbfs_p10']} dBFS) | ")
        + f"silence {D['silence_total_s']} s in {len(D['silence_regions_s'])} regions",
        f"SPECT centroid {Sp['centroid_hz_median']} Hz | rolloff85 {Sp['rolloff85_hz_median']} Hz | "
        f"flatness {Sp['flatness_median']} | top peaks "
        + ", ".join(f"{p['hz']:g}Hz({p['note']})" for p in Sp["dominant_peaks"][:4]),
        f"RHYTHM {Rh['onset_count']} onsets | tempo~{Rh['tempo_bpm_estimate']} BPM (conf {Rh['tempo_confidence']})"
        + (f" | bar {Rh['grid']['bar_period_s']} s, first downbeat {Rh['grid']['first_downbeat_s']} s "
           f"(grid strength {Rh['grid']['grid_strength']})" if Rh.get("grid") else ""),
        f"PITCH voiced {Pi['voiced_fraction']} | f0 median {Pi['f0_median_hz']} Hz ({Pi['f0_median_note']})",
        f"KEY   {res['tonal']['key_estimate']} (r={res['tonal']['key_correlation']})",
    ]
    if "stereo" in res:
        st = res["stereo"]
        lines.append(f"STEREO corr {st['correlation']} | side/mid {st['side_to_mid_db']} dB | "
                     f"L-R {st['balance_L_minus_R_db']} dB")
    ev = res["events"]["list"]
    if ev:
        def one(e):
            note = f" {e['dominant_note']}" if e.get("dominant_note") else ""
            mv = f" {e['pitch_direction']} {e['pitch_change_st']:+.1f}st" if e.get("pitch_change_st") is not None else ""
            return f"#{e['index']} {e['start_s']}-{e['end_s']}s{note}{mv}"
        more = f" ... (+{res['events']['count'] - 8} more)" if res["events"]["count"] > 8 else ""
        lines.append(f"EVENTS {res['events']['count']}: " + " | ".join(one(e) for e in ev[:8]) + more)
    lines.append("FLAGS " + ("; ".join(res["_flags"]) if res["_flags"] else "none"))
    lines.append("FILES " + " ".join(str(p) for p in paths))
    print("\n".join(lines))


def default_out(path: str, suffix="") -> Path:
    return Path("sound-reader-out") / (Path(path).stem + suffix)


def cmd_analyze(a):
    x, sr, info = load_audio(a.file, a.start, a.end)
    if x.shape[0] == 0:
        sys.exit("Selected window is empty.")
    out = Path(a.out) if a.out else default_out(a.file, f"_{a.start or 0:g}-{a.end:g}s" if a.end else "")
    out.mkdir(parents=True, exist_ok=True)
    res = analyze(x, sr, info, music=a.music)
    title = Path(a.file).name
    paths = [out / "overview.png", out / "spectrum.png", out / "detail.png"]
    plot_overview(res, x, sr, paths[0], music=a.music, linear_freq=a.linear_freq, title=title)
    plot_spectrum(res, paths[1], title=title)
    plot_detail(res, x, sr, paths[2], title=title)
    pub = public(res)
    pub["flags"] = pub.pop("_flags")
    pub["images"] = [str(p) for p in paths]
    write_csvs(res, out)
    pub["tables"] = [str(out / "events.csv"), str(out / "frames.csv")]
    (out / "summary.json").write_text(json.dumps(pub, indent=2, ensure_ascii=False))
    print_summary(res, paths + [out / "summary.json", out / "events.csv", out / "frames.csv"])


def cmd_view(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import views

    if a.kind == "list":
        for k, v in views.KINDS.items():
            print(f"{k:10s} {v}")
        return
    loop_kinds = ("loop", "loopfind")
    # for loops, --start/--end are the loop points: keep the whole file so the audio around them is available
    x, sr, info = load_audio(a.file, *((None, None) if a.kind in loop_kinds else (a.start, a.end)))
    if x.shape[0] == 0:
        sys.exit("Selected window is empty.")
    suffix = f"_{a.start or 0:g}-{a.end:g}s" if a.end and a.kind not in loop_kinds else ""
    out = Path(a.out) if a.out else default_out(a.file, suffix)
    out.mkdir(parents=True, exist_ok=True)
    res = analyze(x, sr, info)
    views.VIEWS[a.kind](res, x, sr, out, a)


def aligned_difference(sa, sb, max_shift_s=2.0):
    """Line B up with A (envelope cross-correlation), then diff their log-frequency spectrograms."""
    ea, eb = 10 ** (sa["env"] / 20), 10 ** (sb["env"] / 20)
    step = float(np.median(np.diff(sa["tenv"]))) if len(sa["tenv"]) > 1 else 0.01
    n = min(len(ea), len(eb))
    max_lag = int(min(max_shift_s / step, n // 2))
    best, lag = -np.inf, 0
    if max_lag > 0:
        xa_, xb_ = ea - ea.mean(), eb - eb.mean()
        for L in range(-max_lag, max_lag + 1):
            if L >= 0:
                u, v = xa_[: len(xa_) - L], xb_[L:]
            else:
                u, v = xa_[-L:], xb_[: len(xb_) + L]
            m = min(len(u), len(v))
            if m < 10:
                continue
            c = float(np.dot(u[:m], v[:m]) / (np.linalg.norm(u[:m]) * np.linalg.norm(v[:m]) + EPS))
            if c > best:
                best, lag = c, L
    offset = lag * step  # B content appears `offset` seconds later than in A
    edges, ia = _logfreq_image(sa["f"], sa["S"])
    eb_, ib = _logfreq_image(sb["f"], sb["S"])
    if len(eb_) != len(edges) or not np.allclose(eb_, edges):
        ca, cb = np.sqrt(edges[:-1] * edges[1:]), np.sqrt(eb_[:-1] * eb_[1:])
        ib = np.array([np.interp(ca, cb, ib[:, j]) for j in range(ib.shape[1])]).T
    tb = sb["ts"] - offset
    cols = np.clip(np.searchsorted(tb, sa["ts"]), 0, len(tb) - 1)
    ib_on_a = ib[:, cols]
    # ~50 ms smoothing in time: sub-frame misalignment of sharp transients would otherwise light up as
    # red/blue line pairs that hide the real (sustained) differences
    hop_s = float(np.median(np.diff(sa["ts"]))) if len(sa["ts"]) > 1 else 0.01
    k = max(1, int(round(0.05 / hop_s)))
    if k > 1:
        from scipy.ndimage import uniform_filter1d

        ia = 10 * np.log10(uniform_filter1d(10 ** (ia / 10), k, axis=1) + EPS)
        ib_on_a = 10 * np.log10(uniform_filter1d(10 ** (ib_on_a / 10), k, axis=1) + EPS)
    floor = max(ia.max(), ib.max()) - 70
    diff = np.where((ia > floor) | (ib_on_a > floor), ib_on_a - ia, np.nan)
    diff[:, (sa["ts"] < tb[0]) | (sa["ts"] > tb[-1])] = np.nan  # no overlap after shifting: not a difference
    t, diff_p = _pool_time(sa["ts"], np.nan_to_num(diff, nan=0.0))
    # pooled with max() would hide negative differences: pool by mean of the signed values instead
    if diff_p.shape[1] != diff.shape[1]:
        k = int(math.ceil(diff.shape[1] / diff_p.shape[1]))
        m = diff.shape[1] // k * k
        diff_p = np.nanmean(diff[:, :m].reshape(diff.shape[0], -1, k), axis=2)
        t = sa["ts"][:m:k]
    col_err = np.nan_to_num(np.nanmean(np.abs(diff), axis=0), nan=0.0)
    # rank sustained differences: a 0.3 s moving average stops single-frame transient mismatches from winning
    w = max(1, int(0.3 / max(float(np.median(np.diff(sa["ts"]))) if len(sa["ts"]) > 1 else 0.01, 1e-3)))
    col_err = np.convolve(col_err, np.ones(w) / w, mode="same")
    hot = []
    for i in np.argsort(col_err)[::-1]:
        if all(abs(sa["ts"][i] - h) > 0.5 for h in hot):
            hot.append(float(sa["ts"][i]))
        if len(hot) == 5:
            break
    return {"t": t, "edges": edges, "diff": diff_p, "offset_s": offset, "align_corr": best,
            "mean_abs_diff_db": float(np.nanmean(np.abs(diff))), "hotspots": sorted(hot),
            "hotspot_diff_db": [float(col_err[np.argmin(np.abs(sa["ts"] - h))]) for h in sorted(hot)]}


def cmd_compare(a):
    xa, sra, ia = load_audio(a.file_a, a.start, a.end)
    xb, srb, ib = load_audio(a.file_b, a.start, a.end)
    ra, rb = analyze(xa, sra, ia), analyze(xb, srb, ib)
    out = Path(a.out) if a.out else Path("sound-reader-out") / f"compare_{Path(a.file_a).stem}_vs_{Path(a.file_b).stem}"
    out.mkdir(parents=True, exist_ok=True)
    na, nb = Path(a.file_a).name, Path(a.file_b).name
    if na == nb:
        na, nb = f"A:{a.file_a}", f"B:{a.file_b}"
    sa, sb = ra["_series"], rb["_series"]
    al = aligned_difference(sa, sb)
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), gridspec_kw={"height_ratios": [1, 1, 1, 1.5]})
    ax = axes[0]
    ax.semilogx(sa["wf"][1:], sa["wdb"][1:], label=na, lw=1.2)
    ax.semilogx(sb["wf"][1:], sb["wdb"][1:], label=nb, lw=1.2, alpha=0.8)
    ax.set_xlim(20, min(sa["wf"][-1], sb["wf"][-1]))
    top = max(sa["wdb"].max(), sb["wdb"].max())
    ax.set_ylim(top - 100, top + 5)
    ax.set_title("Average spectrum overlay")
    ax.set_ylabel("dB")
    ax.legend()
    ax = axes[1]
    grid = np.geomspace(20, min(sa["wf"][-1], sb["wf"][-1]), 600)
    sm = lambda f, d: np.convolve(np.interp(grid, f, d), np.ones(9) / 9, mode="same")  # noqa: E731
    diff = sm(sb["wf"], sb["wdb"]) - sm(sa["wf"], sa["wdb"])
    ax.semilogx(grid, diff, color="purple")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(max(-40, diff[5:-5].min() - 3), min(40, diff[5:-5].max() + 3))
    ax.set_title(f"Spectral difference: B - A (dB, smoothed)   [A={na}, B={nb}]")
    ax.set_ylabel("dB")
    for axx in axes[:2]:
        axx.xaxis.set_major_formatter(FuncFormatter(fmt_hz))
        axx.set_xlabel("frequency (Hz)")
    ax = axes[2]
    ax.plot(sa["tz"], sa["lk"], label=na)
    ax.plot(sb["tz"], sb["lk"], label=nb, alpha=0.8)
    ax.set_ylim(max(-80, min(sa["lk"].min(), sb["lk"].min()) - 3), 3)
    ax.set_title("Momentary loudness (LUFS, 400 ms)")
    ax.set_xlabel("time (s)")
    ax.legend()
    ax = axes[3]
    dt = (al["t"][1] - al["t"][0]) if len(al["t"]) > 1 else 0.01
    mesh = ax.pcolormesh(np.concatenate([al["t"] - dt / 2, [al["t"][-1] + dt / 2]]), al["edges"], al["diff"],
                         cmap="RdBu_r", vmin=-20, vmax=20, shading="flat", rasterized=True)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_hz))
    ax.set_ylim(al["edges"][0], al["edges"][-1])
    for t0 in al["hotspots"]:
        ax.axvline(t0, color="k", lw=0.8, ls="--")
    ax.set_title(f"Time-aligned spectrogram difference B - A (red = louder in B, blue = louder in A; "
                 f"B shifted by {al['offset_s']:+.3f} s to line up; dashed = biggest differences)")
    ax.set_xlabel("time in A (s)")
    ax.set_ylabel("Hz")
    fig.colorbar(mesh, ax=ax, label="dB", pad=0.01)
    fig.tight_layout()
    fig.savefig(out / "compare.png")
    plt.close(fig)

    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            if isinstance(v, dict):
                o.update(flat(v, p + k + "."))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                o[p + k] = v
        return o

    fa = flat({k: public(ra)[k] for k in ("level", "dynamics", "spectral", "rhythm", "pitch")})
    fb = flat({k: public(rb)[k] for k in ("level", "dynamics", "spectral", "rhythm", "pitch")})
    delta = {k: {"a": fa[k], "b": fb.get(k), "b_minus_a": r(fb[k] - fa[k], 3)}
             for k in fa if k in fb and fa[k] is not None and fb[k] is not None and fa[k] != fb[k]}
    comp = {"a": na, "b": nb, "flags_a": ra["_flags"], "flags_b": rb["_flags"], "metric_deltas": delta,
            "alignment": {"b_offset_s": r(al["offset_s"], 3), "envelope_corr": r(al["align_corr"], 3),
                          "mean_abs_spectrogram_diff_db": r(al["mean_abs_diff_db"], 1),
                          "biggest_difference_times_s": [r(h, 2) for h in al["hotspots"]],
                          "biggest_difference_db": [r(d, 1) for d in al["hotspot_diff_db"]]},
            "image": str(out / "compare.png")}
    (out / "compare.json").write_text(json.dumps(comp, indent=2, ensure_ascii=False))
    print(f"A = {na}\nB = {nb}")
    for k in ("level.peak_dbfs", "level.integrated_lufs", "level.rms_dbfs", "dynamics.noise_floor_dbfs_p10",
              "spectral.centroid_hz_median", "spectral.rolloff85_hz_median", "rhythm.tempo_bpm_estimate",
              "pitch.f0_median_hz"):
        if k in fa or k in fb:
            print(f"  {k:34s} A={fa.get(k)!s:>10}  B={fb.get(k)!s:>10}")
    print(f"ALIGN B offset {al['offset_s']:+.3f} s (corr {al['align_corr']:.2f}) | mean |B-A| "
          f"{al['mean_abs_diff_db']:.1f} dB | biggest differences at " + ", ".join(f"{h:.2f}s" for h in al["hotspots"]))
    print("FLAGS A: " + ("; ".join(ra["_flags"]) or "none"))
    print("FLAGS B: " + ("; ".join(rb["_flags"]) or "none"))
    print(f"IMAGES {out / 'compare.png'} {out / 'compare.json'}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Render audio as images + metrics for LLM analysis.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("analyze", help="analyse one file")
    pa.add_argument("file")
    pa.add_argument("--out", help="output directory")
    pa.add_argument("--start", type=float, help="start time (s) to zoom into")
    pa.add_argument("--end", type=float, help="end time (s) to zoom into")
    pa.add_argument("--music", action="store_true", help="add chromagram panel")
    pa.add_argument("--linear-freq", action="store_true", help="linear frequency axis on the spectrogram")
    pa.set_defaults(func=cmd_analyze)
    pv = sub.add_parser("view", help="extra on-demand view (run 'view - --kind list' to see all)")
    pv.add_argument("file")
    pv.add_argument("--kind", required=True,
                    choices=["list", "events", "envelope", "pianoroll", "hpss", "structure", "stereo", "harmonics",
                             "reverb", "loop", "loopfind"])
    pv.add_argument("--event", type=int, help="event index (from events.csv) for envelope/harmonics/reverb")
    pv.add_argument("--out")
    pv.add_argument("--start", type=float)
    pv.add_argument("--end", type=float)
    pv.set_defaults(func=cmd_view)
    pc = sub.add_parser("compare", help="compare two files (A vs B)")
    pc.add_argument("file_a")
    pc.add_argument("file_b")
    pc.add_argument("--out")
    pc.add_argument("--start", type=float)
    pc.add_argument("--end", type=float)
    pc.set_defaults(func=cmd_compare)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
