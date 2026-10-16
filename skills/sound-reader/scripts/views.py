"""On-demand views for sound-reader. Run after `analyze` when the interpretation needs a closer look.

    sound_reader.py view FILE --kind KIND [--event N] [--start S --end S] [--out DIR]

Each view writes view_<kind>.png (+ view_<kind>.json with the numbers) and prints a short summary.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy import signal as sps

import sound_reader as core
from sound_reader import EPS, NOTE_NAMES, db, pdb, r, hz_to_note, fmt_hz, plt, FuncFormatter

KINDS = {
    "events": "contact sheet: every detected sound side by side (waveform + spectrogram, same scales)",
    "envelope": "ADSR-style envelope per event with attack / decay / end markers",
    "pianoroll": "semitone-resolution spectrogram with a note grid (which notes, when)",
    "hpss": "harmonic (tonal) vs percussive (transient) split",
    "structure": "self-similarity matrix + section boundaries and repeat labels (A/B/A...)",
    "stereo": "vectorscope, L/R correlation over time, per-band panning",
    "harmonics": "harmonic series of one sound: levels, inharmonicity, odd/even balance",
    "reverb": "decay curves and RT60-style decay times (broadband + 3 bands)",
    "loop": "loop seam check: end -> start continuity (click, level, timbre jumps); --start/--end = loop region",
    "loopfind": "suggest bar-aligned loop points for music (needs a beat grid)",
}


def _save(fig, out: Path, kind: str, data: dict, lines: list[str]):
    png = out / f"view_{kind}.png"
    fig.savefig(png)
    plt.close(fig)
    js = out / f"view_{kind}.json"
    js.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_jsonable))
    print("\n".join(lines + [f"FILES {png} {js}"]))


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return r(float(o), 3)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _pick_event(res, idx):
    evs = res["_series"]["events"]
    if not evs:
        return None
    if idx:
        for e in evs:
            if e.get("index") == idx:
                return e
        raise SystemExit(f"event #{idx} not found (file has {len(evs)} events)")
    return max(evs, key=lambda e: e.get("peak_dbfs", -999))


def _seg(x, sr, off, a, b):
    i0, i1 = max(0, int((a - off) * sr)), min(len(x), int((b - off) * sr))
    return x[i0:i1]


def _stft_db(y, sr, n_fft=None, hop=None):
    n_fft = n_fft or (2048 if sr > 24000 else 1024)
    hop = hop or n_fft // 4
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    f, t, Z = sps.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros", padded=True)
    return f, t, np.abs(Z)


def _pool_mean(t, img, max_cols=900):
    """Average (not max) pooling along time - for signed quantities like pan."""
    if img.shape[1] <= max_cols:
        return t, img
    k = int(math.ceil(img.shape[1] / max_cols))
    m = img.shape[1] // k * k
    return t[:m:k], img[:, :m].reshape(img.shape[0], -1, k).mean(axis=2)


def _log_axis(ax, lo, hi):
    ax.set_yscale("log")
    ticks = [v for v in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000] if lo <= v <= hi]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_hz))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


# ----------------------------------------------------------------------------- events / envelope


def view_events(res, x, sr, out, a):
    evs = res["_series"]["events"]
    mono = res["_series"]["mono"]
    off = res["file"].get("offset_s", 0.0)
    if not evs:
        raise SystemExit("no events detected (continuous or silent audio) - use pianoroll/structure instead")
    evs = evs[:24]
    ncol = min(4, len(evs))
    nrow = math.ceil(len(evs) / ncol)
    fig = plt.figure(figsize=(4.2 * ncol, 4.4 * nrow))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.28)
    vmax = None
    cells = []
    for e in evs:
        a0, b0 = e["start_s"] - 0.02, min(e["end_s"] + 0.02, e["start_s"] + 2.0)
        y = _seg(mono, sr, off, a0, b0)
        f, t, S = _stft_db(y, sr, n_fft=1024 if sr > 24000 else 512, hop=128)
        edges, img = core._logfreq_image(f, S, fmin=50, n_bins=160)
        vmax = max(vmax or -999, float(np.percentile(img, 99.5)))
        cells.append((e, a0, y, t, edges, img))
    for k, (e, a0, y, t, edges, img) in enumerate(cells):
        sub = gs[k // ncol, k % ncol].subgridspec(2, 1, height_ratios=[1, 1.6], hspace=0.05)
        ax1 = fig.add_subplot(sub[0])
        ax2 = fig.add_subplot(sub[1], sharex=ax1)
        tt = a0 + np.arange(len(y)) / sr
        ax1.plot(tt, y, lw=0.5, color="#1f77b4")
        ax1.set_ylim(-1, 1)
        ax1.tick_params(labelbottom=False, labelsize=7)
        mv = ""
        if e.get("pitch_contour"):
            mv = f" {e['pitch_contour']}"
        note = hz_to_note(e["dominant_hz"]).split("+")[0].split("-")[0] if e.get("dominant_hz") else ""
        ax1.set_title(f"#{e['index']} @{e['start_s']:.2f}s  {e['duration_s'] * 1000:.0f} ms\n"
                      f"peak {e.get('peak_dbfs', 0):.1f} dB  {note}{mv}", fontsize=8.5)
        dt = (t[1] - t[0]) if len(t) > 1 else 0.01
        te = np.concatenate([a0 + t - dt / 2, [a0 + t[-1] + dt / 2]])
        ax2.pcolormesh(te, edges, img, cmap="magma", vmin=vmax - 80, vmax=vmax, shading="flat", rasterized=True)
        _log_axis(ax2, edges[0], edges[-1])
        ax2.set_ylim(edges[0], edges[-1])
        ax2.tick_params(labelsize=7)
        ax2.grid(alpha=0.15)
        ax2.set_xlim(a0, a0 + len(y) / sr)
        if e.get("held_notes"):
            ax2.text(0.02, 0.96, e["held_notes"], transform=ax2.transAxes, color="cyan", fontsize=8, va="top",
                     bbox=dict(facecolor="black", alpha=0.6, lw=0, pad=1.5))
    fig.suptitle(f"{Path(a.file).name}  |  event contact sheet ({len(evs)} of {len(res['_series']['events'])}; "
                 "same amplitude and colour scale everywhere)", fontsize=12, fontweight="bold")
    lines = [f"EVENTS {len(res['_series']['events'])} (sheet shows {len(evs)}); details in events.csv / summary.json"]
    _save(fig, out, "events", {"events": [core.compact_event(e) for e in evs]}, lines)


def view_envelope(res, x, sr, out, a):
    s = res["_series"]
    evs = [_pick_event(res, a.event)] if a.event else s["events"][:12]
    if not evs or evs[0] is None:
        raise SystemExit("no events detected")
    ncol = 1 if len(evs) == 1 else min(3, len(evs))
    nrow = math.ceil(len(evs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(16 if ncol == 1 else 5.6 * ncol, 4.2 * nrow), squeeze=False)
    out_rows = []
    for ax, e in zip(axes.flat, evs):
        m = (s["t5"] >= e["start_s"] - 0.02) & (s["t5"] <= e["end_s"] + 0.05)
        t, en = s["t5"][m], s["env5"][m]
        pk = e["peak_dbfs"]
        ax.plot((t - e["start_s"]) * 1000, en, color="#333", lw=1.2)
        tp = (e["peak_time_s"] - e["start_s"]) * 1000
        ax.axvline(tp, color="red", lw=1)
        ax.axhline(pk - 20, color="#999", ls=":", lw=1)
        atk = e.get("attack_ms") or 0
        ax.axvspan(tp - atk, tp, color="#2ca02c", alpha=0.18, label=f"attack {atk:.0f} ms")
        dec = e.get("decay_to_-20db_ms")
        if dec is not None:
            ax.axvspan(tp, tp + dec, color="#ff7f0e", alpha=0.15, label=f"decay to -20 dB {dec:.0f} ms")
        else:
            ax.text(0.98, 0.9, "never decays 20 dB\n(sustained or cut)", transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, color="darkred")
        endl = e.get("end_level_db_rel_peak", 0)
        ax.plot([(e["end_s"] - e["start_s"]) * 1000], [pk + endl], "s", color="purple",
                label=f"end {endl:.0f} dB re peak" + (" (file end!)" if e.get("ends_at_file_end") else ""))
        ax.set_ylim(max(pk - 70, en.min() - 3), pk + 4)
        ax.set_xlabel("ms from event start")
        ax.set_ylabel("dBFS (5 ms RMS)")
        ax.set_title(f"#{e['index']} @{e['start_s']:.3f}s  ({e['duration_s'] * 1000:.0f} ms)", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        out_rows.append({k: core.compact_event(e).get(k) for k in
                         ("index", "start_s", "duration_s", "peak_dbfs", "attack_ms", "decay_to_-20db_ms",
                          "body_level_db_rel_peak", "end_level_db_rel_peak", "ends_at_file_end")})
    for ax in list(axes.flat)[len(evs):]:
        ax.axis("off")
    fig.suptitle(f"{Path(a.file).name}  |  envelopes (green = attack, orange = decay to -20 dB, dotted = peak-20 dB)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    lines = [f"ENVELOPE #{o['index']}: attack {o['attack_ms']} ms, decay(-20dB) {o['decay_to_-20db_ms']} ms, "
             f"end {o['end_level_db_rel_peak']} dB re peak" + (" [at file end]" if o["ends_at_file_end"] else "")
             for o in out_rows]
    _save(fig, out, "envelope", {"events": out_rows}, lines)


# ----------------------------------------------------------------------------- pitch


def view_pianoroll(res, x, sr, out, a):
    s = res["_series"]
    mono, off = s["mono"], res["file"].get("offset_s", 0.0)
    n_fft = 16384 if sr > 64000 else 8192 if sr > 24000 else 4096
    hop = max(256, int(len(mono) / 3000))
    f, t, S = _stft_db(mono, sr, n_fft=n_fft, hop=hop)
    t = t + off
    lo_m, hi_m = 24, min(108, int(69 + 12 * math.log2((sr / 2) / 440)))  # C1 .. C8
    midis = np.arange(lo_m, hi_m + 1)
    P = S.astype(np.float64) ** 2
    roll = np.full((len(midis), P.shape[1]), EPS)
    for i, m in enumerate(midis):
        f_lo = 440 * 2 ** ((m - 0.5 - 69) / 12)
        f_hi = 440 * 2 ** ((m + 0.5 - 69) / 12)
        sel = (f >= f_lo) & (f < f_hi)
        if sel.any():
            roll[i] = P[sel].max(axis=0)
        else:
            roll[i] = P[int(np.argmin(np.abs(f - 440 * 2 ** ((m - 69) / 12))))]
    R = pdb(roll)
    tt, R = core._pool_time(t, R)
    vmax = float(np.percentile(R, 99.7))
    fig, ax = plt.subplots(figsize=(16, 9))
    dt = (tt[1] - tt[0]) if len(tt) > 1 else 0.01
    ax.pcolormesh(np.concatenate([tt - dt / 2, [tt[-1] + dt / 2]]), np.append(midis, midis[-1] + 1) - 0.5, R,
                  cmap="magma", vmin=vmax - 60, vmax=vmax, shading="flat", rasterized=True)
    for m in midis:
        if NOTE_NAMES[m % 12].endswith("#"):
            ax.axhspan(m - 0.5, m + 0.5, color="white", alpha=0.04, lw=0)
        if m % 12 == 0:
            ax.axhline(m - 0.5, color="white", lw=0.6, alpha=0.5)
    ax.set_yticks([m for m in midis if m % 12 in (0, 4, 7, 9)])
    ax.set_yticklabels([f"{NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in midis if m % 12 in (0, 4, 7, 9)], fontsize=8)
    vf = np.isfinite(s["f0"])
    if vf.any() and res["pitch"]["reliable"]:
        ax.plot(s["tf0"], 69 + 12 * np.log2(s["f0"] / 440), color="cyan", lw=1.4, label="f0")
        ax.legend(loc="upper right")
    ax.set_ylim(lo_m - 0.5, hi_m + 0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("note (semitone rows; lines at every C)")
    ax.grid(False)
    tot = roll.sum(axis=1)
    order = np.argsort(tot)[::-1][:10]
    top = [{"note": f"{NOTE_NAMES[midis[i] % 12]}{midis[i] // 12 - 1}", "share_pct": r(100 * tot[i] / tot.sum(), 1)}
           for i in order]
    ax.set_title(f"{Path(a.file).name}  |  piano roll (strongest: " + ", ".join(d["note"] for d in top[:6]) + ")",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "pianoroll", {"top_notes_by_energy": top, "freq_resolution_hz": r(sr / n_fft, 2)},
          ["PIANOROLL strongest notes: " + ", ".join(f"{d['note']} {d['share_pct']}%" for d in top)])


def view_harmonics(res, x, sr, out, a):
    s = res["_series"]
    off = res["file"].get("offset_s", 0.0)
    e = _pick_event(res, a.event)
    if e is not None and e["duration_s"] <= 10:
        a0, b0 = e["start_s"], e["end_s"]
    else:
        a0, b0 = off, off + res["duration_s"]
    # analyse a short stretch right after the peak: long windows smear gliding pitches into fake partials
    pk_t = e["peak_time_s"] if e is not None else a0
    w0, w1 = max(a0, pk_t + 0.01), min(b0, pk_t + 0.13)
    y = _seg(s["mono"], sr, off, w0, w1)
    if len(y) < 1024:
        y = _seg(s["mono"], sr, off, a0, b0)
    nfft = 1 << int(math.ceil(math.log2(max(len(y), 8192) * 2)))
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=nfft))
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    sdb = db(spec / (spec.max() + EPS))
    idx, _ = sps.find_peaks(sdb, prominence=10, height=-70, distance=max(1, int(20 / (fr[1] - fr[0]))))
    peaks = [(float(fr[i]), float(sdb[i])) for i in idx if fr[i] > 30]
    f0, tol = None, 0.03
    if (e or {}).get("pitch_source") == "f0":
        mw = (s["tf0"] >= w0) & (s["tf0"] <= w1) & np.isfinite(s["f0"])
        if mw.any():
            fw = s["f0"][mw]
            f0 = float(np.median(fw))  # pitch inside this window, not the event median
            # a glide inside the window smears harmonic k by k * (pitch spread): widen the match tolerance
            tol = min(0.06, max(0.03, float(np.percentile(fw, 90) / np.percentile(fw, 10)) - 1 + 0.01))
        else:
            f0 = e.get("f0_median_hz")
    if not f0 and peaks:
        strong = [p for p in peaks if p[1] > -40]
        f0 = min(strong or peaks)[0]
    if not f0:
        raise SystemExit("no tonal content found for a harmonic analysis")
    # refine f0 from the nearest peak; if no real peak sits there, the tracker was fooled (inharmonic
    # partials) - fall back to the lowest strong partial
    near = [p for p in peaks if abs(p[0] - f0) < 0.04 * f0]
    if near:
        f0 = max(near, key=lambda p: p[1])[0]
    elif peaks:
        strong = [p for p in peaks if p[1] > -40]
        f0, tol = min(strong or peaks)[0], 0.03
        near = [p for p in peaks if abs(p[0] - f0) < 0.04 * f0]
    f0_db = max([p[1] for p in near], default=float(sdb[int(round(f0 / (fr[1] - fr[0])))]))
    # re-pick peaks no closer than ~half a harmonic spacing: the short window splits peaks into sidelobes
    idx, _ = sps.find_peaks(sdb, prominence=10, height=-70, distance=max(1, int(0.45 * f0 / (fr[1] - fr[0]))))
    peaks = [(float(fr[i]), float(sdb[i])) for i in idx if fr[i] > 30]
    harmonics = []
    for k in range(1, 41):
        target = k * f0
        if target > min(sr / 2, 20000):
            break
        cand = [(abs(p[0] - target), j) for j, p in enumerate(peaks) if abs(p[0] - target) < tol * target]
        if cand:
            _, j = min(cand)
            hz, lv = peaks[j]
            harmonics.append({"k": k, "hz": r(hz, 1), "db_re_f0": r(lv - f0_db, 1),
                              "cents_off": r(1200 * math.log2(hz / target), 0)})
        else:
            harmonics.append({"k": k, "hz": None, "db_re_f0": None, "cents_off": None})

    def on_series(hz):
        k = max(1, round(hz / f0))
        return abs(hz - k * f0) < tol * k * f0

    other = [{"hz": r(p[0], 1), "ratio_to_f0": r(p[0] / f0, 3), "db_re_f0": r(p[1] - f0_db, 1)}
             for p in peaks if p[1] > -50 and p[0] > 1.05 * f0 and not on_series(p[0])][:12]
    found = [h for h in harmonics if h["hz"]]
    odd = sum(10 ** (h["db_re_f0"] / 10) for h in found if h["k"] % 2 == 1 and h["k"] > 1)
    even = sum(10 ** (h["db_re_f0"] / 10) for h in found if h["k"] % 2 == 0)
    tilt = None
    if len(found) >= 3:
        kk = np.log2([h["k"] for h in found])
        tilt = float(np.polyfit(kk, [h["db_re_f0"] for h in found], 1)[0])
    upper = [h for h in found if h["k"] > 1 and h["db_re_f0"] > -40]
    strong_other = [o for o in other if o["db_re_f0"] > -35]
    if strong_other and len(strong_other) >= len(upper):
        verdict = "inharmonic (bell / metal bar / membrane-like partials)"
    elif len(upper) >= 2:
        verdict = "harmonic"
    else:
        verdict = "near-sine (almost only the fundamental)"
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [1.4, 1]})
    m = (fr > 20) & (fr < min(sr / 2, 20000))
    ax.semilogx(fr[m], sdb[m], color="#1f77b4", lw=0.8)
    for h in harmonics[:30]:
        ax.axvline(h["k"] * f0, color="#2ca02c", lw=0.6, ls=":", alpha=0.7)
    for o in other:
        ax.plot(o["hz"], o["db_re_f0"] + f0_db, "v", color="red", ms=6)
        ax.annotate(f"x{o['ratio_to_f0']:.2f}", (o["hz"], o["db_re_f0"] + f0_db), xytext=(2, 4),
                    textcoords="offset points", fontsize=7.5, color="darkred")
    ax.set_ylim(-90, 5)
    ax.set_xlim(max(20, f0 / 2), min(sr / 2, 20000))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_hz))
    ax.set_ylabel("dB re strongest")
    ax.set_title(f"spectrum at {w0:.2f}-{w1:.2f}s  |  f0 = {f0:.1f} Hz ({hz_to_note(f0)})  "
                 "green = expected harmonics k*f0, red = partials off the series (ratio to f0)")
    ks = [h["k"] for h in harmonics[:24]]
    lv = [h["db_re_f0"] if h["db_re_f0"] is not None else -80 for h in harmonics[:24]]
    cols = ["#888" if h["hz"] is None else "#d62728" if abs(h["cents_off"] or 0) > 15 else
            "#1f77b4" if h["k"] % 2 else "#ff7f0e" for h in harmonics[:24]]
    bx.bar(ks, np.array(lv) + 80, bottom=-80, color=cols)
    bx.set_xticks(ks)
    bx.set_xlabel("harmonic number k (blue = odd, orange = even, red = >15 cents off, grey = missing)")
    bx.set_ylabel("dB re f0")
    bx.set_ylim(-80, max(5, max(lv) + 5))
    fig.suptitle(f"{Path(a.file).name}  |  harmonics: {verdict}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    data = {"window_s": [r(w0, 3), r(w1, 3)], "match_tolerance_pct": r(100 * tol, 1), "f0_hz": r(f0, 2), "f0_note": hz_to_note(f0), "verdict": verdict,
            "odd_to_even_db": r(10 * math.log10((odd + EPS) / (even + EPS)), 1) if even > 0 else None,
            "spectral_tilt_db_per_octave": r(tilt, 1), "harmonics": harmonics[:24], "off_series_partials": other}
    lines = [f"HARMONICS f0 {f0:.1f} Hz ({hz_to_note(f0)}) | {verdict} | odd/even {data['odd_to_even_db']} dB | "
             f"tilt {data['spectral_tilt_db_per_octave']} dB/oct",
             "  off-series partials (ratio to f0): " + (", ".join(f"x{o['ratio_to_f0']}" for o in other[:8]) or "none")]
    _save(fig, out, "harmonics", data, lines)


# ----------------------------------------------------------------------------- texture / structure


def view_hpss(res, x, sr, out, a):
    s = res["_series"]
    f, ts, S = s["f"], s["ts"], s["S"].astype(np.float64)
    k_t, k_f = 17, 17
    H = ndimage.median_filter(S, size=(1, k_t))
    Pm = ndimage.median_filter(S, size=(k_f, 1))
    mh = H ** 2 / (H ** 2 + Pm ** 2 + EPS)
    Sh, Sp = S * mh, S * (1 - mh)
    eh, ep = (Sh ** 2).sum(axis=0), (Sp ** 2).sum(axis=0)
    h_pct = float(100 * eh.sum() / (eh.sum() + ep.sum() + EPS))
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
    vmax = float(np.percentile(pdb(S ** 2), 99.8))
    for ax, M, name in ((axes[0], Sh, "harmonic / tonal part (horizontal lines)"),
                        (axes[1], Sp, "percussive / transient part (vertical lines, noise bursts)")):
        edges, img = core._logfreq_image(f, M)
        tt, img = core._pool_time(ts, img)
        dt = (tt[1] - tt[0]) if len(tt) > 1 else 0.01
        ax.pcolormesh(np.concatenate([tt - dt / 2, [tt[-1] + dt / 2]]), edges, img, cmap="magma",
                      vmin=vmax - 90, vmax=vmax, shading="flat", rasterized=True)
        _log_axis(ax, edges[0], edges[-1])
        ax.set_ylim(edges[0], edges[-1])
        ax.set_ylabel("Hz")
        ax.set_title(name)
        ax.grid(alpha=0.15)
    ax = axes[2]
    ax.plot(ts, pdb(eh), label="harmonic energy", color="#1f77b4")
    ax.plot(ts, pdb(ep), label="percussive energy", color="#d62728", alpha=0.8)
    top = max(pdb(eh).max(), pdb(ep).max())
    ax.set_ylim(top - 70, top + 5)
    ax.set_ylabel("dB")
    ax.set_xlabel("time (s)")
    ax.legend(loc="lower right")
    fig.suptitle(f"{Path(a.file).name}  |  HPSS: {h_pct:.0f}% harmonic / {100 - h_pct:.0f}% percussive energy",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    # transient rate from the percussive curve
    pe = pdb(ep)
    pk, _ = sps.find_peaks(pe, prominence=10, distance=max(1, int(0.05 * len(ts) / max(res["duration_s"], EPS))))
    data = {"harmonic_energy_pct": r(h_pct, 1), "percussive_energy_pct": r(100 - h_pct, 1),
            "percussive_hits": int(len(pk)), "percussive_hit_times_s": [r(ts[i], 3) for i in pk[:200]]}
    _save(fig, out, "hpss", data, [f"HPSS harmonic {h_pct:.1f}% | percussive {100 - h_pct:.1f}% | "
                                   f"{len(pk)} percussive hits"])


def _mel_bands(f, n=24, fmin=40.0, fmax=None):
    fmax = fmax or f[-1]
    mel = lambda h: 2595 * np.log10(1 + h / 700)  # noqa: E731
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = imel(np.linspace(mel(fmin), mel(fmax), n + 2))
    W = np.zeros((n, len(f)))
    for i in range(n):
        l, c, h = pts[i], pts[i + 1], pts[i + 2]
        W[i] = np.clip(np.minimum((f - l) / (c - l + EPS), (h - f) / (h - c + EPS)), 0, None)
    return W


def view_structure(res, x, sr, out, a):
    s = res["_series"]
    f, ts, S, C = s["f"], s["ts"], s["S"].astype(np.float64), s["C"]
    dur = res["duration_s"]
    if dur < 8:
        raise SystemExit("structure view needs a longer file (>= 8 s)")
    step = max(0.25, dur / 600)
    nb = int(math.ceil(dur / step))
    t0 = ts[0] if len(ts) else 0
    bins = np.clip(((ts - t0) / step).astype(int), 0, nb - 1)
    mel = np.log1p(100 * (_mel_bands(f) @ (S ** 2)) / ((S ** 2).sum(axis=0).max() + EPS))
    feats = np.vstack([C, mel, pdb((S ** 2).sum(axis=0))[None, :] / 20])
    F = np.zeros((feats.shape[0], nb))
    for b in range(nb):
        m = bins == b
        if m.any():
            F[:, b] = feats[:, m].mean(axis=1)
    F = (F - F.mean(axis=1, keepdims=True)) / (F.std(axis=1, keepdims=True) + EPS)
    Fn = F / (np.linalg.norm(F, axis=0, keepdims=True) + EPS)
    SSM = Fn.T @ Fn
    # checkerboard novelty along the diagonal
    L = max(4, int(round(6.0 / step)))
    g = np.outer(np.r_[np.ones(L), -np.ones(L)], np.r_[np.ones(L), -np.ones(L)])
    win = np.outer(sps.windows.gaussian(2 * L, L / 2), sps.windows.gaussian(2 * L, L / 2))
    K = g * win
    pad = np.pad(SSM, L, mode="edge")
    nov = np.array([np.sum(pad[i:i + 2 * L, i:i + 2 * L] * K) for i in range(nb)])
    nov = np.maximum(nov, 0)
    nov /= nov.max() + EPS
    pk, _ = sps.find_peaks(nov, height=0.15, distance=max(1, int(8.0 / step)), prominence=0.1)
    grid = res["rhythm"].get("grid")
    use_grid = bool(grid) and (res["rhythm"].get("tempo_confidence") or 0) >= 0.5
    if use_grid:
        # snap boundaries to the nearest downbeat: sections change on bar lines
        bar, d0 = grid["bar_period_s"], grid["first_downbeat_s"]
        snapped = []
        for i in pk:
            tb_ = t0 + i * step
            tsnap = d0 + round((tb_ - d0) / bar) * bar
            j = int(round((tsnap - t0) / step))
            if 0 < j < nb and (not snapped or j - snapped[-1] > 1):
                snapped.append(j)
        pk = np.array(snapped, dtype=int)
    bounds = [0] + list(pk) + [nb]
    secs = []
    for i in range(len(bounds) - 1):
        b0, b1 = bounds[i], bounds[i + 1]
        if b1 - b0 < 1:
            continue
        secs.append({"start_s": r(t0 + b0 * step, 2), "end_s": r(t0 + b1 * step, 2), "vec": Fn[:, b0:b1].mean(axis=1)})
    labels, protos = [], []
    for sec in secs:
        v = sec["vec"] / (np.linalg.norm(sec["vec"]) + EPS)
        sims = [float(v @ p) for p in protos]
        if sims and max(sims) > 0.8:
            lab = chr(65 + int(np.argmax(sims)))
        else:
            protos.append(v)
            lab = chr(65 + len(protos) - 1) if len(protos) <= 26 else "?"
        labels.append(lab)
    # merge neighbours that got the same label
    merged_s, merged_l = [], []
    for sec, lab in zip(secs, labels):
        if merged_l and merged_l[-1] == lab:
            merged_s[-1]["end_s"] = sec["end_s"]
        else:
            merged_s.append(sec)
            merged_l.append(lab)
    secs, labels = merged_s, merged_l
    lk_t, lk = s["tz"], s["lk"]
    # repeat periods: how far apart the same material comes back (mean similarity along each diagonal)
    loud_b = np.array([np.mean(lk[(lk_t >= t0 + b_ * step) & (lk_t < t0 + (b_ + 1) * step)]) if np.any(
        (lk_t >= t0 + b_ * step) & (lk_t < t0 + (b_ + 1) * step)) else -99 for b_ in range(nb)])
    valid = loud_b > np.max(loud_b) - 30
    lags = np.arange(max(2, int(4.0 / step)), nb // 2)
    lag_sim = np.array([np.mean(np.diag(SSM, L)[valid[:-L] & valid[L:]]) if np.any(valid[:-L] & valid[L:]) else 0
                        for L in lags])
    repeats = []
    if len(lag_sim) > 3:
        lp, _ = sps.find_peaks(lag_sim, prominence=0.02, distance=max(1, int(4.0 / step)))
        # short patterns (riffs, 1-4 bars) always repeat; phrase-level repeats (verses, choruses) are what
        # loop points and form are built on, so report the best of each scale
        split = 8 * grid["bar_period_s"] if use_grid else 20.0
        for scale, sel in (("short", [i for i in lp if lags[i] * step < split]),
                           ("phrase", [i for i in lp if lags[i] * step >= split])):
            for i in sorted(sel, key=lambda i: -lag_sim[i])[:2]:
                lag_s = float(lags[i] * step)
                item = {"scale": scale, "lag_s": r(lag_s, 2), "similarity": r(float(lag_sim[i]), 3)}
                if use_grid:
                    item["lag_bars"] = r(lag_s / grid["bar_period_s"], 2)
                repeats.append(item)
    for sec, lab in zip(secs, labels):
        sec["label"] = lab
        m = (lk_t >= sec["start_s"]) & (lk_t < sec["end_s"])
        sec["mean_loudness_lufs_m"] = r(float(np.mean(lk[m])) if m.any() else None, 1)
        del sec["vec"]
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 0.8, 1])
    ax = fig.add_subplot(gs[0])
    ext = [t0, t0 + nb * step, t0 + nb * step, t0]
    ax.imshow(SSM, cmap="viridis", extent=ext, vmin=-0.2, vmax=1, aspect="equal", interpolation="nearest")
    for sec in secs[1:]:
        ax.axvline(sec["start_s"], color="white", lw=0.8)
        ax.axhline(sec["start_s"], color="white", lw=0.8)
    ax.set_title("self-similarity (bright blocks = similar material; off-diagonal bright = a repeat)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("time (s)")
    bx = fig.add_subplot(gs[1])
    tb = t0 + (np.arange(nb) + 0.5) * step
    bx.plot(tb, nov, color="#d62728")
    for sec in secs[1:]:
        bx.axvline(sec["start_s"], color="k", lw=0.8, ls="--")
    bx.set_xlim(ext[0], ext[1])
    bx.set_ylabel("novelty")
    cx = fig.add_subplot(gs[2], sharex=bx)
    cx.plot(lk_t, lk, color="#ff7f0e", lw=1)
    palette = plt.get_cmap("tab10")
    for sec in secs:
        col = palette((ord(sec["label"]) - 65) % 10) if sec["label"] != "?" else "#999"
        cx.axvspan(sec["start_s"], sec["end_s"], color=col, alpha=0.2)
        cx.text((sec["start_s"] + sec["end_s"]) / 2, 0.92, sec["label"], transform=cx.get_xaxis_transform(),
                ha="center", va="top", fontsize=12, fontweight="bold")
    cx.set_ylim(max(-60, np.percentile(lk, 2) - 5), max(lk.max() + 3, -5))
    cx.set_ylabel("LUFS (400 ms)")
    cx.set_xlabel("time (s)")
    fig.suptitle(f"{Path(a.file).name}  |  structure: " + " ".join(labels), fontsize=12, fontweight="bold")
    fig.tight_layout()
    lines = ["STRUCTURE " + " | ".join(f"{s_['label']} {s_['start_s']}-{s_['end_s']}s ({s_['mean_loudness_lufs_m']} LUFS)"
                                       for s_ in secs),
             "  repeats every: " + (", ".join(f"[{x_['scale']}] {x_['lag_s']} s" + (f" (= {x_['lag_bars']} bars)" if "lag_bars" in x_ else "")
                                             + f" sim {x_['similarity']}" for x_ in repeats) or "none found")
             + ("  | boundaries snapped to bar lines" if use_grid else "")]
    _save(fig, out, "structure", {"resolution_s": r(step, 3), "snapped_to_bars": use_grid, "grid": grid,
                                  "repeat_lags": repeats, "sections": secs}, lines)


# ----------------------------------------------------------------------------- space


def view_stereo(res, x, sr, out, a):
    if x.shape[1] < 2:
        raise SystemExit("stereo view needs a file with 2+ channels")
    off = res["file"].get("offset_s", 0.0)
    L, R = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    M, Sd = (L + R) / np.sqrt(2), (L - R) / np.sqrt(2)
    win = max(256, int(max(0.1, res["duration_s"] / 600) * sr))  # <= ~600 points so the curve stays readable
    nwin = max(1, len(L) // win)
    corr = np.zeros(nwin)
    for i in range(nwin):
        l, rr = L[i * win:(i + 1) * win], R[i * win:(i + 1) * win]
        corr[i] = np.sum(l * rr) / (np.sqrt(np.sum(l ** 2) * np.sum(rr ** 2)) + EPS)
    tc = off + (np.arange(nwin) + 0.5) * win / sr
    loud = np.array([np.sqrt(np.mean(M[i * win:(i + 1) * win] ** 2)) for i in range(nwin)])
    act = db(loud) > db(loud.max()) - 40
    edges_b = [20, 125, 250, 500, 1000, 2000, 4000, 8000, min(20000, sr / 2)]
    f, t, ZL = sps.stft(L, fs=sr, nperseg=2048, noverlap=1536)
    _, _, ZR = sps.stft(R, fs=sr, nperseg=2048, noverlap=1536)
    pl, pr = np.abs(ZL) ** 2, np.abs(ZR) ** 2
    pan = np.zeros((len(edges_b) - 1, pl.shape[1]))
    band_rows = []
    for i in range(len(edges_b) - 1):
        m = (f >= edges_b[i]) & (f < edges_b[i + 1])
        el, er = pl[m].sum(axis=0), pr[m].sum(axis=0)
        pan[i] = (er - el) / (er + el + EPS)
        cross = np.real(ZL[m] * np.conj(ZR[m])).sum()
        bc = cross / (np.sqrt(el.sum() * er.sum()) + EPS)
        band_rows.append({"band_hz": f"{edges_b[i]}-{int(edges_b[i + 1])}", "pan_mean": r(float(np.average(
            pan[i], weights=el + er + EPS)), 2), "correlation": r(float(bc), 2)})
    t = t + off
    tt, panp = _pool_mean(t, pan)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.2], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    sel = np.random.default_rng(0).choice(len(M), size=min(len(M), 200000), replace=False)
    lim = max(np.abs(M[sel]).max(), np.abs(Sd[sel]).max(), 1e-3)
    ax.hist2d(Sd[sel], M[sel], bins=200, range=[[-lim, lim], [-lim, lim]], cmap="magma", cmin=1, norm="log")
    ax.axhline(0, color="#888", lw=0.6)
    ax.axvline(0, color="#888", lw=0.6)
    ax.set_xlabel("side (L-R)")
    ax.set_ylabel("mid (L+R)")
    ax.set_aspect("equal")
    ax.set_title("vectorscope: vertical line = mono,\nwide blob = wide stereo, horizontal = out of phase")
    bx = fig.add_subplot(gs[0, 1])
    bx.plot(tc, corr, color="#1f77b4", lw=1)
    bx.fill_between(tc, corr, 0, where=corr < 0, color="red", alpha=0.3)
    bx.axhline(0, color="k", lw=0.8)
    bx.set_ylim(-1.05, 1.05)
    bx.set_ylabel(f"L/R correlation ({win / sr * 1000:.0f} ms)")
    bx.set_title("correlation over time (red = out of phase: cancels in mono)")
    cx = fig.add_subplot(gs[1, 1], sharex=bx)
    dt = (tt[1] - tt[0]) if len(tt) > 1 else 0.01
    mesh = cx.pcolormesh(np.concatenate([tt - dt / 2, [tt[-1] + dt / 2]]), np.arange(len(edges_b)), panp,
                         cmap="RdBu_r", vmin=-1, vmax=1, shading="flat", rasterized=True)
    cx.set_yticks(np.arange(len(edges_b) - 1) + 0.5)
    cx.set_yticklabels([b["band_hz"] for b in band_rows], fontsize=8)
    cx.set_ylabel("band (Hz)")
    cx.set_xlabel("time (s)")
    cx.set_title("pan per band (blue = left, red = right, white = centre)")
    fig.colorbar(mesh, ax=cx, label="pan (-1 L ... +1 R)", pad=0.01)
    fig.suptitle(f"{Path(a.file).name}  |  stereo image", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ca = corr[act] if act.any() else corr
    data = {"correlation_median": r(float(np.median(ca)), 3), "correlation_p5": r(float(np.percentile(ca, 5)), 3),
            "time_out_of_phase_pct": r(100 * float(np.mean(ca < 0)), 1), "bands": band_rows}
    lines = [f"STEREO corr median {data['correlation_median']} | p5 {data['correlation_p5']} | "
             f"out-of-phase {data['time_out_of_phase_pct']}% of active time",
             "  pan by band: " + ", ".join(f"{b['band_hz']}:{b['pan_mean']:+.2f}" for b in band_rows)]
    _save(fig, out, "stereo", data, lines)


def _schroeder(y):
    e = np.cumsum((y.astype(np.float64) ** 2)[::-1])[::-1]
    return 10 * np.log10(e / (e[0] + EPS) + EPS)


def _decay_time(edc, sr, lo, hi):
    """Fit the EDC between lo and hi dB (e.g. -5..-25) and extrapolate to -60 dB."""
    i0 = np.argmax(edc <= lo)
    i1 = np.argmax(edc <= hi)
    if edc[-1] > hi or i1 <= i0 + 5:
        return None
    t = np.arange(i0, i1) / sr
    slope = np.polyfit(t, edc[i0:i1], 1)[0]
    return float(-60 / slope) if slope < 0 else None


def view_reverb(res, x, sr, out, a):
    s = res["_series"]
    off = res["file"].get("offset_s", 0.0)
    e = _pick_event(res, a.event)
    if e is None:
        raise SystemExit("no event found to measure a decay")
    y = _seg(s["mono"], sr, off, e["peak_time_s"], e["end_s"])
    if len(y) < int(0.05 * sr):
        raise SystemExit("decay too short to measure (< 50 ms)")
    bands = {"broadband": None, "low <500 Hz": (None, 500), "mid 500-2k": (500, 2000), "high >2k": (2000, None)}
    fig, ax = plt.subplots(figsize=(16, 7))
    rows = {}
    for name, rng in bands.items():
        yy = y
        if rng:
            lo, hi = rng
            if lo and hi:
                sos = sps.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
            elif lo:
                sos = sps.butter(4, lo, btype="high", fs=sr, output="sos")
            else:
                sos = sps.butter(4, hi, btype="low", fs=sr, output="sos")
            yy = sps.sosfilt(sos, y)
        edc = _schroeder(yy)
        t = np.arange(len(edc)) / sr * 1000
        ax.plot(t, edc, lw=1.4 if name == "broadband" else 1, label=name)
        rows[name] = {"EDT_s": r(_decay_time(edc, sr, 0, -10), 3), "T20_s": r(_decay_time(edc, sr, -5, -25), 3),
                      "T30_s": r(_decay_time(edc, sr, -5, -35), 3), "range_db": r(float(-edc[int(len(edc) * 0.9)]), 1)}
    ax.axhline(-5, color="#999", ls=":", lw=0.8)
    ax.axhline(-25, color="#999", ls=":", lw=0.8)
    ax.axhline(-35, color="#999", ls=":", lw=0.8)
    ax.set_ylim(-70, 2)
    ax.set_xlabel("ms after the peak")
    ax.set_ylabel("energy decay (dB, Schroeder)")
    ax.legend()
    bb = rows["broadband"]
    ax.set_title(f"event #{e['index']} @{e['start_s']:.2f}s  |  broadband EDT {bb['EDT_s']} s, T20 {bb['T20_s']} s, "
                 f"T30 {bb['T30_s']} s  (straight line = clean exponential decay; a knee = two decays, e.g. sound + room)")
    fig.suptitle(f"{Path(a.file).name}  |  decay / reverb", fontsize=12, fontweight="bold")
    fig.tight_layout()
    data = {"event": e["index"], "decay_by_band": rows,
            "note": "Times are extrapolated to -60 dB. They describe the decay of this sound (its own ring-out plus "
                    "any room / reverb). They are a real room RT60 only for an impulsive source in a room."}
    lines = ["REVERB " + " | ".join(f"{k}: T20 {v['T20_s']} s, EDT {v['EDT_s']} s" for k, v in rows.items())]
    _save(fig, out, "reverb", data, lines)


def seam_metrics(x, sr, s_idx, e_idx, w_s=0.05):
    """Judge the jump end -> start of a loop [s_idx, e_idx). If the file continues after e_idx, compare
    against that natural continuation: a loop that lands on a downbeat is SUPPOSED to get louder, exactly
    like the original does at the same point."""
    n = len(x)
    w, wl = int(w_s * sr), int(0.2 * sr)
    e_idx = min(e_idx, n)
    # a natural continuation exists only if the file keeps playing after the end point (not silence / fade-out)
    natural = e_idx + w <= n and float(db(np.sqrt(np.mean(x[e_idx:e_idx + w].astype(np.float64) ** 2)))) > -50
    typical = float(np.percentile(np.abs(np.diff(x, axis=0)), 95)) + EPS
    pred = x[e_idx - 1] + (x[e_idx - 1] - x[e_idx - 2])
    click_err = float(np.max(np.abs(x[s_idx] - pred)))
    nat_err = float(np.max(np.abs(x[e_idx] - pred))) if e_idx < n else 0.0
    lvl = lambda seg: float(db(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))  # noqa: E731
    l_loop = lvl(x[s_idx:s_idx + w])
    l_ref = lvl(x[e_idx:e_idx + w]) if natural else lvl(x[e_idx - w:e_idx])

    def spec(seg):
        seg = seg.mean(axis=1)
        return pdb(np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2)
    sl = spec(x[s_idx:s_idx + wl]) if s_idx + wl <= n else None
    sr_ = spec(x[e_idx:e_idx + wl]) if natural and e_idx + wl <= n else spec(x[max(0, e_idx - wl):e_idx])
    spec_diff = 0.0
    if sl is not None and len(sl) == len(sr_):
        m = (np.fft.rfftfreq(wl, 1 / sr) > 40) & (sr_ > sr_.max() - 60)
        spec_diff = float(np.mean(np.abs(sl[m] - sr_[m]))) if m.any() else 0.0
    issues = []
    if click_err > max(0.02, 3 * typical, 2 * nat_err):
        issues.append(f"CLICK at the seam: step {click_err:.3f} vs typical {typical:.4f}"
                      + (f" (natural continuation: {nat_err:.3f})" if e_idx < n else ""))
    if abs(l_loop - l_ref) > 3:
        issues.append(f"LEVEL JUMP {l_loop - l_ref:+.1f} dB vs " + ("what naturally follows the end point"
                                                                     if natural else "the end of the loop"))
    if spec_diff > 6:
        issues.append(f"TIMBRE JUMP {spec_diff:.1f} dB vs " + ("the natural continuation" if natural else "the loop end"))
    return {"start_sample": int(s_idx), "end_sample": int(e_idx), "start_s": r(s_idx / sr, 4), "end_s": r(e_idx / sr, 4),
            "compared_to": "natural continuation after end" if natural else "end of loop (file ends here)",
            "click_step": r(click_err, 4), "natural_step": r(nat_err, 4), "typical_step_p95": r(typical, 4),
            "level_loop_start_dbfs": r(l_loop, 1), "level_reference_dbfs": r(l_ref, 1),
            "spectral_diff_db": r(spec_diff, 1), "issues": issues}


def _zero_cross_near(x, i, sr, max_ms=3.0):
    """Nearest rising zero crossing to sample i (within max_ms) on the mono mix."""
    m = x.mean(axis=1)
    k = int(max_ms / 1000 * sr)
    lo, hi = max(1, i - k), min(len(m) - 1, i + k)
    zc = np.nonzero((m[lo - 1:hi - 1] < 0) & (m[lo:hi] >= 0))[0] + lo
    return int(zc[np.argmin(np.abs(zc - i))]) if len(zc) else i


def view_loop(res, x, sr, out, a):
    n = len(x)
    s_idx = int(round((a.start or 0) * sr))  # round, not truncate: 29.3 * 48000 must be 1406400, not 1406399
    e_idx = int(round(a.end * sr)) if a.end else n
    if e_idx - s_idx < int(0.2 * sr):
        raise SystemExit("loop region too short")
    mt = seam_metrics(x, sr, s_idx, e_idx)
    w = int(0.05 * sr)
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1, 1, 1.4]})
    joined = np.concatenate([x[e_idx - w:e_idx], x[s_idx:s_idx + w]])
    t = (np.arange(len(joined)) - w) / sr * 1000
    for c in range(min(2, x.shape[1])):
        axes[0].plot(t, joined[:, c], lw=0.8, label=["loop L/ch0", "loop R/ch1"][c] if x.shape[1] > 1 else "loop")
    if e_idx + w <= n:
        axes[0].plot(np.arange(w) / sr * 1000, x[e_idx:e_idx + w].mean(axis=1), color="#999", lw=0.8, ls="--",
                     label="what originally follows the end point")
    axes[0].axvline(0, color="red", lw=1.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(f"last 50 ms before {e_idx / sr:.3f}s  |  red line = loop jump  |  first 50 ms after {s_idx / sr:.3f}s")
    axes[0].set_xlabel("ms")
    k = max(8, int(0.002 * sr))
    tz = (np.arange(2 * k) - k) / sr * 1000
    for c in range(min(2, x.shape[1])):
        axes[1].plot(tz, np.concatenate([x[e_idx - k:e_idx, c], x[s_idx:s_idx + k, c]]), ".-", lw=0.8, ms=3)
    axes[1].axvline(0, color="red", lw=1.2)
    axes[1].set_title("sample level, +/- 2 ms around the seam (a smooth curve = no click)")
    axes[1].set_xlabel("ms")
    wl2 = min(int(0.5 * sr), (e_idx - s_idx) // 2)
    y2 = np.concatenate([x[e_idx - wl2:e_idx].mean(axis=1), x[s_idx:s_idx + wl2].mean(axis=1)])
    f, tt, S = _stft_db(y2, sr, n_fft=1024, hop=128)
    edges, img = core._logfreq_image(f, S, fmin=40, n_bins=200)
    vmax = float(np.percentile(img, 99.5))
    dt = tt[1] - tt[0]
    axes[2].pcolormesh(np.concatenate([tt - dt / 2, [tt[-1] + dt / 2]]) - wl2 / sr, edges, img, cmap="magma",
                       vmin=vmax - 80, vmax=vmax, shading="flat", rasterized=True)
    axes[2].axvline(0, color="cyan", lw=1.2)
    axes[2].set_xlim(-wl2 / sr + 0.03, wl2 / sr - 0.03)  # the outer edges carry STFT boundary artefacts
    _log_axis(axes[2], edges[0], edges[-1])
    axes[2].set_ylim(edges[0], edges[-1])
    axes[2].set_title("spectrogram across the loop jump (a vertical line at 0 = click; a colour change = timbre jump)")
    axes[2].set_xlabel("s relative to the loop jump")
    fig.suptitle(f"{Path(a.file).name}  |  loop {s_idx / sr:.3f}s -> {e_idx / sr:.3f}s: "
                 + ("; ".join(mt["issues"]) if mt["issues"] else "seamless"), fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "loop", mt, [f"LOOP {s_idx / sr:.3f}s -> {e_idx / sr:.3f}s (compared to {mt['compared_to']}): "
                                 + ("; ".join(mt["issues"]) if mt["issues"] else "seamless (no click, level or timbre jump)")])


def view_loopfind(res, x, sr, out, a):
    """Suggest musical loop points: bar-aligned (start, end) pairs whose surroundings match, so the jump
    end -> start sounds like the music's own continuation."""
    s = res["_series"]
    grid = res["rhythm"].get("grid")
    if not grid or (res["rhythm"].get("tempo_confidence") or 0) < 0.4:
        raise SystemExit("no reliable beat grid (tempo confidence < 0.4): use `view --kind loop --start S --end E` "
                         "to test hand-picked points instead")
    bar, d0 = grid["bar_period_s"], grid["first_downbeat_s"]
    act = s["tenv"][s["env"] > s["sil_thr"]]
    t_first, t_last = (act[0], act[-1]) if len(act) else (0, res["duration_s"])
    bars_t = np.arange(d0, t_last + bar * 0.5, bar)
    bars_t = bars_t[bars_t >= t_first - bar * 0.25]
    nbar = len(bars_t) - 1
    if nbar < 9:
        raise SystemExit("too few bars for a loop search")
    f, ts, S, C = s["f"], s["ts"], s["S"].astype(np.float64), s["C"]
    mel = np.log1p(100 * (_mel_bands(f) @ (S ** 2)) / ((S ** 2).sum(axis=0).max() + EPS))
    feats = np.vstack([C, mel, pdb((S ** 2).sum(axis=0))[None, :] / 10])
    F = np.zeros((feats.shape[0], nbar))
    for i in range(nbar):
        m = (ts >= bars_t[i]) & (ts < bars_t[i + 1])
        F[:, i] = feats[:, m].mean(axis=1) if m.any() else 0
    Fz = (F - F.mean(axis=1, keepdims=True)) / (F.std(axis=1, keepdims=True) + EPS)
    Fn = Fz / (np.linalg.norm(Fz, axis=0, keepdims=True) + EPS)
    B = Fn.T @ Fn  # bar-to-bar similarity
    music_end = nbar  # index of the bar line where the music stops
    # section boundaries at bar level (checkerboard novelty, 4-bar halves): loops should start/end on them
    nov = np.zeros(nbar + 1)
    for i in range(4, nbar - 3):
        a_, b_ = slice(i - 4, i), slice(i, i + 4)
        nov[i] = np.mean(B[a_, a_]) + np.mean(B[b_, b_]) - 2 * np.mean(B[a_, b_])
    pk, _ = sps.find_peaks(nov, prominence=0.1)
    is_bound = np.zeros(nbar + 1, bool)
    is_bound[pk] = True
    is_bound[0] = is_bound[music_end] = True
    cands = []
    for i in range(0, nbar - 8):
        for j in range(i + 8, music_end + 1, 4):  # loop length in whole 4-bar phrases, >= 8 bars
            # the jump end -> start sounds natural when what follows `end` in the song resembles what follows
            # `start`; only when nothing follows the end (music ends there) compare the lead-ins instead
            if j + 2 <= nbar:
                match = float(np.mean([B[j + t, i + t] for t in range(2)]))
            elif i >= 2:
                match = float(np.mean([B[j - 1 - t, i - 1 - t] for t in range(2)]))
            else:
                continue
            score = match
            if j == music_end and i >= 2:
                score -= 0.05  # ending the loop at the very end loses the follow-up check
            # musical preferences: start/end on section boundaries, and longer loops (less repetition fatigue)
            score += 0.04 * is_bound[i] + 0.03 * is_bound[j] + 0.02 * math.log2((j - i) / 8)
            cands.append((score, i, j, match))
    cands.sort(reverse=True)
    picked = []
    for sc, i, j, match in cands:
        if match < 0.5:
            break
        if any(abs(i - pi) < 2 and abs(j - pj) < 2 for _, pi, pj, _ in picked):
            continue
        s_idx = _zero_cross_near(x, int(round(bars_t[i] * sr)), sr)
        e_idx = _zero_cross_near(x, int(round(bars_t[j] * sr)), sr)
        mt = seam_metrics(x, sr, s_idx, e_idx)
        picked.append((match, i, j, mt))
        if len(picked) == 5:
            break
    rows = [{"rank": k + 1, "start_bar": i, "end_bar": j, "length_bars": j - i, "start_s": mt["start_s"],
             "end_s": mt["end_s"], "length_s": r((mt["end_sample"] - mt["start_sample"]) / sr, 3),
             "start_sample": mt["start_sample"], "end_sample": mt["end_sample"], "match_score": r(sc, 3),
             "starts_on_section_boundary": bool(is_bound[i]), "ends_on_section_boundary": bool(is_bound[j]),
             "seam_issues": mt["issues"]} for k, (sc, i, j, mt) in enumerate(picked)]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2.2, 1]})
    ax.imshow(B, cmap="viridis", vmin=-0.2, vmax=1, origin="upper", interpolation="nearest")
    ax.set_xlabel("bar")
    ax.set_ylabel("bar")
    ax.set_title(f"bar-to-bar similarity (bar = {bar:.4f} s, bar 0 at {bars_t[0]:.3f} s; 4/4 assumed). "
                 "A good loop i->j has bright cells around (j, i).")
    for k, rw in enumerate(rows[:3]):
        ax.plot(rw["start_bar"], rw["end_bar"], "o", ms=12, mfc="none", mec=["red", "orange", "white"][k], mew=2)
        ax.text(rw["start_bar"] + 1, rw["end_bar"], f"#{k + 1}", color=["red", "orange", "white"][k], fontsize=11)
    bx.plot(s["tz"], s["lk"], color="#444", lw=0.8)
    for k, rw in enumerate(rows[:3]):
        bx.axvspan(rw["start_s"], rw["end_s"], ymin=0.05 + 0.1 * k, ymax=0.13 + 0.1 * k,
                   color=["red", "orange", "#2ca02c"][k], alpha=0.8)
        bx.text(rw["start_s"], np.percentile(s["lk"], 5), f"#{k + 1}", fontsize=9)
    for t_ in bars_t[::4]:
        bx.axvline(t_, color="#ccc", lw=0.5)
    bx.set_ylim(max(-60, np.percentile(s["lk"], 2) - 10), s["lk"].max() + 3)
    bx.set_xlabel("time (s) - grey lines every 4 bars")
    bx.set_ylabel("LUFS (400 ms)")
    fig.suptitle(f"{Path(a.file).name}  |  loop point candidates", fontsize=12, fontweight="bold")
    fig.tight_layout()
    lines = ["LOOPFIND (bar-aligned, snapped to zero crossings):"] + [
        f"  #{rw['rank']} {rw['start_s']}s -> {rw['end_s']}s ({rw['length_bars']} bars, {rw['length_s']} s; samples "
        f"{rw['start_sample']}-{rw['end_sample']}) match {rw['match_score']}"
        + (" [starts at a section boundary]" if rw["starts_on_section_boundary"] else "")
        + (f" | seam: {'; '.join(rw['seam_issues'])}" if rw["seam_issues"] else " | seam clean") for rw in rows]
    _save(fig, out, "loopfind", {"grid": grid, "candidates": rows,
                                 "note": "match_score = similarity of the bars around end vs around start (1 = identical). "
                                         "Check the top pick with `view --kind loop --start S --end E` and by ear."},
          lines)


VIEWS = {"events": view_events, "envelope": view_envelope, "pianoroll": view_pianoroll, "hpss": view_hpss,
         "structure": view_structure, "stereo": view_stereo, "harmonics": view_harmonics, "reverb": view_reverb,
         "loop": view_loop, "loopfind": view_loopfind}
