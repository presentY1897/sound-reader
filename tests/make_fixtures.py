"""Generate synthetic test signals with known properties into tests/fixtures/."""
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR = 48000
OUT = Path(__file__).parent / "fixtures"


def t(sec, sr=SR):
    return np.arange(int(sec * sr)) / sr


def write(name, x, sr=SR):
    OUT.mkdir(exist_ok=True)
    wavfile.write(OUT / name, sr, np.asarray(x, dtype=np.float32))


def main():
    rng = np.random.default_rng(0)
    # 1 kHz sine at 0 dBFS peak, mono -> -3.01 LUFS by definition of BS.1770
    write("sine_1k_0dbfs.wav", np.sin(2 * np.pi * 1000 * t(5)))
    # A4 440 Hz tone with harmonics, 0.5 s silence on each side
    tone = sum(0.5 / k * np.sin(2 * np.pi * 440 * k * t(2)) for k in range(1, 5))
    write("a440_padded.wav", np.concatenate([np.zeros(SR // 2), tone, np.zeros(SR // 2)]))
    # Clipped sine: driven 2x into hard clip
    write("clipped.wav", np.clip(2 * np.sin(2 * np.pi * 220 * t(2)), -1, 1))
    # Noise with 60 Hz hum + harmonics
    hum = 0.05 * sum(np.sin(2 * np.pi * 60 * k * t(4)) / k for k in range(1, 5))
    write("hum60.wav", 0.01 * rng.standard_normal(len(t(4))) + hum)
    # Click track at 120 BPM (8 s)
    x = np.zeros(len(t(8)))
    click = np.exp(-np.arange(400) / 60) * np.sin(2 * np.pi * 2000 * np.arange(400) / SR)
    for b in np.arange(0, 8, 0.5):
        i = int(b * SR)
        x[i:i + 400] += 0.8 * click
    write("click_120bpm.wav", x)
    # Stereo with R polarity-inverted
    s = 0.5 * np.sin(2 * np.pi * 300 * t(2))
    write("stereo_inverted.wav", np.stack([s, -s], axis=1))
    # Low-passed noise at 44.1k (simulated "upsampled" / band-limited)
    from scipy.signal import butter, sosfilt

    sos = butter(8, 7000, fs=44100, output="sos")
    write("bandlimited.wav", 0.3 * sosfilt(sos, rng.standard_normal(44100 * 3)), sr=44100)
    # C major chord progression (C-F-G-C), for key estimate
    def chord(freqs, d=1.0):
        return sum(np.sin(2 * np.pi * f * t(d)) for f in freqs) / len(freqs) * 0.6
    prog = [chord([261.63, 329.63, 392.0]), chord([349.23, 440.0, 523.25]),
            chord([392.0, 493.88, 587.33]), chord([261.63, 329.63, 392.0, 523.25])]
    write("c_major_prog.wav", np.concatenate(prog * 2))
    # Rising chirp 100 Hz -> 8 kHz
    from scipy.signal import chirp

    write("chirp.wav", 0.5 * chirp(t(3), 100, 3, 8000, method="logarithmic"))

    # --- fixtures for events / views ---
    def ding(freqs, d=0.3):
        """Notes played one after another inside one sound, with a decaying envelope."""
        y = np.concatenate([np.sin(2 * np.pi * f * t(d / len(freqs))) for f in freqs])
        return 0.5 * y * np.exp(-np.arange(len(y)) / SR * 6)
    gap = np.zeros(SR // 4)
    # 3 separate sounds: rising C5->G5, falling G5->C5, flat A4
    write("three_events.wav", np.concatenate([gap, ding([523.25, 783.99]), gap, ding([783.99, 523.25]), gap,
                                              ding([440.0]), gap]))
    write("loop_good.wav", 0.5 * np.sin(2 * np.pi * 240 * t(2)))      # whole number of cycles
    write("loop_bad.wav", 0.5 * np.sin(2 * np.pi * 243.7 * t(2)))     # cut mid-cycle
    n = int(1.5 * SR)
    rev = rng.standard_normal(n) * np.exp(-6.9078 * t(1.5) / 0.6)    # RT60 = 0.6 s
    write("rt60_0.6.wav", np.concatenate([np.zeros(SR // 10), 0.5 * rev / np.abs(rev).max(), np.zeros(SR // 4)]))
    from scipy.signal import butter as _b, sosfilt as _sf
    hiss = _sf(_b(4, 4000, "high", fs=SR, output="sos"), 0.2 * rng.standard_normal(len(t(2))))
    tone = 0.4 * np.sin(2 * np.pi * 200 * t(2))
    write("pan_highs_right.wav", np.stack([tone + 0.1 * hiss, tone + 0.9 * hiss], axis=1))
    tb = t(1.5)
    bar = sum(a * np.sin(2 * np.pi * 220 * r_ * tb) * np.exp(-tb * d) for a, r_, d in
              [(1, 1, 2), (0.6, 2.756, 3), (0.4, 5.404, 4), (0.25, 8.933, 5)])
    write("bar_inharmonic.wav", 0.4 * bar / np.abs(bar).max())
    write("square_220.wav", 0.3 * np.sign(np.sin(2 * np.pi * 220 * t(1.5))))
    # A-B-A structure (10 s each): tonal chord vs noisy drums
    secA = 0.3 * (np.sin(2 * np.pi * 261.63 * t(10)) + np.sin(2 * np.pi * 329.63 * t(10)))
    drums = np.zeros(len(t(10)))
    for b in np.arange(0, 10, 0.25):
        i = int(b * SR)
        drums[i:i + 2000] += rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300)
    secB = 0.3 * drums + 0.2 * np.sin(2 * np.pi * 98 * t(10))
    write("aba_structure.wav", np.concatenate([secA, secB, secA]) * 0.8)
    # small BGM: 100 BPM (bar = 2.4 s), 4-bar quiet intro, then a 4-bar chord loop played 4 times with drums
    beat, bar_n = 0.6, 4
    prog = [[261.63, 329.63, 392.0], [220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [196.0, 246.94, 293.66]]
    intro = np.concatenate([0.05 * sum(np.sin(2 * np.pi * f_ * t(beat * bar_n)) for f_ in prog[k % 4])
                            for k in range(4)])
    main = []
    for k in range(16):
        tt = t(beat * bar_n)
        y = 0.12 * sum(np.sin(2 * np.pi * f_ * tt) for f_ in prog[k % 4])
        for bt in range(bar_n):
            i = int(bt * beat * SR)
            kick = np.sin(2 * np.pi * 55 * t(0.25)) * np.exp(-t(0.25) * 18)
            y[i:i + len(kick)] += 0.6 * kick
            j = int((bt + 0.5) * beat * SR)
            y[j:j + 1500] += 0.08 * rng.standard_normal(1500) * np.exp(-np.arange(1500) / 200)
        main.append(y)
    write("bgm_loop.wav", np.concatenate([np.zeros(SR // 2), intro, *main, np.zeros(SR // 2)]))


if __name__ == "__main__":
    main()
    print(f"fixtures written to {OUT}")
