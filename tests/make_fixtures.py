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


if __name__ == "__main__":
    main()
    print(f"fixtures written to {OUT}")
