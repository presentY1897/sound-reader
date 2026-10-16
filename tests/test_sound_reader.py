"""Run:  python -m pytest tests/   (or  python tests/test_sound_reader.py)"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "skills" / "sound-reader" / "scripts" / "sound_reader.py"
FIX = Path(__file__).parent / "fixtures"
OUT = Path(__file__).parent / "out"


def setup_module(_=None):
    subprocess.run([sys.executable, str(Path(__file__).parent / "make_fixtures.py")], check=True)


def run(name, *extra):
    out = OUT / Path(name).stem
    subprocess.run([sys.executable, str(SCRIPT), "analyze", str(FIX / name), "--out", str(out), *extra],
                   check=True, capture_output=True, text=True)
    for img in ("overview.png", "spectrum.png", "detail.png"):
        assert (out / img).stat().st_size > 10_000
    return json.loads((out / "summary.json").read_text())


def test_lufs_reference_sine():
    s = run("sine_1k_0dbfs.wav")
    assert abs(s["level"]["integrated_lufs"] - (-3.0)) <= 0.2
    assert abs(s["level"]["peak_dbfs"]) < 0.05
    assert abs(s["spectral"]["dominant_peaks"][0]["hz"] - 1000) < 10


def test_pitch_and_silence():
    s = run("a440_padded.wav")
    assert abs(s["pitch"]["f0_median_hz"] - 440) < 5
    assert s["pitch"]["f0_median_note"].startswith("A4")
    assert abs(s["dynamics"]["leading_silence_s"] - 0.5) < 0.05
    assert abs(s["dynamics"]["trailing_silence_s"] - 0.5) < 0.05


def test_clipping():
    s = run("clipped.wav")
    assert s["level"]["clipped_samples"] > 1000
    assert any(f.startswith("CLIPPING") for f in s["flags"])


def test_hum():
    s = run("hum60.wav")
    assert s["spectral"]["mains_hum"]["suspected"]
    assert s["spectral"]["mains_hum"]["mains_hz"] == 60


def test_tempo():
    s = run("click_120bpm.wav")
    assert abs(s["rhythm"]["tempo_bpm_estimate"] - 120) <= 2
    assert 14 <= s["rhythm"]["onset_count"] <= 17


def test_stereo_phase():
    s = run("stereo_inverted.wav")
    assert s["stereo"]["correlation"] < -0.99
    assert any("NEGATIVE L/R" in f for f in s["flags"])


def test_bandlimited():
    s = run("bandlimited.wav")
    assert any(f.startswith("BAND-LIMITED") for f in s["flags"])


def test_key():
    s = run("c_major_prog.wav", "--music")
    assert s["tonal"]["key_estimate"] in ("C major", "A minor")


def test_window_and_compare():
    s = run("chirp.wav", "--start", "1", "--end", "2")
    assert abs(s["duration_s"] - 1.0) < 0.01
    out = OUT / "cmp"
    subprocess.run([sys.executable, str(SCRIPT), "compare", str(FIX / "clipped.wav"), str(FIX / "a440_padded.wav"),
                    "--out", str(out)], check=True, capture_output=True)
    c = json.loads((out / "compare.json").read_text())
    assert "level.peak_dbfs" in c["metric_deltas"]


if __name__ == "__main__":
    setup_module()
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
