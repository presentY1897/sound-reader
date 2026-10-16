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


def view(name, kind, *extra):
    out = OUT / f"{Path(name).stem}_{kind}"
    subprocess.run([sys.executable, str(SCRIPT), "view", str(FIX / name), "--kind", kind, "--out", str(out), *extra],
                   check=True, capture_output=True, text=True)
    assert (out / f"view_{kind}.png").stat().st_size > 10_000
    return json.loads((out / f"view_{kind}.json").read_text())


def test_events_and_csv():
    s = run("three_events.wav")
    assert s["events"]["count"] == 3
    dirs = [e["pitch_direction"] for e in s["events"]["list"]]
    assert dirs == ["rising", "falling", "flat"], dirs
    assert s["events"]["list"][2]["dominant_note"].startswith("A4")
    out = OUT / "three_events"
    import csv
    rows = list(csv.DictReader(open(out / "events.csv")))
    assert len(rows) == 3 and rows[0]["held_notes"].startswith("C5")
    frames = list(csv.DictReader(open(out / "frames.csv")))
    assert len(frames) > 50 and "f0_hz" in frames[0]


def test_view_events_envelope_pianoroll():
    assert len(view("three_events.wav", "events")["events"]) == 3
    env = view("three_events.wav", "envelope")["events"]
    assert all(e["attack_ms"] <= 15 for e in env)
    top = view("c_major_prog.wav", "pianoroll")["top_notes_by_energy"]
    assert {d["note"] for d in top[:5]} & {"C4", "E4", "G4"}


def test_view_loop():
    assert view("loop_good.wav", "loop")["issues"] == []
    assert any(i.startswith("CLICK") for i in view("loop_bad.wav", "loop")["issues"])


def test_view_reverb():
    t20 = view("rt60_0.6.wav", "reverb")["decay_by_band"]["broadband"]["T20_s"]
    assert abs(t20 - 0.6) < 0.09, t20


def test_view_harmonics():
    bar = view("bar_inharmonic.wav", "harmonics")
    assert bar["verdict"].startswith("inharmonic")
    assert any(abs(p["ratio_to_f0"] - 2.756) < 0.03 for p in bar["off_series_partials"])
    sq = view("square_220.wav", "harmonics")
    assert sq["verdict"] == "harmonic" and sq["odd_to_even_db"] > 20
    assert abs(sq["spectral_tilt_db_per_octave"] + 6) < 1.5


def test_view_stereo_hpss_structure():
    st = view("pan_highs_right.wav", "stereo")
    pans = {b["band_hz"]: b["pan_mean"] for b in st["bands"]}
    assert abs(pans["125-250"]) < 0.1 and pans["4000-8000"] > 0.8
    assert view("click_120bpm.wav", "hpss")["percussive_energy_pct"] > 50
    labels = [s_["label"] for s_ in view("aba_structure.wav", "structure")["sections"]]
    assert labels[0] == labels[-1] and len(set(labels)) >= 2, labels


def test_event_peak_and_attack():
    e = run("three_events.wav")["events"]["list"][0]
    assert abs(e["peak_dbfs"] - (-6.0)) < 0.5  # true sample peak of a 0.5 amplitude tone, not an RMS value
    assert e["attack_ms"] < 5 and e["attack_ms"] != 5.0


def test_beat_grid_and_loops():
    s = run("bgm_loop.wav")
    g = s["rhythm"]["grid"]
    assert g and abs(g["bar_period_s"] - 2.4) < 0.05, g
    assert abs(((g["first_downbeat_s"] - 0.5) / 2.4) - round((g["first_downbeat_s"] - 0.5) / 2.4)) < 0.05
    lf = view("bgm_loop.wav", "loopfind")["candidates"]
    top = lf[0]
    assert top["length_bars"] % 4 == 0 and top["start_s"] >= 0.5 + 4 * 2.4 - 0.1 and not top["seam_issues"], top
    # a bar-aligned loop inside the music must not be flagged just because the downbeat is louder
    out = OUT / "bgm_loop_region"
    subprocess.run([sys.executable, str(SCRIPT), "view", str(FIX / "bgm_loop.wav"), "--kind", "loop",
                    "--start", str(0.5 + 4 * 2.4), "--end", str(0.5 + 12 * 2.4), "--out", str(out)],
                   check=True, capture_output=True)
    assert json.loads((out / "view_loop.json").read_text())["issues"] == []


def test_structure_repeats():
    st = view("bgm_loop.wav", "structure")
    phrase = [x for x in st["repeat_lags"] if x["scale"] == "phrase"] + [x for x in st["repeat_lags"]]
    assert any(abs(x["lag_s"] / 9.6 - round(x["lag_s"] / 9.6)) < 0.05 for x in phrase), st["repeat_lags"]


def test_compare_alignment():
    import numpy as np
    from scipy.io import wavfile
    sr, x = wavfile.read(FIX / "aba_structure.wav")
    shifted = OUT / "aba_shifted.wav"
    OUT.mkdir(exist_ok=True)
    wavfile.write(shifted, sr, np.concatenate([np.zeros(int(0.25 * sr), dtype=x.dtype), x]))
    out = OUT / "cmp_shift"
    subprocess.run([sys.executable, str(SCRIPT), "compare", str(FIX / "aba_structure.wav"), str(shifted),
                    "--out", str(out)], check=True, capture_output=True)
    c = json.loads((out / "compare.json").read_text())
    assert abs(c["alignment"]["b_offset_s"] - 0.25) < 0.02


if __name__ == "__main__":
    setup_module()
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
