---
name: sound-reader
description: Analyse audio files by rendering them into images (waveform, loudness, spectrogram with note axis, spectrum, sample-level zoom, plus on-demand views: event contact sheet, envelope, piano roll, harmonics, HPSS, song structure, stereo, reverb, loop seam) and exact numbers/CSVs (LUFS, clipping, hum, per-sound pitch contour, attack/decay, tempo, key) so a model that cannot hear can still "listen". Use whenever the user gives or mentions an audio/video file (.wav .mp3 .ogg .flac .m4a ...) and wants to know what it sounds like, what is wrong with it, or how two versions differ: game/UI sound effects, music/BGM, voice lines, TTS or generated audio, or debugging an audio pipeline (clicks, clipping, hum, silence, wrong pitch, loudness, phase, loop seams). Also use it for "listen to this", "check this sound", "why does this sound bad", even if they don't ask for images.
---

# sound-reader

You cannot hear audio, but you can read pictures and numbers. This skill converts a sound file into
both, so you can reason about it the way an audio engineer looks at a DAW:

- **Numbers (stdout + `summary.json`)** are the ground truth: peak, true peak, LUFS, clipping, DC offset,
  noise floor, silence regions, onsets, tempo, f0, key, band energy, stereo correlation, and automatic `flags`.
- **Tables (`events.csv`, `frames.csv`)** hold exact values you would otherwise have to read off a curve:
  one row per separate sound (event), and a compact time series. Read them with `head`, `grep` or `cut`.
- **Images** show *shape and pattern* that numbers can't summarise: where things happen in time, harmonic
  structure, transients, noise character, envelope.

Use the numbers when you state values and the images to explain them. Don't guess a value off a picture when
the JSON has it.

## 1. Run the analyzer

The script lives next to this file: `scripts/sound_reader.py` (resolve it relative to this SKILL.md's directory).

```bash
python3 <skill-dir>/scripts/sound_reader.py analyze <audio-file> [--out DIR]
```

- Needs `numpy scipy matplotlib`. If the import fails, run `pip install numpy scipy matplotlib` and retry.
- WAV is read directly. Other formats (mp3, m4a, ogg, flac, opus, video containers) are decoded with `ffmpeg`.
  If ffmpeg is missing, tell the user how to install it (`apt install ffmpeg` / `brew install ffmpeg` / `winget install ffmpeg`).
- Default output dir is `./sound-reader-out/<file-stem>/`. Pass `--out` to a temp/scratch dir when you
  shouldn't litter the user's project.
- Options: `--start S --end S` zooms into a time window (seconds), `--music` adds a chromagram panel
  (use it for music/harmony questions), `--linear-freq` uses a linear frequency axis (better for high-frequency
  detail or harmonic spacing).

The script prints a compact summary. Read it first, especially the `FLAGS` and `EVENTS` lines.

**Events.** The analyzer splits the file into separate sounds wherever there is silence between them (a set of UI
sounds, voice lines, hits). `events.csv` has one row per sound with these columns:
- timing and level: start/end, duration, `peak_dbfs` (true sample peak), `peak_rms10ms_dbfs`
- envelope: `attack_ms` (-20 dB → first peak, 0.5 ms resolution), `decay_to_-20db_ms`, `end_level_db_rel_peak`,
  `ends_at_file_end`
- pitch: `dominant_note`, `pitch_direction` (start vs end: rising/falling/flat), `pitch_contour`
  (rising/falling/rise-fall/fall-rise/flat), `pitch_range_st`, `held_notes` (e.g. `C5 E5 G5 C6`)

`pitch_source` tells you where the pitch came from:
- `f0`: a single voice or tone
- `spectral_peak`: the strongest partial, used for chords and arpeggios

Pitch and envelope fields are left empty for events longer than 10 s, because continuous music has no single start
and end pitch. Use the views below for those. A `~` in front of a note (e.g. `~D2`) means approximate: low notes
fall between wide frequency bins, so no cents are given.

**Rhythm grid.** When a tempo is found, the `RHYTHM` line and `rhythm.grid` give `bar_period_s` and
`first_downbeat_s` (4/4 assumed). Bar *n* starts at `first_downbeat_s + n × bar_period_s`. The downbeat is
chosen from kick/bass onsets and from big level changes (music start, a drop), which usually land on bar 1.
Check it against the overview before quoting bar numbers, because the tempo can be off by ×2.

For continuous material (music, ambience) the `NOISE` line says "n/a". The quietest part of the programme isn't
background noise.

`frames.csv` holds `t_s, rms_dbfs, loudness_lufs_m, f0_hz, peak_hz, centroid_hz, flatness, f0_note` with
at most about 2000 rows.

## 2. Look at the images

Open the PNGs with your image-reading tool (e.g. Read). Pick the ones the question needs:

| Image | Panels | Look at it for |
|---|---|---|
| `overview.png` | waveform envelope · RMS + momentary LUFS with silence shading and onset ticks · log-frequency spectrogram with f0 track and a note axis (C1…C9) on the right (· chromagram with `--music`) | Structure over time: where events, silence, noise, and pitch changes happen. **Almost always look at this one.** |
| `spectrum.png` | average spectrum with labelled peaks (Hz + note) · energy per band | Tonal balance, resonances, hum, missing highs or lows, "muddy", "harsh", "thin" |
| `detail.png` | sample-level zooms: loudest 30 ms, first attack, first clip | Waveform shape, clipping flat-tops, clicks, DC offset, attack sharpness |

For guidance on how to read each panel, and what different problems look like (clicks, hum, aliasing, codec
artefacts, phase issues, sibilance, loudness targets per platform), read `references/interpretation.md`.
Read it the first time you use this skill in a session, or whenever a pattern is unclear.

## 3. Zoom in when needed

The overview squeezes the whole file into about 1800 columns. With files longer than about 20 s, a
short click or glitch can shrink to a single pixel. Zoom in when:

- the file is long and the question concerns a specific moment,
- `rhythm.onsets_s`, `first_clip_s` or a flag points at a timestamp you want to inspect,
- something in the overview looks odd but is too small to read.

```bash
python3 <skill-dir>/scripts/sound_reader.py analyze file.wav --start 12.0 --end 14.5 --out <dir>/zoom_12s
```

Times in a zoomed analysis are still absolute (relative to the start of the file).

## 4. Follow-up views: only when the interpretation needs them

Start with `analyze`. Then, if a question is still open, render one extra view. **Don't render them all.**
Pick the one that answers what you're unsure about.

```bash
python3 <skill-dir>/scripts/sound_reader.py view <audio-file> --kind <kind> [--event N] [--start S --end S] [--out DIR]
```

Each view writes `view_<kind>.png` and `view_<kind>.json` (the numbers) into the same output folder and prints a summary line.

| `--kind` | Shows | Reach for it when |
|---|---|---|
| `events` | every event side by side: waveform + spectrogram on the **same** scales, with held notes | the file has several sounds (SFX set, voice lines) and you need to compare or describe each one |
| `envelope` | per-event dB envelope with attack / decay / end markers | "punchy?", "soft attack?", "is the tail cut off?", "does it click at the start?" |
| `pianoroll` | semitone-resolution spectrogram, one row per note | which notes / chords / melody, tuning, "is it in key" |
| `harmonics` | harmonic series of one sound: level of each k·f0, cents off, off-series partials | timbre: hollow/square (odd-only), bright/dull (tilt), bell/metal (inharmonic ratios like ×2.76), buzz |
| `hpss` | tonal part vs. percussive part, as separate spectrograms + energy curves | separating drums from melody, "how percussive", hidden clicks under a tone |
| `structure` | self-similarity matrix, novelty curve, sections labelled A/B/A… (snapped to bar lines when there is a grid), repeat periods at short and phrase scale | songs/BGM/long ambiences: intro, verse/chorus repeats, drop, where it changes, how long a phrase is |
| `stereo` | vectorscope, L/R correlation over time, pan per frequency band | width, mono compatibility, phase problems, "what is panned where" |
| `reverb` | Schroeder decay curves + EDT/T20/T30 (broadband and low/mid/high) | "too much reverb?", "dry?", room size, how long a sound rings out |
| `loop` | the jump end → start: waveform, sample zoom, spectrogram, compared with what *originally* follows the end point | checking a loop. Without options it tests the whole file. `--start S --end E` tests an in-file loop region (the whole file is still loaded) |
| `loopfind` | bar-to-bar similarity + ranked loop candidates (bar-aligned, snapped to zero crossings, seam-checked, in seconds and samples) | "where should this BGM loop?" Needs a beat grid. Run `loop` on the pick you choose |

`--event N` (an index from `events.csv`) targets `envelope`, `harmonics` and `reverb` at one sound. Without it they use
the loudest event. `harmonics` analyses about 0.12 s right after the peak.

## 5. Compare two files

For before/after, reference vs. render, or "which version is better", run:

```bash
python3 <skill-dir>/scripts/sound_reader.py compare A.wav B.wav [--out DIR]
```

This writes `compare.png` and `compare.json`.

`compare.png` has four panels:
- spectrum overlay
- B−A average spectral difference
- loudness curves
- a **time-aligned spectrogram difference**: B is shifted to line up with A, then red means louder in B and blue means louder in A

`compare.json` has:
- every metric that changed
- each file's flags
- `alignment`: B's time offset and the times of the biggest differences

The aligned difference shows *when* and *where in frequency* two versions differ. That catches a glitch or
a filtered passage that the averages hide. Run `analyze` on each file too if you need its full spectrogram.

## 6. Report back

Answer the user's actual question first, then give the evidence. A good answer:

- quotes concrete measurements with units and timestamps ("clips 37 times, first at 1.204 s; true peak +0.8 dBTP"),
- says what the image shows ("the spectrogram has a steady horizontal line at 60 Hz and its harmonics throughout,
  which is mains hum"),
- separates **measured** facts from **inferences** ("this *suggests* an MP3 encoded at a low bitrate because
  content stops abruptly at 16 kHz"),
- gives actionable fixes when something is wrong (e.g. "reduce gain by 3 dB before the limiter", "high-pass at 80 Hz",
  "add a 5 ms fade-out to remove the end click"),
- is honest about limits (see below).

Mention image paths only when the user might want to open them.

## Limits (be upfront about these)

- **No speech-to-text, by design.** This skill is for understanding a sound through images and measurements.
  It does not extract words. You can tell *that* someone speaks and *how* (pitch contour, syllables, fricatives
  vs. plosives, pauses, level), but not *what* they say.
  If the user needs the words, say that a speech-recognition (ASR / speech-to-text) tool is needed:
  - First check what this environment already offers: a transcription skill or tool in your tool list (Codex, for example,
    may have a `transcribe` skill) or an ASR package that is already installed. Prefer that over suggesting new installs.
  - Don't present any specific product as the current best. Options change quickly. If the user wants a recommendation
    and you can browse, check a maintained comparison such as the Hugging Face Open ASR Leaderboard.
    Otherwise name examples as examples ("e.g. Whisper-family models, cloud STT APIs") and say they may be dated.
- `f0` is a single periodicity track. It is reliable for one voice, one instrument, or tones. For chords or full
  mixes it shows the common periodicity, which may be a "missing fundamental" below the actual notes.
- `tempo_bpm_estimate` can be off by a factor of 2 or 1.5, and `key_estimate` is a statistical guess.
  Both are `null` when the sound has no rhythm or no tonality. Cross-check them against the onset ticks and the chromagram.
- Timbre judgements ("warm", "punchy", "robotic") come from your interpretation of the spectra. Say so.
- Very long files (>10 min) are slow and the overview gets coarse. Analyse sections with `--start/--end`.
