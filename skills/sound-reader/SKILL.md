---
name: sound-reader
description: Analyse audio files by rendering them into images (waveform, loudness curve, spectrogram, spectrum, sample-level zoom) plus exact measurements (peak, LUFS, clipping, noise floor, hum, onsets, tempo, pitch, key, stereo phase) so a model that cannot hear can still "listen". Use this whenever the user gives or mentions an audio or video file (.wav .mp3 .flac .ogg .m4a .aac .opus .aiff .webm .mp4 ...) and wants to know what it sounds like, what is wrong with it, or how it compares to another file - e.g. game/UI sound effects, music, voice recordings, podcasts, TTS/synth output, generated audio, or debugging an audio pipeline (clicks, pops, clipping, distortion, hum, silence, wrong pitch/tempo, too quiet/loud, phase problems, before/after a processing change). Also use it when the user says "listen to this", "check this sound", "why does this sound bad", or asks about loudness, frequencies, or a spectrogram, even if they don't ask for images.
---

# sound-reader

You cannot hear audio, but you can read pictures and numbers. This skill converts a sound file into
both, so you can reason about it the way an audio engineer looks at a DAW:

- **Numbers (stdout + `summary.json`)** are the ground truth: peak, true peak, LUFS, clipping, DC offset,
  noise floor, silence regions, onsets, tempo, f0, key, band energy, stereo correlation, and automatic `flags`.
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

The script prints a compact summary. Read it first, especially the `FLAGS` line.

## 2. Look at the images

Open the PNGs with your image-reading tool (e.g. Read). Pick the ones the question needs:

| Image | Panels | Look at it for |
|---|---|---|
| `overview.png` | waveform envelope · RMS + momentary LUFS with silence shading and onset ticks · log-frequency spectrogram with f0 track (· chromagram with `--music`) | Structure over time: where events, silence, noise, and pitch changes happen. **Almost always look at this one.** |
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

## 4. Compare two files

For before/after, reference vs. render, or "which version is better", run:

```bash
python3 <skill-dir>/scripts/sound_reader.py compare A.wav B.wav [--out DIR]
```

This writes `compare.png` (spectrum overlay, B−A spectral difference, loudness curves) and
`compare.json` (every metric that changed, plus each file's flags). Also run `analyze` on each file if you
need its spectrogram.

## 5. Report back

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
