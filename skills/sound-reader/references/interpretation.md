# Reading sound-reader output

## Contents
1. Reading each panel
2. Visual signatures of common problems
3. Metric reference and typical targets
4. Domain checklists (SFX / music / speech / pipeline debugging)
5. Reading the follow-up views (`view --kind ...`) and the aligned compare

---

## 1. Reading each panel

### Waveform (overview, top)
- Min/max envelope per column; dashed lines are digital full scale (±1.0).
- A block that is flat against the dashed lines is clipping or heavy limiting. Red `|` ticks mark clipped runs.
- Envelope shape shows articulation: a sharp vertical rise is a percussive attack, a slow ramp is a pad or a fade-in.
  An abrupt cut to zero at the end (no fade) often produces an audible click.
- If the envelope is not centred on 0, there is DC offset (also see `level.dc_offset`).
- Stereo: blue = L/ch0, red = R/ch1. When the two colours differ a lot, the balance or the content differs between channels.

### Level panel (overview, 2nd)
- Grey line = RMS in 20 ms frames (dBFS), orange = momentary loudness (LUFS, 400 ms, K-weighted, so it is closer to how loud it feels).
- Grey shaded spans = silence (below `dynamics.silence_threshold_dbfs`).
- Green vertical ticks = detected onsets (note/hit starts).
- A steady line with no dips means a compressed or continuous sound. A large spread means a dynamic sound.
- A sudden step in the RMS floor between notes means the noise or room tone changes (edits, a gate opening, a different take).

### Spectrogram (overview, 3rd)
Time runs along x, frequency along y (log scale by default, so octaves are equally spaced), colour is level in dB
(the colour range is printed in the title: bright = loud, black = about 90 dB below the loudest part).
- **Horizontal lines** are steady tones. Evenly spaced stacks of lines are harmonics of one pitched sound
  (voice, instrument, motor). The lowest line is usually the fundamental.
- **Vertical lines** spanning many frequencies are clicks, pops, transients, or edit discontinuities.
  A thin vertical line where the waveform looks smooth usually means a sample-level discontinuity.
- **Diffuse haze** is noise. Hiss sits mostly above 2 kHz, rumble below 100 Hz, and broadband noise covers everything.
- **A hard ceiling** (everything stops at, say, 16 kHz, 11 kHz, or 8 kHz) means low-pass filtering, a lossy codec,
  or a file that was upsampled from a lower sample rate.
- **Lines that bend downward at the top** or mirror-image lines folding back from the top edge mean aliasing,
  typically from non-band-limited synthesis or a bad resampler.
- **"Watery" or blotchy high frequencies** that switch on and off are low-bitrate codec artefacts.
- **Bright vertical bands at regular intervals** are rhythmic hits. Their spacing gives the tempo.
- Cyan line = f0 estimate. It is drawn only where the sound is periodic.

### Chromagram (overview, with `--music`)
- 12 pitch classes (C..B) over time, dark = strong. Chord changes show up as changes in which rows are dark.
- A major triad lights 3 rows (e.g. C, E, G). Tuning that drifts between rows shows up as smearing.

### Spectrum (spectrum.png)
- Average power across the whole window, log frequency. Red dots are the strongest prominent peaks,
  labelled with Hz and the nearest note ± cents.
- Overall slope: for most music and speech, the level falls gradually toward the high frequencies. A spectrum that
  is flat or rising above 2 kHz sounds bright or harsh, and one that falls steeply sounds dull or muffled.
- Narrow peaks that stand far above their neighbours are resonances, tones, hum, or whine.
- The band-energy bars give the tonal balance at a glance (see section 3).

### Detail (detail.png)
- **Loudest 30 ms**: the shape of individual cycles. A sine is smooth, a square wave is flat-topped,
  clipping looks like a sine with its tops cut off, and noise looks irregular.
- **First attack**: how fast the sound starts. Instant vertical jumps can click. A ramp of a few ms is soft.
- **First clip**: shows the clipped run up close, including how many samples sit flat on the ceiling.

---

## 2. Visual signatures of common problems

| Problem | Where it shows | Evidence |
|---|---|---|
| Clipping / overdrive | waveform flat-tops, detail.png | `clipped_samples`, `clip_events`, CLIPPING flag. New harmonics appear in the spectrogram during loud parts |
| Inter-sample peaks | — | `true_peak_dbtp` > -1 while `peak_dbfs` ≤ 0 |
| Click / pop | thin vertical line in spectrogram, spike in waveform | Zoom in with `--start/--end` around the time. Often at file start/end or loop points |
| Missing fade at start/end | waveform jumps from or to non-zero at the edges | detail "First attack" starts vertically. A vertical line at t=0 or at the end |
| Mains hum | horizontal lines at 50/60 Hz + multiples | `mains_hum.suspected`, peaks at 50/100/150 or 60/120/180 |
| Hiss / noise floor | haze above 2 kHz during "silent" parts | `noise_floor_dbfs_p10`, `snr_estimate_db`, `quiet_frame_flatness` > 0.2 |
| Low-pass / lossy / upsampled | hard ceiling in spectrogram | `bandwidth_hz_upper_-60db`, BAND-LIMITED flag. If `file.lossy_codec` is set (mp3/ogg/aac/opus…), a ceiling around 15-20 kHz is just the encoder and not worth reporting as a problem. In a WAV/FLAC it means the audio was filtered or once passed through a lossy codec |
| Aliasing | lines folding back down from the top | Inharmonic peaks that don't fit the harmonic series |
| DC offset | waveform not centred | `dc_offset` > 0.01, DC OFFSET flag |
| Phase / polarity problem | — | `stereo.correlation` < 0, NEGATIVE L/R flag. Sound cancels in mono |
| Fake stereo | — | `effectively_mono` true |
| Unwanted silence | grey spans at the start/end | `leading_silence_s`, `trailing_silence_s` (matters for game SFX latency and for loops) |
| Wrong pitch | f0 track and labelled peaks | `f0_median_hz` / note vs. expected. A ±100 cent offset is one semitone |
| Wrong speed or sample rate | everything shifted in pitch and time | Ratio of measured f0 to expected f0. 44.1k vs 48k is about +147 cents / -147 cents |
| Over-compression | flat RMS line, small crest factor | `crest_factor_db` < ~8 dB for music, `loudness_range_lu` < 3 |
| Sibilance | bright bursts at 5-9 kHz on "s" sounds in speech | presence/brilliance band share. Short vertical bright patches high up |
| Boomy / muddy | heavy energy at 100-400 Hz | bass + low_mid share high, centroid low |
| Sound too quiet / too loud | level panel | `integrated_lufs` vs target (section 3) |

---

## 3. Metric reference and typical targets

**Level**
- `peak_dbfs`: highest sample. `true_peak_dbtp`: 4× oversampled peak. Most delivery specs want ≤ -1 dBTP.
- `integrated_lufs` (BS.1770 gated loudness). Rough targets:
  - Streaming music (Spotify/YouTube normalisation): about -14 LUFS
  - Apple Music: about -16 LUFS. Podcasts: -16 (stereo) / -19 (mono) LUFS
  - Broadcast EBU R128: -23 LUFS. ATSC A/85: -24 LKFS
  - Games (Sony/Xbox guidance for the overall mix): about -24 to -18 LUFS. Individual SFX vary, so judge them relative to each other
  - Short sounds under 400 ms have unreliable integrated LUFS. Use `momentary_max_lufs` and `peak_dbfs` instead.
- `crest_factor_db` = peak − RMS. Sine ≈ 3 dB, heavily limited master ≈ 6-9 dB, dynamic acoustic recording ≥ 15 dB.
- `loudness_range_lu`: < 5 = very compressed, 5-10 = typical pop, > 15 = cinematic/classical.

**Dynamics**
- `noise_floor_dbfs_p10`: level of the quietest 10% of frames. It only means "noise" when the file has pauses.
- `snr_estimate_db`: active level − floor. Speech recordings: > 40 dB is clean, < 25 dB has noticeable noise.

**Spectral**
- `centroid_hz_median`: the "brightness" centre of mass. Speech is about 1-3 kHz, a bass-heavy mix about 500-1500 Hz, cymbals and hiss above 4 kHz.
- `rolloff85_hz_median`: frequency below which 85% of the energy lies.
- `flatness_median`: 0 = pure tone, about 1 = white noise. Music is usually 0.01-0.3.
- `band_energy_percent`: rough guide for a balanced full-range music mix: sub 5-15, bass 25-45, low_mid 10-20,
  mid 15-30, high_mid 3-10, presence 1-5, brilliance 1-5. Speech has little sub energy and is concentrated in bass and mid.
- `dominant_peaks`: frequency resolution is limited by `stft.freq_resolution_hz` / the Welch window, so treat the values as ±1 bin
  (sub-bin interpolation helps for clean tones).

**Rhythm / pitch / tonal**: see the limits in SKILL.md. Musical note names use A4 = 440 Hz.

---

## 4. Domain checklists

**Game / app SFX and UI sounds**
- Leading silence (adds latency) and trailing silence (wastes memory or breaks loops)
- Clipping and true peak. Consistent peak/LUFS across a set of related sounds (run each one and compare)
- Attack: does it click? Tail: does it fade to true zero, or get cut off?
- Loops: compare the first and last few ms (`--start 0 --end 0.05` and the last 50 ms). Look for a level or phase jump at the seam
- Frequency content suitable for small speakers (energy below about 150 Hz is lost on phones)

**Music**
- Use `--music`. Tempo and key estimates, chord changes in the chromagram, section changes in the level panel
- Mix balance from the band energy, stereo width from `side_to_mid_db`, mono compatibility from `correlation`
- Mastering: LUFS vs target, true peak, crest factor and LRA (over-limiting)

**Speech / voice / podcast / TTS**
- f0 median and range: adult male voices are about 85-180 Hz, female 165-255 Hz, children higher. A flat f0 track can mean a monotone or robotic voice
- Pauses (silence regions) and pace (onset rate), noise floor in the pauses, hum, room reverb
  (the haze tail after each word in the spectrogram)
- TTS or vocoder artefacts: metallic horizontal streaks, buzzy harmonics running up to the top, abrupt spectral cuts between frames

**Pipeline / signal debugging**
- Sample rate and channel count as expected? (`file.sample_rate`, `channels`, `effectively_mono`)
- Pitch shift ratio → sample-rate mismatch. Silence where audio was expected → buffer or underrun bugs (look for
  regularly spaced gaps or clicks in the spectrogram, a sign of buffer-size issues)
- `compare` the output against a known-good reference. A flat B−A difference with a constant offset is just gain.
  A sloped difference means filtering. Large narrow differences mean added tones or artefacts.

---

## 5. Reading the follow-up views

**events (contact sheet)**: every sound uses the same amplitude and colour scale, so you can compare them directly.
Different heights mean inconsistent levels across a set. A cyan label shows the held notes. Use it to describe each sound
in one line (e.g. "#4: E5→G5→C6 arpeggio, 1.05 s, bell-like"). Then check consistency: similar sounds should have
similar peak, length and brightness.

**envelope**:
- **Green** is the attack. Under 10 ms feels instant or percussive, 30–100 ms soft, over 150 ms a swell.
- **Orange** is the decay to −20 dB.
- "never decays 20 dB" means a sustained or cut sound. If `ends_at_file_end` is true and the end level is still above
  about −30 dB re peak, the tail is cut and will click.
- A second bump is a second hit or note inside the same event.

**pianoroll**: rows are semitones and white lines mark each C. Read notes straight off the rows. Harmonics of one note
appear 12 rows up (octave), 19 (octave + fifth) and 24 above it, so don't mistake them for chord tones. Chord tones sit
3–4 / 7 rows apart within the same octave. The time resolution is about 0.2 s, so fast runs smear.

**harmonics**:
- **Harmonic**: green dashed lines with peaks on them, and a bar per k.
- **Odd/even balance**:
  - odd ≫ even (+15 dB or more) means hollow, square, clarinet-like
  - balanced means sawtooth, brass or voice-like
- **Tilt (dB per octave of harmonic number)**:
  - about −6 is neutral (sawtooth or square)
  - −10 or below is dull, soft or filtered
  - −3 or above is bright, harsh or buzzy
- **Inharmonic**: red off-series partials with non-integer ratios.
  - ×2.76 and ×5.40 are a free bar (glockenspiel, xylophone)
  - ×1.59 and ×2.14 are a membrane (drum)
  - irregular ratios point to a bell, metal or FM synthesis
- Cents off above about 15 on low harmonics means detuning or chorus. Rising sharpness at higher k is piano-like stiffness.

**hpss**:
- The top panel keeps sustained horizontal lines (notes, hum, drones).
- The middle panel keeps vertical lines (hits, clicks, consonants) and noise bursts.
- More than about 50% percussive energy means drum- or impact-driven material.
- A percussive line inside a "clean" tone means a click or discontinuity.

**structure**:
- Bright square blocks on the diagonal are sections that stay similar internally.
- Bright blocks *off* the diagonal are repeats. The row and column times say which parts match.
- Peaks in the novelty curve are section changes.
- Labels: the same letter means similar material (A … A = a return, e.g. chorus or main theme). Silence at the
  start and end often shares a letter.
- Combine with loudness to name sections (intro, build, drop/chorus, breakdown, outro). Treat the names as an inference.
- `repeat_lags`: `short` = riff or pattern length, `phrase` = how far apart whole phrases repeat. A phrase lag of exactly
  8, 16 or 32 bars is the natural loop length.

**loopfind**:
- Candidates are ranked by `match_score`, the similarity between what follows the end point in the song and what
  follows the start point. Above about 0.9 the jump sounds like the song's own continuation.
- Prefer candidates that start on a section boundary and that are long, so the loop repeats less often.
- An intro that is played once and never returns is excluded automatically, since nothing matches it.
- Always confirm the pick with `view --kind loop --start S --end E`. Give the user both seconds and sample numbers.

**stereo**:
- **Vectorscope**: a thin vertical line is mono. A tall ellipse is a normal mix. A round blob is very wide or
  decorrelated. A horizontal shape means out of phase (cancels in mono).
- **Correlation curve**: dips below 0 are mono-compatibility risks.
- **Pan map**: blue = left, red = right, white = centre. Bass and kick normally stay white (centred). Anything else
  is a mix choice worth mentioning.

**reverb**:
- The curves are Schroeder energy decay.
- A straight line is one clean exponential decay. Its T20/T30 is the ring-out time.
- A knee (fast drop, then a slower tail) means a dry sound followed by reverb or room. The slow part is the space.
- Typical decay times:
  - under 0.3 s: dry booth or close mic
  - 0.4–0.8 s: normal room
  - 1–2 s: hall or plate
  - over 2.5 s: cathedral or huge FX reverb
- If the high band decays much faster than the low band, the space is damped or dark, which is natural for real rooms.
- These are decay times of *this sound*. A synthesized decay is not a room.

**loop**: "seamless" means no click, level jump or timbre jump at end→start. When audio continues after the end
point, the jump is judged against that *natural continuation*. A loop that lands on a downbeat may get louder, and
that's fine if the song does the same at that point. The grey dashed line in the top panel is the original continuation.
- **CLICK**: the sample zoom shows a step at 0, or the spectrogram shows a vertical line at the loop point.
  Fix with a zero-crossing cut or a 5–20 ms crossfade.
- **LEVEL JUMP**: the fade-in or fade-out doesn't match. Remove the fades or crossfade the loop.
- **TIMBRE JUMP**: the end and start are different material. Choose another loop point.

**compare: aligned difference**: B is shifted to line up with A, then the difference is coloured.
- Red means louder in B, blue means louder in A.
- A blue block above some frequency over a time range means B is low-passed there.
- A thin red vertical line means a click added in B.
- A uniform tint everywhere is just a gain change. Check `level.*` deltas.
- Ignore isolated red/blue line *pairs* at sharp hits: those are sub-frame alignment jitter, not real differences.

