"""Synthesize the 30 s soundtrack for the demo video: no samples, no downloads, no generated audio models.

Everything is computed here with numpy/scipy: a warm pad (D major colour, four chords with slow attack and
release), a soft plucked arpeggio, and quiet effects timed to the scenes in index.html (a noise swell at each
cut, ticks when form values and tool calls appear, a two-note chime on "9/9 checks passed", a low tone on the
end screen). The mix is gain-matched to -16 LUFS integrated with ffmpeg's ebur128 meter and must keep true
peak at or below -1.5 dBTP.

    cd video && uv run python scripts/make_audio.py      # -> assets/audio/soundtrack.wav (48 kHz, 16-bit stereo)
"""

import json
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy import signal

SR = 48_000
DURATION = 30.0
TARGET_LUFS = -16.0
MAX_TRUE_PEAK = -1.5
OUT = Path(__file__).resolve().parents[1] / "assets/audio/soundtrack.wav"

# Scene starts in index.html (seconds); 12.2 is the second beat inside why.html.
CUTS = [2.4, 5.8, 9.6, 12.2, 15.2, 18.6, 22.4, 26.2]
# form.html: caption at +0.15, corn-price outline at +2.7; agent.html: three tool-call rows at +1.0 + 0.18 i,
# rotation count at +0.6, study search at +0.8.
TICKS = [2.4 + 0.15, 2.4 + 2.7, 15.2 + 0.6, 15.2 + 0.8, 15.2 + 1.0, 15.2 + 1.18, 15.2 + 1.36]
CHIME = 18.6 + 2.1  # solution.html: "9/9 checks passed" stamp
END_TONE = 26.2 + 1.6  # bench.html: repository link appears

# D major colour, one chord per 7.5 s: Dmaj9, Bm7(add11), Gmaj7, A7sus4 resolving into the end.
CHORDS = [
    ("D3", "A3", "C#4", "E4", "F#4"),
    ("B2", "F#3", "A3", "D4", "E4"),
    ("G2", "D3", "F#3", "B3", "D4"),
    ("A2", "E3", "G3", "D4", "E4"),
]
NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}

t = np.arange(int(SR * DURATION)) / SR


def hz(name):
    m = re.fullmatch(r"([A-G]#?)(\d)", name)
    return 440.0 * 2 ** ((NOTE[m[1]] + 12 * (int(m[2]) + 1) - 69) / 12)


def envelope(n, attack, release, sustain=None):
    """Linear-in-dB-ish attack/release window of n samples (raised cosine edges)."""
    env = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    env[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    env[n - r:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r))
    return env


def place(buffer, start, sound):
    i = round(start * SR)
    j = min(len(buffer), i + len(sound))
    buffer[i:j] += sound[: j - i]


def pad():
    out = np.zeros((len(t), 2))
    span = DURATION / len(CHORDS)
    for k, chord in enumerate(CHORDS):
        start = max(0.0, k * span - 1.2)
        length = span + (1.2 if k else 0.0) + 1.8
        n = int(length * SR)
        tt = np.arange(n) / SR
        env = envelope(n, attack=2.0, release=2.2)
        voice = np.zeros((n, 2))
        for note in chord:
            f = hz(note)
            for side, detune in ((0, -0.0025), (1, 0.0025)):
                fd = f * (1 + detune)
                wave_ = (np.sin(2 * np.pi * fd * tt) + 0.35 * np.sin(2 * np.pi * 2 * fd * tt + 0.3)
                         + 0.12 * np.sin(2 * np.pi * 3 * fd * tt + 1.1))
                voice[:, side] += wave_ * (0.9 if f < 150 else 1.0)
        tremolo = 1 + 0.06 * np.sin(2 * np.pi * 0.23 * tt)[:, None]
        segment = voice * env[:, None] * tremolo / len(chord)
        i = int(start * SR)
        j = min(len(out), i + n)
        out[i:j] += segment[: j - i]
    sos = signal.butter(2, 1800, "lowpass", fs=SR, output="sos")
    return signal.sosfilt(sos, out, axis=0)


def arpeggio():
    out = np.zeros((len(t), 2))
    step = 0.3125  # eighth notes at 96 bpm
    span = DURATION / len(CHORDS)
    pattern = [0, 2, 3, 4, 3, 2, 1, 2]
    n = int(1.6 * SR)
    tt = np.arange(n) / SR
    k = 0
    time = 2.4  # enter with the first cut
    while time < DURATION - 2.0:
        chord = CHORDS[min(int(time // span), len(CHORDS) - 1)]
        f = hz(chord[pattern[k % len(pattern)]]) * 2
        pluck = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt / 0.32)
        pluck *= np.minimum(1, tt / 0.004)
        pan = 0.5 + 0.25 * np.sin(k * 0.9)
        place(out[:, 0], time, pluck * (1 - pan))
        place(out[:, 1], time, pluck * pan)
        time += step
        k += 1
    sos = signal.butter(2, 2600, "lowpass", fs=SR, output="sos")
    dry = signal.sosfilt(sos, out, axis=0)
    wet = np.zeros_like(dry)  # ping-pong echo
    for i, (delay, gain) in enumerate([(0.375, 0.35), (0.75, 0.2), (1.125, 0.1)]):
        d = int(delay * SR)
        src, dst = (0, 1) if i % 2 == 0 else (1, 0)
        wet[d:, dst] += gain * dry[:-d, src]
    return dry + wet


def whoosh(rng):
    n = int(0.9 * SR)
    tt = np.arange(n) / SR
    noise = rng.standard_normal((n, 2))
    low = signal.sosfilt(signal.butter(2, [200, 900], "bandpass", fs=SR, output="sos"), noise, axis=0)
    high = signal.sosfilt(signal.butter(2, [700, 3000], "bandpass", fs=SR, output="sos"), noise, axis=0)
    mix = np.clip((tt - 0.2) / 0.45, 0, 1)[:, None]  # sweep upwards into the cut
    env = np.where(tt < 0.55, (tt / 0.55) ** 2, np.exp(-(tt - 0.55) / 0.12))[:, None]
    pan = np.stack([1 - 0.4 * tt / tt[-1], 0.6 + 0.4 * tt / tt[-1]], axis=1)
    return ((1 - mix) * low + mix * high) * env * pan


def tick():
    n = int(0.06 * SR)
    tt = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 2200 * tt) * np.exp(-tt / 0.008) + 0.4 * np.sin(2 * np.pi * 3300 * tt) * np.exp(-tt / 0.004)
    tone *= np.minimum(1, tt / 0.0008)
    return np.stack([tone, tone], axis=1)


def bell(f, length=1.8):
    n = int(length * SR)
    tt = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.7) + 0.3 * np.sin(2 * np.pi * 2.76 * f * tt) * np.exp(-tt / 0.25)
            + 0.15 * np.sin(2 * np.pi * 5.4 * f * tt) * np.exp(-tt / 0.1))
    return tone * np.minimum(1, tt / 0.003)


def chime():
    first, second = bell(hz("A5")), bell(hz("D6"))
    out = np.zeros((len(first) + int(0.16 * SR), 2))
    out[: len(first), 0] += first * 0.9
    out[: len(first), 1] += first * 0.6
    out[int(0.16 * SR):, 0] += second * 0.6
    out[int(0.16 * SR):, 1] += second * 0.9
    return out


def end_tone():
    n = int(2.2 * SR)
    tt = np.arange(n) / SR
    tone = sum(np.sin(2 * np.pi * hz(x) * tt) for x in ("D3", "A3", "D4")) / 3
    env = np.minimum(1, tt / 0.6) * np.exp(-np.maximum(0, tt - 0.6) / 0.9)
    return np.stack([tone * env, tone * env], axis=1)


def measure(path):
    """Integrated loudness (LUFS) and true peak (dBTP) from ffmpeg's ebur128 filter."""
    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True, check=True).stderr
    summary = log[log.rindex("Summary:"):]
    lufs = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary)[1])
    peak = float(re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary)[1])
    return lufs, peak


def write(path, audio):
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    rng = np.random.default_rng(2026)
    music = 0.55 * pad() + 0.16 * arpeggio()
    fx = np.zeros_like(music)
    for cut in CUTS:
        place_start = cut - 0.55
        w = whoosh(rng) * 0.45
        i = int(place_start * SR)
        fx[i:i + len(w)] += w[: len(fx) - i]
    for at in TICKS:
        s = tick() * 0.2
        i = int(at * SR)
        fx[i:i + len(s)] += s
    for at, sound, gain in ((CHIME, chime(), 0.25), (END_TONE, end_tone(), 0.3)):
        i = int(at * SR)
        fx[i:i + len(sound)] += gain * sound[: len(fx) - i]
    mix = music + fx
    fade = np.ones(len(t))
    fade[: int(1.0 * SR)] = np.linspace(0, 1, int(1.0 * SR)) ** 2
    fade[-int(1.5 * SR):] = np.linspace(1, 0, int(1.5 * SR)) ** 2
    mix *= fade[:, None]
    mix /= np.abs(mix).max()
    mix *= 10 ** (-6 / 20)
    write(OUT, mix)
    lufs, _ = measure(OUT)
    mix *= 10 ** ((TARGET_LUFS - lufs) / 20)
    write(OUT, mix)
    lufs, peak = measure(OUT)
    report = {"file": str(OUT.name), "seconds": len(t) / SR, "sample_rate": SR, "integrated_lufs": lufs,
              "true_peak_dbtp": peak}
    print(json.dumps(report))
    if peak > MAX_TRUE_PEAK or abs(lufs - TARGET_LUFS) > 0.5:
        raise SystemExit(f"Loudness out of range: {lufs} LUFS, {peak} dBTP")


if __name__ == "__main__":
    main()
