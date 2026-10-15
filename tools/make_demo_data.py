"""Generate SYNTHETIC MIDI files so the pipeline can be smoke-tested without real data.

These are NOT real compositions; each fake 'composer' just has a different random-walk
style (scale, leap size, tempo, texture, instrument). Accuracy on this data says nothing
about real classical music - it only proves the code runs end to end.
"""
import argparse
import random
import zlib
from pathlib import Path

import mido

STYLES = {
    # name: (scale, max_step, tempo, voices, program, vel_mean, note_len_choices)
    "baroque_a": ([0, 2, 4, 5, 7, 9, 11], 2, 110, 2, 6, 70, [0.25, 0.25, 0.5]),
    "baroque_b": ([0, 2, 3, 5, 7, 8, 10], 3, 130, 3, 19, 75, [0.25, 0.5]),
    "classical_a": ([0, 2, 4, 5, 7, 9, 11], 4, 100, 2, 0, 80, [0.5, 1.0, 0.5]),
    "classical_b": ([0, 2, 4, 7, 9], 5, 120, 3, 40, 85, [0.5, 1.0]),
    "romantic_a": (list(range(12)), 7, 70, 4, 0, 60, [1.0, 2.0, 0.5]),
    "romantic_b": ([0, 1, 3, 4, 6, 8, 10], 9, 80, 5, 48, 95, [1.0, 1.5, 0.5]),
}


def make_piece(style, seed, bars=40):
    scale, max_step, tempo, voices, program, vel, lens = style
    rng = random.Random(seed)
    mid = mido.MidiFile(ticks_per_beat=480)
    for v in range(voices):
        tr = mido.MidiTrack()
        mid.tracks.append(tr)
        if v == 0:
            tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo + rng.randint(-10, 10))))
        tr.append(mido.Message("program_change", channel=v, program=program, time=0))
        pitch = 48 + 12 * (voices - v) // 2 + rng.choice(scale)
        for _ in range(bars * 4):
            step = rng.choice([-1, 1]) * rng.randint(0, max_step)
            pitch = min(max(pitch + step, 36), 90)
            while pitch % 12 not in scale:
                pitch += 1
            length = int(rng.choice(lens) * 480)
            v_ = max(1, min(127, int(rng.gauss(vel, 12))))
            tr.append(mido.Message("note_on", channel=v, note=pitch, velocity=v_, time=0))
            tr.append(mido.Message("note_off", channel=v, note=pitch, velocity=0, time=length))
    return mid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_demo")
    ap.add_argument("--per_class", type=int, default=60)
    a = ap.parse_args()
    for name, style in STYLES.items():
        d = Path(a.out) / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(a.per_class):
            make_piece(style, zlib.crc32(f"{name}{i}".encode())).save(d / f"{name}_{i:03d}.mid")
    print(f"wrote {len(STYLES) * a.per_class} synthetic files to {a.out}/")
