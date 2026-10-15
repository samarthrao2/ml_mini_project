"""MIDI parsing used by the feature extractor."""
from dataclasses import dataclass

import mido
import numpy as np


@dataclass
class Notes:
    onset: np.ndarray      # seconds
    offset: np.ndarray     # seconds
    pitch: np.ndarray      # MIDI note number
    velocity: np.ndarray
    channel: np.ndarray
    program: np.ndarray    # GM program (0-127) of the channel when the note started
    tempos: np.ndarray     # tempo values in BPM seen in the file

    def __len__(self):
        return len(self.pitch)


def load_notes(path, skip_drums=True):
    """Parse a MIDI file into note events. Returns None if unreadable/empty."""
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None

    t = 0.0
    program = {}
    active = {}  # (channel, pitch) -> list of (onset, velocity, program)
    rows = []
    tempos = []
    try:
        for msg in mid:  # iterating a MidiFile yields delta times in seconds
            t += msg.time
            if msg.type == "set_tempo":
                tempos.append(mido.tempo2bpm(msg.tempo))
            elif msg.type == "program_change":
                program[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append(
                    (t, msg.velocity, program.get(msg.channel, 0)))
            elif msg.type in ("note_off", "note_on"):
                stack = active.get((msg.channel, msg.note))
                if stack:
                    on, vel, prog = stack.pop(0)
                    rows.append((on, t, msg.note, vel, msg.channel, prog))
    except Exception:
        return None

    if skip_drums:
        rows = [r for r in rows if r[4] != 9]
    if len(rows) < 10:
        return None
    a = np.array(rows, dtype=float)
    a = a[np.argsort(a[:, 0], kind="stable")]
    return Notes(a[:, 0], a[:, 1], a[:, 2].astype(int), a[:, 3].astype(int),
                 a[:, 4].astype(int), a[:, 5].astype(int),
                 np.array(tempos if tempos else [120.0]))
