"""Hand-built music-theory feature vector (a lightweight stand-in for jSymbolic).

Features are grouped (rhythm / melodic / vertical / texture / pitch / dynamics) so the
"which feature group is most distinctive" experiment from the paper can be reproduced.
"""
import numpy as np

from midi_utils import load_notes

DISSONANT = {1, 2, 6, 10, 11}                       # semitone classes: 2nds, tritone, 7ths
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _norm(h):
    s = h.sum()
    return h / s if s > 0 else h


def _stats(x, prefix):
    if len(x) == 0:
        x = np.zeros(1)
    return {f"{prefix}_mean": x.mean(), f"{prefix}_std": x.std(),
            f"{prefix}_min": x.min(), f"{prefix}_max": x.max(),
            f"{prefix}_p25": np.percentile(x, 25), f"{prefix}_p75": np.percentile(x, 75)}


def extract(path):
    """Return {feature_name: value} dict, or None if the file can't be used."""
    n = load_notes(path)
    if n is None:
        return None
    f = {}
    dur = max(n.offset.max() - n.onset.min(), 1e-3)

    # ---------------- pitch ----------------
    ph = np.bincount(n.pitch, minlength=128).astype(float)
    pc = np.bincount(n.pitch % 12, minlength=12).astype(float)
    for i in range(24, 108, 2):                       # coarse 2-semitone pitch histogram
        f[f"pitch:hist_{i}"] = _norm(ph)[i:i + 2].sum()
    # transposition-invariant pitch class histogram (rotate to estimated tonic)
    pcn = _norm(pc)
    scores = [np.corrcoef(np.roll(MAJOR, k), pcn)[0, 1] if pcn.std() > 0 else 0 for k in range(12)] + \
             [np.corrcoef(np.roll(MINOR, k), pcn)[0, 1] if pcn.std() > 0 else 0 for k in range(12)]
    best = int(np.argmax(scores))
    tonic, is_minor = best % 12, best >= 12
    rot = np.roll(pcn, -tonic)
    for i in range(12):
        f[f"pitch:pc_rel_{i}"] = rot[i]
    f["pitch:is_minor"] = float(is_minor)
    f["pitch:key_strength"] = max(scores)
    f["pitch:chromaticity"] = float((pcn > 0.02).sum())
    for k, v in _stats(n.pitch.astype(float), "pitch:val").items():
        f[k] = v
    f["pitch:range"] = n.pitch.max() - n.pitch.min()

    # ---------------- dynamics ----------------
    for k, v in _stats(n.velocity.astype(float), "dyn:vel").items():
        f[k] = v
    vd = np.abs(np.diff(n.velocity.astype(float)))
    f["dyn:avg_change"] = vd.mean() if len(vd) else 0
    f["dyn:distinct_vel"] = len(np.unique(n.velocity))

    # ---------------- rhythm ----------------
    durs = np.clip(n.offset - n.onset, 0, 30)
    for k, v in _stats(np.log2(durs + 1e-2), "rhy:logdur").items():
        f[k] = v
    onsets = np.unique(np.round(n.onset, 3))
    ioi = np.diff(onsets)
    ioi = ioi[ioi > 0]
    for k, v in _stats(np.log2(ioi + 1e-3), "rhy:logioi").items():
        f[k] = v
    ioi_hist, _ = np.histogram(np.log2(ioi + 1e-3) if len(ioi) else [0], bins=np.linspace(-6, 3, 13))
    for i, v in enumerate(_norm(ioi_hist.astype(float))):
        f[f"rhy:ioi_bin_{i}"] = v
    f["rhy:note_density"] = len(n) / dur
    f["rhy:onset_density"] = len(onsets) / dur
    win = np.floor((n.onset - n.onset.min()) / 5.0).astype(int)
    dens = np.bincount(win).astype(float) / 5.0
    f["rhy:density_std"] = dens.std()
    f["rhy:density_cv"] = dens.std() / (dens.mean() + 1e-9)
    f["rhy:tempo_mean"] = n.tempos.mean()
    f["rhy:tempo_std"] = n.tempos.std()
    f["rhy:tempo_changes"] = len(n.tempos)
    f["rhy:duration_sec"] = np.log(dur)
    f["rhy:rest_fraction"] = float(np.clip(1 - (n.offset - n.onset).sum() / (dur * max(len(np.unique(n.channel)), 1)), 0, 1))
    f["rhy:ioi_regularity"] = float(np.mean(np.abs(ioi - np.median(ioi)) < 0.02)) if len(ioi) else 0

    # ---------------- melodic intervals (per channel, note-to-note) ----------------
    mel = []
    for ch in np.unique(n.channel):
        p = n.pitch[n.channel == ch]
        # one note per onset: take the top note of chords so the line stays monophonic
        o = n.onset[n.channel == ch]
        _, first = np.unique(np.round(o, 3), return_index=True)
        top = np.array([p[(np.round(o, 3) == t)].max() for t in np.unique(np.round(o, 3))]) if len(first) < 5000 else p[first]
        if len(top) > 1:
            mel.append(np.diff(top))
    mel = np.concatenate(mel) if mel else np.zeros(1)
    am = np.abs(mel).astype(int)
    mh = np.bincount(np.minimum(am, 24), minlength=25).astype(float)
    for i, v in enumerate(_norm(mh)):
        f[f"mel:interval_{i}"] = v
    f["mel:ascending_frac"] = float((mel > 0).mean())
    f["mel:descending_frac"] = float((mel < 0).mean())
    f["mel:repeat_frac"] = float((mel == 0).mean())
    f["mel:step_frac"] = float(((am >= 1) & (am <= 2)).mean())
    f["mel:leap_frac"] = float((am >= 5).mean())
    f["mel:large_leap_frac"] = float((am >= 12).mean())
    f["mel:dissonant_frac"] = float(np.isin(am % 12, list(DISSONANT)).mean())
    f["mel:mean_abs"] = am.mean()
    f["mel:std"] = mel.std()
    f["mel:direction_changes"] = float((np.diff(np.sign(mel[mel != 0])) != 0).mean()) if (mel != 0).sum() > 2 else 0

    # ---------------- vertical intervals & texture (simultaneity analysis) ----------------
    grid = np.round(n.onset, 2)
    uniq, inv = np.unique(grid, return_inverse=True)
    vert = np.zeros(12)
    n_dis = n_con = 0
    simult = []
    low_notes, high_notes = [], []
    max_groups = 4000
    step = max(1, len(uniq) // max_groups)
    for g in range(0, len(uniq), step):
        t = uniq[g]
        sounding = np.unique(n.pitch[(n.onset <= t + 1e-6) & (n.offset > t + 1e-6)]) if len(n) < 20000 else \
            np.unique(n.pitch[inv == g])
        simult.append(len(sounding))
        if len(sounding) >= 2:
            diffs = (sounding[None, :] - sounding[:, None])[np.triu_indices(len(sounding), 1)] % 12
            vert += np.bincount(diffs, minlength=12)
            d = np.isin(diffs, list(DISSONANT)).sum()
            n_dis += d
            n_con += len(diffs) - d
        low_notes.append(sounding.min() if len(sounding) else -1)
        high_notes.append(sounding.max() if len(sounding) else -1)
    for i, v in enumerate(_norm(vert)):
        f[f"vert:wrapped_{i}"] = v
    f["vert:dissonance_ratio"] = n_dis / (n_con + 1)
    f["vert:dissonant_frac"] = n_dis / max(n_dis + n_con, 1)
    f["vert:perfect_consonance_frac"] = _norm(vert)[[0, 5, 7]].sum()
    f["vert:thirds_sixths_frac"] = _norm(vert)[[3, 4, 8, 9]].sum()
    f["vert:tritone_frac"] = _norm(vert)[6]
    # parallel perfect fifths/octaves between lowest & highest sounding notes
    lo, hi = np.array(low_notes), np.array(high_notes)
    ok = (lo >= 0) & (hi > lo)
    ok = ok[1:] & ok[:-1]
    if ok.any():
        ic = (hi - lo) % 12
        same_ic = (ic[1:] == ic[:-1]) & np.isin(ic[1:], [0, 7])
        moved = (lo[1:] != lo[:-1]) & (np.sign(lo[1:] - lo[:-1]) == np.sign(hi[1:] - hi[:-1]))
        f["vert:parallel_5_8"] = float((same_ic & moved & ok).sum() / ok.sum())
    else:
        f["vert:parallel_5_8"] = 0.0
    f["vert:outer_interval_mean"] = float((hi - lo)[(lo >= 0)].mean()) if (lo >= 0).any() else 0

    simult = np.array(simult, dtype=float)
    f["tex:polyphony_mean"] = simult.mean()
    f["tex:polyphony_max"] = simult.max()
    f["tex:polyphony_std"] = simult.std()
    f["tex:frac_monophonic"] = float((simult <= 1).mean())
    f["tex:frac_dense"] = float((simult >= 4).mean())
    f["tex:n_channels"] = len(np.unique(n.channel))
    f["tex:n_programs"] = len(np.unique(n.program))
    fam = np.bincount(n.program // 8, minlength=16).astype(float)
    for i, v in enumerate(_norm(fam)):
        f[f"tex:instr_family_{i}"] = v
    f["tex:piano_frac"] = _norm(fam)[0]
    f["tex:strings_frac"] = _norm(fam)[4]
    ch_counts = np.bincount(n.channel, minlength=16).astype(float)
    f["tex:channel_balance"] = float(_norm(ch_counts).max())
    return f


FEATURE_GROUPS = ["pitch", "dyn", "rhy", "mel", "vert", "tex"]


def group_of(feature_name):
    return feature_name.split(":")[0]
