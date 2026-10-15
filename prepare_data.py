"""Step 1: scan data/<composer>/**/*.mid, dedupe, extract features, make a 60/20/20 split.

Outputs (in --out, default 'artifacts/'):
    features.csv   one row per usable file: path, composer, split, then all features
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from features import extract


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _job(path, composer):
    try:
        f = extract(path)
    except Exception:
        f = None
    if f is None:
        return None
    return {"path": str(path), "composer": composer, **f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with one sub-folder per composer")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--min_per_class", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = [(p, p.relative_to(a.data).parts[0].lower())
             for p in sorted(Path(a.data).rglob("*"))
             if p.suffix.lower() in (".mid", ".midi")]
    print(f"found {len(files)} MIDI files")

    seen, unique = set(), []
    for p, c in tqdm(files, desc="dedupe"):
        h = md5(p)
        if h not in seen:
            seen.add(h)
            unique.append((p, c))
    print(f"{len(unique)} after removing byte-identical duplicates")

    rows = Parallel(n_jobs=a.jobs)(delayed(_job)(p, c) for p, c in tqdm(unique, desc="features"))
    df = pd.DataFrame([r for r in rows if r])
    print(f"{len(df)} files parsed successfully")

    counts = df.composer.value_counts()
    keep = counts[counts >= a.min_per_class].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print("dropping composers with too few pieces:", dropped)
    df = df[df.composer.isin(keep)].reset_index(drop=True)

    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=df.composer, random_state=a.seed)
    dev, te = train_test_split(rest, test_size=0.5, stratify=df.composer.iloc[rest], random_state=a.seed)
    df["split"] = "train"
    df.loc[dev, "split"] = "dev"
    df.loc[te, "split"] = "test"

    meta = ["path", "composer", "split"]
    df = df[meta + [c for c in df.columns if c not in meta]].fillna(0)
    Path(a.out).mkdir(exist_ok=True)
    df.to_csv(Path(a.out) / "features.csv", index=False)
    print(df.composer.value_counts().to_string())
    print(f"saved {Path(a.out) / 'features.csv'}  ({df.shape[1] - 3} features)")


if __name__ == "__main__":
    main()
