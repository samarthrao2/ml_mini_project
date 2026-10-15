"""Classify a single MIDI file with the saved tabular model:  python predict.py song.mid"""
import sys

import joblib
import numpy as np

from features import extract

bundle = joblib.load("artifacts/best_tabular_model.joblib")
f = extract(sys.argv[1])
if f is None:
    sys.exit("could not parse MIDI file")
x = np.array([[f.get(c, 0.0) for c in bundle["features"]]])
x = np.nan_to_num(bundle["scaler"].transform(x))
m = bundle["model"]
if hasattr(m, "predict_proba"):
    p = m.predict_proba(x)[0]
    for i in np.argsort(p)[::-1][:5]:
        print(f"{bundle['label_encoder'].classes_[i]:15s} {p[i]:.3f}")
else:
    print(bundle["label_encoder"].inverse_transform(m.predict(x))[0])
