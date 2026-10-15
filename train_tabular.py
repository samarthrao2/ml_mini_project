"""Step 2: classical models on the feature vectors + the paper's analyses.

 - model comparison (softmax, SVM, LDA, kNN, MLP, LightGBM)   -> results/model_comparison.csv
 - confusion matrix + feature importance of best model        -> results/*.png
 - SVM vs number of PCA components                            -> results/svm_pca.csv
 - PCA 2-D scatter                                            -> results/pca_scatter.png
 - pairwise binary classification (logistic regression)       -> results/pairwise.png
 - era classification using limited feature groups            -> results/era_feature_groups.csv
"""
import argparse
import itertools
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from features import FEATURE_GROUPS, group_of

# eras from the paper; composers outside this map (or "transitional" ones) are skipped in the era experiment
ERA = {"bach": "baroque", "handel": "baroque", "vivaldi": "baroque", "telemann": "baroque",
       "mozart": "classical", "haydn": "classical",
       "brahms": "romantic", "chopin": "romantic", "debussy": "romantic", "mendelssohn": "romantic",
       "liszt": "romantic", "dvorak": "romantic"}
GROUP_NAMES = {"rhy": "rhythm", "mel": "melodic intervals", "vert": "vertical intervals", "tex": "texture",
               "pitch": "pitch", "dyn": "dynamics"}


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return accuracy_score(y, p), pr, rc, f1


def make_models(n_feat, n_classes, fast, seed):
    return {
        "Softmax": LogisticRegression(C=0.1, max_iter=2000),
        "SVM (rbf)": SVC(kernel="rbf", C=5.5, gamma=1.0 / n_feat),
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "k-NN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "MLP (1x150)": MLPClassifier((150,), alpha=1e-3, solver="sgd", momentum=0.9, learning_rate_init=0.04,
                                    batch_size=50, max_iter=200, early_stopping=False, random_state=seed),
        "LightGBM": lgb.LGBMClassifier(n_estimators=150 if fast else 700, reg_lambda=10.0, subsample=0.6,
                                       subsample_freq=1, colsample_bytree=0.5, learning_rate=0.05,
                                       random_state=seed, verbose=-1, n_jobs=-1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/features.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models_dir", default="artifacts")
    ap.add_argument("--fast", action="store_true", help="fewer boosting rounds (quick demo)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    df = pd.read_csv(a.features)
    feat_cols = [c for c in df.columns if c not in ("path", "composer", "split")]
    X = df[feat_cols].to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder().fit(df.composer)
    y = le.transform(df.composer)
    split = df.split.to_numpy()
    tr, dv, te = split == "train", split == "dev", split == "test"

    scaler = StandardScaler().fit(X[tr])
    Xs = np.nan_to_num(scaler.transform(X))
    classes = list(le.classes_)
    print(f"{len(df)} pieces, {len(classes)} composers, {len(feat_cols)} features")

    # ---------- 1. model comparison ----------
    rows, fitted = [], {}
    for name, m in make_models(len(feat_cols), len(classes), a.fast, a.seed).items():
        m.fit(Xs[tr], y[tr])
        fitted[name] = m
        r = {"model": name, "train_acc": accuracy_score(y[tr], m.predict(Xs[tr])),
             "dev_acc": accuracy_score(y[dv], m.predict(Xs[dv]))}
        acc, pr, rc, f1 = metrics(y[te], m.predict(Xs[te]))
        r.update(test_acc=acc, avg_precision=pr, avg_recall=rc, avg_f1=f1)
        rows.append(r)
        print(f"  {name:12s} train {r['train_acc']:.3f}  dev {r['dev_acc']:.3f}  test {acc:.3f}")
    comp = pd.DataFrame(rows).round(3)
    comp.to_csv(out / "model_comparison.csv", index=False)
    print(comp.to_string(index=False))

    best_name = comp.sort_values("dev_acc", ascending=False).iloc[0].model     # pick on dev, report test
    best = fitted[best_name]
    print("best on dev:", best_name)
    joblib.dump({"model": best, "scaler": scaler, "label_encoder": le, "features": feat_cols},
                Path(a.models_dir) / "best_tabular_model.joblib")

    # ---------- 2. confusion matrix ----------
    cm = confusion_matrix(y[te], best.predict(Xs[te]), labels=range(len(classes)))
    plt.figure(figsize=(0.7 * len(classes) + 3, 0.6 * len(classes) + 2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", xticklabels=classes, yticklabels=classes)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"{best_name}: test confusion matrix")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=130); plt.close()

    # ---------- 3. feature importance (LightGBM gain) ----------
    gb = fitted["LightGBM"]
    imp = pd.Series(gb.booster_.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv", header=["gain"])
    top = imp.head(30)[::-1]
    plt.figure(figsize=(8, 8))
    plt.barh(top.index, top.values, color="#3b6ea5")
    plt.title("LightGBM top-30 features (gain)"); plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=130); plt.close()
    grp_gain = imp.groupby(lambda c: GROUP_NAMES.get(group_of(c), group_of(c))).sum()
    print("\nGain share by feature group:\n", (grp_gain / grp_gain.sum()).round(3).sort_values(ascending=False).to_string())

    # ---------- 4. SVM + PCA ----------
    pc_rows = []
    for k in [2, 10, 50, 100, 500, Xs.shape[1]]:
        k = min(k, Xs.shape[1], tr.sum() - 1)
        if k < Xs.shape[1]:
            pca = PCA(k, random_state=a.seed).fit(Xs[tr])
            Z = pca.transform(Xs)
        else:
            Z = Xs
        s = SVC(kernel="rbf", C=5.5, gamma="scale").fit(Z[tr], y[tr])
        pc_rows.append({"n_components": k, "train": s.score(Z[tr], y[tr]), "dev": s.score(Z[dv], y[dv])})
    pd.DataFrame(pc_rows).round(3).drop_duplicates("n_components").to_csv(out / "svm_pca.csv", index=False)

    # ---------- 5. PCA scatter ----------
    Z2 = PCA(2, random_state=a.seed).fit(Xs[tr]).transform(Xs)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Z2[:, 0], y=Z2[:, 1], hue=df.composer, s=18, alpha=0.7)
    plt.title("First two principal components"); plt.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(out / "pca_scatter.png", dpi=130); plt.close()

    # ---------- 6. pairwise binary classification ----------
    n = len(classes)
    M = np.full((n, n), np.nan)
    trv = tr | dv
    for i, j in itertools.combinations(range(n), 2):
        sel = np.isin(y, [i, j])
        lr = LogisticRegression(C=0.1, max_iter=1000).fit(Xs[sel & trv], y[sel & trv] == j)
        M[i, j] = M[j, i] = lr.score(Xs[sel & te], y[sel & te] == j)
    plt.figure(figsize=(0.7 * n + 3, 0.6 * n + 2))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="viridis", xticklabels=classes, yticklabels=classes)
    plt.title(f"Pairwise logistic regression test accuracy (mean {np.nanmean(M):.3f})")
    plt.tight_layout(); plt.savefig(out / "pairwise.png", dpi=130); plt.close()
    print(f"\nmean pairwise accuracy: {np.nanmean(M):.3f}")

    # ---------- 7. era classification w/ restricted feature groups ----------
    era = df.composer.map(ERA)
    if era.nunique() >= 2:
        m_era = era.notna().to_numpy()
        era_le = LabelEncoder().fit(era[m_era])
        ye = np.full(len(df), -1)
        ye[m_era] = era_le.transform(era[m_era])
        er = []
        for g in FEATURE_GROUPS + ["ALL"]:
            cols = [i for i, c in enumerate(feat_cols) if g == "ALL" or group_of(c) == g]
            clf = lgb.LGBMClassifier(n_estimators=100 if a.fast else 300, reg_lambda=10.0, subsample=0.6,
                                     subsample_freq=1, learning_rate=0.05, random_state=a.seed, verbose=-1)
            clf.fit(Xs[np.ix_(tr & m_era, cols)], ye[tr & m_era])
            er.append({"feature_group": GROUP_NAMES.get(g, g), "n_features": len(cols),
                       "era_test_acc": clf.score(Xs[np.ix_(te & m_era, cols)], ye[te & m_era])})
        era_df = pd.DataFrame(er).round(3).sort_values("era_test_acc", ascending=False)
        era_df.to_csv(out / "era_feature_groups.csv", index=False)
        print("\nEra classification by feature group:\n", era_df.to_string(index=False))
    else:
        print("\n(skipping era experiment: folder names don't match known composers)")

    print(f"\nall outputs in {out}/")


if __name__ == "__main__":
    main()
