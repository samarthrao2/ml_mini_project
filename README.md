# Composer Classification from Symbolic (MIDI) Music

UE24CS352A Machine Learning mini-project. Predicts the composer of a classical MIDI piece using **six
machine-learning models** on 186 hand-built music-theory features. Based on *"Analysis and Classification of
Symbolic Western Classical Music by Composer"* (Jain, Smit, Yngesjö, Stanford CS229, 2019).

**Models:** softmax regression, SVM (RBF), LDA, k-NN, MLP (1 hidden layer), LightGBM.

**Analyses:** model comparison, confusion matrix, LightGBM feature importance and gain share by feature group,
SVM vs. number of PCA components, PCA scatter, pairwise composer classification, era classification from
restricted feature groups.

## Repository layout
| Path | What it is |
|---|---|
| `prepare_data.py` | Step 1: read MIDI, extract 186 features, make the stratified 60/20/20 split |
| `train_tabular.py` | Step 2: train and compare the six models, run the analyses |
| `predict.py` | Step 3: name the composer of one MIDI file |
| `features.py`, `midi_utils.py` | Feature extraction and MIDI parsing used by the scripts above |
| `Composer_Classification_Report.pdf` | One-page write-up |
| `Composer_Classification_Slides.pptx` | Presentation slides |
| `data/` | 785 MAESTRO piano pieces, one sub-folder per composer |
| `results/` | Output tables and plots from `train_tabular.py` |
| `artifacts/` | `features.csv` and the saved best model |
| `tools/make_demo_data.py` | Optional: generates fake MIDI for a quick smoke test |

## Setup
```
pip install -r requirements.txt
```
Python 3.11 was used.

## Run
Run everything from this folder.
```
python prepare_data.py --data data      # ~4 min: features + split -> artifacts/features.csv
python train_tabular.py                 # ~20 s: six models + analyses -> results/   (--fast for fewer boosting rounds)
python predict.py data/bach/<file>.midi # top-5 composer guesses with probabilities
```

## Data
`data/` holds MIDI files from the [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) dataset,
one folder per composer (`data/bach/*.midi`, ...). One performance was kept per piece, and two-composer
arrangements were removed. Composers with fewer than 20 pieces (`--min_per_class`) are excluded, leaving
13 composers and 725 pieces.

To test the code quickly on synthetic data:
```
python tools/make_demo_data.py --out data_demo
python prepare_data.py --data data_demo --out artifacts_demo
python train_tabular.py --features artifacts_demo/features.csv --out results_demo --models_dir artifacts_demo --fast
```
Synthetic accuracy (~100%) only shows the code runs; it says nothing about real music.

## Results (MAESTRO, 145 test pieces)
| Model | Test accuracy | Macro-F1 |
|---|---|---|
| LightGBM | 0.869 | 0.800 |
| SVM (RBF) | 0.855 | 0.807 |
| Softmax | 0.855 | 0.790 |
| MLP | 0.841 | 0.784 |
| LDA | 0.814 | 0.759 |
| k-NN | 0.759 | 0.660 |

Scores are probably a little optimistic: movements of one work can fall in different splits, and no
cross-validation was done. Dataset is piano-only, so texture features carry little information.
