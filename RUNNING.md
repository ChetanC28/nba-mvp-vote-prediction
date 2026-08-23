# Running this project

Everything here runs on CPU. No GPU, no API keys, no external data downloads —
the dataset is committed to the repo.

## Requirements

- **Python 3.11** (developed on 3.11.13)
- ~2 GB RAM free; the largest single step holds 17,976 × 24 floats plus five
  fitted tree ensembles
- No GPU

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is deliberately unpinned. The versions this was last verified
against, in August 2026:

| Package | Version |
|---|---|
| scikit-learn | 1.8.0 |
| pandas | 2.3.3 |
| numpy | 2.3.5 |
| scipy | 1.17.0 |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| catboost | 1.2.10 |
| matplotlib | 3.10.8 |
| streamlit | ≥1.30 |

## Data

Both CSVs are committed — nothing to download.

| File | Size | What it is |
|---|---|---|
| `cleaned_data.csv` | 3.9 MB | 17,976 player-seasons, 1980–2021, 45 columns. Pre-cleaned release of the Kaggle *NBA Stats since 1980* dataset (originally scraped from Basketball Reference). Contains the `Pts Won` target. |
| `api_data.csv` | 70 KB | 582 players for the 2025–26 season, pulled via the `nba_api` package and manually aligned to the historical schema. **No ground-truth MVP votes** — voting had not concluded. Inference/demo only. |

The `nba_api` pull was done by hand and its output committed; there is no
script in this repo that regenerates `api_data.csv`. Re-pulling it means
redoing the schema alignment described in §2.2 of the report: reconstruct
`PS/G = PTS` and `PA/G = PTS - PLUS_MINUS`, merge team stats onto the player
table via `TEAM_ID`, and rename headers to the historical dataset's names.

## Ordering

The three entry points are independent — none consumes another's output. Run
whichever you want.

### 1. Streamlit dashboard

```bash
streamlit run app.py          # must be run from the repo root
```

**Cold start trains nine models before the first page render** — expect roughly
**3–6 minutes** of blank screen on first load. Results are cached by
`@st.cache_resource`, so later interactions are instant *until* you restart the
process or edit `app.py`.

One slow path to know about: on the **New Predictions** tab, choosing
`Stacked Ensemble` refits the entire stack (5-fold GroupKFold × 3 base learners,
then the Ridge meta-model) on the full dataset on every button press — about
**2–4 minutes per click**. Every other model on that tab refits a single
estimator and returns in seconds.

### 2. Benchmark the adaptive weighted ensemble

```bash
python run_adaptive_benchmark.py      # from the repo root
```

Runtime **~10–15 minutes** on first run — it fits 5 base learners × 5 folds,
then runs SLSQP weight optimisation with 30 random restarts, three separate
times (global, high-agreement, low-agreement). Each objective evaluation sweeps
all 40 training seasons, so the optimisation is the slow half, not the fitting.

The out-of-fold predictions are cached to `results/oof_cache.npz` (~700 KB, not
committed) on the first run, so **subsequent runs finish in about a minute**.
Delete that file to force a full refit.

Writes `results/model_comparison.csv` and `figures/model_comparison.png`, and
prints the gating diagnostic and base-learner weight analysis quoted in the
README.

### 3. Notebooks

```bash
jupyter lab
```

- `notebooks/01_baseline_models.ipynb` — data exploration and five standalone
  tree models on an earlier 2017–21 split
- `notebooks/02_stacked_ensemble.ipynb` — the pipeline reported in the paper:
  season-aware split, nine models, stacking, error analysis

Both take **5–10 minutes** to execute end to end (CatBoost and the
GroupKFold OOF loops dominate).

## Known rough edges

Things that will bite someone running this fresh. None are silently worked
around — they're listed so you know what you're hitting.

1. **The notebooks read `cleaned_data.csv` from the current working directory,
   but they live in `notebooks/`.** Launch Jupyter from the repo root and they
   resolve. If you `cd notebooks` first, cell 3 raises `FileNotFoundError`.
   Same for `app.py` and `run_adaptive_benchmark.py` — all paths are relative
   to the repo root.

2. **`requirements.txt` was missing `scipy` and `seaborn`.** Both are now
   listed. `scipy` powers the SLSQP weight optimiser in
   `adaptive_ensemble.py`; `seaborn` is imported by notebook 01. `scipy` would
   have arrived transitively via scikit-learn, but `seaborn` would not — the
   notebook hard-failed on a clean install before this was fixed.

3. **Notebook 01 opens with `!pip install` cells.** Harmless if you already
   installed from `requirements.txt`, but they will hit the network and can
   upgrade packages out from under your venv. Skip cell 0.

4. **Unpinned dependencies.** `pandas` 3.0 changed enough that the notebooks'
   stored outputs show `str` dtypes where older pandas showed `object`. Nothing
   breaks, but re-running will not reproduce byte-identical output cells.

5. **`app.py` retrains from scratch rather than loading saved models.** There is
   no persistence layer — every cold start re-fits everything. Fine for a demo,
   not for a deployment. `saved_models/` is gitignored for the same reason: the
   pickles are ~24 MB and regenerable.

6. **The held-out test set is two seasons.** `Top-1 Accuracy` therefore only
   takes the values 0, 0.5, or 1.0 — a single season flipping moves it by 0.5.
   Treat the ranking metrics as illustrative, not as a precise estimate. See
   the findings section of the README.

7. **The weight optimiser is not seeded, so its weights vary between runs.**
   `AdaptiveWeightedEnsemble._optimize_weights` draws 30 restart points from
   `np.random.dirichlet` without a fixed seed. Across two consecutive runs the
   high-agreement weights came out `[0.455, 0.395, 0.05, 0.05, 0.05]` and then
   `[0.421, 0.429, 0.05, 0.05, 0.05]`. Final test RMSE was stable to four
   decimal places (32.2055 both times), so the conclusions hold, but the exact
   weight vector printed will not match this document. Seeding the RNG in
   `_optimize_weights` would make it reproducible; the algorithm is left as
   originally written.

## Repo layout

```
.
├── app.py                        # Streamlit dashboard (trains on startup)
├── adaptive_ensemble.py          # from-scratch adaptive weighted ensemble
├── run_adaptive_benchmark.py     # benchmarks it on the report's split
├── cleaned_data.csv              # 17,976 player-seasons, 1980–2021
├── api_data.csv                  # 2025–26 season, no ground truth
├── notebooks/
│   ├── 01_baseline_models.ipynb
│   └── 02_stacked_ensemble.ipynb
├── figures/
└── results/
```
