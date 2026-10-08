# 🎬 MovieSphere — Movie Recommendation System

A machine-learning movie recommender built on the MovieLens *ml-latest-small* dataset
(100,836 ratings · 610 users · 9,724 movies).
It predicts the rating a user would give to a movie they have not seen, recommends the
highest-scoring movies, finds similar movies, and handles brand-new users (cold start).

**Techniques (as required by the course brief):** K-Nearest Neighbours · Cosine Similarity ·
Collaborative Filtering (item-based and user-based).
**Extras:** content-based filtering, matrix factorisation (ALS), a learnt hybrid blend,
Top-N ranking metrics and an interactive Streamlit app.

## Quick start
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

python train.py                # tune + evaluate everything (~5 min) -> results/ and assets/img/
streamlit run app.py           # launch the web app
```
`train.py --quick` runs tiny grids in about 3 minutes if you only want a smoke test.
The app works even before `train.py` (it falls back to default settings) but the *Model Lab* tab needs `results/results.json`.

## Project layout
| File | Purpose |
|---|---|
| `recsys.py` | cleaning, features, all models, evaluation metrics |
| `train.py` | split → tune → test → cross-validate → rank → error analysis → figures |
| `engine.py` | fits the final models on all ratings and serves the app |
| `app.py` | Streamlit interface (6 tabs) |
| `plots.py` | all charts (light for the report, dark for the app) |
| `tests/` | unit tests (`python tests/test_recsys.py`) and an app smoke-test |
| `data/` | `ratings.csv`, `movies.csv` (MovieLens) |
| `results/results.json` | every number quoted in the report |

## Method in one paragraph
Ratings are split 70 / 10 / 20 (train / validation / test, seed 42). Each model predicts
`baseline(u,i) + weighted average of neighbour residuals`, where the baseline is
`μ + b_u + b_i` and neighbours are found with cosine similarity on mean-centred ratings
(shrunk when two movies share few raters). Hyper-parameters are chosen by grid search on the validation
set only; the test set is touched once. The models are also compared with 5-fold cross-validation.

## Tests
```bash
python tests/test_recsys.py        # 8 unit tests of the engine
python tests/run_app_stub.py       # runs app.py against a fake Streamlit (no browser needed)
```

Dataset: F. M. Harper and J. A. Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4).
