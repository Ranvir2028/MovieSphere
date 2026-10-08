"""
recsys.py  -  MovieSphere core engine
=====================================

Everything that is "machine learning" in the project lives in this file:

    1. Data loading and cleaning
    2. Content features (genres, decade, title words)
    3. Recommendation models
         GlobalMean            - the dumbest possible baseline
         BiasBaseline          - mu + user bias + movie bias
         ItemKNN               - item-based collaborative filtering (cosine)
         UserKNN               - user-based collaborative filtering (cosine)
         ContentKNN            - content-based filtering (cosine on features)
         MatrixFactorALS       - latent-factor model (bonus, trained by ALS)
         Hybrid                - non-negative blend of the models above
    4. Evaluation helpers (regression, classification and ranking metrics)

All models share one tiny interface so that train.py and the web app can treat
them identically:

    model.fit(train)                              -> self
    model.predict(u, i)                           -> ratings for (user, item) pairs
    model.user_scores(u, targets=None)            -> scores for an existing user
    model.profile_scores(items, ratings, targets) -> scores for a brand-new user
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

SEED = 42
LIKE_THRESHOLD = 3.5          # a rating >= 3.5 (the global mean) counts as "liked"
RELEVANT_THRESHOLD = 4.0      # a rating >= 4.0 counts as relevant for top-N ranking
MIN_RATING, MAX_RATING = 0.5, 5.0
EPS = 1e-9


# =============================================================================
# 1. DATA
# =============================================================================
@dataclass
class Ratings:
    """A set of ratings stored as three parallel integer/float arrays."""
    u: np.ndarray          # user index   (0 .. n_users-1)
    i: np.ndarray          # item index   (0 .. n_items-1)
    r: np.ndarray          # rating       (0.5 .. 5.0)
    n_users: int
    n_items: int

    def __len__(self):
        return len(self.r)

    def take(self, idx) -> "Ratings":
        return Ratings(self.u[idx], self.i[idx], self.r[idx], self.n_users, self.n_items)


@dataclass
class Dataset:
    ratings: Ratings
    movies: pd.DataFrame          # one row per item index, aligned with Ratings.i
    user_ids: np.ndarray          # user index -> original userId
    clean_report: dict


def load_data(data_dir: str = "data") -> Dataset:
    """Load, clean and index the MovieLens files (see report, Section 4)."""
    raw_r = pd.read_csv(f"{data_dir}/ratings.csv")
    raw_m = pd.read_csv(f"{data_dir}/movies.csv")
    rep = {"raw_ratings": len(raw_r), "raw_movies": len(raw_m)}

    # --- invalid / duplicate ratings ----------------------------------------
    ok = raw_r["rating"].between(MIN_RATING, MAX_RATING)
    rep["invalid_ratings_removed"] = int((~ok).sum())
    r = raw_r[ok].copy()
    before = len(r)
    r = r.sort_values("timestamp").drop_duplicates(["userId", "movieId"], keep="last")
    rep["duplicate_pairs_removed"] = before - len(r)
    r = r[r["movieId"].isin(raw_m["movieId"])]
    rep["orphan_ratings_removed"] = before - rep["duplicate_pairs_removed"] - len(r)

    # --- movies: keep only rated movies, parse year, clean title -------------
    m = raw_m[raw_m["movieId"].isin(r["movieId"])].copy().sort_values("movieId")
    m = m.reset_index(drop=True)
    year = m["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    rep["titles_without_year"] = int(year.isna().sum())
    m["year"] = year.fillna(year.median()).astype(int)           # median imputation
    m["clean_title"] = (m["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
                        .str.replace(r"\s*\(.*?\)\s*$", "", regex=True))
    m["genre_list"] = m["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|"))
    rep["movies_without_genre"] = int((m["genre_list"].str.len() == 0).sum())

    # --- integer indexing ----------------------------------------------------
    user_ids = np.sort(r["userId"].unique())
    umap = {v: k for k, v in enumerate(user_ids)}
    imap = {v: k for k, v in enumerate(m["movieId"].values)}
    ratings = Ratings(r["userId"].map(umap).values.astype(np.int32),
                      r["movieId"].map(imap).values.astype(np.int32),
                      r["rating"].values.astype(np.float64),
                      len(user_ids), len(m))
    rep.update(clean_ratings=len(ratings), clean_users=len(user_ids), clean_movies=len(m))
    return Dataset(ratings, m, user_ids, rep)


def train_test_split(rt: Ratings, test_size: float = 0.2, seed: int = SEED):
    """Random split of individual ratings (fixed seed => reproducible)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rt))
    n_test = int(round(len(rt) * test_size))
    return rt.take(np.sort(perm[n_test:])), rt.take(np.sort(perm[:n_test]))


def kfold_indices(n: int, k: int = 5, seed: int = SEED):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for f in range(k):
        val = np.sort(folds[f])
        tr = np.sort(np.concatenate([folds[j] for j in range(k) if j != f]))
        yield tr, val


# =============================================================================
# 2. CONTENT FEATURES
# =============================================================================
def build_content_features(movies: pd.DataFrame, title_weight: float = 0.5,
                           decade_weight: float = 0.7):
    """Multi-hot genres + one-hot decade + TF-IDF title words -> L2-normalised matrix.
    Returns (matrix, feature_names, feature_groups)."""
    all_genres = sorted({g for gl in movies["genre_list"] for g in gl})
    G = np.zeros((len(movies), len(all_genres)), dtype=np.float32)
    gi = {g: k for k, g in enumerate(all_genres)}
    for row, gl in enumerate(movies["genre_list"]):
        for g in gl:
            G[row, gi[g]] = 1.0

    decades = ((movies["year"].values // 10) * 10).clip(1910, 2010)
    dec_vals = sorted(set(decades))
    D = np.zeros((len(movies), len(dec_vals)), dtype=np.float32)
    for k, d in enumerate(dec_vals):
        D[decades == d, k] = decade_weight

    tf = TfidfVectorizer(token_pattern=r"[A-Za-z]{3,}", stop_words="english",
                         min_df=2, max_features=800, lowercase=True)
    T = tf.fit_transform(movies["clean_title"]).toarray().astype(np.float32) * title_weight

    X = np.hstack([G, D, T])
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + EPS)
    names = ([f"genre:{g}" for g in all_genres] + [f"decade:{d}s" for d in dec_vals] +
             [f"title:{w}" for w in tf.get_feature_names_out()])
    groups = (["genre"] * len(all_genres) + ["decade"] * len(dec_vals) +
              ["title"] * T.shape[1])
    return X.astype(np.float32), names, np.array(groups)


# =============================================================================
# 3. MODELS
# =============================================================================
def knn_aggregate(sims: np.ndarray, vals: np.ndarray, k: int, axis: int = 1, damp: float = 0.0):
    """For every row of `sims`, keep the k largest *positive* similarities and
    return the similarity-weighted average of the matching `vals` and the sum of
    the weights used. `damp` is added to the denominator so that predictions resting on a few
    weak neighbours are pulled towards the baseline. (axis=1: neighbours are columns)"""
    sims = np.maximum(sims, 0.0)
    n = sims.shape[axis]
    if n > k:
        idx = np.argpartition(sims, n - k, axis=axis)
        idx = idx[:, n - k:] if axis == 1 else idx[n - k:, :]
        w = np.take_along_axis(sims, idx, axis=axis)
        v = vals[idx] if vals.ndim == 1 else np.take_along_axis(vals, idx, axis=axis)
    else:
        w = sims
        v = np.broadcast_to(vals, sims.shape) if vals.ndim == 1 else vals
    den = w.sum(axis=axis)
    num = (w * v).sum(axis=axis)
    return np.where(den > EPS, num / np.maximum(den + damp, EPS), 0.0), den


class Recommender:
    """Shared plumbing: stores the training ratings in CSR form and implements
    `predict` on top of `user_scores`."""
    name = "recommender"

    def _prep(self, train: Ratings):
        self.n_users, self.n_items = train.n_users, train.n_items
        self.csr = sp.csr_matrix((train.r, (train.u, train.i)),
                                 shape=(train.n_users, train.n_items))
        self.item_count = np.bincount(train.i, minlength=train.n_items)
        self.user_count = np.bincount(train.u, minlength=train.n_users)

    def _user_profile(self, u):
        a, b = self.csr.indptr[u], self.csr.indptr[u + 1]
        return self.csr.indices[a:b], self.csr.data[a:b]

    # ---- to be provided by sub-classes ------------------------------------
    def user_scores(self, u, targets=None):
        raise NotImplementedError

    def profile_scores(self, items, ratings, targets=None):
        raise NotImplementedError

    # ---- shared ------------------------------------------------------------
    def predict(self, u, i):
        u, i = np.asarray(u), np.asarray(i)
        out = np.empty(len(u), dtype=np.float64)
        order = np.argsort(u, kind="stable")
        us = u[order]
        bounds = np.flatnonzero(np.diff(us)) + 1
        for grp in np.split(order, bounds):
            if len(grp) == 0:
                continue
            out[grp] = self.user_scores(int(u[grp[0]]), i[grp])
        return np.clip(out, MIN_RATING, MAX_RATING)

    def predict_one(self, u, i):
        return float(np.clip(self.user_scores(u, np.array([i]))[0], MIN_RATING, MAX_RATING))


class GlobalMean(Recommender):
    name = "Global Mean"

    def fit(self, train):
        self._prep(train)
        self.mu = float(train.r.mean())
        return self

    def user_scores(self, u, targets=None):
        n = self.n_items if targets is None else len(targets)
        return np.full(n, self.mu)

    def profile_scores(self, items, ratings, targets=None):
        return self.user_scores(0, targets)


class BiasBaseline(Recommender):
    """r_ui = mu + b_u + b_i, with damped (regularised) bias estimates."""
    name = "Bias Baseline"

    def __init__(self, reg_i: float = 10.0, reg_u: float = 15.0, n_iter: int = 10):
        self.reg_i, self.reg_u, self.n_iter = reg_i, reg_u, n_iter

    def fit(self, train):
        self._prep(train)
        self.mu = float(train.r.mean())
        self.b_i = np.zeros(self.n_items)
        self.b_u = np.zeros(self.n_users)
        for _ in range(self.n_iter):
            res = train.r - self.mu - self.b_u[train.u]
            self.b_i = np.bincount(train.i, res, self.n_items) / np.maximum(self.reg_i + self.item_count, EPS)
            res = train.r - self.mu - self.b_i[train.i]
            self.b_u = np.bincount(train.u, res, self.n_users) / np.maximum(self.reg_u + self.user_count, EPS)
        return self

    def base(self, bu, targets=None):
        bi = self.b_i if targets is None else self.b_i[targets]
        return self.mu + bu + bi

    def profile_bias(self, items, ratings):
        """Damped user bias for a brand-new user from a handful of ratings."""
        items, ratings = np.asarray(items), np.asarray(ratings, dtype=float)
        if len(items) == 0:
            return 0.0
        return float(np.sum(ratings - self.mu - self.b_i[items]) / (self.reg_u + len(items)))

    def user_scores(self, u, targets=None):
        return self.base(self.b_u[u], targets)

    def profile_scores(self, items, ratings, targets=None):
        return self.base(self.profile_bias(items, ratings), targets)


class _BaselineKNN(Recommender):
    """Common code for models that predict   baseline + weighted neighbour residuals."""

    def _fit_base(self, train, base: BiasBaseline | None):
        self._prep(train)
        self.base_model = base if base is not None else BiasBaseline().fit(train)
        b = self.base_model
        self.resid = train.r - (b.mu + b.b_u[train.u] + b.b_i[train.i])   # r - baseline
        self.resid_csr = sp.csr_matrix((self.resid, (train.u, train.i)),
                                       shape=(train.n_users, train.n_items))

    def user_scores(self, u, targets=None):
        J, rJ = self._user_profile(u)
        return self._score(self.base_model.b_u[u], J, rJ, targets, exclude_user=u)

    def profile_scores(self, items, ratings, targets=None):
        items = np.asarray(items, dtype=int)
        ratings = np.asarray(ratings, dtype=float)
        bu = self.base_model.profile_bias(items, ratings)
        return self._score(bu, items, ratings, targets, exclude_user=None)

    def _score(self, bu, J, rJ, targets, exclude_user):
        raise NotImplementedError


class ItemKNN(_BaselineKNN):
    """Item-based collaborative filtering.

    similarity(i, j) = cosine of the two movies' *mean-centred* rating vectors,
                       shrunk by  n_ij / (n_ij + shrink)   (n_ij = users who rated both)
    prediction(u, i) = baseline(u, i) + sum_j sim(i,j) * resid(u,j) / sum_j sim(i,j)
                       over the k most similar movies j already rated by u.

    mode="raw" reproduces the naive version (zero-filled raw ratings, plain weighted
    average) so the benefit of mean-centring can be measured.
    """

    def __init__(self, k=40, shrink=10.0, min_support=5, mode="centered", damp=0.0):
        self.k, self.shrink, self.min_support, self.mode, self.damp = k, shrink, min_support, mode, damp
        self.name = "Item-based KNN" if mode == "centered" else "Item KNN (raw cosine)"

    def fit(self, train, base: BiasBaseline | None = None):
        self._fit_base(train, base)
        X = self.resid_csr if self.mode == "centered" else self.csr
        self.items = np.flatnonzero(self.item_count >= self.min_support)
        self.pos = np.full(self.n_items, -1, dtype=np.int64)
        self.pos[self.items] = np.arange(len(self.items))
        Xs = X[:, self.items].tocsc()
        G = (Xs.T @ Xs).toarray()
        nrm = np.sqrt(np.diag(G))
        self._cos = (G / (nrm[:, None] * nrm[None, :] + EPS)).astype(np.float32)
        B = (self.csr[:, self.items] > 0).astype(np.float32).tocsc()
        self._co = (B.T @ B).toarray().astype(np.float32)
        self._built_shrink = None
        self._rebuild()
        return self

    def _rebuild(self):
        """(Re)compute the shrunk similarity matrix; cheap, so shrink can be tuned."""
        if self._built_shrink == self.shrink:
            return
        self.S = self._cos * (self._co / (self._co + self.shrink + EPS))
        np.fill_diagonal(self.S, 0.0)
        self._built_shrink = self.shrink

    def set_params(self, k=None, shrink=None, damp=None):
        if k is not None:
            self.k = k
        if damp is not None:
            self.damp = damp
        if shrink is not None:
            self.shrink = shrink
            self._rebuild()
        return self

    def _score(self, bu, J, rJ, targets, exclude_user):
        b = self.base_model
        tg = np.arange(self.n_items) if targets is None else np.asarray(targets)
        out = b.base(bu, tg).copy()
        Jm = self.pos[J]
        keep = Jm >= 0
        Jm, Jk, rk = Jm[keep], J[keep], rJ[keep]
        tm = self.pos[tg]
        ok = tm >= 0
        if Jm.size == 0 or not ok.any():
            return out
        vals = rk if self.mode == "raw" else rk - (b.mu + bu + b.b_i[Jk])
        sub = self.S[np.ix_(tm[ok], Jm)]
        agg, den = knn_aggregate(sub, vals, self.k, damp=0.0 if self.mode == "raw" else self.damp)
        if self.mode == "raw":
            tmp = out[ok]
            out[ok] = np.where(den > EPS, agg, tmp)
        else:
            out[ok] += agg
        return out

    # ---- extras used by the app ---------------------------------------------
    def similar_items(self, item, n=10):
        """Most similar movies to `item` (collaborative similarity)."""
        p = self.pos[item]
        if p < 0:
            return np.array([], dtype=int), np.array([])
        s = self.S[p]
        top = np.argsort(-s)[:n]
        return self.items[top], s[top]

    def explain(self, items, ratings, bu, target, n=3):
        """The n movies the user rated that pushed the prediction for `target` most."""
        b = self.base_model
        items = np.asarray(items, dtype=int)
        p = self.pos[target]
        Jm = self.pos[items]
        keep = Jm >= 0
        if p < 0 or not keep.any():
            return []
        sims = self.S[p, Jm[keep]]
        order = np.argsort(-sims)[:n]
        out = []
        for o in order:
            if sims[o] <= 0:
                continue
            j = items[keep][o]
            out.append((int(j), float(sims[o]), float(np.asarray(ratings)[keep][o])))
        return out


class UserKNN(_BaselineKNN):
    """User-based collaborative filtering: neighbours are *users* with similar
    (mean-centred) taste who have rated the target movie."""
    name = "User-based KNN"

    def __init__(self, k=40, shrink=0.0, damp=0.0):
        self.k, self.shrink, self.damp = k, shrink, damp

    def fit(self, train, base: BiasBaseline | None = None):
        self._fit_base(train, base)
        # item-major copies (items x users) keep the per-target neighbour search cache-friendly
        self.XdT = np.ascontiguousarray(self.resid_csr.toarray().astype(np.float32).T)
        self.BdT = np.ascontiguousarray((self.csr.toarray() > 0).astype(np.float32).T)
        self.unorm = np.sqrt((self.XdT ** 2).sum(0)) + EPS
        return self

    def _sims(self, bu, J, rJ, exclude_user):
        b = self.base_model
        res = (rJ - (b.mu + bu + b.b_i[J])).astype(np.float32)
        dots = res @ self.XdT[J]                                        # (users,)
        cos = dots / (self.unorm * (np.linalg.norm(res) + EPS))
        co = self.BdT[J].sum(0)
        sims = cos * (co / (co + self.shrink + EPS))
        if exclude_user is not None:
            sims[exclude_user] = 0.0
        return np.maximum(sims, 0.0)

    def _score(self, bu, J, rJ, targets, exclude_user):
        b = self.base_model
        tg = np.arange(self.n_items) if targets is None else np.asarray(targets)
        out = b.base(bu, tg).copy()
        if len(J) == 0:
            return out
        sims = self._sims(bu, J, rJ, exclude_user)
        W = self.BdT[tg] * sims[None, :]                               # targets x users
        agg, den = knn_aggregate(W, self.XdT[tg], self.k, axis=1, damp=self.damp)
        return out + agg

    def similar_users(self, u, n=5):
        J, rJ = self._user_profile(u)
        cos = self._sims(self.base_model.b_u[u], J, rJ, u)
        top = np.argsort(-cos)[:n]
        return top, cos[top]


class ContentKNN(_BaselineKNN):
    """Content-based filtering: a movie is predicted from the user's ratings of
    movies with similar genres / decade / title words (cosine similarity)."""
    name = "Content-based KNN"

    def __init__(self, features: np.ndarray, k=30, damp=0.0):
        self.F, self.k, self.damp = features.astype(np.float32), k, damp

    def fit(self, train, base: BiasBaseline | None = None):
        self._fit_base(train, base)
        return self

    def _score(self, bu, J, rJ, targets, exclude_user):
        b = self.base_model
        tg = np.arange(self.n_items) if targets is None else np.asarray(targets)
        out = b.base(bu, tg).copy()
        if len(J) == 0:
            return out
        vals = rJ - (b.mu + bu + b.b_i[J])
        sub = self.F[tg] @ self.F[J].T
        sub[tg[:, None] == np.asarray(J)[None, :]] = 0.0               # never use the movie itself
        agg, _ = knn_aggregate(sub, vals, self.k, damp=self.damp)
        return out + agg

    def similar_items(self, item, n=10):
        s = self.F @ self.F[item]
        s[item] = -1
        top = np.argsort(-s)[:n]
        return top, s[top]


class MatrixFactorALS(_BaselineKNN):
    """Latent-factor model fitted by Alternating Least Squares on baseline residuals:
        r_ui  ~  baseline(u,i) + p_u . q_i
    (bonus model - not required by the brief, included as a stronger CF reference)."""
    name = "Matrix Factorisation (ALS)"

    def __init__(self, factors=20, reg=0.1, n_iter=12, seed=SEED):
        self.factors, self.reg, self.n_iter, self.seed = factors, reg, n_iter, seed

    def fit(self, train, base: BiasBaseline | None = None):
        self._fit_base(train, base)
        f = self.factors
        rng = np.random.RandomState(self.seed)
        self.P = np.zeros((self.n_users, f))
        self.Q = rng.normal(0, 0.1, (self.n_items, f))
        self.Q[self.item_count == 0] = 0
        csr = self.resid_csr.tocsr()
        csc = self.resid_csr.tocsc()
        eye = np.eye(f)
        for _ in range(self.n_iter):
            for u in range(self.n_users):
                a, b = csr.indptr[u], csr.indptr[u + 1]
                if b == a:
                    continue
                Qj = self.Q[csr.indices[a:b]]
                self.P[u] = np.linalg.solve(Qj.T @ Qj + self.reg * (b - a) * eye,
                                            Qj.T @ csr.data[a:b])
            for i in range(self.n_items):
                a, b = csc.indptr[i], csc.indptr[i + 1]
                if b == a:
                    continue
                Pu = self.P[csc.indices[a:b]]
                self.Q[i] = np.linalg.solve(Pu.T @ Pu + self.reg * (b - a) * eye,
                                            Pu.T @ csc.data[a:b])
        return self

    def user_scores(self, u, targets=None):
        tg = np.arange(self.n_items) if targets is None else np.asarray(targets)
        return self.base_model.base(self.base_model.b_u[u], tg) + self.Q[tg] @ self.P[u]

    def _score(self, bu, J, rJ, targets, exclude_user):
        b = self.base_model
        tg = np.arange(self.n_items) if targets is None else np.asarray(targets)
        out = b.base(bu, tg).copy()
        if len(J) == 0:
            return out
        Qj = self.Q[J]
        res = rJ - (b.mu + bu + b.b_i[J])
        p = np.linalg.solve(Qj.T @ Qj + self.reg * len(J) * np.eye(self.factors), Qj.T @ res)
        return out + self.Q[tg] @ p


class Hybrid(Recommender):
    """Weighted blend of fitted component models (weights learnt on validation data)."""
    name = "Hybrid (blend)"

    def __init__(self, components: dict, weights: dict):
        self.components, self.weights = components, weights

    def user_scores(self, u, targets=None):
        return sum(w * self.components[k].user_scores(u, targets) for k, w in self.weights.items())

    def profile_scores(self, items, ratings, targets=None):
        return sum(w * self.components[k].profile_scores(items, ratings, targets)
                   for k, w in self.weights.items())

    @property
    def csr(self):
        return next(iter(self.components.values())).csr

    @property
    def n_items(self):
        return next(iter(self.components.values())).n_items


class Popularity(Recommender):
    """Non-personalised ranking baseline: most-rated movies first."""
    name = "Popularity"

    def fit(self, train):
        self._prep(train)
        return self

    def user_scores(self, u, targets=None):
        c = self.item_count.astype(float)
        return c if targets is None else c[targets]


class PopularityBlend(Recommender):
    """Re-ranks a rating predictor with a popularity prior:
         score = predicted rating + alpha * log(1 + number of ratings)
    Rating predictors alone love obscure, rarely-rated films; the prior pulls well-known
    films back up. Used for Top-N lists (not for rating prediction)."""

    def __init__(self, model, item_count, alpha=0.2):
        self.model, self.alpha = model, alpha
        self.logpop = np.log1p(item_count).astype(float)
        self.name = f"{getattr(model, 'name', 'model')} + popularity prior"

    @property
    def csr(self):
        return self.model.csr

    @property
    def n_items(self):
        return self.model.n_items

    def user_scores(self, u, targets=None):
        lp = self.logpop if targets is None else self.logpop[targets]
        return np.clip(self.model.user_scores(u, targets), MIN_RATING, MAX_RATING) + self.alpha * lp

    def profile_scores(self, items, ratings, targets=None):
        lp = self.logpop if targets is None else self.logpop[targets]
        return np.clip(self.model.profile_scores(items, ratings, targets), MIN_RATING, MAX_RATING) + self.alpha * lp


class ScoreCache(Recommender):
    """Memoises user_scores(u, targets) so that several rankings (e.g. different
    popularity-prior strengths) can re-use one expensive set of model scores.
    Only valid while the same `targets` array is used for every call."""

    def __init__(self, model):
        self.model, self._c = model, {}
        self.name = getattr(model, "name", "model")

    @property
    def csr(self):
        return self.model.csr

    @property
    def n_items(self):
        return self.model.n_items

    def user_scores(self, u, targets=None):
        if u not in self._c:
            self._c[u] = self.model.user_scores(u, targets)
        return self._c[u]


def fit_hybrid_weights(models: dict, val: Ratings) -> dict:
    """Non-negative least squares on validation predictions, normalised to sum to 1."""
    from scipy.optimize import nnls
    keys = list(models)
    A = np.column_stack([models[k].predict(val.u, val.i) for k in keys])
    w, _ = nnls(A, val.r)
    w = w / w.sum()
    return {k: float(v) for k, v in zip(keys, w) if v > 1e-6}


# =============================================================================
# 4. EVALUATION HELPERS
# =============================================================================
def regression_metrics(y, p) -> dict:
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def classification_metrics(y, p, thr=LIKE_THRESHOLD) -> dict:
    yt, yp = (y >= thr).astype(int), (p >= thr).astype(int)
    try:
        auc = float(roc_auc_score(yt, p))
    except ValueError:
        auc = float("nan")
    return {"Accuracy": float(accuracy_score(yt, yp)),
            "Precision": float(precision_score(yt, yp, zero_division=0)),
            "Recall": float(recall_score(yt, yp, zero_division=0)),
            "F1": float(f1_score(yt, yp, zero_division=0)),
            "ROC_AUC": auc}


def ranking_metrics(model: Recommender, train: Ratings, test: Ratings, k: int = 10,
                    min_votes: int = 10, rel_thr: float = RELEVANT_THRESHOLD):
    """Top-K evaluation. For every user with at least one relevant test movie, rank all
    movies they have not rated in training (restricted to movies with >= min_votes
    training ratings) and compare the top-K with their held-out liked movies."""
    csr_tr = sp.csr_matrix((train.r, (train.u, train.i)), shape=(train.n_users, train.n_items))
    item_count = np.bincount(train.i, minlength=train.n_items)
    eligible = item_count >= min_votes
    elig_idx = np.flatnonzero(eligible)
    rel = {}
    for u, i, r in zip(test.u, test.i, test.r):
        if r >= rel_thr and eligible[i]:
            rel.setdefault(int(u), set()).add(int(i))
    prec, rec, ndcg, hit, recommended = [], [], [], [], set()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    pop_sum, pop_n = 0.0, 0
    for u, rel_items in rel.items():
        sc = np.full(train.n_items, -np.inf)
        sc[elig_idx] = np.asarray(model.user_scores(u, elig_idx), dtype=float)
        a, b = csr_tr.indptr[u], csr_tr.indptr[u + 1]
        sc[csr_tr.indices[a:b]] = -np.inf
        top = np.argpartition(-sc, k)[:k]
        top = top[np.argsort(-sc[top])]
        hits = np.array([t in rel_items for t in top], dtype=float)
        prec.append(hits.mean())
        rec.append(hits.sum() / len(rel_items))
        hit.append(float(hits.sum() > 0))
        ideal = disc[:min(len(rel_items), k)].sum()
        ndcg.append((hits * disc).sum() / ideal)
        recommended.update(top.tolist())
        pop_sum += np.log1p(item_count[top]).sum()
        pop_n += k
    n_catalog = int(eligible.sum())
    return {f"Precision@{k}": float(np.mean(prec)), f"Recall@{k}": float(np.mean(rec)),
            f"NDCG@{k}": float(np.mean(ndcg)), f"HitRate@{k}": float(np.mean(hit)),
            "Coverage": len(recommended) / n_catalog,
            "AvgLogPopularity": pop_sum / max(pop_n, 1), "UsersEvaluated": len(rel)}
