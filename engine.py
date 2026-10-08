"""
engine.py - the "brain" behind the MovieSphere web app.

It fits every model on ALL ratings (using the best settings found by train.py) and
exposes a small, friendly API for the interface: recommend for a user, recommend for a
brand-new visitor, find similar movies, predict a rating, explain a recommendation.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import recsys as rs

MODEL_NAMES = ["Hybrid (blend)", "Item-based KNN", "User-based KNN", "Content-based KNN",
               "Matrix Factorisation (ALS)"]

DEFAULT_PARAMS = {
    "bias": {"reg_i": 5, "reg_u": 5},
    "item": {"k": 40, "shrink": 100, "damp": 0.0},
    "user": {"k": 20, "shrink": 0, "damp": 0.5},
    "content": {"title_weight": 0.0, "k": 30, "damp": 0.0},
    "mf": {"factors": 40, "reg": 0.2},
    "hybrid_weights": {"Item-based KNN": 0.4, "User-based KNN": 0.1, "Content-based KNN": 0.5},
    "pop_alpha": 0.8,
}


def load_results(path="results/results.json"):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


class Engine:
    def __init__(self, data_dir="data", results_path="results/results.json"):
        self.results = load_results(results_path)
        self.params = {**DEFAULT_PARAMS, **(self.results["best_params"] if self.results else {})}
        p = self.params

        ds = rs.load_data(data_dir)
        self.ds, self.rt, self.movies = ds, ds.ratings, ds.movies.copy()
        rt = self.rt
        self.item_count = np.bincount(rt.i, minlength=rt.n_items)
        self.item_avg = np.bincount(rt.i, rt.r, rt.n_items) / np.maximum(self.item_count, 1)
        self.user_count = np.bincount(rt.u, minlength=rt.n_users)
        self.user_avg = np.bincount(rt.u, rt.r, rt.n_users) / np.maximum(self.user_count, 1)
        self.movies["votes"] = self.item_count
        self.movies["avg"] = self.item_avg.round(2)
        self.movie_pos = {mid: k for k, mid in enumerate(self.movies["movieId"].values)}
        self.user_pos = {uid: k for k, uid in enumerate(ds.user_ids)}
        self.all_genres = sorted({g for gl in self.movies["genre_list"] for g in gl})

        # ---- fit the models on everything ---------------------------------------------------
        self.base = rs.BiasBaseline(**p["bias"]).fit(rt)
        self.item = rs.ItemKNN(**p["item"]).fit(rt, self.base)
        self.user = rs.UserKNN(**p["user"]).fit(rt, self.base)
        cp = p["content"]
        self.F, self.fnames, _ = rs.build_content_features(self.movies, title_weight=cp["title_weight"])
        self.content = rs.ContentKNN(self.F, k=cp["k"], damp=cp.get("damp", 0.0)).fit(rt, self.base)
        self.mf = rs.MatrixFactorALS(p["mf"]["factors"], p["mf"]["reg"], 12).fit(rt, self.base)
        comps = {"Bias Baseline": self.base, "Item-based KNN": self.item, "User-based KNN": self.user,
                 "Content-based KNN": self.content, "Matrix Factorisation (ALS)": self.mf}
        w = p["hybrid_weights"]
        self.hybrid = rs.Hybrid({k: comps[k] for k in w}, w)
        self.models = {"Hybrid (blend)": self.hybrid, "Item-based KNN": self.item,
                       "User-based KNN": self.user, "Content-based KNN": self.content,
                       "Matrix Factorisation (ALS)": self.mf}
        self.logpop = np.log1p(self.item_count)

    # ------------------------------------------------------------------ helpers ---
    @property
    def pop_alpha(self):
        return float(self.params["pop_alpha"])

    def label(self, item):
        return self.movies.at[item, "title"]

    def _frame(self, items, scores, extra=None):
        m = self.movies.loc[items, ["movieId", "title", "genres", "year", "votes", "avg"]].copy()
        m["predicted"] = np.clip(scores, rs.MIN_RATING, rs.MAX_RATING)
        if extra is not None:
            for k, v in extra.items():
                m[k] = v
        return m.reset_index(drop=True)

    def _filter_mask(self, min_votes, genres, years):
        mask = self.item_count >= min_votes
        if genres:
            gs = set(genres)
            mask &= np.array([bool(gs & set(gl)) for gl in self.movies["genre_list"]])
        if years:
            mask &= (self.movies["year"].values >= years[0]) & (self.movies["year"].values <= years[1])
        return mask

    def _rank(self, scores, exclude, mask, n, alpha):
        s = np.clip(scores, rs.MIN_RATING, rs.MAX_RATING) + alpha * self.logpop
        s = np.where(mask, s, -np.inf)
        s[list(exclude)] = -np.inf
        n = min(n, int(np.isfinite(s).sum()))
        if n <= 0:
            return np.array([], dtype=int)
        top = np.argpartition(-s, n - 1)[:n]
        return top[np.argsort(-s[top])]

    # ------------------------------------------------------------- existing user ---
    def users(self):
        return [int(u) for u in self.ds.user_ids]

    def user_summary(self, user_id):
        u = self.user_pos[user_id]
        J, r = self.item._user_profile(u)
        order = np.argsort(-r, kind="stable")
        fav = self._frame(J[order][:8], r[order][:8])
        fav = fav.rename(columns={"predicted": "your_rating"})
        # genre affinity: average rating this user gives per genre (needs >= 3 movies)
        rows = {}
        for j, rr in zip(J, r):
            for g in self.movies.at[j, "genre_list"]:
                rows.setdefault(g, []).append(rr)
        aff = sorted(((g, float(np.mean(v)), len(v)) for g, v in rows.items() if len(v) >= 3),
                     key=lambda x: (-x[2]))[:8]
        return {"n_rated": int(len(J)), "avg": float(r.mean()), "favourites": fav, "genre_affinity": aff,
                "bias": float(self.base.b_u[u])}

    def recommend_user(self, user_id, model="Hybrid (blend)", n=10, min_votes=20, genres=None,
                       years=None, alpha=None):
        u = self.user_pos[user_id]
        alpha = self.pop_alpha if alpha is None else alpha
        scores = self.models[model].user_scores(u)
        J, rJ = self.item._user_profile(u)
        mask = self._filter_mask(min_votes, genres, years)
        top = self._rank(scores, set(J.tolist()), mask, n, alpha)
        df = self._frame(top, scores[top])
        # explain: which movies the user already rated pushed this one up (item-based neighbours)
        why = []
        for it in top:
            ex = self.item.explain(J, rJ, 0.0, it, n=2)
            why.append([(self.label(j), s, r) for j, s, r in ex])
        df["because"] = why
        return df

    # ----------------------------------------------------------------- new visitor ---
    def profile_from_ids(self, rated: dict):
        items = np.array([self.movie_pos[m] for m in rated], dtype=int)
        ratings = np.array([rated[m] for m in rated], dtype=float)
        return items, ratings

    def recommend_profile(self, rated: dict, model="Hybrid (blend)", n=10, min_votes=20, genres=None,
                          years=None, alpha=None):
        items, ratings = self.profile_from_ids(rated)
        alpha = self.pop_alpha if alpha is None else alpha
        scores = self.models[model].profile_scores(items, ratings)
        mask = self._filter_mask(min_votes, genres, years)
        top = self._rank(scores, set(items.tolist()), mask, n, alpha)
        df = self._frame(top, scores[top])
        bu = self.base.profile_bias(items, ratings)
        why = []
        for it in top:
            ex = self.item.explain(items, ratings, bu, it, n=2)
            why.append([(self.label(j), s, r) for j, s, r in ex])
        df["because"] = why
        return df, bu

    # ------------------------------------------------------------- similar movies ---
    def similar_movies(self, movie_id, n=10, mix=0.5, min_votes=10):
        """mix = 0 -> only 'people who rated this also rated...' (collaborative);
           mix = 1 -> only genre/decade/title similarity (content)."""
        i = self.movie_pos[movie_id]
        collab = np.zeros(self.rt.n_items)
        p = self.item.pos[i]
        if p >= 0:
            collab[self.item.items] = self.item.S[p]
        content = self.F @ self.F[i]
        collab[i] = content[i] = 0
        c1 = collab / (collab.max() + rs.EPS)
        c2 = content / (content.max() + rs.EPS)
        score = (1 - mix) * c1 + mix * c2
        score[self.item_count < min_votes] = -1
        score[i] = -1
        top = np.argsort(-score)[:n]
        df = self._frame(top, np.zeros(len(top)))
        df["similarity"] = score[top]
        df["collab"] = collab[top]
        df["content"] = content[top]
        base_g = set(self.movies.at[i, "genre_list"])
        df["shared_genres"] = [sorted(base_g & set(self.movies.at[t, "genre_list"])) for t in top]
        return df

    # ------------------------------------------------------------ predict a rating ---
    def predict_all(self, user_id, movie_id):
        u, i = self.user_pos[user_id], self.movie_pos[movie_id]
        out = {name: m.predict_one(u, i) for name, m in self.models.items()}
        actual = self.rt.r[(self.rt.u == u) & (self.rt.i == i)]
        return out, (float(actual[0]) if len(actual) else None)

    def explain(self, user_id, movie_id, n=3):
        u, i = self.user_pos[user_id], self.movie_pos[movie_id]
        J, rJ = self.item._user_profile(u)
        return [(self.label(j), s, r) for j, s, r in self.item.explain(J, rJ, 0.0, i, n=n)]

    def similar_users(self, user_id, n=3):
        top, sims = self.user.similar_users(self.user_pos[user_id], n)
        return [(int(self.ds.user_ids[t]), float(s)) for t, s in zip(top, sims)]

    # --------------------------------------------------------------- dataset facts ---
    def stats(self):
        return {"users": self.rt.n_users, "movies": self.rt.n_items, "ratings": len(self.rt),
                "sparsity": 1 - len(self.rt) / (self.rt.n_users * self.rt.n_items)}

    def popular_titles(self, min_votes=25):
        m = self.movies[self.movies["votes"] >= min_votes].sort_values("votes", ascending=False)
        return m[["movieId", "title"]]
