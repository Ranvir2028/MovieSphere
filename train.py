"""
train.py - the complete, reproducible ML pipeline for MovieSphere.

    python train.py            # full run  (about 3-5 minutes on a normal laptop)
    python train.py --quick    # tiny grids, for a fast smoke test

What it does
    1. loads + cleans the MovieLens data
    2. splits it:  70% train  |  10% validation  |  20% test   (seed 42)
    3. tunes every model by grid search on the validation set
    4. refits on train (80%) and scores the untouched test set
    5. runs 5-fold cross-validation, ranking (Top-N) evaluation and error analysis
    6. writes results/results.json and every figure in assets/img/
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time

import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import confusion_matrix, roc_curve

import recsys as rs
from plots import Plotter, MODEL_ORDER

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--data", default="data")
args = ap.parse_args()
Q = args.quick
P = Plotter("assets/img")
os.makedirs("results", exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def rmse_of(model, rt):
    return rs.regression_metrics(rt.r, model.predict(rt.u, rt.i))["RMSE"]


# ------------------------------------------------------------------ 1. DATA --
ds = rs.load_data(args.data)
rt = ds.ratings
movies = ds.movies
log(f"data loaded: {len(rt):,} ratings, {rt.n_users} users, {rt.n_items:,} movies")

train, test = rs.train_test_split(rt, 0.20)
train_sub, val = rs.train_test_split(train, 0.125)          # 0.8 * 0.125 = 10% of all ratings
log(f"split -> train_sub {len(train_sub):,} | val {len(val):,} | test {len(test):,}   (train = {len(train):,})")

# ------------------------------------------------- dataset statistics + EDA --
item_counts_all = np.bincount(rt.i, minlength=rt.n_items)
user_counts_all = np.bincount(rt.u, minlength=rt.n_users)
genre_counts = {}
for gl in movies["genre_list"]:
    for g in gl:
        genre_counts[g] = genre_counts.get(g, 0) + 1
decade_counts = movies["year"].floordiv(10).mul(10).value_counts().to_dict()
liked_share_test = float((test.r >= rs.LIKE_THRESHOLD).mean())
dataset_stats = {
    **ds.clean_report,
    "sparsity": 1 - len(rt) / (rt.n_users * rt.n_items),
    "mean_rating": float(rt.r.mean()), "median_rating": float(np.median(rt.r)),
    "std_rating": float(rt.r.std()),
    "min_ratings_per_user": int(user_counts_all.min()), "max_ratings_per_user": int(user_counts_all.max()),
    "median_ratings_per_user": float(np.median(user_counts_all)),
    "movies_with_lt5_ratings": int((item_counts_all < 5).sum()),
    "movies_with_1_rating": int((item_counts_all == 1).sum()),
    "movies_with_ge10_ratings": int((item_counts_all >= 10).sum()),
    "year_min": int(movies["year"].min()), "year_max": int(movies["year"].max()),
    "genres": genre_counts, "n_genres": len(genre_counts),
    "rating_counts": {str(k): int(v) for k, v in zip(*np.unique(rt.r, return_counts=True))},
    "liked_share_all": float((rt.r >= rs.LIKE_THRESHOLD).mean()), "liked_share_test": liked_share_test,
    "n_train": len(train), "n_train_sub": len(train_sub), "n_val": len(val), "n_test": len(test),
    "test_pairs_unseen_movie": int((np.bincount(train.i, minlength=rt.n_items)[test.i] == 0).sum()),
    "test_pairs_item_below_min_support": int((np.bincount(train.i, minlength=rt.n_items)[test.i] < 5).sum()),
}
# heavy-user / outlier detection with the IQR rule (reported in Section 4)
q1, q3 = np.percentile(user_counts_all, [25, 75])
iqr_cut = q3 + 1.5 * (q3 - q1)
dataset_stats["heavy_users_iqr"] = int((user_counts_all > iqr_cut).sum())
dataset_stats["iqr_cutoff"] = float(iqr_cut)
mi_pop = np.corrcoef(item_counts_all, np.bincount(rt.i, rt.r, rt.n_items) / np.maximum(item_counts_all, 1))[0, 1]
dataset_stats["corr_popularity_avg_rating"] = float(mi_pop)

P.all("eda_ratings", rt.r)
P.all("eda_long_tail", item_counts_all, user_counts_all)
P.all("eda_genres", genre_counts, decade_counts)
P.all("architecture")

# ------------------------------------------------------------ 3. TUNING ----
tuning = {}
best = {}

log("tuning: bias baseline")
grid = list(itertools.product([0, 5, 10, 25], [0, 5, 15, 25]))
if Q:
    grid = grid[:4]
res = []
for ri, ru in grid:
    res.append((rmse_of(rs.BiasBaseline(ri, ru).fit(train_sub), val), ri, ru))
res.sort()
best["bias"] = {"reg_i": res[0][1], "reg_u": res[0][2]}
tuning["bias"] = {"values_tried": {"reg_i": sorted({g[0] for g in grid}), "reg_u": sorted({g[1] for g in grid})},
                  "best": best["bias"], "best_val_rmse": res[0][0], "worst_val_rmse": res[-1][0]}
base_sub = rs.BiasBaseline(**best["bias"]).fit(train_sub)

log("tuning: item-based KNN")
ks = [5, 10, 20, 40, 80] if not Q else [20, 40]
shr = [0, 10, 25, 50, 100, 200] if not Q else [10, 50]
ik = rs.ItemKNN(k=20, shrink=0).fit(train_sub, base_sub)
default_item = rmse_of(ik, val)                                   # naive first guess: k=20, no shrinkage
g_item = []
for k in ks:
    row = []
    for s in shr:
        ik.set_params(k=k, shrink=s, damp=0.0)
        row.append(rmse_of(ik, val))
    g_item.append(row)
a_ = np.unravel_index(np.argmin(g_item), np.shape(g_item))
best["item"] = {"k": ks[a_[0]], "shrink": shr[a_[1]], "damp": 0.0}
damps_i = [0.0, 0.25, 0.5, 1.0] if not Q else [0.0, 0.5]
d_item = []
for dm in damps_i:
    ik.set_params(k=best["item"]["k"], shrink=best["item"]["shrink"], damp=dm)
    d_item.append(rmse_of(ik, val))
best["item"]["damp"] = damps_i[int(np.argmin(d_item))]
tuning["item"] = {"values_tried": {"k": ks, "shrink": shr, "damp": damps_i}, "best": best["item"], "grid": g_item,
                  "damp_stage": d_item, "best_val_rmse": float(min(np.min(g_item), np.min(d_item))),
                  "default_val_rmse": default_item}
P.all("heatmap", "tuning_item", "Item-based KNN", g_item, ks, shr, "k (neighbours)", "shrinkage")
ik.set_params(**best["item"])

log("tuning: user-based KNN")
ks_u = [10, 20, 40, 80] if not Q else [20, 40]
dmp_u = [0.0, 0.25, 0.5, 1.0, 2.0] if not Q else [0.0, 0.5]
uk = rs.UserKNN(k=20, shrink=0, damp=0).fit(train_sub, base_sub)
default_user = rmse_of(uk, val)                                   # naive first guess: k=20, no shrink, no damping
g_user = []
for k in ks_u:
    row = []
    for dm in dmp_u:
        uk.k, uk.shrink, uk.damp = k, 0, dm
        row.append(rmse_of(uk, val))
    g_user.append(row)
a_ = np.unravel_index(np.argmin(g_user), np.shape(g_user))
best["user"] = {"k": ks_u[a_[0]], "damp": dmp_u[a_[1]], "shrink": 0}
shr_u = [0, 10, 25, 50] if not Q else [0, 10]
s_user = []
for s_ in shr_u:
    uk.k, uk.shrink, uk.damp = best["user"]["k"], s_, best["user"]["damp"]
    s_user.append(rmse_of(uk, val))
best["user"]["shrink"] = shr_u[int(np.argmin(s_user))]
tuning["user"] = {"values_tried": {"k": ks_u, "damp": dmp_u, "shrink": shr_u}, "best": best["user"], "grid": g_user,
                  "shrink_stage": s_user, "best_val_rmse": float(min(np.min(g_user), np.min(s_user))),
                  "default_val_rmse": default_user}
P.all("heatmap", "tuning_user", "User-based KNN", g_user, ks_u, dmp_u, "k (neighbours)", "damping")
uk.k, uk.shrink, uk.damp = best["user"]["k"], best["user"]["shrink"], best["user"]["damp"]

log("tuning: content-based KNN")
tws = [0.0, 0.5, 1.0] if not Q else [0.0, 0.5]
kc = [10, 30, 60, 100] if not Q else [30]
g_cont = []
feat_cache = {}
for tw in tws:
    F, fnames, fgroups = rs.build_content_features(movies, title_weight=tw)
    feat_cache[tw] = (F, fnames, fgroups)
    row = []
    for k in kc:
        row.append(rmse_of(rs.ContentKNN(F, k=k).fit(train_sub, base_sub), val))
    g_cont.append(row)
default_content = rmse_of(rs.ContentKNN(feat_cache[0.5][0], k=10).fit(train_sub, base_sub), val)
a_ = np.unravel_index(np.argmin(g_cont), np.shape(g_cont))
best["content"] = {"title_weight": tws[a_[0]], "k": kc[a_[1]], "damp": 0.0}
F, fnames, fgroups = feat_cache[best["content"]["title_weight"]]
damps_c = [0.0, 0.5, 1.0, 2.0] if not Q else [0.0, 0.5]
d_cont = [rmse_of(rs.ContentKNN(F, k=best["content"]["k"], damp=dm).fit(train_sub, base_sub), val) for dm in damps_c]
best["content"]["damp"] = damps_c[int(np.argmin(d_cont))]
tuning["content"] = {"values_tried": {"title_weight": tws, "k": kc, "damp": damps_c}, "best": best["content"], "grid": g_cont,
                     "damp_stage": d_cont, "best_val_rmse": float(min(np.min(g_cont), np.min(d_cont))),
                     "default_val_rmse": default_content}
P.all("tuning_lines", "tuning_content", "Content-based KNN", kc,
      {f"title weight {tw}": g_cont[i] for i, tw in enumerate(tws)}, "k (neighbours)")

log("tuning: matrix factorisation (ALS)")
fs = [10, 20, 40, 80] if not Q else [10]
regs = [0.05, 0.1, 0.2, 0.4] if not Q else [0.1]
g_mf = []
for f in fs:
    row = []
    for rg in regs:
        row.append(rmse_of(rs.MatrixFactorALS(f, rg, 10).fit(train_sub, base_sub), val))
    g_mf.append(row)
a = np.unravel_index(np.argmin(g_mf), np.shape(g_mf))
best["mf"] = {"factors": fs[a[0]], "reg": regs[a[1]]}
default_mf = rmse_of(rs.MatrixFactorALS(10, 0.05, 10).fit(train_sub, base_sub), val)
tuning["mf"] = {"values_tried": {"factors": fs, "reg": regs}, "best": best["mf"], "grid": g_mf,
                "best_val_rmse": float(np.min(g_mf)), "default_val_rmse": default_mf}
P.all("heatmap", "tuning_mf", "Matrix factorisation (ALS)", g_mf, fs, regs, "latent factors", "regularisation")


# ---------------------------------------------- model factory (any training set)
def build_models(tr, with_raw=True):
    base = rs.BiasBaseline(**best["bias"]).fit(tr)
    m = {"Global Mean": rs.GlobalMean().fit(tr), "Bias Baseline": base}
    if with_raw:
        m["Item KNN (raw cosine)"] = rs.ItemKNN(mode="raw", **best["item"]).fit(tr, base)
    m["Item-based KNN"] = rs.ItemKNN(**best["item"]).fit(tr, base)
    m["User-based KNN"] = rs.UserKNN(**best["user"]).fit(tr, base)
    m["Content-based KNN"] = rs.ContentKNN(F, k=best["content"]["k"], damp=best["content"]["damp"]).fit(tr, base)
    m["Matrix Factorisation (ALS)"] = rs.MatrixFactorALS(best["mf"]["factors"], best["mf"]["reg"], 12).fit(tr, base)
    return m


# ------------------------------- hybrid weights: learnt on validation data ----
log("learning hybrid weights on the validation set")
sub_models = build_models(train_sub, with_raw=False)
hy_keys = ["Bias Baseline", "Item-based KNN", "User-based KNN", "Content-based KNN", "Matrix Factorisation (ALS)"]
hy_w = rs.fit_hybrid_weights({k: sub_models[k] for k in hy_keys}, val)
best["hybrid_weights"] = hy_w
log(f"hybrid weights: { {k: round(v, 3) for k, v in hy_w.items()} }")
tuning["hybrid"] = {"weights": hy_w,
                    "val_rmse": rmse_of(rs.Hybrid({k: sub_models[k] for k in hy_w}, hy_w), val)}
P.all("hybrid_weights", hy_w)

# ------------------------------------------------------ 4. FINAL TEST SCORES --
log("fitting final models on the full training set (80%)")
models, timing = {}, {}
for name, fn in [("all", lambda: build_models(train))]:
    t0 = time.time()
    models = fn()
    timing["fit_all_models_s"] = time.time() - t0
models["Hybrid (blend)"] = rs.Hybrid({k: models[k] for k in hy_w}, hy_w)

results, preds, curves, cms = {}, {}, {}, {}
rng = np.random.RandomState(rs.SEED)
for name in MODEL_ORDER:
    m = models[name]
    t0 = time.time()
    p = m.predict(test.u, test.i)
    dt = time.time() - t0
    preds[name] = p
    met = {**rs.regression_metrics(test.r, p), **rs.classification_metrics(test.r, p)}
    se = (p - test.r) ** 2
    boots = [np.sqrt(se[rng.randint(0, len(se), len(se))].mean()) for _ in range(500 if not Q else 50)]
    met["RMSE_CI"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    met["predict_time_s"] = dt
    met["predict_ms_per_rating"] = 1000 * dt / len(p)
    results[name] = met
    yt, yp = (test.r >= rs.LIKE_THRESHOLD).astype(int), (p >= rs.LIKE_THRESHOLD).astype(int)
    cms[name] = confusion_matrix(yt, yp).tolist()
    if len(np.unique(p)) > 1:
        fpr, tpr, _ = roc_curve(yt, p)
        keep = np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int)
        curves[name] = (fpr[keep].tolist(), tpr[keep].tolist())
    log(f"  {name:28s} RMSE {met['RMSE']:.4f}  MAE {met['MAE']:.4f}  F1 {met['F1']:.4f}  AUC {met['ROC_AUC']:.4f}")

best_name = min((n for n in results if n not in ("Hybrid (blend)", "Global Mean", "Bias Baseline", "Item KNN (raw cosine)")),
                key=lambda n: results[n]["RMSE"])
best_overall = min(results, key=lambda n: results[n]["RMSE"])
# paired bootstrap: is the best single model really better than the bias baseline?
d = (preds["Bias Baseline"] - test.r) ** 2 - (preds[best_name] - test.r) ** 2
paired = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(500 if not Q else 50)]
significance = {"model": best_name, "versus": "Bias Baseline",
                "mean_mse_reduction": float(d.mean()),
                "ci95": [float(np.percentile(paired, 2.5)), float(np.percentile(paired, 97.5))]}

aucs = {n: results[n]["ROC_AUC"] for n in results}
P.all("comparison", results)
P.all("classification_bars", results)
P.all("roc", curves, aucs)
P.all("confusion", cms)

# ------------------------------------------------------ 5. CROSS-VALIDATION --
log("5-fold cross-validation on the 80% training portion")
cv = {n: {"RMSE_folds": [], "MAE_folds": [], "F1_folds": [], "AUC_folds": []} for n in MODEL_ORDER if n != "Item KNN (raw cosine)"}
n_folds = 5 if not Q else 2
for f, (tri, vai) in enumerate(rs.kfold_indices(len(train), n_folds)):
    tr_f, va_f = train.take(tri), train.take(vai)
    ms = build_models(tr_f, with_raw=False)
    ms["Hybrid (blend)"] = rs.Hybrid({k: ms[k] for k in hy_w}, hy_w)
    for n in cv:
        p = ms[n].predict(va_f.u, va_f.i)
        r_ = rs.regression_metrics(va_f.r, p)
        c_ = rs.classification_metrics(va_f.r, p)
        cv[n]["RMSE_folds"].append(r_["RMSE"]); cv[n]["MAE_folds"].append(r_["MAE"])
        cv[n]["F1_folds"].append(c_["F1"]); cv[n]["AUC_folds"].append(c_["ROC_AUC"])
    log(f"  fold {f + 1}/{n_folds} done")
for n in cv:
    for key in ["RMSE", "MAE", "F1", "AUC"]:
        v = np.array(cv[n][f"{key}_folds"])
        cv[n][f"{key}_mean"], cv[n][f"{key}_std"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
P.all("cv_plot", cv)

# ------------------------------------------------------- 6. RANKING (TOP-N) --
log("tuning the popularity prior for Top-N lists (validation ranking)")
hy_sub = rs.ScoreCache(rs.Hybrid({k: sub_models[k] for k in hy_w}, hy_w))
alphas = [0.0, 0.2, 0.5, 0.8, 1.2, 2.0] if not Q else [0.0, 0.3]
cnt_sub = np.bincount(train_sub.i, minlength=rt.n_items)
alpha_scores = []
for al in alphas:
    rk_ = rs.ranking_metrics(rs.PopularityBlend(hy_sub, cnt_sub, al), train_sub, val, k=10)
    alpha_scores.append(rk_["NDCG@10"])
    log(f"  alpha {al}: NDCG@10 {rk_['NDCG@10']:.4f}  P@10 {rk_['Precision@10']:.4f}")
best["pop_alpha"] = alphas[int(np.argmax(alpha_scores))]
tuning["pop_alpha"] = {"values_tried": alphas, "val_ndcg": alpha_scores, "best": best["pop_alpha"]}
log("top-N ranking evaluation (K = 10)")
rank_models = {"Popularity": rs.Popularity().fit(train), **{n: models[n] for n in
               ["Bias Baseline", "Item-based KNN", "User-based KNN", "Content-based KNN",
                "Matrix Factorisation (ALS)"]}}
rank_models["Hybrid (blend)"] = rs.ScoreCache(models["Hybrid (blend)"])
rank_models["Hybrid + popularity prior"] = rs.PopularityBlend(
    rank_models["Hybrid (blend)"], np.bincount(train.i, minlength=rt.n_items), best["pop_alpha"])
ranking = {}
for n, m in rank_models.items():
    t0 = time.time()
    ranking[n] = rs.ranking_metrics(m, train, test, k=10)
    log(f"  {n:28s} P@10 {ranking[n]['Precision@10']:.4f}  R@10 {ranking[n]['Recall@10']:.4f}  "
        f"NDCG@10 {ranking[n]['NDCG@10']:.4f}  coverage {ranking[n]['Coverage']:.2%}  ({time.time() - t0:.0f}s)")
P.all("ranking", ranking)

# ----------------------------------------------------------- 7. ERROR ANALYSIS --
log("error analysis")
train_item_cnt = np.bincount(train.i, minlength=rt.n_items)
train_user_cnt = np.bincount(train.u, minlength=rt.n_users)
bp = preds[best_overall]
err = bp - test.r


def bucket_stats(values, edges, labels, p=bp):
    out = {}
    for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        if mask.sum():
            out[lab] = {"n": int(mask.sum()), "RMSE": float(np.sqrt(((p - test.r)[mask] ** 2).mean()))}
    return out


pop_edges = [0, 1, 5, 20, 100, 10 ** 9]
pop_labels = ["0 (new)", "1-4", "5-19", "20-99", "100+"]
uq = np.percentile(train_user_cnt[train_user_cnt > 0], [25, 50, 75])
usr_edges = [0, uq[0], uq[1], uq[2], 10 ** 9]
usr_labels = [f"<{int(uq[0])}", f"{int(uq[0])}-{int(uq[1]) - 1}", f"{int(uq[1])}-{int(uq[2]) - 1}", f">={int(uq[2])}"]
by_true = {}
for v in np.unique(test.r):
    mk = test.r == v
    by_true[float(v)] = {"n": int(mk.sum()), "bias": float(err[mk].mean()), "RMSE": float(np.sqrt((err[mk] ** 2).mean()))}

# cold-start movies (never seen in training): who copes best?
cold = train_item_cnt[test.i] == 0
cold_rmse = {n: float(np.sqrt(((preds[n] - test.r)[cold] ** 2).mean())) for n in MODEL_ORDER} if cold.sum() else {}
# per-model RMSE by movie popularity bucket
pop_by_model = {n: bucket_stats(train_item_cnt[test.i], pop_edges, pop_labels, preds[n]) for n in MODEL_ORDER}

# largest mistakes
order = np.argsort(-np.abs(err))[:10]
worst = [{"user": int(ds.user_ids[test.u[o]]), "movie": movies.loc[test.i[o], "title"], "actual": float(test.r[o]),
          "predicted": round(float(bp[o]), 2)} for o in order]

# RMSE by genre
genre_rmse = {}
for g in genre_counts:
    mask_items = np.array([g in gl for gl in movies["genre_list"]])
    mk = mask_items[test.i]
    if mk.sum() >= 200:
        genre_rmse[g] = {"n": int(mk.sum()), "RMSE": float(np.sqrt((err[mk] ** 2).mean()))}

error_analysis = {"model": best_overall,
                  "by_item_popularity": bucket_stats(train_item_cnt[test.i], pop_edges, pop_labels),
                  "by_user_activity": bucket_stats(train_user_cnt[test.u], usr_edges, usr_labels),
                  "by_true_rating": by_true, "cold_start_movie_rmse": cold_rmse,
                  "cold_start_pairs": int(cold.sum()), "by_model_and_popularity": pop_by_model,
                  "worst_predictions": worst, "by_genre": genre_rmse,
                  "residuals": err.tolist(),
                  "within_half_star": float((np.abs(err) <= 0.5).mean()),
                  "within_one_star": float((np.abs(err) <= 1.0).mean())}
P.all("error_analysis", error_analysis, best_overall)

# ------------------------------------------------------- 8. FEATURE IMPORTANCE --
log("feature importance for the content model")
base_full = models["Bias Baseline"]
item_mean = np.bincount(train.i, train.r, rt.n_items) / np.maximum(train_item_cnt, 1)
sel = train_item_cnt >= 5
Xf, yf = F[sel], item_mean[sel]
keep_cols = Xf.std(0) > 1e-6
mi = mutual_info_regression(Xf[:, keep_cols], yf, random_state=rs.SEED)
fs_, _ = f_regression(Xf[:, keep_cols], yf)
names_k = np.array(fnames)[keep_cols]
mi_rank = sorted(zip(names_k.tolist(), mi.tolist()), key=lambda x: -x[1])
f_rank = sorted(zip(names_k.tolist(), np.nan_to_num(fs_).tolist()), key=lambda x: -x[1])
ablation = {}
variants = {"All features": {"genre", "decade", "title"}, "Genre + decade": {"genre", "decade"},
            "Genre only": {"genre"}, "Decade only": {"decade"}, "Title words only": {"title"}}
for lab, groups in variants.items():
    Fm = F.copy()
    Fm[:, ~np.isin(fgroups, list(groups))] = 0
    Fm /= (np.linalg.norm(Fm, axis=1, keepdims=True) + rs.EPS)
    ablation[lab] = rmse_of(rs.ContentKNN(Fm, k=best["content"]["k"], damp=best["content"]["damp"]).fit(train_sub, base_sub), val)
fi = {"mutual_info": mi_rank[:25], "f_score": f_rank[:25], "ablation": ablation,
      "n_features": int(F.shape[1]), "n_features_used": int(keep_cols.sum()),
      "groups": {g: int((fgroups == g).sum()) for g in ["genre", "decade", "title"]}}
P.all("feature_importance", fi)

# genre appeal after removing user generosity
resid_u = train.r - (base_full.mu + base_full.b_u[train.u])
ga = {}
for g in genre_counts:
    mk = np.array([g in gl for gl in movies["genre_list"]])[train.i]
    if mk.sum() >= 300:
        ga[g] = float(resid_u[mk].mean())
P.all("genre_appeal", ga)

# ----------------------------------------------------------- 9. SAVE EVERYTHING --
env = {"python": sys.version.split()[0], "numpy": np.__version__, "scikit_learn": sklearn.__version__,
       "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
       "cpu_count": os.cpu_count()}
try:
    import pandas, scipy
    env.update(pandas=pandas.__version__, scipy=scipy.__version__)
    import matplotlib
    env["matplotlib"] = matplotlib.__version__
    with open("/proc/meminfo") as fh:
        env["ram_gb"] = round(int(fh.readline().split()[1]) / 1024 / 1024, 1)
except Exception:
    pass

# similarity-weighted example: what the engine recommends for one user (for the report)
example_user = int(np.argsort(-user_counts_all)[5])
em = models["Hybrid (blend)"]
sc = em.user_scores(example_user).copy()
seen = set(rt.i[rt.u == example_user].tolist())
cand = np.flatnonzero((item_counts_all >= 20))
cand = [c for c in cand if c not in seen]
top = sorted(cand, key=lambda c: -sc[c])[:8]
ex_rated = np.argsort(-rt.r[rt.u == example_user], kind="stable")[:6]
ex_items = rt.i[rt.u == example_user][ex_rated]
example = {"user_id": int(ds.user_ids[example_user]), "n_rated": int((rt.u == example_user).sum()),
           "favourites": [{"title": movies.loc[i, "title"], "genres": movies.loc[i, "genres"],
                           "rating": float(rt.r[(rt.u == example_user) & (rt.i == i)][0])} for i in ex_items],
           "recommended": [{"title": movies.loc[c, "title"], "genres": movies.loc[c, "genres"],
                            "predicted": round(float(np.clip(sc[c], .5, 5)), 2),
                            "votes": int(item_counts_all[c])} for c in top]}

out = {"generated_in_s": time.time() - T0, "quick_mode": Q, "env": env, "dataset": dataset_stats,
       "best_params": best, "tuning": tuning, "test_results": results, "confusion_matrices": cms,
       "cross_validation": cv, "ranking": ranking, "error_analysis": {k: v for k, v in error_analysis.items() if k != "residuals"},
       "feature_importance": fi, "genre_appeal": ga, "significance": significance,
       "best_single_model": best_name, "best_overall": best_overall, "example": example,
       "timing": timing, "label_threshold": rs.LIKE_THRESHOLD, "relevant_threshold": rs.RELEVANT_THRESHOLD}
with open("results/results.json", "w") as fh:
    json.dump(out, fh, indent=1)
np.savez_compressed("results/test_predictions.npz", user=ds.user_ids[test.u], movie=movies["movieId"].values[test.i],
                    actual=test.r, **{n.replace(" ", "_"): preds[n] for n in preds})
log(f"DONE. results/results.json written. best model on test RMSE: {best_overall} ({results[best_overall]['RMSE']:.4f})")
