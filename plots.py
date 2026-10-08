"""
plots.py - every figure used by the report and by the web app's "Model Lab".

Each figure is drawn twice: a light version (assets/img/, for the printed report)
and a dark "HUD" version (assets/img/dark/, for the Streamlit app).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

THEMES = {
    "light": dict(bg="#FFFFFF", fg="#1B1F2A", grid="#D9DEE7", muted="#6B7385",
                  accent="#B3261E", gold="#D99A00", blue="#1F4E79", teal="#0E8A7D",
                  panel="#F4F6FA", dir=""),
    "dark": dict(bg="#0B1220", fg="#DBE7FF", grid="#22304A", muted="#8FA3C7",
                 accent="#35E0FF", gold="#FFCC4D", blue="#4D8DFF", teal="#3DDBB0",
                 panel="#111B2E", dir="dark"),
}

MODEL_ORDER = ["Global Mean", "Bias Baseline", "Item KNN (raw cosine)", "Item-based KNN",
               "User-based KNN", "Content-based KNN", "Matrix Factorisation (ALS)",
               "Hybrid (blend)"]


def model_colors(t):
    return {"Global Mean": t["muted"], "Bias Baseline": "#9AA5B8",
            "Item KNN (raw cosine)": "#C77D5A", "Item-based KNN": t["accent"],
            "User-based KNN": t["blue"], "Content-based KNN": t["teal"],
            "Matrix Factorisation (ALS)": "#8E6BD8", "Hybrid (blend)": t["gold"]}


def short(name):
    return {"Item KNN (raw cosine)": "Item raw", "Matrix Factorisation (ALS)": "MF-ALS",
            "Hybrid (blend)": "Hybrid", "Hybrid + popularity prior": "Hybrid + pop. prior", "Content-based KNN": "Content KNN",
            "Item-based KNN": "Item KNN", "User-based KNN": "User KNN"}.get(name, name)


def _apply(t):
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "axes.edgecolor": t["grid"], "axes.labelcolor": t["fg"], "text.color": t["fg"],
        "xtick.color": t["fg"], "ytick.color": t["fg"], "axes.grid": True,
        "grid.color": t["grid"], "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.axisbelow": True, "legend.frameon": False,
    })


class Plotter:
    def __init__(self, out_dir="assets/img"):
        self.out_dir = out_dir

    def _save(self, fig, name, t):
        d = os.path.join(self.out_dir, t["dir"])
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name + ".png"), dpi=170, bbox_inches="tight")
        plt.close(fig)

    def all(self, fn_name, *args, **kw):
        for t in THEMES.values():
            _apply(t)
            getattr(self, fn_name)(t, *args, **kw)

    # ------------------------------------------------------------------ EDA --
    def eda_ratings(self, t, ratings):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        vals, cnt = np.unique(ratings, return_counts=True)
        ax[0].bar(vals, cnt, width=0.4, color=t["accent"])
        ax[0].set_title("Rating distribution"); ax[0].set_xlabel("Rating (stars)")
        ax[0].set_ylabel("Number of ratings")
        liked = cnt[vals >= 3.5].sum() / cnt.sum()
        ax[1].pie([liked, 1 - liked], labels=[f"Liked (>= 3.5)\n{liked:.1%}", f"Not liked\n{1 - liked:.1%}"],
                  colors=[t["gold"], t["blue"]], startangle=90, wedgeprops=dict(width=0.38, edgecolor=t["bg"]),
                  textprops=dict(color=t["fg"]))
        ax[1].set_title("Class balance (threshold 3.5)")
        fig.tight_layout(); self._save(fig, "eda_ratings", t)

    def eda_long_tail(self, t, item_counts, user_counts):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ic = np.sort(item_counts)[::-1]
        ax[0].plot(np.arange(1, len(ic) + 1), ic, color=t["accent"], lw=2)
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_title("Long tail of movie popularity"); ax[0].set_xlabel("Movie rank (log)")
        ax[0].set_ylabel("Ratings received (log)")
        ax[1].hist(user_counts, bins=40, color=t["blue"])
        ax[1].set_yscale("log"); ax[1].set_title("Ratings given per user")
        ax[1].set_xlabel("Number of ratings"); ax[1].set_ylabel("Users (log)")
        fig.tight_layout(); self._save(fig, "eda_long_tail", t)

    def eda_genres(self, t, genre_counts, decade_counts):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.3, 1]})
        g = sorted(genre_counts.items(), key=lambda x: x[1])
        ax[0].barh([a for a, _ in g], [b for _, b in g], color=t["teal"])
        ax[0].set_title("Movies per genre"); ax[0].set_xlabel("Movies")
        d = sorted(decade_counts.items())
        ax[1].bar([f"{a}s" for a, _ in d], [b for _, b in d], color=t["gold"])
        ax[1].set_title("Movies per decade"); ax[1].tick_params(axis="x", rotation=60)
        fig.tight_layout(); self._save(fig, "eda_genres", t)

    # --------------------------------------------------------- ARCHITECTURE --
    def architecture(self, t):
        fig, ax = plt.subplots(figsize=(11, 4.6))
        ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off"); ax.grid(False)

        def box(x, y, w, h, title, sub, color):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                        fc=t["panel"], ec=color, lw=2))
            ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)
            ax.text(x + w / 2, y + h / 2 - 2.2, sub, ha="center", va="center", fontsize=8, color=t["fg"],
                    linespacing=1.5)

        def arrow(x1, y1, x2, y2):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                         color=t["muted"], lw=1.6))
        cols = [t["blue"], t["teal"], t["gold"], t["accent"]]
        box(1, 27, 20, 14, "1  DATA", "MovieLens\nratings.csv\nmovies.csv", cols[0])
        box(26, 27, 22, 14, "2  PREPARE", "clean - index\nsplit 70/10/20\ncontent features", cols[1])
        box(53, 27, 22, 14, "3  TUNE", "grid search on the\nvalidation set", cols[2])
        box(79, 27, 20, 14, "4  MODELS", "Item / User / Content\nKNN - ALS - Hybrid", cols[3])
        for a_, b_ in [(21.6, 25.6), (48.6, 52.4), (75.6, 78.4)]:
            arrow(a_, 34, b_, 34)
        box(79, 3, 20, 14, "5  EVALUATE", "RMSE - MAE - F1 - ROC\nTop-N - 5-fold CV", cols[3])
        box(53, 3, 22, 14, "6  ENGINE", "refit on all ratings\nbest settings", cols[2])
        box(26, 3, 22, 14, "7  WEB APP", "Streamlit 'JARVIS'\nrecommend - explain", cols[1])
        arrow(89, 26.4, 89, 18.2)
        arrow(78.4, 10, 75.6, 10); arrow(52.4, 10, 48.6, 10)
        ax.set_title("MovieSphere - system architecture", pad=6)
        self._save(fig, "architecture", t)

    # --------------------------------------------------------------- TUNING --
    def heatmap(self, t, name, title, grid, rows, cols, row_label, col_label):
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.grid(False)
        g = np.array(grid)
        cmap = "YlOrRd_r" if t["dir"] == "" else "viridis_r"
        im = ax.imshow(g, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
        ax.set_xlabel(col_label); ax.set_ylabel(row_label); ax.set_title(title)
        bi = np.unravel_index(np.argmin(g), g.shape)
        for (r, c), v in np.ndenumerate(g):
            ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if t["dir"] == "" else "white",
                    fontweight="bold" if (r, c) == bi else "normal")
        ax.add_patch(plt.Rectangle((bi[1] - .5, bi[0] - .5), 1, 1, fill=False, ec=t["accent"], lw=2.5))
        fig.colorbar(im, ax=ax, label="validation RMSE")
        self._save(fig, name, t)

    def tuning_lines(self, t, name, title, xs, series: dict, xlabel):
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        pal = [t["accent"], t["blue"], t["teal"], t["gold"]]
        for (lab, ys), c in zip(series.items(), pal):
            ax.plot(xs, ys, marker="o", lw=2, color=c, label=lab)
        ax.set_xlabel(xlabel); ax.set_ylabel("validation RMSE"); ax.set_title(title)
        ax.legend()
        self._save(fig, name, t)

    # --------------------------------------------------------------- RESULTS --
    def comparison(self, t, res):
        names = [m for m in MODEL_ORDER if m in res]
        col = model_colors(t)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, metric in zip(ax, ["RMSE", "MAE"]):
            vals = [res[m][metric] for m in names]
            bars = a.barh([short(n) for n in names], vals, color=[col[n] for n in names])
            if metric == "RMSE":
                for n, b in zip(names, bars):
                    lo, hi = res[n]["RMSE_CI"]
                    a.plot([lo, hi], [b.get_y() + b.get_height() / 2] * 2, color=t["fg"], lw=1.5)
            a.set_xlim(min(vals) * 0.8, max(vals) * 1.09)
            for n, b, v in zip(names, bars, vals):
                x = res[n]["RMSE_CI"][1] if metric == "RMSE" else v
                a.text(x + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
            a.invert_yaxis(); a.set_title(f"{metric} on the test set (lower is better)")
        fig.tight_layout(); self._save(fig, "comparison", t)

    def classification_bars(self, t, res):
        names = [m for m in MODEL_ORDER if m in res and m != "Item KNN (raw cosine)"]
        metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
        pal = [t["blue"], t["teal"], t["gold"], t["accent"], "#8E6BD8"]
        fig, ax = plt.subplots(figsize=(11, 4))
        w = 0.16
        x = np.arange(len(names))
        for k, (m, c) in enumerate(zip(metrics, pal)):
            ax.bar(x + (k - 2) * w, [res[n][m] for n in names], w, label=m.replace("_", "-"), color=c)
        ax.set_xticks(x); ax.set_xticklabels([short(n) for n in names], rotation=15)
        ax.set_ylim(0.4, 1.0); ax.set_title("Liked / not-liked classification (threshold 3.5)")
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        fig.tight_layout(); self._save(fig, "classification_bars", t)

    def roc(self, t, curves, aucs):
        col = model_colors(t)
        fig, ax = plt.subplots(figsize=(6.2, 5))
        for n in MODEL_ORDER:
            if n in curves and n != "Item KNN (raw cosine)":
                fpr, tpr = curves[n]
                ax.plot(fpr, tpr, color=col[n], lw=2.2 if n == "Hybrid (blend)" else 1.6,
                        label=f"{short(n)}  (AUC {aucs[n]:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color=t["muted"], label="Chance")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves - predicting 'liked' (>= 3.5)")
        ax.legend(loc="lower right", fontsize=8)
        self._save(fig, "roc_curves", t)

    def confusion(self, t, cms):
        names = [m for m in MODEL_ORDER if m in cms and m != "Item KNN (raw cosine)"]
        n = len(names)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(11, 2.9 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, name in zip(axes, names):
            cm = np.array(cms[name])
            ax.grid(False)
            light = t["dir"] == ""
            frac = cm / cm.sum(1, keepdims=True)
            ax.imshow(frac, cmap="Blues" if light else "cividis", vmin=0, vmax=1)
            for (r, c), v in np.ndenumerate(cm):
                if light:
                    tc = "white" if frac[r, c] > .5 else t["fg"]
                else:
                    tc = "black" if frac[r, c] > .6 else "white"
                ax.text(c, r, f"{v:,}\n{frac[r, c]:.0%}", ha="center", va="center", fontsize=9, color=tc)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Not liked", "Liked"], fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Not liked", "Liked"], fontsize=8)
            ax.set_title(short(name), fontsize=10); ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
        for ax in axes[n:]:
            ax.axis("off")
        fig.tight_layout(); self._save(fig, "confusion_matrices", t)

    def cv_plot(self, t, cv):
        col = model_colors(t)
        names = [m for m in MODEL_ORDER if m in cv]
        fig, ax = plt.subplots(figsize=(9, 4))
        data = [cv[n]["RMSE_folds"] for n in names]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color=t["fg"], lw=2))
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(col[n]); patch.set_alpha(0.85); patch.set_edgecolor(t["fg"])
        for el in ("whiskers", "caps"):
            for line in bp[el]:
                line.set_color(t["fg"])
        ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels([short(n) for n in names], rotation=15)
        ax.set_ylabel("RMSE per fold"); ax.set_title("5-fold cross-validation (RMSE, lower is better)")
        fig.tight_layout(); self._save(fig, "cv_rmse", t)

    def ranking(self, t, rk):
        col = model_colors(t)
        names = [m for m in ["Popularity", "Bias Baseline", "Item-based KNN", "User-based KNN",
                             "Content-based KNN", "Matrix Factorisation (ALS)", "Hybrid (blend)",
                             "Hybrid + popularity prior"] if m in rk]
        col["Popularity"] = t["muted"]
        col["Hybrid + popularity prior"] = t["gold"]
        k = [x for x in rk[names[0]] if x.startswith("Precision@")][0].split("@")[1]
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
        for a, key in zip(ax, [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"]):
            vals = [rk[n][key] for n in names]
            a.barh([short(n) for n in names], vals, color=[col[n] for n in names])
            for i, v in enumerate(vals):
                a.text(v, i, f" {v:.3f}", va="center", fontsize=8)
            a.invert_yaxis(); a.set_title(key); a.set_xlim(0, max(vals) * 1.25)
        fig.tight_layout(); self._save(fig, "ranking_metrics", t)

    # -------------------------------------------------------- ERROR ANALYSIS --
    def error_analysis(self, t, ea, best):
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        a = ax[0, 0]; d = ea["by_item_popularity"]
        a.bar(list(d), [v["RMSE"] for v in d.values()], color=t["accent"])
        for i, v in enumerate(d.values()):
            a.text(i, v["RMSE"] + 0.005, f"n={v['n']:,}", ha="center", fontsize=8)
        a.set_title(f"{short(best)}: RMSE by movie popularity"); a.set_xlabel("Ratings the movie has in training")
        a.set_ylabel("RMSE"); a.set_ylim(0.6, max(v["RMSE"] for v in d.values()) * 1.1)
        a = ax[0, 1]; d = ea["by_user_activity"]
        a.bar(list(d), [v["RMSE"] for v in d.values()], color=t["blue"])
        for i, v in enumerate(d.values()):
            a.text(i, v["RMSE"] + 0.005, f"n={v['n']:,}", ha="center", fontsize=8)
        a.set_title("RMSE by user activity"); a.set_xlabel("Ratings the user has in training")
        a.set_ylim(0.6, max(v["RMSE"] for v in d.values()) * 1.1)
        a = ax[1, 0]; d = ea["by_true_rating"]
        a.bar([str(k) for k in d], [v["bias"] for v in d.values()],
              color=[t["accent"] if v["bias"] > 0 else t["teal"] for v in d.values()])
        a.axhline(0, color=t["fg"], lw=1)
        a.set_title("Mean signed error by true rating"); a.set_xlabel("True rating")
        a.set_ylabel("predicted - actual")
        a = ax[1, 1]
        a.hist(ea["residuals"], bins=50, color=t["gold"])
        a.axvline(0, color=t["fg"], lw=1)
        a.set_title("Residual distribution (predicted - actual)"); a.set_xlabel("Error")
        fig.tight_layout(); self._save(fig, "error_analysis", t)

    def feature_importance(self, t, fi):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
        for a, key, col, ttl in [(ax[0], "mutual_info", t["accent"], "Mutual information (top 12)"),
                                 (ax[1], "f_score", t["blue"], "F-score (top 12)")]:
            items = fi[key][:12][::-1]
            a.barh([n.split(":")[1] + f" [{n.split(':')[0][0]}]" for n, _ in items], [v for _, v in items], color=col)
            a.set_title(ttl)
        a = ax[2]
        ab = fi["ablation"]
        a.barh(list(ab)[::-1], list(ab.values())[::-1], color=t["gold"])
        lo = min(ab.values())
        a.set_xlim(lo - 0.01, max(ab.values()) + 0.01)
        for i, v in enumerate(list(ab.values())[::-1]):
            a.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        a.set_title("Content KNN: feature-group ablation\n(validation RMSE)")
        fig.tight_layout(); self._save(fig, "feature_importance", t)

    def genre_appeal(self, t, ga):
        g = sorted(ga.items(), key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.barh([a for a, _ in g], [b for _, b in g],
                color=[t["teal"] if b > 0 else t["accent"] for _, b in g])
        ax.axvline(0, color=t["fg"], lw=1)
        ax.set_title("Average rating vs. what each user normally gives\n(by genre)")
        ax.set_xlabel("stars above (+) / below (-) the user's own average")
        fig.tight_layout(); self._save(fig, "genre_appeal", t)

    def hybrid_weights(self, t, w):
        col = model_colors(t)
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        names = list(w)
        ax.bar([short(n) for n in names], [w[n] for n in names], color=[col[n] for n in names])
        for i, n in enumerate(names):
            ax.text(i, w[n] + 0.01, f"{w[n]:.2f}", ha="center", fontsize=9)
        ax.set_title("Hybrid blend weights (learnt on validation)"); ax.set_ylim(0, max(w.values()) * 1.2)
        ax.tick_params(axis="x", rotation=15)
        self._save(fig, "hybrid_weights", t)
