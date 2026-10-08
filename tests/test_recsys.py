"""Unit tests for the core engine.   Run:  python -m pytest tests -q   (or  python tests/test_recsys.py)"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import recsys as rs

DS = rs.load_data(os.path.join(ROOT, "data"))
TR, TE = rs.train_test_split(DS.ratings)
BASE = rs.BiasBaseline(5, 5).fit(TR)


def rmse(m):
    return rs.regression_metrics(TE.r, m.predict(TE.u, TE.i))["RMSE"]


def test_cleaning_keeps_everything_valid():
    r = DS.ratings
    assert r.r.min() >= 0.5 and r.r.max() <= 5.0
    assert len(set(zip(r.u.tolist(), r.i.tolist()))) == len(r)          # no duplicate user-movie pairs


def test_split_is_disjoint_and_reproducible():
    a, b = rs.train_test_split(DS.ratings)
    assert len(a) + len(b) == len(DS.ratings)
    assert np.array_equal(b.r, TE.r)                                    # same seed -> same split
    s1 = set(zip(TR.u.tolist(), TR.i.tolist())); s2 = set(zip(TE.u.tolist(), TE.i.tolist()))
    assert not (s1 & s2)


def test_predictions_are_in_range():
    for m in [rs.GlobalMean().fit(TR), BASE, rs.ItemKNN(40, 100).fit(TR, BASE), rs.UserKNN(20, 10, 0.25).fit(TR, BASE)]:
        p = m.predict(TE.u[:500], TE.i[:500])
        assert p.min() >= 0.5 and p.max() <= 5.0 and np.isfinite(p).all()


def test_models_beat_global_mean_and_item_knn_beats_bias():
    g = rmse(rs.GlobalMean().fit(TR))
    ik = rmse(rs.ItemKNN(40, 100).fit(TR, BASE))
    assert ik < rmse(BASE) < g


def test_mean_centering_helps():
    raw = rmse(rs.ItemKNN(40, 100, mode="raw").fit(TR, BASE))
    cen = rmse(rs.ItemKNN(40, 100).fit(TR, BASE))
    assert cen < raw


def test_similarity_is_symmetric_and_bounded():
    ik = rs.ItemKNN(40, 10).fit(TR, BASE)
    assert np.allclose(ik.S, ik.S.T, atol=1e-5)
    assert ik.S.max() <= 1.0 + 1e-6 and ik.S.min() >= -1.0 - 1e-6


def test_new_user_profile_gets_recommendations():
    ik = rs.ItemKNN(40, 10).fit(TR, BASE)
    sc = ik.profile_scores(np.array([0, 1, 2]), np.array([5.0, 4.0, 4.5]))
    assert sc.shape == (TR.n_items,) and np.isfinite(sc).all()


def test_ranking_metrics_are_valid():
    m = rs.ranking_metrics(rs.Popularity().fit(TR), TR, TE, k=10)
    assert 0 <= m["Precision@10"] <= 1 and 0 <= m["NDCG@10"] <= 1


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except AssertionError as e:
                fails += 1; print("FAIL", name, e)
    sys.exit(fails)
