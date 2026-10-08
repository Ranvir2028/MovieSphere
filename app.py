"""
MovieSphere - Stark-grade movie recommendation interface.

Run with:   streamlit run app.py
(train.py is optional: without results/results.json the app uses sensible default
 settings and the Model Lab tab simply asks you to run train.py first.)
"""
import html
import os

import numpy as np
import streamlit as st

from engine import Engine, MODEL_NAMES

st.set_page_config(page_title="MovieSphere | Recommendation Core", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# THEME  (arc-reactor cyan, Stark gold, deep-space navy)
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');
:root{--cyan:#35e0ff;--gold:#ffcc4d;--red:#ff4d5e;--navy:#0b1220;--panel:#111b2e;--line:#22304a;--text:#dbe7ff;--muted:#8fa3c7;}
.stApp{background:radial-gradient(1200px 600px at 85% -10%,rgba(53,224,255,.10),transparent 60%),radial-gradient(900px 500px at -10% 10%,rgba(255,204,77,.07),transparent 55%),#070d18;}
html,body,[class*="css"],.stMarkdown,.stSelectbox,.stSlider{font-family:'Rajdhani','Segoe UI',system-ui,sans-serif;}
.block-container{padding-top:1.4rem;max-width:1250px;}
#MainMenu,footer{visibility:hidden;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1322,#070d18);border-right:1px solid var(--line);}
h1,h2,h3{font-family:'Orbitron','Segoe UI',sans-serif;letter-spacing:.04em;}
.hero{display:flex;align-items:center;gap:22px;padding:18px 24px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(120deg,rgba(17,27,46,.95),rgba(11,18,32,.6));box-shadow:0 0 40px rgba(53,224,255,.08) inset,0 10px 40px rgba(0,0,0,.4);margin-bottom:14px;position:relative;overflow:hidden;}
.hero:after{content:"";position:absolute;inset:0;background:repeating-linear-gradient(0deg,rgba(255,255,255,.015) 0 1px,transparent 1px 3px);pointer-events:none;}
.reactor{width:92px;height:92px;flex:none;filter:drop-shadow(0 0 10px rgba(53,224,255,.7));}
.reactor .spin{transform-origin:50px 50px;animation:spin 14s linear infinite;}
.reactor .spin2{transform-origin:50px 50px;animation:spin 9s linear infinite reverse;}
.reactor .core{animation:pulse 2.4s ease-in-out infinite;transform-origin:50px 50px;}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes pulse{0%,100%{opacity:.75;transform:scale(1);}50%{opacity:1;transform:scale(1.08);}}
.eyebrow{font-size:.78rem;letter-spacing:.32em;color:var(--cyan);font-weight:600;}
.hero h1{margin:.1rem 0 .1rem 0;font-size:2.5rem;font-weight:900;color:#fff;line-height:1.05;}
.hero h1 span{color:var(--gold);}
.hero p{margin:0;color:var(--muted);font-size:1.05rem;}
.status{margin-left:auto;text-align:right;font-family:'Orbitron',sans-serif;font-size:.72rem;letter-spacing:.2em;color:var(--cyan);}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#2dff9b;box-shadow:0 0 10px #2dff9b;margin-right:8px;animation:pulse 1.6s infinite;}
.hud{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:8px 0 16px 0;}
.hud .cell{border:1px solid var(--line);border-radius:14px;padding:12px 16px;background:linear-gradient(160deg,rgba(17,27,46,.95),rgba(11,18,32,.8));position:relative;}
.hud .cell:before{content:"";position:absolute;left:0;top:14%;bottom:14%;width:3px;border-radius:3px;background:var(--cyan);box-shadow:0 0 10px var(--cyan);}
.hud .k{font-size:.72rem;letter-spacing:.2em;color:var(--muted);text-transform:uppercase;}
.hud .v{font-family:'Orbitron',sans-serif;font-size:1.5rem;color:#fff;font-weight:700;}
.hud .s{font-size:.82rem;color:var(--muted);}
.hud .gold:before{background:var(--gold);box-shadow:0 0 10px var(--gold);}
.sect{font-family:'Orbitron',sans-serif;letter-spacing:.14em;color:var(--cyan);font-size:.9rem;margin:18px 0 8px 0;text-transform:uppercase;}
.sect:after{content:"";display:block;height:1px;margin-top:6px;background:linear-gradient(90deg,var(--cyan),transparent);opacity:.5;}
.mcard{display:flex;align-items:center;gap:16px;padding:12px 16px;margin:9px 0;border:1px solid var(--line);border-radius:14px;background:linear-gradient(100deg,rgba(17,27,46,.96),rgba(11,18,32,.7));transition:all .18s ease;position:relative;}
.mcard:hover{border-color:var(--cyan);box-shadow:0 0 22px rgba(53,224,255,.18);transform:translateX(3px);}
.mcard .rank{font-family:'Orbitron',sans-serif;font-size:1.25rem;color:var(--gold);width:38px;text-align:center;font-weight:700;}
.mcard .body{flex:1;min-width:0;}
.mcard .title{font-size:1.2rem;font-weight:700;color:#fff;line-height:1.2;}
.mcard .year{color:var(--muted);font-weight:500;font-size:.95rem;}
.mcard .meta{color:var(--muted);font-size:.9rem;margin-top:2px;}
.mcard .why{color:#b7c6e4;font-size:.9rem;margin-top:5px;border-left:2px solid var(--gold);padding-left:8px;}
.mcard .why b{color:var(--gold);font-weight:600;}
.chip{display:inline-block;padding:1px 9px;margin:4px 5px 0 0;border-radius:999px;border:1px solid rgba(53,224,255,.45);color:var(--cyan);font-size:.74rem;letter-spacing:.04em;background:rgba(53,224,255,.07);}
.chip.gold{border-color:rgba(255,204,77,.5);color:var(--gold);background:rgba(255,204,77,.07);}
.ring{width:64px;height:64px;border-radius:50%;flex:none;background:conic-gradient(var(--cyan) var(--p),rgba(255,255,255,.08) 0);display:grid;place-items:center;box-shadow:0 0 14px rgba(53,224,255,.25);}
.ring.gold{background:conic-gradient(var(--gold) var(--p),rgba(255,255,255,.08) 0);box-shadow:0 0 14px rgba(255,204,77,.25);}
.ring .in{width:52px;height:52px;border-radius:50%;background:#0b1220;display:grid;place-items:center;text-align:center;line-height:1;}
.ring .in b{font-family:'Orbitron',sans-serif;font-size:.95rem;color:#fff;}
.ring .in i{font-style:normal;color:var(--muted);font-size:.58rem;letter-spacing:.1em;}
.bar{display:flex;align-items:center;gap:10px;margin:7px 0;}
.bar .lab{width:150px;color:var(--text);font-size:.95rem;flex:none;}
.bar .track{flex:1;height:10px;background:rgba(255,255,255,.07);border-radius:6px;overflow:hidden;}
.bar .fill{height:100%;border-radius:6px;background:linear-gradient(90deg,var(--cyan),#7a8cff);box-shadow:0 0 10px rgba(53,224,255,.5);}
.bar .fill.gold{background:linear-gradient(90deg,var(--gold),#ff9d4d);box-shadow:0 0 10px rgba(255,204,77,.4);}
.bar .val{width:70px;text-align:right;font-family:'Orbitron',sans-serif;font-size:.85rem;color:#fff;flex:none;}
table.lb{width:100%;border-collapse:collapse;font-size:.95rem;}
table.lb th{font-family:'Orbitron',sans-serif;font-size:.68rem;letter-spacing:.14em;color:var(--cyan);text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);text-transform:uppercase;}
table.lb th:first-child,table.lb td:first-child{text-align:left;}
table.lb td{padding:8px 10px;border-bottom:1px solid rgba(34,48,74,.6);text-align:right;color:var(--text);}
table.lb tr.best td{color:#fff;background:linear-gradient(90deg,rgba(255,204,77,.10),transparent);}
table.lb tr.best td:first-child{border-left:3px solid var(--gold);}
.note{border:1px dashed var(--line);border-radius:12px;padding:10px 14px;color:var(--muted);font-size:.95rem;margin:8px 0;}
.note b{color:var(--gold);}
button[data-baseweb="tab"]{font-family:'Orbitron',sans-serif;letter-spacing:.08em;font-size:.82rem;}
div[data-testid="stTabs"] button[aria-selected="true"]{color:var(--cyan);}
.side-title{font-family:'Orbitron',sans-serif;color:#fff;letter-spacing:.12em;font-size:1.05rem;}
.side-sub{color:var(--muted);font-size:.85rem;}
</style>
"""

REACTOR = """
<svg class="reactor" viewBox="0 0 100 100">
 <circle cx="50" cy="50" r="46" fill="none" stroke="#35e0ff" stroke-width="2" opacity=".9"/>
 <g class="spin"><circle cx="50" cy="50" r="40" fill="none" stroke="#35e0ff" stroke-width="3" stroke-dasharray="6 5"/></g>
 <g class="spin2"><circle cx="50" cy="50" r="31" fill="none" stroke="#ffcc4d" stroke-width="2" stroke-dasharray="2 4"/>
  <path d="M50 19 L50 31 M50 69 L50 81 M19 50 L31 50 M69 50 L81 50 M28 28 L36 36 M64 64 L72 72 M72 28 L64 36 M28 72 L36 64" stroke="#35e0ff" stroke-width="3" stroke-linecap="round"/></g>
 <circle class="core" cx="50" cy="50" r="15" fill="#35e0ff" opacity=".9"/>
 <circle cx="50" cy="50" r="9" fill="#e9fdff"/>
</svg>
"""


# =============================================================================
# SMALL HTML HELPERS  (collapse whitespace so Streamlit never treats it as code)
# =============================================================================
def H(s: str) -> str:
    return "".join(line.strip() for line in s.strip().splitlines())


def esc(x) -> str:
    return html.escape(str(x))


def chips(genres: str, gold=False) -> str:
    cls = "chip gold" if gold else "chip"
    return "".join(f'<span class="{cls}">{esc(g)}</span>' for g in str(genres).split("|") if g)


def ring(value: float, gold=False, label="PRED") -> str:
    pct = max(0.0, min(1.0, value / 5.0)) * 100
    cls = "ring gold" if gold else "ring"
    return (f'<div class="{cls}" style="--p:{pct:.0f}%"><div class="in"><span><b>{value:.2f}</b><br>'
            f'<i>{label}</i></span></div></div>')


def bar(label: str, value: float, vmax: float, text: str, gold=False) -> str:
    pct = max(0.0, min(1.0, value / vmax)) * 100 if vmax else 0
    cls = "fill gold" if gold else "fill"
    return (f'<div class="bar"><div class="lab">{esc(label)}</div><div class="track">'
            f'<div class="{cls}" style="width:{pct:.0f}%"></div></div><div class="val">{esc(text)}</div></div>')


def hud(cells) -> str:
    out = ""
    for k, v, s, gold in cells:
        out += (f'<div class="cell{" gold" if gold else ""}"><div class="k">{esc(k)}</div>'
                f'<div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>')
    return f'<div class="hud">{out}</div>'


def movie_cards(df, show_why=True, score_label="PRED", gold_first=True) -> str:
    out = ""
    for k, row in enumerate(df.itertuples(index=False), start=1):
        why = ""
        if show_why and getattr(row, "because", None):
            parts = [f"<b>{esc(t)}</b> (you gave {r:g}★)" for t, s, r in row.because]
            why = f'<div class="why">Because you liked {" and ".join(parts)}</div>'
        out += (f'<div class="mcard"><div class="rank">{k:02d}</div><div class="body">'
                f'<div class="title">{esc(row.title)}</div>'
                f'<div>{chips(row.genres)}</div>'
                f'<div class="meta">community average {row.avg:.2f}★ &nbsp;·&nbsp; {int(row.votes)} ratings</div>'
                f'{why}</div>{ring(row.predicted, gold=(gold_first and k == 1), label=score_label)}</div>')
    return out


# =============================================================================
# LOAD THE ENGINE ONCE (trains every model on all ratings, ~10-30 s the first time)
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    return Engine(data_dir="data", results_path="results/results.json")


st.markdown(CSS, unsafe_allow_html=True)
with st.spinner("Booting the recommendation core - training models on 100k ratings..."):
    eng = get_engine()

S = eng.stats()
R = eng.results
best_name = R["best_overall"] if R else "Hybrid (blend)"
best_rmse = R["test_results"][best_name]["RMSE"] if R else None

st.markdown(H(f"""
<div class="hero">{REACTOR}
 <div><div class="eyebrow">STARK-GRADE RECOMMENDATION CORE</div>
 <h1>MOVIE<span>SPHERE</span></h1>
 <p>Tell the core what you loved - it predicts what you'll love next. KNN · cosine similarity · collaborative filtering.</p></div>
 <div class="status"><span class="dot"></span>CORE ONLINE<br><span style="color:#8fa3c7">{S['ratings']:,} RATINGS LOADED</span></div>
</div>"""), unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar ---
with st.sidebar:
    st.markdown('<div class="side-title">🎬 MOVIESPHERE</div><div class="side-sub">Machine Learning mini-project</div>',
                unsafe_allow_html=True)
    st.markdown(hud([("Users", f"{S['users']:,}", "in the dataset", False),
                     ("Movies", f"{S['movies']:,}", "rated at least once", False),
                     ("Ratings", f"{S['ratings']:,}", f"{S['sparsity']:.1%} of the matrix is empty", True),
                     ("Best RMSE", f"{best_rmse:.3f}" if best_rmse else "run train.py", best_name, True)]),
                unsafe_allow_html=True)
    st.markdown('<div class="sect">Models online</div>', unsafe_allow_html=True)
    st.markdown(H("""
    <div class="side-sub">
    <b style="color:#35e0ff">Item KNN</b> - movies liked by the same people<br>
    <b style="color:#35e0ff">User KNN</b> - people with your taste<br>
    <b style="color:#35e0ff">Content KNN</b> - genres &amp; decade<br>
    <b style="color:#35e0ff">Matrix Factorisation</b> - hidden taste factors<br>
    <b style="color:#ffcc4d">Hybrid</b> - the blend of all four
    </div>"""), unsafe_allow_html=True)
    st.caption("Dataset: MovieLens ml-latest-small (GroupLens Research).")

tab_user, tab_new, tab_sim, tab_pred, tab_lab, tab_how = st.tabs(
    ["FOR YOU", "NEW VISITOR", "SIMILAR MOVIES", "PREDICT A RATING", "MODEL LAB", "HOW IT WORKS"])

year_lo, year_hi = int(eng.movies["year"].min()), int(eng.movies["year"].max())


def filter_controls(prefix):
    """Shared filters: genres, minimum number of ratings, release years, ranking style."""
    with st.expander("Filters and ranking style"):
        c1, c2 = st.columns(2)
        genres = c1.multiselect("Only these genres", eng.all_genres, key=f"{prefix}_genres")
        votes = c2.slider("Minimum number of ratings a movie must have", 5, 150, 20, key=f"{prefix}_votes")
        years = st.slider("Release years", year_lo, year_hi, (year_lo, year_hi), key=f"{prefix}_years")
        style = st.slider("Ranking style:  hidden gems  <->  crowd favourites", 0, 100, 50, key=f"{prefix}_style",
                          help="0 = rank purely by predicted rating. 100 = lean heavily on how popular a movie is. "
                               "The middle is the setting tuned in train.py.")
    alpha = eng.pop_alpha * 2 * style / 100.0
    years = None if tuple(years) == (year_lo, year_hi) else years
    return genres or None, votes, years, alpha


# =============================================================================
# TAB 1 - EXISTING USER
# =============================================================================
with tab_user:
    c1, c2, c3 = st.columns([1, 1.3, 1])
    uid = c1.selectbox("User ID", eng.users(), index=eng.users().index(414) if 414 in eng.users() else 0,
                       format_func=lambda x: f"User {x}")
    model = c2.selectbox("Model", MODEL_NAMES, index=0, key="u_model")
    n_rec = c3.slider("How many recommendations", 5, 20, 10, key="u_n")
    genres, votes, years, alpha = filter_controls("u")

    info = eng.user_summary(uid)
    sims = eng.similar_users(uid, 3)
    gen = "generous" if info["bias"] > 0.1 else "tough" if info["bias"] < -0.1 else "typical"
    st.markdown(hud([("Movies rated", f"{info['n_rated']}", "by this user", False),
                     ("Average rating", f"{info['avg']:.2f}★", f"{gen} rater ({info['bias']:+.2f}★ vs. average)", False),
                     ("Taste twins", f"{sims[0][0]}", f"similarity {sims[0][1]:.2f}  (also {sims[1][0]}, {sims[2][0]})", True)]),
                unsafe_allow_html=True)

    left, right = st.columns([1, 1.6])
    with left:
        st.markdown('<div class="sect">Taste profile</div>', unsafe_allow_html=True)
        if info["genre_affinity"]:
            st.markdown("".join(bar(g, a, 5.0, f"{a:.2f}★ · {n}") for g, a, n in info["genre_affinity"]),
                        unsafe_allow_html=True)
        st.markdown('<div class="sect">Top-rated by this user</div>', unsafe_allow_html=True)
        fav = info["favourites"].head(6).copy()
        fav["avg"] = fav["avg"]
        fav["predicted"] = fav["your_rating"]
        fav["because"] = None
        st.markdown(movie_cards(fav, show_why=False, score_label="YOU", gold_first=False), unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="sect">Recommended for User {uid} · {esc(model)}</div>', unsafe_allow_html=True)
        recs = eng.recommend_user(uid, model, n_rec, votes, genres, years, alpha)
        if len(recs):
            st.markdown(movie_cards(recs), unsafe_allow_html=True)
        else:
            st.warning("No movies match these filters - try relaxing them.")
        st.markdown('<div class="note"><b>PRED</b> is the rating the model expects this user to give (0.5-5★). '
                    'Movies the user has already rated are never recommended.</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2 - NEW VISITOR (cold start)
# =============================================================================
with tab_new:
    st.markdown('<div class="sect">Cold start - no account needed</div>', unsafe_allow_html=True)
    st.write("Pick at least **two** movies you know and rate them. The core builds a taste profile on the spot.")
    pop = eng.popular_titles(25)
    id_to_title = dict(zip(pop["movieId"], pop["title"]))
    c1, c2 = st.columns([2.2, 1])
    picked = c1.multiselect("Movies you have seen (type to search)", list(id_to_title), format_func=lambda m: id_to_title[m],
                            key="n_pick")
    model_n = c2.selectbox("Model", MODEL_NAMES, index=0, key="n_model")
    rated = {}
    if picked:
        cols = st.columns(min(3, len(picked)))
        for k, mid in enumerate(picked):
            rated[mid] = cols[k % len(cols)].slider(id_to_title[mid], 0.5, 5.0, 4.0, 0.5, key=f"rate_{mid}")
    n_new = st.slider("How many recommendations", 5, 20, 10, key="n_n")
    g2, v2, y2, a2 = filter_controls("n")
    if len(rated) < 2:
        st.info("Add at least two movies above to get recommendations.")
    else:
        recs, bu = eng.recommend_profile(rated, model_n, n_new, v2, g2, y2, a2)
        mood = "generous" if bu > 0.1 else "tough" if bu < -0.1 else "typical"
        st.markdown(f'<div class="sect">Your picks</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="note">From your ratings the core reads you as a <b>{mood}</b> rater '
                    f'({bu:+.2f}★ vs. the average viewer).</div>', unsafe_allow_html=True)
        st.markdown(movie_cards(recs), unsafe_allow_html=True)

# =============================================================================
# TAB 3 - SIMILAR MOVIES
# =============================================================================
with tab_sim:
    pool = eng.popular_titles(10)
    ids = list(pool["movieId"])
    titles = dict(zip(pool["movieId"], pool["title"]))
    default = next((m for m in ids if titles[m].startswith("Toy Story (1995)")), ids[0])
    c1, c2, c3 = st.columns([2, 1.6, 1])
    mid = c1.selectbox("Movie", ids, index=ids.index(default), format_func=lambda m: titles[m], key="s_movie")
    mix = c2.slider("Similarity type:   0 = what people rate alike   ·   1 = genre / decade", 0.0, 1.0, 0.5, 0.05, key="s_mix")
    n_sim = c3.slider("How many", 5, 20, 8, key="s_n")
    sim = eng.similar_movies(mid, n_sim, mix)
    st.markdown(f'<div class="sect">Because you picked {esc(titles[mid])}</div>', unsafe_allow_html=True)
    out = ""
    for k, r in enumerate(sim.itertuples(index=False), start=1):
        shared = "".join(f'<span class="chip gold">{esc(g)}</span>' for g in r.shared_genres)
        out += (f'<div class="mcard"><div class="rank">{k:02d}</div><div class="body">'
                f'<div class="title">{esc(r.title)}</div><div>{chips(r.genres)}</div>'
                f'<div class="meta">community average {r.avg:.2f}★ · {int(r.votes)} ratings &nbsp;|&nbsp; shared: {shared or "-"}</div>'
                f'{bar("rated alike", r.collab, 1.0, f"{r.collab:.2f}")}{bar("same genre/decade", r.content, 1.0, f"{r.content:.2f}", gold=True)}'
                f'</div>{ring(min(5.0, r.similarity * 5), gold=(k == 1), label="MATCH")}</div>')
    st.markdown(out, unsafe_allow_html=True)

# =============================================================================
# TAB 4 - PREDICT A RATING
# =============================================================================
with tab_pred:
    c1, c2 = st.columns([1, 2])
    u4 = c1.selectbox("User", eng.users(), format_func=lambda x: f"User {x}", key="p_user",
                      index=eng.users().index(414) if 414 in eng.users() else 0)
    m4 = c2.selectbox("Movie", ids, index=ids.index(default), format_func=lambda m: titles[m], key="p_movie")
    preds, actual = eng.predict_all(u4, m4)
    mrow = eng.movies.loc[eng.movie_pos[m4]]
    st.markdown(f'<div class="sect">What would User {u4} give "{esc(mrow["title"])}"?</div>', unsafe_allow_html=True)
    rows = "".join(bar(name, v, 5.0, f"{v:.2f}★", gold=(name == "Hybrid (blend)")) for name, v in preds.items())
    st.markdown(rows, unsafe_allow_html=True)
    cells = [("Community average", f"{mrow['avg']:.2f}★", f"{int(mrow['votes'])} ratings", False)]
    if actual is not None:
        err = abs(preds["Hybrid (blend)"] - actual)
        cells.append(("Actual rating", f"{actual:g}★", f"hybrid missed by {err:.2f}★", True))
    else:
        cells.append(("Actual rating", "unseen", "this user has not rated it", True))
    st.markdown(hud(cells), unsafe_allow_html=True)
    ex = eng.explain(u4, m4, 3)
    if ex:
        parts = "<br>".join(f"<b>{esc(t)}</b> - you gave {r:g}★ (similarity {s:.2f})" for t, s, r in ex)
        st.markdown(f'<div class="note">Item-KNN looked at the movies this user rated that are most like this one:<br>{parts}</div>',
                    unsafe_allow_html=True)
    if actual is not None:
        st.caption("Heads-up: this user already rated the movie, so the Bias and Matrix-Factorisation parts of the "
                   "models have seen that rating. The KNN parts exclude it. Honest accuracy numbers are in the Model Lab.")

# =============================================================================
# TAB 5 - MODEL LAB
# =============================================================================
IMG = "assets/img/dark"


def show(name, caption=None):
    path = os.path.join(IMG, name + ".png")
    if os.path.exists(path):
        st.image(path)
        if caption:
            st.caption(caption)


with tab_lab:
    if not R:
        st.warning("No results found yet. Run  `python train.py`  once to generate the evaluation dashboard.")
    else:
        T = R["test_results"]
        order = ["Global Mean", "Bias Baseline", "Item KNN (raw cosine)", "Item-based KNN", "User-based KNN",
                 "Content-based KNN", "Matrix Factorisation (ALS)", "Hybrid (blend)"]
        best_rmse_v = min(T[m]["RMSE"] for m in order)
        ds_ = R["dataset"]
        st.markdown(hud([("Test ratings", f"{ds_['n_test']:,}", "never seen during training or tuning", False),
                         ("Best RMSE", f"{best_rmse_v:.3f}", R["best_overall"], True),
                         ("Beats global mean by", f"{(1 - best_rmse_v / T['Global Mean']['RMSE']):.0%}", "lower RMSE", False),
                         ("Cross-validated", f"{R['cross_validation'][R['best_overall']]['RMSE_mean']:.3f}",
                          f"± {R['cross_validation'][R['best_overall']]['RMSE_std']:.3f} over 5 folds", True)]),
                    unsafe_allow_html=True)
        head = ("<tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Accuracy</th><th>Precision</th>"
                "<th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>")
        body = ""
        for m in order:
            t = T[m]
            body += (f'<tr class="{"best" if m == R["best_overall"] else ""}"><td>{esc(m)}</td><td>{t["RMSE"]:.3f}</td>'
                     f'<td>{t["MAE"]:.3f}</td><td>{t["R2"]:.3f}</td><td>{t["Accuracy"]:.3f}</td><td>{t["Precision"]:.3f}</td>'
                     f'<td>{t["Recall"]:.3f}</td><td>{t["F1"]:.3f}</td><td>{t["ROC_AUC"]:.3f}</td></tr>')
        st.markdown(f'<div class="sect">Leaderboard (held-out test set)</div><table class="lb">{head}{body}</table>',
                    unsafe_allow_html=True)
        sig = R["significance"]
        st.markdown(f'<div class="note">Paired bootstrap: <b>{esc(sig["model"])}</b> lowers squared error versus the bias baseline by '
                    f'{sig["mean_mse_reduction"]:.3f} on average (95% CI {sig["ci95"][0]:.3f} to {sig["ci95"][1]:.3f}). '
                    f'"Liked" means a rating of 3.5★ or more.</div>', unsafe_allow_html=True)
        sub = st.tabs(["Accuracy", "Classification", "ROC & confusion", "Cross-validation", "Tuning", "Top-N",
                       "Errors", "Features", "Data"])
        with sub[0]:
            show("comparison", "RMSE (with 95% bootstrap interval) and MAE, lower is better.")
        with sub[1]:
            show("classification_bars")
        with sub[2]:
            show("roc_curves")
            show("confusion_matrices")
        with sub[3]:
            show("cv_rmse")
        with sub[4]:
            a, b = st.columns(2)
            with a:
                show("tuning_item"); show("tuning_content")
            with b:
                show("tuning_user"); show("tuning_mf")
            show("hybrid_weights")
        with sub[5]:
            show("ranking_metrics", "Rating predictors alone rank obscure films too high; a small popularity prior fixes it.")
        with sub[6]:
            show("error_analysis")
        with sub[7]:
            show("feature_importance")
            show("genre_appeal")
        with sub[8]:
            show("eda_ratings"); show("eda_long_tail"); show("eda_genres")

# =============================================================================
# TAB 6 - HOW IT WORKS
# =============================================================================
with tab_how:
    st.markdown('<div class="sect">The idea in one minute</div>', unsafe_allow_html=True)
    st.write("MovieSphere turns a table of star ratings into predictions. For every user and every movie they have not "
             "seen, it estimates the rating they would give, then recommends the highest ones.")
    st.markdown('<div class="sect">Cosine similarity</div>', unsafe_allow_html=True)
    st.latex(r"\mathrm{sim}(i,j)=\frac{\sum_u r'_{ui}\,r'_{uj}}{\sqrt{\sum_u r'^2_{ui}}\;\sqrt{\sum_u r'^2_{uj}}}")
    st.caption("r' is the rating minus what we would expect from the user's and the movie's habits (mean-centring).")
    st.markdown('<div class="sect">K-nearest-neighbour prediction (item-based collaborative filtering)</div>', unsafe_allow_html=True)
    st.latex(r"\hat r_{ui}=b_{ui}+\frac{\sum_{j\in N_k(i)}\mathrm{sim}(i,j)\,(r_{uj}-b_{uj})}{\sum_{j\in N_k(i)}\mathrm{sim}(i,j)}")
    st.latex(r"b_{ui}=\mu+b_u+b_i")
    st.write("`N_k(i)` are the k movies most similar to *i* that user *u* has rated. User-based KNN swaps the roles of "
             "users and movies; content-based KNN measures similarity from genres, decade and title words instead of ratings.")
    st.markdown('<div class="sect">Why mean-centre?</div>', unsafe_allow_html=True)
    st.write("Some people rate everything 4-5★, others rarely go above 3. Plain cosine on raw ratings treats a missing rating as a "
             "zero and mostly rediscovers 'popular movies'. Centring ratings first makes the similarity about taste, not mood.")
    st.markdown('<div class="note"><b>Honest limits.</b> Only 610 users; offline numbers do not prove real-world performance; '
                'new users need at least two ratings; popular movies get more accurate predictions than rarely-rated ones.</div>',
                unsafe_allow_html=True)
