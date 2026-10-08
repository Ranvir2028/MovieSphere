"""
Smoke-test for app.py that does NOT need Streamlit installed.

It replaces the `streamlit` module with a tiny fake whose widgets return sensible
defaults, then executes app.py top to bottom. If the script raises an exception
(NameError, KeyError, bad HTML helper, wrong engine call ...) this test fails.

    python tests/run_app_stub.py
"""
import contextlib
import os
import runpy
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = []          # everything the app tried to render (for inspection)


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return _make(name)


def _make(name):
    def fn(*args, **kw):
        if name in ("markdown", "write", "caption", "info", "warning", "error", "success", "latex", "image"):
            OUT.append((name, args[0] if args else ""))
        if name == "selectbox":
            opts = list(args[1]) if len(args) > 1 else list(kw["options"])
            idx = kw.get("index", 0) or 0
            return opts[idx]
        if name == "multiselect":
            opts = list(args[1]) if len(args) > 1 else list(kw["options"])
            return list(SCENARIO.get(kw.get("key"), []) or kw.get("default", []))
        if name == "slider":
            if "value" in kw:
                return kw["value"]
            return args[3] if len(args) > 3 else args[1]
        if name == "columns":
            n = args[0]
            n = len(n) if hasattr(n, "__len__") else n
            return [_Ctx() for _ in range(n)]
        if name == "tabs":
            return [_Ctx() for _ in args[0]]
        if name in ("expander", "container", "spinner", "form", "empty", "status"):
            return _Ctx()
        if name in ("button", "checkbox", "toggle", "form_submit_button"):
            return False
        return None
    return fn


SCENARIO = {}


def make_fake():
    st = types.ModuleType("streamlit")
    st.__getattr__ = _make                      # any st.xxx(...) works

    def cache_resource(*a, **kw):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f
    st.cache_resource = cache_resource
    st.cache_data = cache_resource
    st.sidebar = _Ctx()
    st.session_state = {}
    return st


def run(scenario=None):
    OUT.clear()
    SCENARIO.clear()
    SCENARIO.update(scenario or {})
    sys.modules["streamlit"] = make_fake()
    runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
    return list(OUT)


if __name__ == "__main__":
    print("== scenario 1: defaults (existing-user tab, similar movies, predict, lab) ==")
    out = run()
    print(f"rendered {len(out)} elements, no exceptions")

    from engine import Engine
    e = Engine()
    pop = e.popular_titles(25)
    picks = [int(x) for x in pop["movieId"].values[:3]]
    print("== scenario 2: new visitor with 3 rated movies ==")
    out = run({"n_pick": picks})
    html_blob = " ".join(str(o[1]) for o in out)
    assert "mcard" in html_blob
    # the HTML must contain no blank lines / indented lines (Streamlit would treat them as code)
    for kind, text in out:
        if kind == "markdown" and "<div" in str(text):
            assert "\n" not in str(text).strip(), "HTML block contains a newline"
    print("OK")
