# -*- coding: utf-8 -*-
"""run: the whole lab, stage by stage. Every stage survives a missing source and says so.

    from polisy_lab import run_all
    run_all()                                  # fetch -> inventory -> profile -> adapt -> link -> analyze -> site
    run_all(stages=("adapt", "link", "analyze", "site"), fetch=False)
"""
from __future__ import annotations

import time
import traceback

from .core import dirs, log, save_results, RESULTS, LAB, __version__, pc
from . import sources
from .adapters import ADAPTERS
from .link import build_panels
from .analyses import ANALYSES

STAGES = ("fetch", "inventory", "profile", "adapt", "link", "analyze", "site")


def _safe(label, f, *a, **k):
    t = time.time()
    try:
        out = f(*a, **k)
        log(f"-- {label} ({time.time() - t:.0f}s)")
        return out
    except Exception as e:
        log(f"-- {label} FAILED: {type(e).__name__}: {e}")
        log(traceback.format_exc(limit=3))
        RESULTS["run"].setdefault("failed", []).append(f"{label}: {type(e).__name__}: {e}")
        return None


def run_all(stages=STAGES, fetch=True, patentsview=False, adapters=None, analyses=None):
    dirs()
    RESULTS["run"].pop("failed", None)
    RESULTS["run"].update({"started": time.strftime("%Y-%m-%d %H:%M"), "version": __version__, "polisy_core": pc.__version__,
                           "stages": [s for s in stages if s != "fetch" or fetch], "analyses_only": list(analyses or [])})
    if "fetch" in stages and fetch:
        _safe("fetch open datasets", sources.fetch_all, patentsview=patentsview)
    if "inventory" in stages:
        inv = _safe("inventory", sources.inventory)
        if inv is not None:
            print(inv[["source", "role", "files"]].to_string(index=False))
    if "profile" in stages:
        _safe("profile every file found", sources.profile_all)
    if "adapt" in stages:
        status = {}
        for name, f in ADAPTERS.items():
            if adapters and name not in adapters:
                continue
            status[name] = bool(_safe(f"adapter {name}", f))
        RESULTS["run"]["adapters"] = status
    if "link" in stages:
        _safe("build panels", build_panels)
    if "analyze" in stages:
        for mod in ANALYSES:
            name = mod.__name__.split(".")[-1]
            if analyses and name not in analyses:
                continue
            _safe(f"analysis {name}", mod.run)
        from .analyses import apply_priority
        apply_priority(RESULTS["findings"])
    path = save_results()
    if "site" in stages:
        from .site import build_site
        _safe("build site", build_site)
    return path


if __name__ == "__main__":
    run_all()
