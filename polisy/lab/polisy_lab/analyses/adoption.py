# -*- coding: utf-8 -*-
"""AI adoption (BTOS): who uses AI, how fast use is growing, and exposure versus use."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import read, finding, dataset, view, wcorr, wls, log
from . import needs


def run():
    b = read("btos_ai")
    if b is None or b.empty:
        return needs("adoption", "Actual AI adoption by firms (BTOS)", "AI adoption", "sector/state", "Census BTOS data downloads",
                     "Where is AI actually used, compared with where it could be (exposure)?")
    b = b.copy()
    b["date"] = pd.to_datetime(b.date, errors="coerce")
    nat = b[(b.level == "national")].sort_values("date")
    if len(nat):
        dataset("btos_national", nat[["date", "measure", "rate"]].assign(date=lambda d: d.date.dt.strftime("%Y-%m-%d")))
        view("btos-trend", "line", "Share of US firms using AI (BTOS, %)", "btos_national", x="date", y="rate", group="measure")
        now = nat[nat.measure == "ai_use_now"]
        if len(now) >= 2:
            finding("btos-trend", "AI use among US firms over the survey period",
                    f"The share of firms reporting AI use in producing goods or services went from {now.rate.iloc[0]:.1f}% "
                    f"({now.date.iloc[0]:%b %Y}) to {now.rate.iloc[-1]:.1f}% ({now.date.iloc[-1]:%b %Y}).",
                    theme="AI adoption", level="national", datasets=["BTOS"], strength="descriptive",
                    stats={"first": float(now.rate.iloc[0]), "last": float(now.rate.iloc[-1])},
                    question="Which firms (size, age, leadership, workforce politics) drive the rise?", next_data="BTOS microdata (FSRDC)",
                    views=["btos-trend"], rank=14)
    for lvl in ("sector", "size", "msa"):
        x = b[(b.level == lvl) & (b.measure == "ai_use_now")]
        if len(x):
            x = x[x.date == x.date.max()].sort_values("rate")
            dataset(f"btos_{lvl}", x[["geo", "rate"]])
            view(f"btos-{lvl}", "bar", f"Firms using AI by {lvl} (latest period, %)", f"btos_{lvl}", x="rate", y="geo", orientation="h")
    s = read("panel_state_year", "PANELS")
    if s is not None and "btos_ai_use_now" in s and s.btos_ai_use_now.notna().sum() > 20:
        y = int(s.loc[s.btos_ai_use_now.notna(), "year"].max())
        x = s[s.year == y].dropna(subset=["btos_ai_use_now"]).copy()
        parts = []
        if "aige" in x:
            r1 = wcorr(x.aige, x.btos_ai_use_now, x.get("jobs_2019"))
            parts.append(f"AI exposure (AIGE) correlates {r1:+.2f} with the share of firms using AI")
            k = x.dropna(subset=["aige"])
            beta = np.polyfit(k.aige, k.btos_ai_use_now, 1)
            x.loc[k.index, "use_beyond_exposure"] = k.btos_ai_use_now - np.polyval(beta, k.aige)
        for pol, lab in (("rep_vote_share", "Republican vote share"), ("vr_rep_share", "Republican share of the workforce")):
            if pol in x and x[pol].notna().sum() > 20:
                parts.append(f"{lab} correlates {wcorr(x[pol], x.btos_ai_use_now):+.2f} with use"
                             + (f" and {wcorr(x[pol], x.use_beyond_exposure):+.2f} with use beyond exposure" if "use_beyond_exposure" in x else ""))
        dataset("btos_state", x[[c for c in ("state", "name", "btos_ai_use_now", "aige", "use_beyond_exposure", "rep_vote_share", "vr_rep_share", "jobs_2019") if c in x]])
        view("btos-states", "scatter", "States: AI exposure against actual AI use", "btos_state",
             x=[c for c in ("aige", "rep_vote_share", "vr_rep_share") if c in x], y=[c for c in ("btos_ai_use_now", "use_beyond_exposure") if c in x],
             size="jobs_2019" if "jobs_2019" in x else None, text="name",
             labels={"aige": "AI exposure (AIGE)", "btos_ai_use_now": "Firms using AI (%)", "use_beyond_exposure": "Use beyond exposure",
                     "rep_vote_share": "Republican vote share", "vr_rep_share": "Republican share of the workforce"})
        finding("btos-states", "Exposure versus actual AI use across states, and where politics enters",
                f"In {y}: " + "; ".join(parts) + f" ({len(x)} states).", theme="AI adoption", level="state", datasets=["BTOS", "AIOE (AIGE)", "votes", "VRscores"],
                strength="suggestive", question="Does the political environment speed or slow AI adoption beyond what exposure predicts?",
                next_data="BTOS state x sector; state AI policies (NCSL); CSPP", views=["btos-states", "state-map"], rank=6)
    log("adoption: done")
    return True
