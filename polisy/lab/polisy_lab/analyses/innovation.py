# -*- coding: utf-8 -*-
"""AI innovation (PatentsView): growth, geography, concentration, technology network, inventor moves."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import read, finding, dataset, view, wcorr, gini, moran, knn_weights, log, FIPS_ABBR
from . import needs


def run():
    pt = read("patents")
    if pt is None or pt.empty:
        return needs("patents", "AI patenting: growth, geography and technology", "AI & innovation", "patent/county/state",
                     "PatentsView g_patent, g_cpc_current, g_inventor_disambiguated, g_location_disambiguated",
                     "Where is AI invented, how fast is it spreading, and does it follow the political map?")
    y = pt.groupby("year").agg(patents=("patent_id", "size"), ai=("ai", "sum"), ai_broad=("ai_broad", "sum"),
                               ml=("sub_ml", "sum"), vision=("sub_vision", "sum"), language=("sub_language", "sum"),
                               control=("sub_control", "sum"), robotics=("sub_robotics", "sum")).reset_index()
    y = y[y.patents >= 1000] if (y.patents >= 1000).sum() > 5 else y
    y["ai_share"], y["ai_broad_share"] = y.ai / y.patents * 100, y.ai_broad / y.patents * 100
    dataset("patents_year", y)
    view("pat-trend", "line", "AI share of US utility patents by grant year (%)", "patents_year", x="year", y=["ai_share", "ai_broad_share"])
    sub = y.melt(id_vars="year", value_vars=["ml", "vision", "language", "control", "robotics"], var_name="subfield", value_name="patents")
    dataset("patents_subfield", sub)
    view("pat-subfields", "area", "AI patents by subfield and grant year", "patents_subfield", x="year", y="patents", group="subfield")
    a, z = y.iloc[max(0, len(y) - 11)], y.iloc[-1]
    finding("pat-growth", "AI's share of US patents keeps climbing",
            f"AI patents (broad CPC definition) were {a.ai_broad_share:.1f}% of utility patents granted in {int(a.year)} and {z.ai_broad_share:.1f}% in "
            f"{int(z.year)}; the narrow machine-learning definition went from {a.ai_share:.2f}% to {z.ai_share:.2f}%.",
            theme="AI & innovation", level="national", datasets=["PatentsView"], strength="descriptive",
            stats={"first_year": int(a.year), "last_year": int(z.year), "broad_first": a.ai_broad_share, "broad_last": z.ai_broad_share},
            question="Grant-year counts lag filing by 2-3 years: does the post-2022 generative-AI wave show up in applications?",
            next_data="PatentsView pregrant applications (pg_published_application), USPTO AI Patent Dataset", views=["pat-trend", "pat-subfields"], rank=8)
    ps, pc_ = read("patents_state_year"), read("patents_county_year")
    if ps is not None and len(ps):
        rows = []
        for yr, g in ps.groupby("year"):
            if g.patents.sum() > 100:
                rows.append({"year": yr, "gini_all": gini(g.patents), "gini_ai": gini(g.ai_broad_patents)})
        conc = pd.DataFrame(rows)
        if pc_ is not None:
            cc = [{"year": yr, "gini_county_all": gini(g.patents), "gini_county_ai": gini(g.ai_broad_patents)} for yr, g in pc_.groupby("year") if g.patents.sum() > 100]
            conc = conc.merge(pd.DataFrame(cc), on="year", how="left")
        dataset("patent_concentration", conc)
        view("pat-conc", "line", "Geographic concentration of patenting (Gini across states and counties)", "patent_concentration", x="year",
             y=[c for c in ("gini_all", "gini_ai", "gini_county_all", "gini_county_ai") if c in conc])
        if len(conc) > 5:
            c0, c1 = conc.iloc[max(0, len(conc) - 11)], conc.iloc[-1]
            finding("pat-concentration", "Is AI invention concentrating or spreading out?",
                    f"Across states, the Gini of AI patents moved from {c0.gini_ai:.2f} ({int(c0.year)}) to {c1.gini_ai:.2f} ({int(c1.year)}), "
                    f"against {c0.gini_all:.2f} to {c1.gini_all:.2f} for all patents" +
                    (f"; across counties AI {c0.gini_county_ai:.2f} to {c1.gini_county_ai:.2f}." if "gini_county_ai" in conc else "."),
                    theme="AI & innovation", level="state/county", datasets=["PatentsView"], strength="descriptive",
                    stats={"gini_ai_first": c0.gini_ai, "gini_ai_last": c1.gini_ai},
                    question="Does AI diffuse along existing tech hubs or open new places, and do new places differ politically?",
                    next_data="Assignee data (g_assignee_disambiguated) to separate firms from places", views=["pat-conc"], rank=15)
    if pc_ is not None and len(pc_):
        _county_geography(pc_)
    _network()
    _inventor_moves()
    log("innovation: done")
    return True


def _county_geography(pc_):
    recent = pc_[pc_.year >= pc_.year.max() - 4].groupby("county_fips")[["patents", "ai_broad_patents", "inventors"]].sum()
    early = pc_[(pc_.year >= pc_.year.max() - 14) & (pc_.year < pc_.year.max() - 9)].groupby("county_fips")[["patents", "ai_broad_patents"]].sum()
    c = recent.join(early, rsuffix="_early", how="left")
    c["ai_share"] = c.ai_broad_patents / c.patents
    c["ai_growth"] = np.log1p(c.ai_broad_patents) - np.log1p(c.ai_broad_patents_early.fillna(0))
    pl = read("patent_places", columns=["county_fips", "lat", "lon"])
    if pl is not None:
        c = c.join(pl.dropna().groupby("county_fips")[["lat", "lon"]].median())
    for extra, cols in (("county_exposure", ["county_fips", "aige"]),):
        e = read(extra)
        if e is not None:
            c = c.join(e[cols].set_index("county_fips"))
    v = read("votes_county_year")
    if v is not None:
        c = c.join(v[v.year == v.year.max()].set_index("county_fips")[["rep_vote_share"]])
    c = c[c.patents >= 20].reset_index()
    c["state"] = c.county_fips.str[:2].map(FIPS_ABBR)
    dataset("county_patents", c)
    view("county-pat", "points", "Counties: AI patenting in the last five years", "county_patents", lat="lat", lon="lon", size="ai_broad_patents",
         value=[x for x in ("ai_share", "ai_growth", "rep_vote_share", "aige") if x in c], text="county_fips",
         labels={"ai_share": "AI share of patents", "ai_growth": "Growth of AI patenting (log)", "rep_vote_share": "Republican vote share", "aige": "AIGE"})
    view("county-pat-scatter", "scatter", "Counties: AI patenting against politics and exposure", "county_patents",
         x=[x for x in ("rep_vote_share", "aige") if x in c], y=["ai_share", "ai_growth"], size="patents", color="state", text="county_fips",
         labels={"ai_share": "AI share of patents", "ai_growth": "Growth of AI patenting (log)", "rep_vote_share": "Republican vote share", "aige": "AIGE"})
    parts = []
    if "rep_vote_share" in c:
        parts.append(f"AI share of a county's patents correlates {wcorr(c.rep_vote_share, c.ai_share, c.patents):+.2f} with its Republican vote share "
                     f"(weighted by patents) and growth {wcorr(c.rep_vote_share, c.ai_growth, c.patents):+.2f}")
    if "aige" in c:
        parts.append(f"with AI exposure (AIGE) {wcorr(c.aige, c.ai_share, c.patents):+.2f}")
    if "lat" in c and c.lat.notna().sum() > 50:
        g = c.dropna(subset=["lat", "lon"])
        mo = moran(g.ai_share.values, knn_weights(g.lat, g.lon, 8))
        parts.append(f"Moran's I of the AI share across counties = {mo['I']:.2f} (p = {mo['p']:.3f})")
    finding("pat-politics", "AI invention and the political map of counties", "; ".join(parts) + f" ({len(c):,} counties with 20+ patents).",
            theme="Political ideology x AI", level="county", datasets=["PatentsView", "county returns", "AIOE (AIGE)"], strength="suggestive",
            question="Do AI inventors cluster in Democratic counties because of amenities, universities, or the firms located there?",
            next_data="Assignee data; university locations; county education (ACS)", views=["county-pat", "county-pat-scatter"], rank=5)


def _network():
    e = read("ai_cpc_edges")
    if e is None or e.empty:
        return
    import networkx as nx
    per = e.period.unique()
    last = sorted(per, key=lambda p: ("before" not in p, p))[-1]
    x = e[e.period == last]
    G = nx.Graph()
    for r in x.itertuples(index=False):
        G.add_edge(r.a, r.b, weight=r.weight)
    if G.number_of_nodes() < 5:
        return
    strength = dict(G.degree(weight="weight"))
    keep = sorted(strength, key=strength.get, reverse=True)[:60]
    H = G.subgraph(keep).copy()
    comms = list(nx.algorithms.community.greedy_modularity_communities(H, weight="weight"))
    comm = {n: i for i, cset in enumerate(comms) for n in cset}
    pos = nx.spring_layout(H, weight="weight", seed=7, k=0.6)
    nodes = pd.DataFrame([{"id": n, "x": pos[n][0], "y": pos[n][1], "strength": strength[n], "community": f"group {comm[n] + 1}"} for n in H.nodes])
    edges = pd.DataFrame([{"a": a, "b": b, "weight": d["weight"]} for a, b, d in H.edges(data=True)])
    dataset("ai_network_nodes", nodes)
    dataset("ai_network_edges", edges)
    view("ai-network", "network", f"What AI patents combine with: CPC subclasses on AI patents ({last})", "ai_network_nodes",
         edges="ai_network_edges", x="x", y="y", size="strength", color="community", text="id")
    core = {"G06N", "G06V", "G06F", "G10L", "G05B", "G06T"}
    partners = sorted(((n, s) for n, s in strength.items() if n not in core), key=lambda t: -t[1])[:8]
    finding("ai-network", "The technologies AI is being built into",
            f"On AI patents granted {last}, the subclasses most often combined with the AI core are: " + ", ".join(f"{n} ({int(s):,})" for n, s in partners) +
            f". The network splits into {len(comms)} communities.", theme="AI & innovation", level="technology", datasets=["PatentsView"],
            strength="descriptive", question="Which application domains (health, vehicles, finance) pull AI, and do their inventors' places differ politically?",
            next_data="CPC titles; assignee industry codes", views=["ai-network"], rank=17)


def _inventor_moves():
    mv = read("inventor_moves")
    if mv is None or mv.empty:
        return
    recent = mv[mv.year >= mv.year.max() - 9]
    rows = []
    for ai, g in recent.groupby("ai"):
        inn = g.groupby("dest").moves.sum()
        out = g.groupby("origin").moves.sum()
        net = inn.sub(out, fill_value=0)
        rows += [{"state_fips": k, "ai": bool(ai), "net_moves": v, "in": inn.get(k, 0), "out": out.get(k, 0)} for k, v in net.items()]
    n = pd.DataFrame(rows)
    n["state"] = n.state_fips.map(FIPS_ABBR)
    s = read("panel_state_year", "PANELS")
    if s is not None:
        last = s[s.year == s.year.max()].set_index("state_fips")
        for c in ("rep_vote_share", "vr_rep_share", "jobs_2019", "name"):
            if c in last:
                n[c] = n.state_fips.map(last[c])
    dataset("inventor_net", n)
    view("inv-net", "bar", "Net inventor moves by state, last ten years (AI vs other inventors)", "inventor_net", x="net_moves", y="state",
         orientation="h", group="ai")
    a = n[n.ai]
    if "rep_vote_share" in a and a.rep_vote_share.notna().sum() > 20:
        rate = a.net_moves / a.jobs_2019 * 1e5 if "jobs_2019" in a else a.net_moves
        r = wcorr(a.rep_vote_share, rate)
        b = n[~n.ai]
        rate_b = b.net_moves / b.jobs_2019 * 1e5 if "jobs_2019" in b else b.net_moves
        rb = wcorr(b.rep_vote_share, rate_b)
        finding("inventor-moves", "Where AI inventors move, compared with other inventors",
                f"Over the last ten years of grants, states' net gain of AI inventors (per 100k jobs) correlates {r:+.2f} with their Republican vote "
                f"share, against {rb:+.2f} for other inventors. Biggest net gains: " +
                ", ".join(a.nlargest(4, "net_moves").state.astype(str)) + "; biggest losses: " + ", ".join(a.nsmallest(4, "net_moves").state.astype(str)) + ".",
                theme="Migration x AI", level="state", datasets=["PatentsView", "county returns"], strength="suggestive",
                stats={"r_ai": r, "r_other": rb}, question="Are AI inventors sorting politically, or following firms (Texas, Florida relocations)?",
                next_data="Assignee histories; IRS migration of high-income households", views=["inv-net", "state-map"], rank=9)
