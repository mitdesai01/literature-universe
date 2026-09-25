# -*- coding: utf-8 -*-
"""Migration (IRS SOI): who moves away from AI-exposed places, the partisan direction of moves, the state network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import read, finding, dataset, view, wcorr, wls, log, FIPS_ABBR
from . import needs

PERIODS = {"2013-16": (2013, 2016), "2017-19": (2017, 2019), "2020-22": (2020, 2022)}


def run():
    cy, fc, fs, sy = read("irs_county_year"), read("irs_flows_county"), read("irs_flows_state"), read("irs_state_year")
    if cy is None and fs is None:
        return needs("migration", "Migration, politics and AI places", "Migration x AI", "state/county", "IRS SOI county (or state) migration files",
                     "Do households and income move towards AI-exposed or innovative places, and towards which politics?")
    ae = read("county_exposure")
    if cy is not None and ae is not None:
        _county_ai(cy, ae)
        if fc is not None:
            _exposure_gap(fc, ae)
    if fs is not None and len(fs):
        _state_network(fs, sy)
        _partisan_direction_state(fs)
    if fc is not None and len(fc):
        _partisan_direction(fc)
    log("migration: done")
    return True


def _county_ai(cy, ae):
    """Net domestic migration of households against the AI exposure of the county's jobs (AIGE, 2019 employment)."""
    names = read("irs_county_names")
    c = cy.merge(ae[["county_fips", "aige"]], on="county_fips", how="inner").dropna(subset=["net_migration_rate", "aige", "base_returns"]).copy()
    if c.county_fips.nunique() < 500:
        return
    c["net_pp"] = c.net_migration_rate * 100
    c["log_size"] = np.log(c.base_returns)
    c["log_income"] = np.log(c.stay_agi_per_return)
    c["quintile"] = c.groupby("year").aige.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop") + 1)
    qn = {1: "1 least exposed", 2: "2", 3: "3", 4: "4", 5: "5 most exposed"}
    q = c.groupby(["year", "quintile"]).apply(lambda g: pd.Series({
        "net_pp": np.average(g.net_pp, weights=g.base_returns),
        "income_gap": np.average(g.mover_income_gap.fillna(0), weights=g.base_returns)}), include_groups=False).reset_index()
    q["quintile"] = q.quintile.map(qn)
    dataset("mig_aige_quintiles", q)
    view("mig-quintiles", "line", "Net domestic migration (% of households a year) by AI exposure of the county, fifths of counties", "mig_aige_quintiles",
         x="year", y="net_pp", group="quintile", palette="sequential", labels={"net_pp": "Net migration (% of households)", "quintile": "AIGE fifth"})
    view("mig-income-gap", "line", "Income of households moving in minus moving out ($000 AGI per return), by AI exposure of the county",
         "mig_aige_quintiles", x="year", y="income_gap", group="quintile", palette="sequential",
         labels={"income_gap": "In-movers minus out-movers ($000 per return)"})
    yr = pd.DataFrame([{"year": y, "r_weighted": wcorr(g.aige, g.net_pp, g.base_returns), "r_counties": wcorr(g.aige, g.net_pp),
                        "r_income_gap": wcorr(g.aige, g.mover_income_gap, g.base_returns)} for y, g in c.groupby("year")])
    dataset("mig_aige_corr", yr)
    view("mig-aige-corr", "line", "Correlation of county AI exposure with net migration and with movers' income gap, by year", "mig_aige_corr",
         x="year", y=["r_weighted", "r_counties", "r_income_gap"],
         labels={"r_weighted": "Net migration, weighted by households", "r_counties": "Net migration, counties weighted equally",
                 "r_income_gap": "Movers' income gap, weighted"})
    models, fe = [], {}
    for per, (a, b) in PERIODS.items():
        d = c[c.year.between(a, b)].groupby("county_fips").agg(net_pp=("net_pp", "mean"), aige=("aige", "first"), base=("base_returns", "mean"),
                                                              log_size=("log_size", "mean"), log_income=("log_income", "mean"),
                                                              state=("state_fips", "first")).dropna()
        for name, xs, f in (("AIGE alone", ["aige"], None), ("+ state fixed effects", ["aige"], "state"),
                            ("+ size, income, state fixed effects", ["aige", "log_size", "log_income"], "state")):
            m = wls(d, "net_pp", xs, "base", fe=f)
            if m:
                t = m["terms"]["aige"]
                models.append({"model": name, "term": per, "coef": t["coef"], "lo": t["coef"] - 1.96 * t["se"], "hi": t["coef"] + 1.96 * t["se"],
                               "t": t["t"], "n": m["n"]})
                if name == "+ state fixed effects":
                    fe[per] = t
    dataset("mig_aige_models", pd.DataFrame(models))
    view("mig-aige-models", "coef", "Net migration (points a year) per SD of county AI exposure, by period", "mig_aige_models",
         x="coef", y="term", group="model", lo="lo", hi="hi", xlabel="Percentage points of households a year per SD of AIGE, 95% interval")
    last = c[c.year.between(2020, 2022)].groupby("county_fips").agg(net_pp=("net_pp", "mean"), aige=("aige", "first"), base_returns=("base_returns", "mean"),
                                                                    income_gap=("mover_income_gap", "mean"), state=("state_fips", "first"))
    if names is not None:
        last = last.join(names.set_index("county_fips")[["county_name"]])
    last = last.reset_index()
    last["state"] = last.state.map(FIPS_ABBR)
    last["label"] = last.get("county_name", last.county_fips).fillna(last.county_fips) + ", " + last.state.fillna("")
    big = last[last.base_returns >= 200000]
    losers = big.nsmallest(5, "net_pp")
    exposed_gainers = big[big.aige > big.aige.quantile(0.75)].nlargest(3, "net_pp")
    cm = c.merge(last[["county_fips", "label"]], on="county_fips", how="left")
    cm = cm[cm.groupby("county_fips").base_returns.transform("mean") >= 10000]    # the map shows counties with 10,000+ households
    dataset("county_migration", cm[["county_fips", "label", "year", "net_pp", "aige", "base_returns", "mover_income_gap", "net_agi_rate"]]
            .assign(net_agi_rate=lambda d: d.net_agi_rate * 100), note="counties with 10,000 or more households (tax returns)")
    view("county-migration-map", "points", "Counties: net domestic migration by year (IRS)", "county_migration", text="label", size="base_returns", year="year",
         value=["net_pp", "net_agi_rate", "mover_income_gap", "aige"],
         labels={"net_pp": "Net migration (% of households)", "net_agi_rate": "Net income migration (% of AGI)",
                 "mover_income_gap": "In-movers minus out-movers ($000 per return)", "aige": "AI exposure of the county (AIGE)",
                 "base_returns": "Households (returns)"})
    dataset("county_migration_summary", last[["county_fips", "label", "state", "net_pp", "aige", "base_returns", "income_gap"]])
    view("county-migration-explorer", "scatter", "Counties, 2020-22: AI exposure against net domestic migration", "county_migration_summary",
         x=["aige"], y=["net_pp", "income_gap"], size="base_returns", color="state", text="label", filter=["state"],
         labels={"aige": "AI exposure (AIGE)", "net_pp": "Net migration 2020-22 (% of households a year)", "income_gap": "In-movers minus out-movers ($000)",
                 "base_returns": "Households"})
    y0, y1 = yr.iloc[0], yr.iloc[-1]
    peak = yr.loc[yr.r_weighted.idxmin()]
    q5 = q[q.quintile == qn[5]].set_index("year").net_pp
    f = lambda per: f"{fe[per]['coef']:+.2f} (t = {fe[per]['t']:+.1f})" if per in fe else "n/a"  # noqa: E731
    finding("mig-ai-exodus", "Households are leaving AI-exposed counties, faster every year and within the same state",
            f"Across {c.county_fips.nunique():,} counties, the correlation between AI exposure (AIGE) and net domestic migration went from "
            f"{y0.r_weighted:+.2f} ({int(y0.year) - 1}-{str(int(y0.year))[2:]}) to {peak.r_weighted:+.2f} ({int(peak.year) - 1}-{str(int(peak.year))[2:]}), weighted by "
            f"households. The most exposed fifth of counties lost {-q5.iloc[0]:.2f}% of households a year at the start and {-q5.min():.2f}% at the worst. "
            f"Comparing counties within the same state, one SD more exposure means {f('2013-16')} points a year in 2013-16, {f('2017-19')} in 2017-19 "
            f"and {f('2020-22')} in 2020-22. Biggest losers 2020-22 among large counties: {'; '.join(f'{r.label} ({r.net_pp:+.1f}%)' for r in losers.itertuples())}. "
            f"Exposed counties still gaining: {'; '.join(f'{r.label} ({r.net_pp:+.1f}%)' for r in exposed_gainers.itertuples())}.",
            theme="Migration x AI", level="county", datasets=["IRS SOI migration", "AIOE (AIGE)"], strength="robust",
            stats={"r_first": y0.r_weighted, "r_peak": peak.r_weighted, **{f"fe_{k}": v["coef"] for k, v in fe.items()}},
            question="Is this remote work and housing costs pulling knowledge workers out of dense cores, and does it move AI-exposed work, "
                     "and its politics, into less exposed and more Republican places?",
            next_data="County votes (MIT) for the partisan side; ACS remote-work shares and housing costs; movers' occupations (ACS migration microdata)",
            caveats=["AIGE describes the county's 2019 jobs, not the movers themselves.",
                     "IRS counts tax returns (households); year = the second year of each filing pair."],
            views=["mig-quintiles", "mig-aige-models", "mig-aige-corr", "county-migration-map", "county-migration-explorer"], rank=1)
    q5g = q[q.quintile == qn[5]].set_index("year").income_gap
    q1g = q[q.quintile == qn[1]].set_index("year").income_gap
    finding("mig-income", "AI-exposed counties lose richer households than they gain",
            f"In the most exposed fifth of counties, households moving out report more income than those moving in every year: "
            f"{q5g.iloc[0]:+.1f} thousand dollars of AGI per return in {int(q5g.index[0])} and {q5g.iloc[-1]:+.1f} in {int(q5g.index[-1])}. "
            f"In the least exposed fifth the gap is {q1g.iloc[0]:+.1f} and {q1g.iloc[-1]:+.1f}. The correlation of exposure with the income gap is "
            f"{y1.r_income_gap:+.2f} in {int(y1.year)}.",
            theme="Migration x AI", level="county", datasets=["IRS SOI migration", "AIOE (AIGE)"], strength="robust",
            stats={"gap_exposed_first": q5g.iloc[0], "gap_exposed_last": q5g.iloc[-1], "gap_least_first": q1g.iloc[0], "gap_least_last": q1g.iloc[-1]},
            question="Do high earners in AI-exposed jobs carry their (remote) work with them, spreading AI-exposed employment to new places?",
            next_data="IRS migration by income class (gross migration files); LinkedIn or Revelio worker moves",
            views=["mig-income-gap", "county-migration-map"], rank=7)


def _exposure_gap(fc, ae):
    a = ae.set_index("county_fips").aige
    x = fc[fc.origin != fc.dest].assign(o=lambda d: d.origin.map(a), d_=lambda d: d.dest.map(a)).dropna(subset=["o", "d_", "returns"])
    if len(x) < 1000:
        return
    rows = []
    for y, g in x.groupby("year"):
        inter = g.origin.str[:2] != g.dest.str[:2]
        k = g.dropna(subset=["agi"])
        rows.append({"year": int(y), "gap": np.average(g.d_ - g.o, weights=g.returns) / a.std(),
                     "gap_interstate": np.average((g.d_ - g.o)[inter], weights=g.returns[inter]) / a.std(),
                     "gap_income_weighted": np.average(k.d_ - k.o, weights=k.agi.clip(lower=0) + 1e-9) / a.std()})
    t = pd.DataFrame(rows)
    dataset("mig_exposure_gap", t)
    view("mig-exposure-gap", "line", "Destination minus origin AI exposure of movers (SD of county AIGE)", "mig_exposure_gap", x="year",
         y=["gap", "gap_interstate", "gap_income_weighted"],
         labels={"gap": "All county-to-county moves", "gap_interstate": "Moves across states", "gap_income_weighted": "Weighted by income moved"})
    finding("mig-exposure-gap", "Movers trade down in AI exposure, and more so since 2020",
            f"Weighted by households, the average county-to-county move ends in a county {abs(t.gap.iloc[0]):.3f} SD of AIGE less exposed than its "
            f"origin in {int(t.year.iloc[0])} and {abs(t.gap.iloc[-1]):.3f} SD in {int(t.year.iloc[-1])} (largest: {abs(t.gap.min()):.3f} in "
            f"{int(t.year.iloc[t.gap.idxmin()])}); weighted by income moved it is {abs(t.gap_income_weighted.min()):.3f} at its largest.",
            theme="Migration x AI", level="county", datasets=["IRS SOI migration", "AIOE (AIGE)"], strength="descriptive",
            stats={"gap_first": t.gap.iloc[0], "gap_last": t.gap.iloc[-1], "gap_min": t.gap.min()},
            question="Does the down-trading in exposure change the local AI adoption and politics of destination counties?",
            next_data="BTOS AI use by state and MSA; county votes", views=["mig-exposure-gap"], rank=16,
            caveats=["County pairs with fewer than 20 returns are suppressed by the IRS, so small flows are missing."])


def _state_network(fs, sy):
    import networkx as nx
    last = int(fs.year.max())
    x = fs[(fs.year == last) & (fs.origin != fs.dest)]
    G = nx.DiGraph()
    for r in x.itertuples(index=False):
        G.add_edge(r.origin, r.dest, weight=r.returns)
    pr = nx.pagerank(G, weight="weight")
    nodes = pd.DataFrame({"state_fips": list(pr)}).assign(state=lambda d: d.state_fips.map(FIPS_ABBR), pagerank=lambda d: d.state_fips.map(pr))
    if sy is not None:          # complete interstate totals, not only the visible pairs
        s = sy[sy.year == last].set_index("state_fips")
        nodes["net_returns"] = nodes.state_fips.map(s.net_returns)
        nodes["net_migration_pct"] = nodes.state_fips.map(s.net_migration_rate * 100)
        nodes["net_agi"] = nodes.state_fips.map(s.net_agi)
    else:
        inn, out = x.groupby("dest").returns.sum(), x.groupby("origin").returns.sum()
        nodes["net_returns"] = nodes.state_fips.map(inn.sub(out, fill_value=0))
        nodes["net_agi"] = nodes.state_fips.map(x.groupby("dest").agi.sum().sub(x.groupby("origin").agi.sum(), fill_value=0))
    und = G.to_undirected()
    comms = list(nx.algorithms.community.greedy_modularity_communities(und, weight="weight"))
    nodes["community"] = nodes.state_fips.map({n: f"group {i + 1}" for i, c in enumerate(comms) for n in c})
    pos = nx.spring_layout(und, weight="weight", seed=3)
    nodes["x"], nodes["y"] = nodes.state_fips.map(lambda n: pos[n][0]), nodes.state_fips.map(lambda n: pos[n][1])
    nodes["id"] = nodes.state
    top = x.nlargest(120, "returns")[["origin", "dest", "returns", "agi"]].assign(a=lambda d: d.origin.map(FIPS_ABBR), b=lambda d: d.dest.map(FIPS_ABBR))
    dataset("mig_nodes", nodes)
    dataset("mig_edges", top.rename(columns={"returns": "weight"})[["a", "b", "weight", "agi"]])
    view("mig-network", "network", f"State migration network {last - 1}-{str(last)[2:]} (120 largest corridors between counties' states)", "mig_nodes",
         edges="mig_edges", x="x", y="y", size="pagerank", color="community", text="id")
    corr = x.assign(o=x.origin.map(FIPS_ABBR), d=x.dest.map(FIPS_ABBR)).nlargest(5, "returns")
    groups = nodes.groupby("community").state.apply(lambda s: ", ".join(sorted(s.astype(str))))
    finding("mig-network", "The US migration network in the latest year",
            f"{last - 1}-{str(last)[2:]}: the largest net gainers of households were " + ", ".join(nodes.nlargest(4, "net_returns").state.astype(str)) +
            ", the largest net losers " + ", ".join(nodes.nsmallest(4, "net_returns").state.astype(str)) + ". Biggest corridors: " +
            "; ".join(f"{a} -> {b} ({int(r):,} returns)" for a, b, r in zip(corr.o, corr.d, corr.returns)) +
            f". The network splits into {len(comms)} communities: " + " | ".join(groups) + ".",
            theme="Migration x AI", level="state", datasets=["IRS SOI migration"], strength="descriptive",
            question="Do migration communities follow political regions, and do AI hubs sit at their centre or their edge?",
            next_data="IRS state-to-state files (complete flows); earlier years for trends", views=["mig-network"], rank=18,
            caveats=["Corridors come from county-to-county pairs, which show about half of interstate moves; net totals use all moves."])


def _partisan_direction_state(fs):
    """Destination minus origin Republican presidential share (CSPP), holding each state's political map at the last
    election before the move (CSPP ends in 2017, so moves after 2017 use the 2016 election)."""
    cs = read("cspp_state_year")
    if cs is None or "propgoppres" not in cs:
        return
    pres = cs[["state_fips", "year", "propgoppres"]].dropna()
    rows = []
    for y, g in fs[fs.origin != fs.dest].groupby("year"):
        v = pres[pres.year <= y].sort_values("year").groupby("state_fips").propgoppres.last()
        k = g.assign(o=g.origin.map(v), d=g.dest.map(v)).dropna(subset=["o", "d", "returns"])
        if k.empty:
            continue
        ka = k.dropna(subset=["agi"])
        rows.append({"year": int(y), "shift_pp": np.average(k.d - k.o, weights=k.returns),
                     "shift_agi_pp": np.average(ka.d - ka.o, weights=ka.agi.clip(lower=0) + 1e-9), "moves": k.returns.sum(),
                     "map_year": int(pres[pres.year <= y].year.max())})
    t = pd.DataFrame(rows)
    if t.empty:
        return
    dataset("mig_partisan_state", t)
    view("mig-partisan-state", "line", "Interstate movers: destination minus origin Republican presidential share (points)", "mig_partisan_state",
         x="year", y=["shift_pp", "shift_agi_pp"], labels={"shift_pp": "Weighted by households", "shift_agi_pp": "Weighted by income moved"})
    direction = "more Republican" if t.shift_pp.mean() > 0 else "more Democratic"
    finding("mig-partisan-state", f"Interstate movers go to {direction} states, and the gap widened after 2019",
            f"Weighted by households, the average interstate move ends in a state whose last presidential vote was {t.shift_pp.iloc[0]:+.1f} points "
            f"more Republican than the origin in {int(t.year.iloc[0])}, {t.shift_pp.max():+.1f} at the peak ({int(t.year.iloc[t.shift_pp.idxmax()])}) and "
            f"{t.shift_pp.iloc[-1]:+.1f} in {int(t.year.iloc[-1])}; weighted by the income moved, {t.shift_agi_pp.iloc[0]:+.1f} to {t.shift_agi_pp.max():+.1f}. "
            f"Moves from {int(t.year[t.map_year == t.map_year.max()].min())} on hold the map at the {t.map_year.max() - 1} election, so the widening "
            f"comes from where people moved, not from how states voted.",
            theme="Migration x AI", level="state", datasets=["IRS SOI migration", "CSPP"], strength="robust" if (t.shift_pp > 0).all() or (t.shift_pp < 0).all() else "suggestive",
            stats={"shift_first": t.shift_pp.iloc[0], "shift_max": t.shift_pp.max(), "shift_agi_max": t.shift_agi_pp.max()},
            question="Is partisan re-sorting by migration large enough to move state electorates, and is it driven by policy, housing or jobs?",
            next_data="IRS state-to-state files (all interstate moves); county votes for the within-state picture; movers' ages",
            caveats=["CSPP's presidential share is the state's result in the last election before each year (through the 2016 election).",
                     "Flows are county-to-county pairs summed to states (about half of interstate moves)."],
            views=["mig-partisan-state", "mig-network"], rank=3)


def _partisan_direction(fc):
    """County version: needs county presidential returns (POLISY_DA COUNTYPRES)."""
    v = read("votes_county_year")
    if v is None:
        return
    rows = []
    for y, g in fc[fc.origin != fc.dest].groupby("year"):
        vv = v[v.year <= y]
        if vv.empty:
            continue
        vv = vv[vv.year == vv.year.max()].set_index("county_fips").rep_vote_share
        k = g.assign(o=g.origin.map(vv), d=g.dest.map(vv)).dropna(subset=["o", "d"])
        if k.returns.sum() == 0:
            continue
        ka = k.dropna(subset=["agi"])
        rows.append({"year": int(y), "shift_pp": np.average(k.d - k.o, weights=k.returns) * 100,
                     "shift_agi_pp": np.average(ka.d - ka.o, weights=ka.agi.clip(lower=0) + 1e-9) * 100, "returns": k.returns.sum()})
    t = pd.DataFrame(rows)
    if t.empty:
        return
    dataset("mig_partisan", t)
    view("mig-partisan", "line", "County movers: destination minus origin Republican vote share (points)", "mig_partisan", x="year", y=["shift_pp", "shift_agi_pp"],
         labels={"shift_pp": "Weighted by households", "shift_agi_pp": "Weighted by income moved"})
    direction = "more Republican" if t.shift_pp.mean() > 0 else "more Democratic"
    finding("mig-partisan", f"County movers go to {direction} counties than they leave",
            f"Weighted by households, the average destination county is {t.shift_pp.mean():+.2f} points more Republican than the origin "
            f"(years {t.year.min()}-{t.year.max()}, latest {t.shift_pp.iloc[-1]:+.2f}); weighted by income moved, {t.shift_agi_pp.mean():+.2f}.",
            theme="Migration x AI", level="county", datasets=["IRS SOI migration", "county returns"],
            strength="suggestive" if abs(t.shift_pp.mean()) > 0.5 else "descriptive", stats={"mean_shift_pp": t.shift_pp.mean()},
            question="Is this political sorting, housing costs, or jobs? Do AI-exposed workers move differently?",
            next_data="IRS flows by income and age; occupation of movers (ACS migration microdata)", views=["mig-partisan"], rank=11)
