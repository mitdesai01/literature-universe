# POLISY Lab: technical report

Version 0.1.0 of `polisy_lab`, 25 September 2026. Research theme 1: Political Ideology × AI × Innovation.

This report explains how to run the lab in Google Colab, what each stage does, which datasets it uses and how they are linked, what the outputs mean, what the first run found, and how to rebuild and publish the website. The site's Methods tab shows this same text.

## 1. What POLISY is

POLISY (Politics, Organizations, Leadership, Strategy & Innovation) is a research lab built as code. It links VRscores measures of workforce partisanship (the share of an employer's, occupation's, industry's or metro's workers registered or modelled as Republican) to open data on AI exposure, AI adoption, migration, state policy and innovation. It then keeps every pattern it finds as a graded finding: the claim, the numbers behind it, the research question it raises and the data that would settle it.

Three rules shape the design:

1. **Nothing is hard-wired to a file name.** Every input is found by what its name looks like and what its header contains, inside folders or inside zip files.
2. **Every join is measured.** Each link between two datasets records how much matched, so a result that rests on half the data says so.
3. **The website is generated.** Analyses register findings, charts and tables; the site draws whatever is registered. Adding an analysis never requires touching the site code.

## 2. What you need

The lab runs in a Colab notebook (`notebooks/POLISY_AI_Innovation_Lab.ipynb`, or the same code as a script in `notebooks/polisy_lab_colab.py`). It needs:

- **The code**: `POLISY_lab_code.zip`, which holds `POLISY_DA/` (the file finder `polisy_core.py` and modules 01-10) and `lab/` (the `polisy_lab` package, this report, the notebook and tools).
- **Your data files**, under any names, in one Drive folder (for example `MyDrive/POLISY/data`). The first run used these:

| Dataset | Files used in the first run | Needed for |
|---|---|---|
| VRscores | `vrscores_report.html` (the report generated 22 Sep 2026), or the canonical Parquet files POLISY_DA writes | all partisanship measures |
| AIOE | fetched from GitHub by the lab | occupation, industry and county AI exposure |
| DAIOE v1.0.0 | `daioe-v1.0.0-scores.zip` | dynamic exposure and the comparison of exposure measures |
| IRS SOI county migration | `IRS_SOI_County_Migration.zip` (Inflow and Outflow folders, 2012-13 to 2021-22) | migration findings |
| Correlates of State Policy | `cspp_data_2026-09-24_academic and policy addition.csv` (in a zip) | state political and policy environment |

Optional sources the lab uses when present: county presidential returns and the Census CBSA delineation file (found by POLISY_DA), the BTOS data downloads (fetched automatically in Colab), IRS state-to-state files (fetched automatically) and the four PatentsView tables (deferred in this version).

## 3. Run it in Google Colab, step by step

### Step 1. Put the files in Drive

Create `MyDrive/POLISY/` with a `data/` folder inside. Put your downloads in `data/` as they came: zips do not need unpacking and names do not need changing. Upload `POLISY_lab_code.zip` to `MyDrive/POLISY/` (or to the Colab file panel).

### Step 2. Open the notebook and run Setup

Open `POLISY_AI_Innovation_Lab.ipynb` in Colab and run the first cell. It installs the few packages Colab lacks (DuckDB, Polars, RapidFuzz, `naics`, `py7zr`, `markdown`) and mounts Drive.

### Step 3. Load the code and point the lab at your files

The second cell unzips the code into `/content/polisy_code`, loads `polisy_core` and `polisy_lab`, and prints both versions. It sets:

- `POLISY_LAB_ROOT` = `MyDrive/POLISY/lab`: everything the lab writes (kept in Drive, so it survives the Colab session);
- `pc.CONFIG["SEARCH_DIRS"]` = your `data/` folder first, then `/content`.

If the inventory later picks the wrong file for an input, point it at the right one, for example `pc.CONFIG["LAB_CSPP_DATA"] = "/content/drive/MyDrive/POLISY/data/cspp.csv"`. The key is `LAB_<SOURCE>_<ROLE>`; source and role names appear in the inventory table.

### Step 4. Run the stages

One cell per stage, so you can stop and inspect:

1. `fetch` downloads what is open and missing: the AIOE repository, DAIOE from Zenodo, BTOS and CSPP files from their pages, and IRS state and county files. Anything already found is skipped.
2. `inventory` lists, for every source and role, how many files were found and which.
3. `profile` samples each file and records its columns, fill rates and example values.
4. `adapt` turns each source into canonical tables (one Parquet file per table, one declared grain each).
5. `link` joins the canonical tables into five panels and records the match rates.
6. `analyze` runs every analysis module and registers findings, charts and tables.
7. `site` writes the website.

A full run on the first run's files takes about a minute after downloads.

Optional first step: if you have the raw VRscores files and want the richer panels (occupation and metro panels by year, employer histories), run POLISY_DA modules 01 to 04 first (`pc.run("01")` ... `pc.run("04")`, or the POLISY_Lab notebook). Run module 05 too: it downloads the Census CBSA delineation file and O*NET titles, which let the lab link metros to CBSAs and occupations by more titles. Without them the lab reads the VRscores report instead and skips metro-level AI exposure.

### Step 5. Look at the results

The notebook prints the inventory, the link rates and the findings table, and serves the site inside Colab (a link opens it in a new tab). Everything is also on Drive under `POLISY/lab/`.

### Step 6. Publish on GitHub Pages

Copy the folder `POLISY/lab/site/` into a GitHub repository, for example as `docs/`. In the repository go to Settings, then Pages, choose "Deploy from a branch", pick the branch and the `/docs` folder, and save. The site appears at `https://<user>.github.io/<repository>/` a minute later. `index.html` loads Plotly and the fonts from public CDNs; `polisy_lab_offline.html` carries Plotly inside and opens without internet.

## 4. The pipeline, stage by stage

All paths below are relative to the lab root (`POLISY_LAB_ROOT`).

| Stage | Code | Writes |
|---|---|---|
| fetch | `sources.fetch_all` | `raw/` (downloads only; your own files are never moved) |
| inventory | `sources.inventory` | the data catalog in `results/results.json` |
| profile | `sources.profile_all` | `results/profiles/profiles.json` |
| adapt | `adapters/*.py` | `canonical/*.parquet` |
| link | `link.py` | `panels/*.parquet`, linkage diagnostics |
| analyze | `analyses/*.py` | findings, views and datasets in `results/results.json`, `results/findings.csv`, `results/linkage_diagnostics.csv` |
| site | `site.py` with `site_assets/` | `site/index.html`, `site/polisy_lab_offline.html`, `site/data/` |

Every stage survives a missing source. An analysis whose data are absent registers a "needs data" finding naming the files it wants, so the site always shows what the lab is waiting for. Running only some stages keeps the earlier stages' results: re-running `analyze` does not erase the catalog or the file profiles.

### Discovery: how files are found

Each source in `sources.SOURCES` declares its roles (for example the IRS source has `county` and `state`). Each role gives a regular expression for file names (tested on a cleaned name: lower case, copy markers such as "(1)" removed), the accepted extensions, and header tokens that must appear in the first rows. A zip is opened and every table inside is tested on its own name, so `IRS_SOI_County_Migration.zip` yields its 20 yearly files. The inventory prints what was found for each role.

### Canonical tables

Adapters write one table per concept, each with a declared grain (`core.GRAINS`); a duplicate key raises a warning in the log. The main tables:

| Table | Grain | From |
|---|---|---|
| `occ_exposure` | SOC 2010 occupation | AIOE, rebuilt from abilities and the application matrix (matches the published scores at r = 0.99999) |
| `ind_exposure`, `county_exposure`, `state_exposure` | NAICS-4, county, state | AIOE (AIIE, AIGE) |
| `oes_staffing`, `qcew_area_total`, `qcew_area_industry` | industry x occupation; area; area x industry | AIOE repository (OES staffing, QCEW 2019) |
| `occ_daioe` | SOC 2010 x year, 2010-2024 | DAIOE 2024 refresh: levels, subdomains and within-year z-scores |
| `occ_soc2018_panel`, `occ_measures_soc2018` | SOC 2018 x year; SOC 2018 | DAIOE SOC 2018 panel, with the comparison measures |
| `vr_occupation_year`, `vr_industry_year`, `vr_metro_year`, `vr_state_year`, `vr_employer_summary`, `vr_sorting` | as named | VRscores (POLISY_DA panels, else the report) |
| `irs_county_year` | county x year | IRS: households, people and income moving in, out and staying, with rates |
| `irs_state_year` | state x year | the counties' interstate totals |
| `irs_flows_county`, `irs_flows_state` | origin x destination x year | IRS county pairs; states summed from them (or IRS state files) |
| `cspp_state_year`, `cspp_catalog`, `cspp_state_profile` | state x year; variable; state | CSPP |
| `geo_state`, `geo_county` | state; county | lookup tables, CBSA delineation when available |

## 5. Datasets

**VRscores** (Kagan, Frake and Hurst, *Organization Science* 2026). Workforce partisanship from voter files matched to employment profiles. The lab reads POLISY_DA's canonical panels when they exist; otherwise it parses the figures inside the VRscores HTML report: occupations (2024), industries with their 2012-2024 change, 359 metros by year, states, 6,000 large employers and the sorting indices. Partisanship is party registration where states register party, and primary participation or L2's model elsewhere; several findings are about this difference.

**AIOE, AIIE, AIGE** (Felten, Raj and Seamans 2021). Exposure of occupations, industries and counties to ten AI applications. The lab clones the repository, rebuilds occupation exposure from O*NET abilities and the application matrix (so per-application exposures and ability shares are available), and expands the "combined industry" codes of the OES and AIIE files into their 4-digit members.

**DAIOE v1.0.0** (Engberg et al. 2026; Zenodo 10.5281/zenodo.21873968). Occupation-year exposure built from measured progress on 140 AI benchmarks, 2010-2024, for nine capability subdomains and a generative-AI composite. The lab uses the 2024 refresh on SOC 2010 (it prefers `refresh-2024` over `frozen-2010-2023` and reads the fastest format present) and the SOC 2018 panel. The index is cumulative: levels rise about fifty-fold between 2012 and 2024 for every occupation alike, so the lab keeps within-year z-scores next to the levels. The SOC 2018 panel also carries published comparison measures: AIOE (Felten, Raj and Seamans 2018, 2021), Webb's (2020) AI, software and robot patent scores, the GPT exposure measures of Eloundou et al. (2024) and Frey and Osborne's (2017) computerisation probability.

**IRS SOI migration data**. For each county and pair of filing years, the files count tax returns (households), exemptions (people) and adjusted gross income moving between counties. Summary rows carry codes in the "other county" columns: 96 total, 97 domestic (000 all, 001 same state, 003 other states), 98 foreign, 57-59 foreign detail and "other flows"; the row whose origin equals its destination counts non-movers. Cells under the disclosure threshold hold -1 and are treated as missing, and county pairs with fewer than 20 returns are folded into "other flows". The lab labels each file by its second year (2021-22 is 2022). Header case and zero-padding differ between years; the adapter normalises both.

**Correlates of State Policy** (Grossmann, Jordan and McCrain, IPPSR). About 3,000 state-year variables, 1900-2020. The file carries no descriptions, so the lab uses a curated set whose meaning was checked against known states (for example `propgoppres` for Texas reads 55.5 in 2012, McCain's 2008 share, and 57.2 from 2014, Romney's), keeps all others in a searchable catalog, and builds a 2012-2016 state profile, because most political variables end between 2014 and 2017.

| Curated CSPP variable | Meaning |
|---|---|
| `propgoppres` | Republican share of the last presidential vote (%) |
| `propgopleg` | Republican share of state legislators (%) |
| `ranney4_control` | Democratic control of state government (Ranney index, 0-1) |
| `inst6014_nom` | State government ideology (0 conservative to 100 liberal) |
| `policyeconlib_est`, `policysociallib_est` | Economic and social policy liberalism (Caughey and Warshaw) |
| `masseconlib_est`, `masssociallib_est` | Public economic and social liberalism (Caughey and Warshaw) |
| `grtw` | Right-to-work law |
| `perc_college` | Adults with a college degree (%) |
| `x_top_corporateincometaxrate` | Top corporate income tax rate (%) |
| `hincomemed`, `incomepcap`, `unemployment` | Median household income, income per capita, unemployment rate |

**Waiting**: Census BTOS (actual AI use by firms; the adapter reads the question-by-answer-by-period workbooks and the Colab fetch downloads them) and PatentsView (AI patents by CPC codes, inventor locations and moves; deferred in this version).

## 6. How the datasets connect

Keys, and how much matched in the first run:

| Join | Key | Matched |
|---|---|---|
| AIOE rebuilt from abilities -> published AIOE | SOC 2010 | 774 of 774 occupations (r = 0.99999) |
| VRscores occupations -> AIOE | O*NET title (exact 63.4%, fuzzy 20.2%) | 83.6% of workers |
| VRscores occupations -> DAIOE | SOC 2010 code from the AIOE link | 83.4% of workers |
| VRscores occupations -> SOC 2018 measures | SOC 2018 title (exact 85.6%, fuzzy 0.2%), else the same SOC code (5.9%) | 88.1% of workers |
| VRscores industry titles -> NAICS-6 | NAICS 2017 titles | 99.98% of workers |
| VRscores industries (NAICS-4) -> AIIE | NAICS-4, combined codes expanded | 72.6% of workers |
| IRS county totals -> AIGE | county FIPS | 3,130 counties |
| IRS county pairs -> interstate totals | state, year | 51.4% of interstate moves |
| CSPP -> states | state FIPS | 51 of 51 |

Two joins deserve care. Occupations are linked by title because the VRscores report shows titles, not codes; the unlinked 12-16% of workers sit in occupations whose titles changed between SOC vintages. And IRS county pairs show only about half of all interstate moves, because the IRS suppresses small county pairs; the lab computes state and county rates from the summary rows (which include everyone) and uses pairs only for directions and corridors.

Metro-level AI exposure (the AIIE of a metro's industry mix, from QCEW by MSA) needs the CBSA delineation file, which POLISY_DA module 05 downloads; it was not available in the first run.

## 7. Methods

- **Weighted correlations** use workers, households or jobs as weights, as each finding states; 95% intervals are percentile bootstraps (1,000 draws, seed 20260925).
- **Regressions** are weighted least squares with heteroskedasticity-robust (HC1) standard errors. Continuous regressors are standardised, so coefficients read as "per standard deviation"; 0/1 indicators stay raw. Fixed effects (occupation group, sector, state) enter as dummies.
- **Spatial statistics**: Moran's I with 8-nearest-neighbour weights on great-circle distance, 999 permutations; local indicators give the high-high and low-low clusters on the metro map.
- **Structure**: principal components; k-means with the number of groups chosen by silhouette; isolation forests for unusual profiles; the migration network uses PageRank and greedy modularity communities (networkx, fixed seeds).
- **Panels**: two-way fixed effects (state and year) with standard errors clustered by state, and Benjamini-Hochberg false-discovery control across all tested variable-outcome pairs.

Each finding carries a grade:

| Grade | Meaning |
|---|---|
| Robust | Holds across weights, controls and samples |
| Suggestive | Consistent sign and a t-statistic of 2 or more in the preferred model; not yet stress-tested |
| Fragile | Depends on weighting or controls; a lead, not a result |
| Descriptive | A pattern worth describing; no test implied |
| Artifact | Most likely produced by how the data were built |
| Needs data | An analysis that runs as soon as its dataset is added |

## 8. What the first run found

The first run used the VRscores report, the AIOE repository, DAIOE v1.0.0, IRS county migration 2012-13 to 2021-22 and the CSPP extract. The site's Findings tab always shows the latest run; the numbers below are from 25 September 2026.

**1. Households are leaving AI-exposed counties, faster every year and within the same state (robust).** Across 3,130 counties, the household-weighted correlation between a county's AI exposure (AIGE) and its net domestic migration went from -0.19 (2012-13) to -0.51 (2020-21). The most exposed fifth of counties lost 0.16% of households a year at the start and 0.80% at the worst. Comparing counties within the same state, one standard deviation more exposure means -0.16 points a year in 2013-16 (t = -3.9), -0.56 in 2017-19 (t = -11.1) and -0.86 in 2020-22 (t = -13.6). The biggest losers are San Francisco, Boston's Suffolk County, Manhattan, the Bronx and Brooklyn; exposed Collin County, Texas, still gained 2.4% a year. *Question:* is this remote work and housing costs pulling knowledge workers out of dense cores, and does it carry AI-exposed work, and its politics, into less exposed and more Republican places? *Data to add:* county votes, remote-work shares and housing costs (ACS), movers' occupations.

**2. AI exposure measures disagree about who is exposed (robust).** On the same 825 occupations, the worker-weighted correlation with the Republican share of workers runs from -0.18 (DAIOE generative AI) to +0.23 (Webb's AI patent score). Weighting occupations equally, benchmark and GPT measures sit near -0.32 and patent and automation measures near +0.25. Holding education and wage equal, Frey and Osborne's computerisation risk keeps a Republican lean (+2.0 points per SD, t = 2.5) and DAIOE's generative-AI score a Democratic one (-1.9, t = -2.7); within occupation groups, Webb's AI patent score leans Republican (+2.1, t = 2.3). The political incidence of "AI exposure" is a choice of measure. *Question:* which measure predicts what happens to workers since 2022, and do exposed occupations shift politically, as robot exposure did after 2016? *Data to add:* the VRscores occupation panel by year, OEWS 2019-2025.

**3. Interstate movers go to more Republican states, and the gap widened after 2019 (robust).** The average interstate move ends in a state whose last presidential vote was 0.8 points more Republican than the origin in 2012-13, 2.5 in 2020-21 and 2.2 in 2021-22; weighted by the income moved, 1.4 to 4.2 points. The political map is held at the 2016 election from 2017 on, so the widening comes from where people moved.

**4. AI exposure looks Democratic across occupations, but the link runs through education (robust).** r = -0.30 across occupations; weighted by workers -0.03 (95% CI -0.15 to +0.08).

**5. Households left states with liberal economic policy and AI-exposed jobs; at the state level the two are hard to separate (suggestive).** Net interstate migration correlates -0.75 with economic policy liberalism, -0.64 with income per capita, +0.62 with right-to-work laws. With policy liberalism and AIGE together, policy carries the association (-0.42 points a year per SD, t = -5.5; AIGE -0.05, t = -0.5), which is why the county comparison within states (finding 1) matters.

**6. AI-exposed counties lose richer households than they gain (robust).** In the most exposed fifth of counties, households moving out report 3.6 thousand dollars more AGI per return than those moving in (2012-13) and 7.5 thousand more (2021-22); in the least exposed fifth, arrivals out-earn departures by 16.1 thousand in 2021-22.

**7-26.** The VRscores findings of the earlier exploration carry on: education pulls occupations Democratic and pay Republican; partisans sort increasingly across employers but less across industries and occupations; AI-exposed industries lean Democratic mostly through education; big-tech workforces barely joined the Democratic drift; metros and states where party is inferred drift Democratic faster than registration states (a measurement regime effect); drift clusters in space along state lines; VRscores coverage is thin in New Jersey, Delaware, New Hampshire and Wyoming (an artifact of multi-state metros); and the largest employer drifts are restructured firms. DAIOE adds that dynamic exposure mostly rises for every occupation alike (levels in 2010 and 2024 correlate 0.98), and that occupations whose standing rose since 2012 have more Democratic workforces (r = -0.26). Movers trade down in county AI exposure, more so since 2020. The migration network splits into five regional communities; the 2021-22 corridors are led by California to Texas and New York to New Jersey and Florida.

**Waiting for data**: AI adoption (BTOS) and AI patenting (PatentsView).

## 9. Outputs and file formats

- `results/results.json`: everything the site shows, in six parts. `findings` is a list of records (id, title, claim, theme, level, datasets, strength, stats, question, next_data, caveats, views, rank). `views` are chart specifications keyed by id, each naming a dataset. `datasets` are tables (columns and rows). `diagnostics` are the joins. `catalog` holds the sources and files found. `profiles` hold column summaries per file. `run` holds versions, times and the adapters' status.
- `results/findings.csv` and `results/linkage_diagnostics.csv`: the same findings and joins as flat tables.
- `canonical/*.parquet` and `panels/*.parquet`: every intermediate table, for your own analyses (`pandas.read_parquet`, DuckDB or Polars).
- `site/data/<table>.csv`: every table behind the charts.

## 10. Regenerating the site

The site is rebuilt from `results.json` in a second:

```python
from polisy_lab.site import build_site
build_site()                                     # latest results -> LAB/site
build_site("path/to/results.json", out="docs")   # any results file, any folder
```

or `run_all(stages=("site",), fetch=False)`. `index.html` is self-contained apart from Plotly (pinned to version 4.1.1 on jsDelivr) and the Google fonts; `polisy_lab_offline.html` inlines Plotly from the installed Python package.

## 11. Extending the lab

**Add a dataset.** Add an entry to `SOURCES` in `sources.py` (title, publisher, url, grain, keys, and for each role the name pattern, extensions and header tokens). Write an adapter in `adapters/` that reads the discovered files with `pc.read_table(path, member)`, uses `find_col` for columns (so a renamed column is logged, not fatal) and writes canonical tables with `write(df, name)`. Register it in `adapters/__init__.py` and its grain in `core.GRAINS`. Then join it in `link.py` with a `diagnostic(...)` line.

**Add an analysis.** Create a module in `analyses/` with a `run()` function. Read panels with `read(name, "PANELS")`; register tables with `dataset(id, df)`, charts with `view(id, kind, title, dataset, **spec)` and results with `finding(...)`; return `needs(...)` when inputs are missing. Add the module to `ANALYSES` and its findings to `PRIORITY` in `analyses/__init__.py`.

**Chart kinds** the site draws, and their fields:

| Kind | Fields |
|---|---|
| `scatter` | `x` and `y` (lists offered as axis choices), `size`, `color`, `text`, `filter`, `labels` |
| `bar` | `x`, `y`, `orientation`, `color`, `group` |
| `coef` | `x` (estimate), `y` (term), `group` (model), `lo`, `hi`, `xlabel` |
| `line`, `area` | `x`, `y`, `group`, `facet`, `text`, `palette` ("sequential" for ordered groups) |
| `network` | node table with `x`, `y`, `size`, `color`, `text`; `edges` names the edge table (a, b, weight) |
| `choropleth` | `location` (state), `year`, `value` (list), `text` |
| `points` | `lat` and `lon`, or `county_fips`; `size`, `year`, `value` (list), `text` |
| `table` | `columns` |

A new chart kind is one renderer function in `site_assets/app.js` (`RENDER.<kind>`).

## 12. Limitations

- Nothing here is causal. The designs (within-state comparisons, controls, fixed effects) rule out some explanations, not all.
- VRscores from the report are a 2024 cross-section for occupations. Metro and state series come from the report's figures, and coverage varies by state.
- Partisanship where party is inferred drifts differently from registered party; findings 11 and 18 measure this.
- AIGE describes a county's 2019 jobs, not the people who move.
- IRS counts households that file tax returns and misses non-filers; small county pairs are suppressed.
- Most CSPP political variables end between 2014 and 2017.
- Titles link occupations across classifications; about 12-16% of workers stay unlinked.

## 13. Reproducibility

- Code versions: `polisy_lab` 0.1.0 (25 Sep 2026), `polisy_core` "2026-09-24 file finder". Tested with Python 3.11 (pandas 3.0) and 3.12 (pandas 2.2).
- Random seeds are fixed (bootstrap 20260925; clustering, layouts and isolation forests seeded).
- Every run records its stages, versions and times in `results.json` (`run`), and every join's match rate.
- The site pins Plotly 4.1.1.

## 14. Sources

- VRscores: Kagan, Frake and Hurst (2026), *Organization Science* 37(2): 444-465, https://ideas.repec.org/a/inm/ororsc/v37y2026i2p444-465.html; data at https://politicsatwork.org/download-data
- AIOE: Felten, Raj and Seamans (2021), *Strategic Management Journal*; https://github.com/AIOE-Data/AIOE
- DAIOE v1.0.0: Engberg, Görg, Hellsten, Javed, Lodefalk, Längkvist, Monteiro, Kyvik Nordås, Pulito, Schroeder and Tang (2026), "AI Unboxed: Capability Arrival and the Clerical Decline"; data https://zenodo.org/records/21873968; code https://github.com/Magnus-L/daioe-pipeline
- Comparison measures in the DAIOE SOC 2018 panel: Eloundou, Manning, Mishkin and Rock (2024), "GPTs are GPTs", *Science*; Webb (2020), "The Impact of Artificial Intelligence on the Labor Market"; Frey and Osborne (2017), "The Future of Employment", *Technological Forecasting and Social Change*
- IRS SOI migration data: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- Correlates of State Policy: https://ippsr.msu.edu/public-policy/correlates-state-policy; policy liberalism from Caughey and Warshaw (2016), *American Journal of Political Science*
- Census Business Trends and Outlook Survey: https://www.census.gov/hfp/btos/data_downloads
- PatentsView: https://patentsview.org/download/data-download-tables
- Map shapes: US Census Bureau cartographic boundaries via us-atlas, https://github.com/topojson/us-atlas
