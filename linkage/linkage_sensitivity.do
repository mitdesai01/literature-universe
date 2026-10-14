* linkage_sensitivity.do: the thesis model once per link set (made by polisy_triangulate.py)
* Before running: edit the five lines marked EDIT. Run it only after the links are frozen and the audit is coded.
clear all
set more off

global LINK   "C:/path/to/POLISY/linkage"            // EDIT: the linkage folder (download it from Google Drive)
global THESIS "C:/path/to/thesis_firm_year.dta"      // EDIT: your firm-year panel with gvkey, year, patents, DIPI, controls
global CEO    ceo_liberalism                         // EDIT: your DIPI CEO liberalism variable (0 = Republican, 1 = Democratic)
global controls "lnsize leverage capint firmage psm"  // EDIT: your control variables
global MODEL  "nbreg patents incongruence $controls i.year, vce(cluster gvkey)"   // EDIT: Table 5, Model 1 of the thesis

* Your thesis file needs one row per gvkey and year (variables named gvkey and year). It may store gvkey as a number or
* as text: the workforce files carry both (gvkey as six-digit text, gvkey_n as a number).
foreach set in strict main broad baseline {
    use "$THESIS", clear
    capture confirm numeric variable gvkey
    if !_rc {
        rename gvkey gvkey_n
        merge 1:1 gvkey_n year using "$LINK/workforce_`set'.dta", keep(match) nogenerate
    }
    else {
        replace gvkey = string(real(gvkey), "%06.0f") if real(gvkey) < .   // "1004" -> "001004", as in Compustat
        merge 1:1 gvkey year using "$LINK/workforce_`set'.dta", keep(match) nogenerate
    }
    capture drop incongruence
    generate incongruence = abs($CEO - vr_dem_share)
    display as text _n "==== link set: `set' ===="
    $MODEL
    estimates store m_`set'
}
estimates table m_strict m_main m_broad m_baseline, keep(incongruence) b(%9.3f) se(%9.3f) stats(N)
* The coefficient on incongruence should be similar across the four sets. Report all four in the linkage appendix.
