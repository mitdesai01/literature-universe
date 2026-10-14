# Linkage: VRscores employers to Compustat firms (GVKEY), triangulated and audited

Turns the VRscores-to-Compustat name match into a documented linkage with a measured error rate, following
Akcigit & Goldschlag (2025): a link is trusted when independent sources agree (Compustat names, the patent route
through PatentsView and DISCERN 2.0, and the size trajectory), every link is labelled by how it was confirmed,
selection into the linked sample is shown, and precision is measured by a blind two-coder audit.
No political variable is read while links are built.

| File | What it is |
|---|---|
| `polisy_triangulate.py` | the code; put it in your POLISY folder in Google Drive |
| `POLISY_linkage.ipynb` | the Colab notebook that runs the steps in order |
| `Linkage_instructions.pdf` (`.tex`) | step-by-step instructions, including what goes in the paper |
| `linkage_sensitivity.do` | Stata template: the thesis model once per link set |
| `tests/test_triangulate.py` | code-path test on small made-up files: `python tests/test_triangulate.py . --with-core <POLISY_DA folder>` |

It reads the outputs of the POLISY pipeline and lab (module 04's name match, the VRscores Parquet, the lab's patent
layer and DISCERN tables) and writes everything to POLISY/linkage in Google Drive.
