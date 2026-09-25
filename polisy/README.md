# POLISY pipeline

- `POLISY_DA/`: `polisy_core.py` and modules `01`–`10`. Zip the files in this folder as `POLISY_DA.zip` for Colab.
- `POLISY_Lab.ipynb`: runs the pipeline in Colab. It loads the code only from `POLISY_DA.zip` and runs module 05 (downloads) before 01–10.
- `tests/smoke_test.py`: builds small fake downloads with realistic names ("(1)" copies, renamed or unzipped zips, a WRDS random name, a `.tab` county file) and runs every module on them. Run it with `python tests/smoke_test.py`.

## How input files are found

Every input is declared once, in `FILES` in `polisy_core.py`: what it is, the name it downloads as, the looser names that are accepted, and what has to be inside it. `locate(key)` and `find(key)` look for it in this order:

1. `CONFIG[key]`, if that path exists and its contents fit;
2. the download name, in `data/raw` or any `CONFIG["SEARCH_DIRS"]` folder (default: `/content`, `/content/drive/MyDrive`, `~/Downloads`, each one subfolder deep);
3. a similar name: "(1)" copies, "Copy of …", spaces or dashes instead of underscores, other capitals, another extension;
4. the contents alone: zip members (`employer_panel_year_*`, …) or header columns (`gvkey` + `fyear` + `conm`, …).

Module 01 (or `show_files()`) prints which file each input resolved to and how it was found. Inputs it cannot find are listed with their download name and source. When two different files fit equally well, it reports `AMBIGUOUS`.

The checklist of file names to double-check at each step: https://claude.ai/code/artifact/1aa6867a-4183-4491-a34a-c7f41a86be48

## POLISY lab

`lab/` extends the pipeline into a research lab on Political Ideology × AI × Innovation: it links VRscores to AI
exposure (AIOE, DAIOE), IRS migration and state policy (CSPP), grades every finding and writes an interactive site for
GitHub Pages. Start with `lab/notebooks/POLISY_AI_Innovation_Lab.ipynb`; `lab/docs/TECHNICAL_REPORT.md` explains it.
