# Publishing this repo to GitHub

This directory is a self-contained, git-initialised snapshot of the `btate`
library (with the vendored `bayes_tda`, `tcda_uq`, and `ffscb` source) ready to
push to GitHub. It was prepared locally because the `gh` CLI was not available in
the build environment.

The Phase-4 Colab notebooks clone **`https://github.com/hugogobato/btate`**. To
publish under that URL:

## Option A — with the GitHub CLI

```bash
cd dist/btate-github
gh repo create hugogobato/btate --public --source=. --remote=origin --push
```

## Option B — plain git (create the empty repo on github.com first)

1. On github.com, create a new **empty** public repo named `btate` under your
   account (no README/License — this repo already has them).
2. Then:

```bash
cd dist/btate-github
git remote add origin https://github.com/hugogobato/btate.git
git branch -M main
git push -u origin main
```

If you publish under a different owner/name, update `GITHUB_URL` at the top of
each `notebooks/P4_*_colab.ipynb` (a single variable per notebook) and the
`[project.urls]` in `pyproject.toml`.

## Sanity check after cloning

```bash
pip install "git+https://github.com/hugogobato/btate.git#egg=btate[dev]"
python -c "import btate, bayes_tda, tcda_uq; print(btate.__version__)"
pytest -q
```
