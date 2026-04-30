# AOM-Ridge end-to-end reproducibility Makefile.
#
# Goal: from a fresh checkout, build every artefact required by the
# manuscript with one command. The targets are layered so individual
# steps can be re-run without restarting the entire pipeline.
#
# Run from the repository root (recommended):
#
#     make -C bench/AOM_v0/Ridge all
#
# or directly from this directory:
#
#     cd bench/AOM_v0/Ridge && make all

# Resolve the repo root robustly — directory containing this Makefile + 3 levels up.
HERE       := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT  := $(realpath $(HERE)/../../..)
PY         := $(REPO_ROOT)/.venv/bin/python
PYTHONPATH_BASE := $(REPO_ROOT)/bench/AOM_v0:$(REPO_ROOT)/bench/AOM_v0/Ridge

ROOT             := $(HERE)
RUNS_DIR         := $(ROOT)/benchmark_runs
COHORT_FULL      := $(REPO_ROOT)/bench/AOM_v0/benchmarks/cohort_regression.csv
COHORT_CURATED   := $(RUNS_DIR)/curated_cohort.csv
RESULTS_CURATED_V2 := $(RUNS_DIR)/curated_v2/results.csv
RESULTS_CURATED  := $(RUNS_DIR)/curated/results.csv
PUB_DIR          := $(ROOT)/publication
SCRIPTS          := $(PUB_DIR)/scripts
FIG_DIR          := $(PUB_DIR)/figures
TAB_DIR          := $(PUB_DIR)/tables
MS_DIR           := $(PUB_DIR)/manuscript

# Publication results are taken from curated_v2 (10 lean variants × 39
# datasets, all five selection modes including branch_global and mkl).
# Override to compare against the older curated cohort with:
#     make RESULTS=$(RESULTS_CURATED) all
RESULTS ?= $(RESULTS_CURATED_V2)

.PHONY: all clean test lint bench-smoke bench-curated figures tables paper

all: lint test figures tables paper

## ---------------------------------------------------------------- code

lint:
	cd $(REPO_ROOT) && PYTHONPATH=$(PYTHONPATH_BASE) $(REPO_ROOT)/.venv/bin/ruff check bench/AOM_v0/Ridge

test:
	cd $(REPO_ROOT) && PYTHONPATH=$(PYTHONPATH_BASE) $(PY) -m pytest bench/AOM_v0/Ridge/tests -q

## ---------------------------------------------------------------- benchmarks

bench-smoke: $(COHORT_CURATED)
	cd $(REPO_ROOT) && PYTHONPATH=$(PYTHONPATH_BASE) $(PY) \
	  bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py \
	  --workspace bench/AOM_v0/Ridge/benchmark_runs/smoke6 \
	  --cohort smoke6 --variants lean --cv 3 --cv-kind spxy

bench-curated: $(COHORT_CURATED)
	cd $(REPO_ROOT) && PYTHONPATH=$(PYTHONPATH_BASE) $(PY) \
	  bench/AOM_v0/Ridge/benchmarks/run_aomridge_benchmark.py \
	  --workspace bench/AOM_v0/Ridge/benchmark_runs/curated_v2 \
	  --cohort full --variants lean --cv 3 --cv-kind spxy \
	  --cohort-path bench/AOM_v0/Ridge/benchmark_runs/curated_cohort.csv

# Curated cohort: all datasets in cohort_regression.csv with a paper Ridge
# baseline and (n_train <= 400, p <= 2500) for fast iteration. Regenerate
# any time the upstream cohort changes.
$(COHORT_CURATED):
	cd $(REPO_ROOT) && PYTHONPATH=$(PYTHONPATH_BASE) $(PY) -c "import pandas as pd; df=pd.read_csv('$(COHORT_FULL)'); df=df[(df['status']=='ok')&df['ref_rmse_ridge'].notna()&(df['n_train']<=400)&(df['p']<=2500)]; df.to_csv('$(COHORT_CURATED)', index=False); print(f'wrote {len(df)} rows to $(COHORT_CURATED)')"

## ---------------------------------------------------------------- figures

figures: $(FIG_DIR)/.timestamp

$(FIG_DIR)/.timestamp: $(RESULTS) $(SCRIPTS)/make_aomridge_figures.py $(SCRIPTS)/make_cumulative_irmsep.py
	mkdir -p $(FIG_DIR)
	cd $(REPO_ROOT) && PYTHONPATH=$(PYTHONPATH_BASE) $(PY) \
	  bench/AOM_v0/Ridge/publication/scripts/make_aomridge_figures.py --results $(RESULTS)
	cd $(REPO_ROOT) && PYTHONPATH=$(PYTHONPATH_BASE) $(PY) \
	  bench/AOM_v0/Ridge/publication/scripts/make_cumulative_irmsep.py --aomridge $(RESULTS)
	touch $@

## ---------------------------------------------------------------- tables

tables: $(TAB_DIR)/.timestamp

$(TAB_DIR)/.timestamp: $(RESULTS) $(SCRIPTS)/make_aomridge_tables.py
	mkdir -p $(TAB_DIR)
	cd $(REPO_ROOT) && PYTHONPATH=$(PYTHONPATH_BASE) $(PY) \
	  bench/AOM_v0/Ridge/publication/scripts/make_aomridge_tables.py --results $(RESULTS)
	touch $@

## ---------------------------------------------------------------- paper

paper: $(MS_DIR)/aomridge_paper.pdf

$(MS_DIR)/aomridge_paper.pdf: $(MS_DIR)/aomridge_paper.tex $(MS_DIR)/refs.bib $(FIG_DIR)/.timestamp $(TAB_DIR)/.timestamp
	cd $(MS_DIR) && pdflatex -interaction=nonstopmode aomridge_paper.tex >/dev/null
	cd $(MS_DIR) && bibtex aomridge_paper
	cd $(MS_DIR) && pdflatex -interaction=nonstopmode aomridge_paper.tex >/dev/null
	cd $(MS_DIR) && pdflatex -interaction=nonstopmode aomridge_paper.tex >/dev/null
	@echo "[paper] $(MS_DIR)/aomridge_paper.pdf updated"

## ---------------------------------------------------------------- clean

clean:
	rm -f $(MS_DIR)/aomridge_paper.pdf
	rm -f $(MS_DIR)/*.aux $(MS_DIR)/*.log $(MS_DIR)/*.bbl $(MS_DIR)/*.blg
	rm -f $(MS_DIR)/*.out $(MS_DIR)/*.toc
	rm -f $(FIG_DIR)/.timestamp $(TAB_DIR)/.timestamp
