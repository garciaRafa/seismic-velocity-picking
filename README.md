# Seismic Velocity Picking
Automated seismic velocity picking using metaheuristics.
## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests
```bash
pip install pytest
pytest
```
The suite checks the NMO equation against an analytic value, the semblance
limits (1 for identical traces, 0 for cancelling ones), the validity rules of
a solution, a three-layer sanity model, and that runs are reproducible.

## Experiments
Experiments are described by JSON files in `configs/` and run from the
project root:
```bash
python scripts/run_experiment.py configs/hill_climbing_layers_quick.json   # quick check
python scripts/run_experiment.py configs/hill_climbing_cdp6800.json        # 30 runs, formulation v1
python scripts/run_experiment.py configs/hill_climbing_cdp6800_v2.json     # 30 runs, formulation v2
```

**Hill climbing vs. random search** (same seeds, same budget of 33,000
evaluations, formulation v2), then a paired comparison with the Wilcoxon
signed-rank test:
```bash
python scripts/run_experiment.py configs/hill_climbing_cdp6800_v2_budget.json
python scripts/run_experiment.py configs/random_search_cdp6800_v2_budget.json
python scripts/compare_experiments.py results/experiments/hill_climbing_cdp6800_v2_budget_<date> \
                                      results/experiments/random_search_cdp6800_v2_budget_<date>
```
Each run also reports `n_correct`: picks whose velocity is within 2% of the
true RMS velocity (`experiment.correct_tol`).

**Formulations:** v1 only requires distinct pick times. v2 adds a minimum
separation between picks (`min_sep_ms`) and an allowed time window
(`t_min_ms`, `t_max_ms`), which excludes the water layer and the end of
the record. The defaults of these parameters reproduce v1 exactly.
Each run writes `results/experiments/<name>_<timestamp>/` with the config,
the machine/git metadata, one record per run and a summary.

**Reproducibility:** the data noise uses `experiment.data_seed` (fixed for
all runs); the run seeds are derived from `experiment.master_seed` with
`numpy.random.SeedSequence`. Commit your changes before running, so that
`metadata.json` points to the exact code that produced the results.
