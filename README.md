# CoDraft-Bench

Code for **CoDraft: Taxonomy-Grounded Enrichment for Benchmarking Ordinal Product Similarity
in Intellectual Property Litigation**.

Given two trademark product names, predict the 5-level ordinal similarity label EUIPO
opposition rulings use: `Dissimilar (0)`, `Low similar (1)`, `Similar (2)`,
`High similar (3)`, `Identical (4)`.

## Layout

```
runner.py             runs any experiment end to end: python runner.py --run-id 7
config/runs.py        every experiment in the paper, keyed by run ID (17 runs)
codraft_enrichment/   LLM enrichment: a product name T plus its NICE class heading C
                      becomes (Nature, Purpose, expanded name)
config/               CONFIG_DATA (NICE class map, class tokens), CONFIG_MODEL (hyperparameters)
preprocess/           DataManager: loads the splits, builds the inputs and datasets
model/                MultiTask (the main model), the baselines, Rank-Aware Focal Loss
utils/                metrics, prediction helpers, seeding
scripts/run.ipynb     the one notebook: set RUN_ID, then Run All
data/                 the splits (gitignored)
```

## Data

`data/` is gitignored, so fetch the dataset separately and unpack it there:

```
data/
  codraft/   train.csv  val.csv  test.csv    name + LLM Nature/Purpose + NICE heading
  plain/     train.csv  val.csv  test.csv    bare product name
  category/  train.csv  val.csv  test.csv    name + NICE heading only
  expanded/  train.csv  val.csv  test.csv    LLM-rewritten name
  term_attributes.csv                        attribute cache, one row per unique term
```

21,340 pairs split 70/10/20 (14,939 / 2,135 / 4,266), stratified by label. **The four
variants are row-aligned**: same `Pair ID` set, same order, same labels, so any two runs
compare pair by pair. `data/README.md` has the full description.

## Running

Every experiment has an ID in `config/runs.py`:

```bash
python runner.py --list              # the 17 runs
python runner.py --run-id 7          # the full model
python runner.py --run-id 7 --smoke  # tiny subset, one epoch: checks the pipeline
```

or open `scripts/run.ipynb`, set `RUN_ID`, and Run All. On Kaggle the notebook clones this
repo at branch `new`, installs `requirements.txt` and finds the dataset under
`/kaggle/input`. Pick the **GPU T4 x2** accelerator for every run so all runs share the same
hardware.

**Every run uses exactly one GPU**, so each family trains under the same conditions; only the
multi-task model could use two through `Trainer`, and the baselines cannot. A list such as
`RUN_ID = [7, 12]` runs two experiments at once, one per card.

Each run writes, under its tag `<ID>_<name>_seed42`:

```
results/<tag>_preds.csv      per-example predictions, carrying Pair ID
results/<tag>_metrics.json   every metric, the run spec, the settings and the environment
results/<tag>_logits.npy     averaged logits (multi-task only)
results/cm_<tag>.pdf/.png    confusion matrix
weights/<tag>/               the trained model
logs/<tag>.log               full console output
```

## Enrichment

```python
from codraft_enrichment import get_client, run_enrichment
```

Needs an API key for the endpoint in `CONFIG_DATA.CODRAFT_CONFIG`; put it in `.env`, which
is gitignored. `data/term_attributes.csv` is the cache of what the pipeline already
produced, so the experiments reproduce without API calls.

## Licence

MIT, see `LICENSE`.
