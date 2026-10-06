# CoDraft-Bench

Code for **CoDraft: Taxonomy-Grounded Enrichment for Benchmarking Ordinal Product Similarity in
Intellectual Property Litigation**.

The task: given two product names from EUIPO opposition decisions, predict their similarity on a
5-level scale: Dissimilar (0), Low similar (1), Similar (2), High similar (3), Identical (4).

## Layout

```
runner.py              trains and scores one experiment; --install unpacks a Kaggle zip
llm_runner.py          runs an LLM: enrichment (Nature, Purpose) or direct classification
config/runs.py         every experiment, by run ID, with the folder it is kept in
config/llms.py         every LLM, by tag
config/config_model.py hyperparameters
config/config_data.py  NICE class headings and class tokens
preprocess/            data loading and model inputs
model/                 the Multi-Task Cross-Encoder and the baselines
codraft_enrichment/    the LLM client, the enrichment prompt and the LLM classifier
utils/                 metrics and prediction helpers
analysis/group_a.py    CPU analyses of the saved predictions
scripts/run.ipynb      notebook for runner.py
scripts/llm.ipynb      notebook for llm_runner.py
```

## Data

The data is not in git. Put it in `data/`:

```
data/
  plain/                 product name only
  category/              product name + NICE class heading
  codraft/               product name + Nature + Purpose (Gemini 2.5 Flash) + heading
  expanded/              product name rewritten by Gemini 2.5 Flash
  codraft_<llm>/         like codraft/, with Nature and Purpose from an open LLM
  term_attributes.csv    Gemini attributes, one row per term
  attributes/            open-LLM attributes (<llm>_attributes.csv) and run details (<llm>_enrich.json)
```

Each folder has `train.csv`, `val.csv` and `test.csv`: 14,939 / 2,135 / 4,266 pairs (21,340 in
total), split 70/10/20 and stratified by label. All folders have the same pairs in the same order,
so any two runs can be compared pair by pair.

| label | pairs |
|---|---|
| Dissimilar | 11,799 |
| Low similar | 807 |
| Similar | 2,718 |
| High similar | 388 |
| Identical | 5,628 |

## Train and score

```bash
pip install -r requirements.txt
python runner.py --list              # all runs
python runner.py --run-id 7          # the full model
python runner.py --run-id 7 --smoke  # a quick check on a tiny subset
```

On Kaggle, open `scripts/run.ipynb`, set `RUN_ID` and `SMOKE`, choose GPU T4 x2, then Run All.
Each run uses one GPU; `RUN_ID = [7, 12]` runs two at once. Download the zip at the end and unpack
it:

```bash
python runner.py --install codraft_runs_7-12.zip
```

## LLM runs

```bash
pip install -r requirements-llm.txt vllm==0.30.0
python llm_runner.py --list
python llm_runner.py --llm qwen3-8b --task enrich classify
```

`enrich` writes the attributes and the `codraft_<llm>/` data folder. `classify` asks the LLM for
the label of every test pair, with 0 and 10 examples, on the `plain`, `codraft` and `own` inputs
(`own` is the LLM's own attributes). On Kaggle use `scripts/llm.ipynb`: set the variables in the
first cell, choose GPU T4 x2, then Run All. Gated models need the Kaggle secret `HF_TOKEN`. A run
that stopped can continue from its zip with `RESUME_FROM`.

## Results

`python runner.py --install <zip>` puts every result in `weights/`, named by what it is:

```
weights/
  codraft/                                  the full model (run 7)
  baselines/<model>_<backbone>_<input>/     runs 1-6, 8-11, 20
  ablations/<what is removed>/              runs 12-19
  enrichment-llms/<llm>/                    run 7 on an open LLM's attributes (runs 21-24)
  binary/codraft_binary-trained/            run 7 on the binary target (run 25)
  llm-classifiers/<llm>/<k>shot_<input>/    the LLM classifier
```

Each folder holds `preds.csv` (one row per test pair) and `metrics.json`. The full folders, with
`model/`, `logits.npy` and `run.log`, are in the release zips, one per subfolder of `weights/`.

## Checks and analyses

```bash
python llm_runner.py --check         # data/codraft/ rebuilds from term_attributes.csv; metrics match
python -m analysis.group_a           # writes analysis/out/group_a.md
```

## License

MIT, see `LICENSE`.
