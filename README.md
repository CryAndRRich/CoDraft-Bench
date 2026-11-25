# CoDraft

Code for *Five Shades of Grey: Taxonomy-grounded Semantic Enrichment for Ordinal Product
Similarity*: CoDraft enrichment, the Multi-Task Cross-Encoder, all baselines, and the experiments
reported in the paper.

## Structure

```
runner.py              trains and evaluates one experiment, selected by run ID (--list shows all)
llm_runner.py          runs an LLM for CoDraft enrichment or as a classifier
config/
  runs.py              every experiment of the paper, by run ID
  llms.py              the LLMs used for enrichment and classification
  config_model.py      hyperparameters
  config_data.py       NICE class headings and class tokens
preprocess/            data loading and model inputs
model/                 the Multi-Task Cross-Encoder, the Rank-Aware Focal Loss and the baselines
codraft_enrichment/    the CoDraft prompt and schema, the LLM client and the LLM classifier
utils/                 metrics and prediction helpers
scripts/               notebooks for runner.py and llm_runner.py on Kaggle
requirements.txt       dependencies for training and evaluation
requirements-llm.txt   dependencies for the LLM runs (with vLLM)
```

## License

MIT, see `LICENSE`.
