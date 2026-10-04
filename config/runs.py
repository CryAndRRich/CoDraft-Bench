"""Every experiment in the paper, keyed by run ID.

The notebook only picks an ID; everything that distinguishes one run from another lives
here, so two runs can differ only in what this table says they differ in.

family      what trains it
  xgboost   TF-IDF features + XGBoost (CPU or GPU)
  cross     sentence-transformers CrossEncoder
  bi        bi-encoder (dense retriever) + classifier over [u, v, |u - v|]
  multi     the Multi-Task Cross-Encoder (JointClassSimBGE)

variant     which data/<variant>/ split the model reads
  codraft   name + LLM Nature/Purpose + NICE heading   (the full input)
  plain     the bare product name
  category  name + NICE heading, Nature/Purpose blanked
  expanded  the LLM-rewritten name

Multi-task overrides (only on the "multi" family):
  aux_weight  lambda in the paper; 0 switches the masked class head off
  loss_type   "rank_aware" (focal + alpha * rank MSE) or "ce" (class-weighted CE)
  alpha       weight of the rank penalty inside the rank-aware loss
"""

DEBERTA = "microsoft/deberta-v3-base"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
BGE_M3 = "BAAI/bge-m3"
LEGAL_BERT = "nlpaueb/legal-bert-base-uncased"

RUNS = {
    # ---- Table 2: every baseline with and without CoDraft ------------------------
    1:  dict(name="xgboost_plain",         family="xgboost", model=None,         variant="plain",
             table="Table 2", row="XGBoost via TF-IDF (vanilla)"),
    2:  dict(name="xgboost_codraft",       family="xgboost", model=None,         variant="codraft",
             table="Table 2", row="XGBoost via TF-IDF (w/ CoDraft)"),
    3:  dict(name="ce_deberta_plain",      family="cross",   model=DEBERTA,      variant="plain",
             table="Table 2", row="Cross-Encoder deberta (vanilla); also Fig. 4A"),
    4:  dict(name="ce_deberta_codraft",    family="cross",   model=DEBERTA,      variant="codraft",
             table="Table 2", row="Cross-Encoder deberta (w/ CoDraft)"),
    5:  dict(name="ce_bge_codraft",        family="cross",   model=BGE_RERANKER, variant="codraft",
             table="Table 2", row="Cross-Encoder, same backbone as ours (Reviewer 2)"),
    6:  dict(name="ce_bge_plain",          family="cross",   model=BGE_RERANKER, variant="plain",
             table="Table 2", row="Cross-Encoder, same backbone as ours (vanilla)"),
    7:  dict(name="multi_codraft",         family="multi",   model=BGE_RERANKER, variant="codraft",
             table="Table 2/3", row="Multi-Task Cross-Encoder, full model; also Fig. 3 and 4B"),

    # ---- baselines Reviewer 3 named -------------------------------------------------
    8:  dict(name="bi_bgem3_plain",        family="bi",      model=BGE_M3,       variant="plain",
             table="Table 2", row="Bi-encoder / dense retriever (vanilla)"),
    9:  dict(name="bi_bgem3_codraft",      family="bi",      model=BGE_M3,       variant="codraft",
             table="Table 2", row="Bi-encoder / dense retriever (w/ CoDraft)"),
    10: dict(name="ce_legalbert_plain",    family="cross",   model=LEGAL_BERT,   variant="plain",
             table="Table 2", row="Cross-Encoder legal-BERT (vanilla)"),
    11: dict(name="ce_legalbert_codraft",  family="cross",   model=LEGAL_BERT,   variant="codraft",
             table="Table 2", row="Cross-Encoder legal-BERT (w/ CoDraft)"),

    # ---- Table 3: ablations of the full model (run 7) ----------------------------
    12: dict(name="multi_plain",           family="multi",   model=BGE_RERANKER, variant="plain",
             table="Table 3", row="w/o CoDraft Enrichment"),
    13: dict(name="multi_category",        family="multi",   model=BGE_RERANKER, variant="category",
             table="Table 3", row="Term + NICE heading only (is the gain the free lookup?)"),
    14: dict(name="multi_expanded",        family="multi",   model=BGE_RERANKER, variant="expanded",
             table="Table 3", row="free-text rewrite instead of Nature/Purpose (is the schema needed?)"),
    15: dict(name="multi_codraft_noaux",   family="multi",   model=BGE_RERANKER, variant="codraft",
             aux_weight=0.0,
             table="Table 3", row="w/o Multi-Task Head"),
    16: dict(name="multi_codraft_ce",      family="multi",   model=BGE_RERANKER, variant="codraft",
             loss_type="ce",
             table="Table 3", row="w/o Rank-Aware Loss (class-weighted CE)"),
    17: dict(name="multi_codraft_alpha0",  family="multi",   model=BGE_RERANKER, variant="codraft",
             alpha=0.0,
             table="Table 3", row="focal loss only, no rank penalty"),

    # ---- added after the first 17 runs ------------------------------------------------
    # Removing the masked head (15) or the rank-aware loss (16) alone costs no macro-F1,
    # yet the full model beats a plain cross-encoder on the same backbone (5) by +4.7.
    # This run removes both, leaving a plain cross-encoder trained with the multi-task
    # recipe (two masked views averaged at test time, class tokens, 10 epochs, cosine
    # schedule). If it matches run 7, the gap is the recipe, not the architecture.
    18: dict(name="multi_codraft_noaux_ce", family="multi",  model=BGE_RERANKER, variant="codraft",
             aux_weight=0.0, loss_type="ce",
             table="Table 3", row="neither head nor rank-aware loss: the multi-task recipe alone"),
    # Inference only: run 7's trained model on test inputs with parts of the enrichment
    # removed or shuffled. Says which part of the input the model actually relies on.
    # Needs run 7's checkpoint (weights/07_multi_codraft_seed42/model) available locally or
    # attached on Kaggle as a dataset.
    19: dict(name="probe_run07_inputs",    family="probe",   model=BGE_RERANKER, variant="codraft",
             source="07_multi_codraft_seed42",
             table="Analysis", row="run 7 at test time with Nature / Purpose / heading removed or shuffled"),
    # Le Nir et al. (2026)'s best design, re-headed to five classes: frozen sentence
    # embeddings of both names, one-hot NICE classes and a same-class flag, into an
    # MLP 1024-512-256. The direct rival to CoDraft: taxonomy as one-hot categoricals
    # instead of LLM-generated attributes.
    20: dict(name="hybrid_mlp_bgem3_plain", family="hybrid", model=BGE_M3,       variant="plain",
             table="Table 2", row="Hybrid MLP (P1): embeddings + one-hot NICE classes"),
}


def get_run(run_id):
    if run_id not in RUNS:
        raise KeyError(f"unknown run {run_id}; valid IDs are {sorted(RUNS)}")
    spec = dict(RUNS[run_id])
    spec["id"] = run_id
    return spec


def describe():
    """The run table, for printing at the top of the notebook."""
    lines = [f"{'ID':>3}  {'name':24s} {'family':8s} {'variant':9s} {'table':10s} row"]
    for i in sorted(RUNS):
        r = RUNS[i]
        lines.append(f"{i:>3}  {r['name']:24s} {r['family']:8s} {r['variant']:9s} "
                     f"{r['table']:10s} {r['row']}")
    return "\n".join(lines)
