DEBERTA = "microsoft/deberta-v3-base"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
BGE_M3 = "BAAI/bge-m3"
LEGAL_BERT = "nlpaueb/legal-bert-base-uncased"

RUNS = {
    1: dict(
        name="xgboost_plain",
        family="xgboost",
        model=None,
        variant="plain",
        table="Table 3",
        row="XGBoost (TF-IDF), vanilla",
        path="baselines/xgboost-tfidf_plain"
    ),
    2: dict(
        name="xgboost_codraft",
        family="xgboost",
        model=None,
        variant="codraft",
        table="Table 3",
        row="XGBoost (TF-IDF), CoDraft",
        path="baselines/xgboost-tfidf_codraft"
    ),
    3: dict(
        name="ce_deberta_plain",
        family="cross",
        model=DEBERTA,
        variant="plain",
        table="Table 3",
        row="Cross-Encoder DeBERTa-v3-base, vanilla",
        path="baselines/cross-encoder_deberta-v3-base_plain"
    ),
    4: dict(
        name="ce_deberta_codraft",
        family="cross",
        model=DEBERTA,
        variant="codraft",
        table="Table 3",
        row="Cross-Encoder DeBERTa-v3-base, CoDraft",
        path="baselines/cross-encoder_deberta-v3-base_codraft"
    ),
    5: dict(
        name="ce_bge_codraft",
        family="cross",
        model=BGE_RERANKER,
        variant="codraft",
        table="Table 3",
        row="Cross-Encoder BGE-Reranker-v2-M3 (our backbone), CoDraft",
        path="baselines/cross-encoder_bge-reranker-v2-m3_codraft"
    ),
    6: dict(
        name="ce_bge_plain",
        family="cross",
        model=BGE_RERANKER,
        variant="plain",
        table="Table 3",
        row="Cross-Encoder BGE-Reranker-v2-M3 (our backbone), vanilla",
        path="baselines/cross-encoder_bge-reranker-v2-m3_plain"
    ),
    7: dict(
        name="multi_codraft",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        table="Table 3/4",
        row="Multi-Task Cross-Encoder, full model; also Figure 2",
        path="codraft"
    ),
    8: dict(
        name="bi_bgem3_plain",
        family="bi",
        model=BGE_M3,
        variant="plain",
        table="Table 3",
        row="Bi-Encoder BGE-M3, vanilla",
        path="baselines/bi-encoder_bge-m3_plain"
    ),
    9: dict(
        name="bi_bgem3_codraft",
        family="bi",
        model=BGE_M3,
        variant="codraft",
        table="Table 3",
        row="Bi-Encoder BGE-M3, CoDraft",
        path="baselines/bi-encoder_bge-m3_codraft"
    ),
    10: dict(
        name="ce_legalbert_plain",
        family="cross",
        model=LEGAL_BERT,
        variant="plain",
        table="Table 3",
        row="Cross-Encoder LEGAL-BERT-base, vanilla",
        path="baselines/cross-encoder_legal-bert_plain"
    ),
    11: dict(
        name="ce_legalbert_codraft",
        family="cross",
        model=LEGAL_BERT,
        variant="codraft",
        table="Table 3",
        row="Cross-Encoder LEGAL-BERT-base, CoDraft",
        path="baselines/cross-encoder_legal-bert_codraft"
    ),
    12: dict(
        name="multi_plain",
        family="multi",
        model=BGE_RERANKER,
        variant="plain",
        table="Table 4",
        row="w/o CoDraft Enrichment",
        path="ablations/no-codraft-enrichment"
    ),
    13: dict(
        name="multi_category",
        family="multi",
        model=BGE_RERANKER,
        variant="category",
        table="Table 4",
        row="w/o Nature & Purpose (class heading kept)",
        path="ablations/no-llm-attributes"
    ),
    14: dict(
        name="multi_expanded",
        family="multi",
        model=BGE_RERANKER,
        variant="expanded",
        table="Table 4",
        row="w/ Free-text Expansion",
        path="ablations/free-text-expansion"
    ),
    15: dict(
        name="multi_codraft_noaux",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        aux_weight=0.0,
        table="Table 4",
        row="w/o Multi-Task Head",
        path="ablations/no-multitask-head"
    ),
    16: dict(
        name="multi_codraft_ce",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        loss_type="ce",
        table="Table 4",
        row="w/o Rank-Aware Loss (class-weighted cross-entropy)",
        path="ablations/no-rank-aware-loss"
    ),
    17: dict(
        name="multi_codraft_alpha0",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        alpha=0.0,
        table="Table 4",
        row="w/o Rank Penalty (alpha = 0)",
        path="ablations/no-rank-penalty"
    ),
    18: dict(
        name="multi_codraft_noaux_ce",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        aux_weight=0.0,
        loss_type="ce",
        table="Table 4",
        row="w/o Head & Rank-Aware Loss",
        path="ablations/no-head-no-rank-aware-loss"
    ),
    19: dict(
        name="probe_run07_inputs",
        family="probe",
        model=BGE_RERANKER,
        variant="codraft",
        source=7,
        table="Table 4",
        row="test-time input probe of the full model",
        path="ablations/input-probe"
    ),
    20: dict(
        name="hybrid_mlp_bgem3_plain",
        family="hybrid",
        model=BGE_M3,
        variant="plain",
        table="Table 3",
        row="Hybrid MLP of Le Nir et al. (2026), vanilla",
        path="baselines/hybrid-mlp_bge-m3_plain"
    ),
    21: dict(
        name="multi_codraft_qwen2.5-7b",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft_qwen2.5-7b",
        table="Table 5",
        row="full model on attributes from Qwen2.5-7B-Instruct",
        path="enrichment-llms/qwen2.5-7b"
    ),
    22: dict(
        name="multi_codraft_qwen3-8b",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft_qwen3-8b",
        table="Table 5",
        row="full model on attributes from Qwen3-8B",
        path="enrichment-llms/qwen3-8b"
    ),
    23: dict(
        name="multi_codraft_llama3.1-8b",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft_llama3.1-8b",
        table="Table 5",
        row="full model on attributes from Llama-3.1-8B-Instruct",
        path="enrichment-llms/llama3.1-8b"
    ),
    24: dict(
        name="multi_codraft_nemotron-nano-8b",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft_nemotron-nano-8b",
        table="Table 5",
        row="full model on attributes from Nemotron-Nano-8B",
        path="enrichment-llms/nemotron-nano-8b"
    ),
    25: dict(
        name="multi_codraft_binary",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        binary=True,
        table="Table 7",
        row="full model trained on the binary target",
        path="binary/codraft_binary-trained"
    ),
    26: dict(
        name="ce_recipe_bge_plain",
        family="ce_recipe",
        model=BGE_RERANKER,
        variant="plain",
        table="Rebuttal E1",
        row="Cross-Encoder bge-reranker-v2-m3, vanilla, trained with the full model's recipe (E1b: two orders averaged)",
        path="rebuttal/ce-same-recipe/bge-reranker-v2-m3_plain",
        rebuttal="E1"
    ),
    27: dict(
        name="ce_recipe_bge_codraft",
        family="ce_recipe",
        model=BGE_RERANKER,
        variant="codraft",
        table="Rebuttal E1",
        row="Cross-Encoder bge-reranker-v2-m3, CoDraft, trained with the full model's recipe (E1b: two orders averaged)",
        path="rebuttal/ce-same-recipe/bge-reranker-v2-m3_codraft",
        rebuttal="E1"
    ),
    28: dict(
        name="ce_recipe_deberta_plain",
        family="ce_recipe",
        model=DEBERTA,
        variant="plain",
        table="Rebuttal E1",
        row="Cross-Encoder deberta-v3-base, vanilla, trained with the full model's recipe (E1b: two orders averaged)",
        path="rebuttal/ce-same-recipe/deberta-v3-base_plain",
        rebuttal="E1"
    ),
    29: dict(
        name="ce_recipe_deberta_codraft",
        family="ce_recipe",
        model=DEBERTA,
        variant="codraft",
        table="Rebuttal E1",
        row="Cross-Encoder deberta-v3-base, CoDraft, trained with the full model's recipe (E1b: two orders averaged)",
        path="rebuttal/ce-same-recipe/deberta-v3-base_codraft",
        rebuttal="E1"
    ),
    30: dict(
        name="multi_codraft",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        seed=13,
        table="Rebuttal E2",
        row="full model, seed 13",
        path="rebuttal/seeds/codraft_seed13",
        rebuttal="E2"
    ),
    31: dict(
        name="multi_codraft",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        seed=2026,
        table="Rebuttal E2",
        row="full model, seed 2026",
        path="rebuttal/seeds/codraft_seed2026",
        rebuttal="E2"
    ),
    32: dict(
        name="multi_plain",
        family="multi",
        model=BGE_RERANKER,
        variant="plain",
        seed=13,
        table="Rebuttal E2",
        row="w/o CoDraft Enrichment, seed 13",
        path="rebuttal/seeds/no-codraft-enrichment_seed13",
        rebuttal="E2"
    ),
    33: dict(
        name="multi_plain",
        family="multi",
        model=BGE_RERANKER,
        variant="plain",
        seed=2026,
        table="Rebuttal E2",
        row="w/o CoDraft Enrichment, seed 2026",
        path="rebuttal/seeds/no-codraft-enrichment_seed2026",
        rebuttal="E2"
    ),
    34: dict(
        name="hybrid_mlp_bgem3_codraft",
        family="hybrid",
        model=BGE_M3,
        variant="codraft",
        table="Rebuttal E6",
        row="Hybrid MLP of Le Nir et al. (2026), CoDraft input",
        path="rebuttal/hybrid-mlp_bge-m3_codraft",
        rebuttal="E6"
    ),
    35: dict(
        name="multi_codraft_mask_nohead",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        mask_hides_heading=True,
        table="Rebuttal E9b",
        row="full model, masked side also loses its class heading",
        path="rebuttal/mask-hides-heading",
        rebuttal="E9b"
    ),
    36: dict(
        name="multi_codraft_coral",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        loss_type="coral",
        aux_weight=0.0,
        table="Rebuttal E7",
        row="CORAL ordinal head instead of the 5-way head and rank-aware loss, no masked class head (compare with run 18)",
        path="rebuttal/ordinal-heads/coral",
        rebuttal="E7"
    ),
    37: dict(
        name="multi_codraft_corn",
        family="multi",
        model=BGE_RERANKER,
        variant="codraft",
        loss_type="corn",
        aux_weight=0.0,
        table="Rebuttal E7",
        row="CORN ordinal head instead of the 5-way head and rank-aware loss, no masked class head (compare with run 18)",
        path="rebuttal/ordinal-heads/corn",
        rebuttal="E7"
    ),
    38: dict(
        name="ce_recipe_bge_plain",
        family="ce_recipe",
        model=BGE_RERANKER,
        variant="plain",
        seed=13,
        table="Rebuttal E2 (optional)",
        row="run 26, seed 13",
        path="rebuttal/seeds/ce-same-recipe_bge-reranker-v2-m3_plain_seed13",
        rebuttal="E2"
    ),
    39: dict(
        name="ce_recipe_bge_codraft",
        family="ce_recipe",
        model=BGE_RERANKER,
        variant="codraft",
        seed=13,
        table="Rebuttal E2 (optional)",
        row="run 27, seed 13",
        path="rebuttal/seeds/ce-same-recipe_bge-reranker-v2-m3_codraft_seed13",
        rebuttal="E2"
    ),
    40: dict(
        name="ce_recipe_bge_plain",
        family="ce_recipe",
        model=BGE_RERANKER,
        variant="plain",
        seed=2026,
        table="Rebuttal E2 (optional)",
        row="run 26, seed 2026",
        path="rebuttal/seeds/ce-same-recipe_bge-reranker-v2-m3_plain_seed2026",
        rebuttal="E2"
    ),
    41: dict(
        name="ce_recipe_bge_codraft",
        family="ce_recipe",
        model=BGE_RERANKER,
        variant="codraft",
        seed=2026,
        table="Rebuttal E2 (optional)",
        row="run 27, seed 2026",
        path="rebuttal/seeds/ce-same-recipe_bge-reranker-v2-m3_codraft_seed2026",
        rebuttal="E2"
    )
}


def get_run(run_id: int) -> dict:
    if run_id not in RUNS:
        raise KeyError(f"Unknown run {run_id}. Valid IDs are {sorted(RUNS)}.")
    return {**RUNS[run_id], "id": run_id}


def describe() -> str:
    cols = ("name", "family", "variant", "table")
    width = {c: max(len(c), *(len(r[c]) for r in RUNS.values())) for c in cols}
    lines = [f"{'ID':>3}  " + " ".join(f"{c:{width[c]}s}" for c in cols) + " row"]
    for run_id in sorted(RUNS):
        run = RUNS[run_id]
        cells = " ".join(f"{run[c]:{width[c]}s}" for c in cols)
        lines.append(f"{run_id:>3}  {cells} {run['row']}")
    return "\n".join(lines)
