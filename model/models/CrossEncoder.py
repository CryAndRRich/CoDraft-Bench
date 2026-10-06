import torch
import torch.nn as nn
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer


def get_model_cross_encoder(
    model_name: str,
    num_classes: int,
    max_len: int,
    weights_tensor: torch.Tensor,
) -> CrossEncoder:
    model = CrossEncoder(
        model_name,
        num_labels=num_classes,
        max_length=max_len,
        automodel_args={"ignore_mismatched_sizes": True},
    )
    model.model.resize_token_embeddings(len(AutoTokenizer.from_pretrained(model_name)))
    model.loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
    return model
