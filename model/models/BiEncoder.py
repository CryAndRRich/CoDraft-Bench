import os

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class BasicBiEncoderClassifier(nn.Module):
    def __init__(self, base_model_path: str, num_classes: int = 5) -> None:
        super().__init__()
        self.encoder = SentenceTransformer(base_model_path, trust_remote_code=True)
        dim = self.encoder.get_sentence_embedding_dimension()
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        input_ids1: torch.Tensor,
        attention_mask1: torch.Tensor,
        input_ids2: torch.Tensor,
        attention_mask2: torch.Tensor,
    ) -> torch.Tensor:
        u = self.encoder({"input_ids": input_ids1, "attention_mask": attention_mask1})[
            "sentence_embedding"
        ]
        v = self.encoder({"input_ids": input_ids2, "attention_mask": attention_mask2})[
            "sentence_embedding"
        ]
        return self.classifier(torch.cat([u, v, torch.abs(u - v)], dim=1))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "siamese_state.pth"))


def get_model_bi_encoder_baseline(
    input_model_path: str, num_classes: int
) -> BasicBiEncoderClassifier:
    return BasicBiEncoderClassifier(input_model_path, num_classes=num_classes)
