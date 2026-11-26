from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PreTrainedTokenizer, XLMRobertaModel, XLMRobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from model.loss.RankAwareFocalLoss import RankAwareFocalLoss

RANK_HEADS = ("coral", "corn")


@dataclass
class JointOutput(SequenceClassifierOutput):
    aux_logits: torch.FloatTensor | None = None


def coral_loss(rank_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    levels = (labels.unsqueeze(1) > torch.arange(rank_logits.size(1), device=labels.device)).float()
    return F.binary_cross_entropy_with_logits(
        rank_logits.float(), levels, reduction="none"
    ).sum(1).mean()


def corn_loss(rank_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    total, count = rank_logits.new_zeros((), dtype=torch.float32), 0
    for k in range(rank_logits.size(1)):
        keep = labels >= k
        if keep.any():
            total = total + F.binary_cross_entropy_with_logits(
                rank_logits[keep, k].float(), (labels[keep] > k).float(), reduction="sum"
            )
            count += int(keep.sum())
    return total / max(count, 1)


def rank_scores(rank_logits: torch.Tensor, loss_type: str) -> torch.Tensor:
    """K-way scores whose argmax is the number of thresholds passed (P(y > k) > 0.5).

    s_0 = 0 and s_c = sum_{k<c} logit P(y > k). With P(y > k) decreasing in k (always for CORN,
    and for CORAL once its biases are ordered) the argmax equals the count rule, and averaging
    the scores of two views is the same as averaging their log-odds.
    """
    if loss_type == "coral":
        log_odds = rank_logits.float()
    else:
        log_p = torch.cumsum(F.logsigmoid(rank_logits.float()), dim=1)
        log_odds = log_p - torch.log(-torch.expm1(log_p).clamp(max=-1e-7))
    zero = log_odds.new_zeros(log_odds.size(0), 1)
    return torch.cat([zero, torch.cumsum(log_odds, dim=1)], dim=1)


class JointClassSimBGE(XLMRobertaPreTrainedModel):
    def __init__(self, config: AutoConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_product_classes = config.num_product_classes
        self.mask_token_id = config.mask_token_id
        self.alpha = config.alpha
        self.aux_weight = config.aux_weight
        self.loss_type = config.loss_type
        class_weights = getattr(config, "class_weights", None)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.roberta = XLMRobertaModel(config)
        if self.loss_type == "coral":
            self.classifier = nn.Linear(config.hidden_size, 1, bias=False)
            self.rank_bias = nn.Parameter(torch.zeros(self.num_labels - 1))
        elif self.loss_type == "corn":
            self.classifier = nn.Linear(config.hidden_size, self.num_labels - 1)
        else:
            self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.aux_classifier = nn.Linear(config.hidden_size, self.num_product_classes)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        aux_labels: torch.Tensor | None = None,
        num_items_in_batch: int | None = None,
        **kwargs: object
    ) -> JointOutput:
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs
        )
        hidden = outputs.last_hidden_state
        head = self.classifier(hidden[:, 0, :])
        if self.loss_type == "coral":
            head = head + self.rank_bias
        logits_sim = rank_scores(head, self.loss_type) if self.loss_type in RANK_HEADS else head
        mask_positions = (input_ids == self.mask_token_id).int().argmax(dim=-1)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        logits_aux = self.aux_classifier(hidden[batch_indices, mask_positions, :])

        loss = None
        if labels is not None:
            labels = labels.view(-1)
            if self.loss_type == "coral":
                loss = coral_loss(head, labels)
            elif self.loss_type == "corn":
                loss = corn_loss(head, labels)
            else:
                if self.loss_type == "ce":
                    loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                else:
                    loss_fct = RankAwareFocalLoss(
                        num_classes=self.num_labels, gamma=2.0, alpha=self.alpha
                    )
                loss = loss_fct(logits_sim.view(-1, self.num_labels), labels)
            if aux_labels is not None and self.training and self.aux_weight:
                loss_aux = nn.CrossEntropyLoss()(
                    logits_aux.view(-1, self.num_product_classes), aux_labels.view(-1)
                )
                loss = loss + self.aux_weight * loss_aux
        return JointOutput(
            loss=loss, logits=logits_sim, aux_logits=None if self.training else logits_aux
        )


def get_model_multi_task(
    model_name: str,
    num_classes: int,
    num_product_classes: int,
    alpha: float,
    aux_weight: float,
    device: torch.device,
    tokenizer: PreTrainedTokenizer,
    loss_type: str = "rank_aware",
    class_weights: list | None = None,
    rank_bias_init: list | None = None
) -> JointClassSimBGE:
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_classes
    config.num_product_classes = num_product_classes
    config.mask_token_id = tokenizer.mask_token_id
    config.alpha = alpha
    config.aux_weight = aux_weight
    config.loss_type = loss_type
    if class_weights is not None:
        config.class_weights = [float(x) for x in class_weights]
    model = JointClassSimBGE.from_pretrained(
        model_name, config=config, ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    if rank_bias_init is not None:
        if loss_type != "coral":
            raise ValueError("rank_bias_init applies to the CORAL head only")
        with torch.no_grad():
            model.rank_bias.copy_(torch.tensor(rank_bias_init, dtype=torch.float32))
    return model.to(device)
