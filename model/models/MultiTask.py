import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedTokenizer, XLMRobertaModel, XLMRobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from model.loss.RankAwareFocalLoss import RankAwareFocalLoss


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
        **kwargs: object,
    ) -> SequenceClassifierOutput:
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        logits_sim = self.classifier(hidden[:, 0, :])
        mask_positions = (input_ids == self.mask_token_id).int().argmax(dim=-1)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        logits_aux = self.aux_classifier(hidden[batch_indices, mask_positions, :])

        loss = None
        if labels is not None:
            if self.loss_type == "ce":
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = RankAwareFocalLoss(
                    num_classes=self.num_labels, gamma=2.0, alpha=self.alpha
                )
            loss = loss_fct(logits_sim.view(-1, self.num_labels), labels.view(-1))
            if aux_labels is not None and self.training:
                loss_aux = nn.CrossEntropyLoss()(
                    logits_aux.view(-1, self.num_product_classes),
                    aux_labels.view(-1),
                )
                loss = loss + self.aux_weight * loss_aux
        return SequenceClassifierOutput(loss=loss, logits=logits_sim)


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
    return model.to(device)
