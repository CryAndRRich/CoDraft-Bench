import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    XLMRobertaPreTrainedModel,
    XLMRobertaModel,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from ..loss.RankAwareFocalLoss import RankAwareFocalLoss

class JointClassSimBGE(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 5
        self.num_product_classes = config.num_product_classes

        self.mask_token_id = getattr(config, "mask_token_id", 250001)
        self.alpha = getattr(config, "alpha", 0.5)
        self.aux_weight = getattr(config, "aux_weight", 0.3)
        self.loss_type = getattr(config, "loss_type", "rank_aware")
        class_weights_tensor = getattr(config, "class_weights", None)
        if class_weights_tensor is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights_tensor, dtype=torch.float32))
        else:
            self.register_buffer("class_weights", None)

        self.roberta = XLMRobertaModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.aux_classifier = nn.Linear(config.hidden_size, self.num_product_classes)

        if getattr(config, "gradient_checkpointing", False):
            self.roberta.gradient_checkpointing = True
            self.roberta.config.use_cache = False

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, aux_labels=None,
                num_items_in_batch=None,
                **kwargs):

        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)

        cls_output = outputs.last_hidden_state[:, 0, :]
        logits_sim = self.classifier(cls_output)

        is_mask_token = (input_ids == self.mask_token_id).int()
        mask_positions = is_mask_token.argmax(dim=-1)

        batch_size = input_ids.size(0)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        mask_output = outputs.last_hidden_state[batch_indices, mask_positions, :]

        logits_aux = self.aux_classifier(mask_output)

        loss = None
        if labels is not None:
            if self.loss_type == "ce":
                loss_fct_sim = nn.CrossEntropyLoss(weight=self.class_weights)
                loss_sim = loss_fct_sim(logits_sim.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct_sim = RankAwareFocalLoss(num_classes=self.num_labels, gamma=2.0, alpha=self.alpha)
                loss_sim = loss_fct_sim(logits_sim.view(-1, self.num_labels), labels.view(-1))

            loss_aux = torch.tensor(0.0).to(logits_sim.device)

            if aux_labels is not None and self.training:
                loss_fct_aux = nn.CrossEntropyLoss()
                loss_aux = loss_fct_aux(logits_aux.view(-1, self.num_product_classes), aux_labels.view(-1))
            loss = loss_sim + self.aux_weight * loss_aux
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits_sim
        )
    
def get_training_args(**kwargs):
    training_args = TrainingArguments(**kwargs)
    return training_args

def get_model_multi_task(model_name, num_classes, num_product_classes, alpha, aux_weight, device, loss_type="rank_aware",class_weights=None, tokenizer=None):
    # Resize against the tokenizer the data was encoded with. This used to load a fresh
    # tokenizer here, so the embedding matrix never grew to cover the class tokens.
    if tokenizer is None:
        from model.get_tokenizer import get_tokenizer
        tokenizer = get_tokenizer(model_name, add_class_tokens=True)
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_classes
    config.num_product_classes = num_product_classes
    config.mask_token_id = tokenizer.mask_token_id
    config.alpha = alpha
    config.aux_weight = aux_weight
    config.loss_type = loss_type
    if class_weights is not None:
        if hasattr(class_weights, "tolist"):
            class_weights = class_weights.tolist()
        config.class_weights = [float(x) for x in class_weights]
    model = JointClassSimBGE.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model, tokenizer
