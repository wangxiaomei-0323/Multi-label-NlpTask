import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import ElectraPreTrainedModel, ElectraModel
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)
from component.loss import MultiLabelCategoricalCrossentropy
from utils.utils import *


class ElectraMultiCategoricalClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "multi_Categorical_classification":
                loss_fct = MultiLabelCategoricalCrossentropy()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    def predict(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            thresholds: Union[float, list[float], None] = 0.5,
    ) -> Union[List[int], List[List[int]]]:
        """
        输入input，输出 label_id
        :return: label_text
        """
        logits = self.forward(input_ids, attention_mask).logits
        is_mul = True if self.config.problem_type == "multi_Categorical_classification" else False
        label_prob = logits2probabilities(logits, is_mul)
        label_id = probabilities2classes(label_prob, is_mul, thresholds)
        return label_id

    def score(
            self,
            pred_label_ids: Union[list[int], list[list[int]]] = None,
            references: Union[list[str], list[list[str]]] = None,
            classes: Union[list[int], list[str]] = None
    ) -> pd.DataFrame:
        is_mul = True if self.config.problem_type == "multi_Categorical_classification" else False
        # 返回label
        if is_mul:
            prediction_labels = [[classes[i] for i in prediction if i != 0] for prediction in pred_label_ids]
            print(prediction_labels)
        else:
            prediction_labels = [classes[i] for i in pred_label_ids if i != 0]
        return classification_fscore(
            prediction_labels,
            references,
            classes[1:],  # 第一个是自己加的NULL，不算
        )
