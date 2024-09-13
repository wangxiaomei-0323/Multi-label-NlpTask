"""
1.padding
2.处理label 和所有label的0,1
"""
from typing import Dict, Any
import torch


class NlpCollator(object):
    def __init__(self, tokenizer, max_seq_length, is_training=True):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_training = is_training

    def __call__(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max([len(x['input_ids']) for x in batch])
        batch_max_len = min(max_len, self.max_seq_length)
        input_ids_batch, attention_mask_batch, labels, label_texts = [], [], [], []
        for x in batch:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]

            padding_len = batch_max_len - len(input_ids)

            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
            attention_mask = attention_mask + [self.tokenizer.pad_token_id] * padding_len

            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[: self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            label_texts.append(x["label_text"])
            labels.append(x["label"])
        if self.is_training:
            return {
                "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),  # electra model 要求long
                "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        else:
            return {
                "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                'label_texts': label_texts
            }
