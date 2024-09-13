import torch
from torch.utils.data import Dataset
import pandas as pd


class NlpDataset(Dataset):
    def __init__(self, labels, file_path, tokenizer, max_length):
        self.labels = labels
        self.label2id = dict(zip(self.labels, range(len(self.labels))))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(file_path, sep="\t", dtype=str)
        self.data.fillna("", inplace=True)
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_id = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取文本和标签
        text = str(self.data.iloc[idx]['text'])
        label = set(str(self.data.iloc[idx]['label']).split("|"))
        label.discard("")
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            # return_tensors='pt' # 默认list
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            "label": [1 if l in label else 0 for l in self.labels],
            'label_text': list(label)
        }
