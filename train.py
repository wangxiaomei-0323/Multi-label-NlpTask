import json
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, ElectraConfig, AdamW, TrainingArguments, Trainer
import torch
import os
import pandas as pd
from component.collator import NlpCollator
from component.dataset import NlpDataset
from component.model import ElectraMultiCategoricalClassification
from utils.utils import logits2classesId


def get_config(num_labels, config_file: str = "config/config.json"):
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    config_dict["num_labels"] = num_labels
    return config_dict


def get_model_token(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    # 模型需要的参数problem_type 会被更新，这个写法只有内部有的参数才会被更新。
    # 如果要添加参数，可以config.update(dict)
    model_config = ElectraConfig.from_pretrained(config["model_path"], **config)
    model = ElectraMultiCategoricalClassification.from_pretrained(config["model_path"], config=model_config)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    return model, tokenizer, optimizer


def get_dataloader(config, labels, data_file, tokenizer):
    dataset = NlpDataset(labels, data_file, tokenizer, config["max_seq_length"])
    train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    datacollator = NlpCollator(tokenizer, config["max_seq_length"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=datacollator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=datacollator)
    return datacollator, train_dataset, eval_dataset, train_dataloader, eval_dataloader


def old_train():
    print(os.path.abspath("./"))
    # set 可以去重
    train_data_file = "./data/dev.txt"
    labels = sorted(
        list(set(label for labels in pd.read_csv(train_data_file, sep="\t")["label"] for label in labels.split("|")))
    )
    labels = ["NULL"] + labels
    print(labels)
    num_labels = len(labels)
    device = "cpu"
    config = get_config(num_labels)
    model, tokenizer, optimizer = get_model_token(config)
    datacollator, train_dataset, eval_dataset, train_dataloader, eval_dataloader = get_dataloader(config, labels,
                                                                                                  train_data_file,
                                                                                                  tokenizer)
    epoch = config["epoch"]
    global_step = 0
    for e in range(0, epoch):
        losses = 0
        accuracy = 0
        model.train()
        for batch in train_dataloader:
            # 1.梯度清零
            optimizer.zero_grad()
            # 2.前向传播
            input_ids = batch['input_ids']
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['label_ids'].to(device),
            )
            loss = output.loss
            losses += loss.sum().item()

            # 3.反向传播，计算梯度
            loss.backward()
            # 4.优化，更新梯度
            optimizer.step()

            if global_step % 20 == 0:
                print(f"Epoch {epoch}, global_step {global_step}, loss: {loss.mean().item()}")
            global_step += 1
        print(f"Epoch {epoch}, loss: {losses / len(train_dataloader)}")
    torch.save(model.state_dict(), config["model_save"])


# 必须是 predictions=all_preds, label_ids=all_labels
def eval_metric(eval_predict):
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_predict
    predictions = logits2classesId(torch.Tensor(logits), True, 0.5)
    predictions = np.array(predictions).flatten().tolist()
    labels = labels.flatten().tolist()
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc


def get_argments(config):
    return TrainingArguments(
        output_dir=config["model_save"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        save_strategy="epoch",
        evaluation_strategy="epoch",
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        logging_steps=5,
        num_train_epochs=config["epoch"]
    )


def transformers_train():
    train_data_file = "./data/train.txt"
    data = pd.read_csv(train_data_file, sep="\t", dtype=str)
    data.fillna("", inplace=True)
    labels = set(label for labels in data["label"].astype(str) for label in labels.split("|"))
    labels.discard("")
    labels = ["NULL"] + list(labels)
    print(labels)
    num_labels = len(labels)
    device = "cpu"
    config = get_config(num_labels)

    model, tokenizer, optimizer = get_model_token(config)
    datacollator, train_dataset, eval_dataset, train_dataloader, eval_dataloader = get_dataloader(
        config, labels,
        train_data_file,
        tokenizer
    )
    args = get_argments(config)

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # 会被分成train和test集
        eval_dataset=eval_dataset,  # 验证集
        compute_metrics=eval_metric,
        data_collator=datacollator,
    )
    trainer.train()
    model.save_pretrained("./models/trained_model", safe_serialization=False)
    # trainer.save_model()


if __name__ == '__main__':
    # old_train()
    transformers_train()
