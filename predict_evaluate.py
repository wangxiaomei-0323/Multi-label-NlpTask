import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, ElectraConfig
from component.dataset import NlpDataset
from component.collator import NlpCollator
from component.model import ElectraMultiCategoricalClassification

device = "cpu"
labels = ['NULL', 'nan', '丰巢催单', '关机黑屏', '取件码错误', '取消寄件',
          '同意入柜', '寄件催收', '寄件包装问题咨询', '寄件填错信息', '寄件收费标准',
          '寄件放错格口', '寄件被取消原因', '屏幕故障', '广告合作', '快递破损',
          '快递遗失', '投柜需求', '拒绝入柜', '描述取件', '描述取返件',
          '描述寄件',  '支付异常', '断电', '断网', '未开门',
          '未收到取件码', '柜中无件', '柜门故障', '查监控', '格口尺寸问题',
          '物业问题', '要求退款', '警方调取视频', '询问会员资费', '询问保管费收费标准',
          '误关门', '误开通会员', '请求远程开门', '请求远程重启', '超时收费',
          '超时滞留', '转人工-丰巢文本', '错取他人快递']

num_labels = len(labels)
max_seq_length = 512
model_path = "D:/code/pythocode/4.aispeech/sknlp/pretrain/electra_small"
model_save = "D:/code/pythonProject/aihive_2024/models/trained_model_sFalse/pytorch_model.bin"
# model_save = "./models/best_model.pkl"
with open("./config/config.json", "r") as f:
    config = json.load(f)
config["num_labels"] = num_labels
model_config = ElectraConfig.from_pretrained(model_path, **config)
print(model_config)
state_dict = torch.load(model_save, map_location=device)
model = ElectraMultiCategoricalClassification.from_pretrained(model_save, config=model_config)  #
model.load_state_dict(state_dict)
model.eval()

eval_file = "./data/dev.txt"
tokenizer = AutoTokenizer.from_pretrained(model_path)
eval_dataset = NlpDataset(labels, eval_file, tokenizer, max_length=512)
data_collator = NlpCollator(tokenizer, max_seq_length, is_training=False)
eval_dataloader = DataLoader(eval_dataset, batch_size=3, shuffle=False, collate_fn=data_collator)
"""
[[1, 5, 7, 8], [1, 7, 8], [1, 5, 7, 8]]
[[1, 7, 8], [0, 1, 7, 8], [0, 1, 6, 7, 8]]
[[1, 7, 8], [0, 1, 7, 8], [0, 1, 5, 7, 8]]
"""
predict_ids = []
y_true = []
for batch in eval_dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    label = batch["label_texts"]
    p_result = model.predict(input_ids, attention_mask, thresholds=0.45)
    print(p_result)
    print(label)
    predict_ids = predict_ids + p_result
    y_true = y_true + label
print(y_true)
s = model.score(predict_ids,
                references=y_true,
                classes=labels
                )
print(s)
