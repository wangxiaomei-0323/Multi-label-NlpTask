import torch
import torch.nn as nn

print(torch.ones((0,)))
# loss = nn.BCEWithLogitsLoss()
# input = torch.randn(3, requires_grad=False)
# print(input)
# print(torch.em)
# target = torch.empty(3).random_(2)
# print(target)
# output = loss(input, target)
# # output.backward()
# print(output)
import pandas as pd
import numpy as np

labels = ["t1", "t2", "t3"]
probabilities = torch.Tensor([[0.2, 0.4, 0.9], [0.7, 0.6, 0.1], [0.2, 0.8, 0.3]])
labels = np.argmax(probabilities.tolist(), axis=1)
thresholds = 0.5
is_multilabel = True
if probabilities.ndim != 2:
    raise ValueError(
        "probabilities should have shape(`n_samples`, `n_classes`), "
        "but shape%s was given" % str(probabilities.shape)
    )
# 选择prob > threshold的作为最终预测的label_index
label_indexs = [np.where(probabilities[i, :] > thresholds)[0].tolist() for i in range(probabilities.shape[0])]
print(label_indexs)
# label_text = [[labels[i] for i in label] for label in label_indexs]
# print(label_text)
false_labels = np.argmax(probabilities.tolist(), axis=1)
# print(false_labels.tolist().astype(int))
print(list(map(int, false_labels)))
# (label[] for label in false_labels)

label_set = ['NULL', 'nan', '丰巢催单', '关机黑屏', '取件码错误', '取消寄件',
             '同意入柜', '寄件催收', '寄件包装问题咨询', '寄件填错信息', '寄件收费标准',
             '寄件放错格口', '寄件被取消原因', '屏幕故障', '广告合作', '快递破损',
             '快递遗失', '投柜需求', '拒绝入柜', '描述取件', '描述取返件',
             '描述寄件', '支付异常', '断电', '断网', '未开门',
             '未收到取件码', '柜中无件', '柜门故障', '查监控', '格口尺寸问题',
             '物业问题', '要求退款', '警方调取视频', '询问会员资费', '询问保管费收费标准',
             '误关门', '误开通会员', '请求远程开门', '请求远程重启', '超时收费',
             '超时滞留', '转人工-丰巢文本', '错取他人快递']
train_data_file = "./data/train.txt"
data = pd.read_csv(train_data_file, sep="\t", dtype=str)
data.fillna("", inplace=True)
labels = set(label for labels in data["label"].astype(str) for label in labels.split("|"))
labels.discard("")
labels = ["NULL"] + list(labels)
print(labels)
