from typing import Optional

import transformers
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelCategoricalCrossentropy(_WeightedLoss):
    """
    苏剑林. (Apr. 25, 2020). 《将“softmax+交叉熵”推广到多标签分类问题 》[Blog post]. Retrieved from https://spaces.ac.cn/archives/7359
    """

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> torch.Tensor:
        """
        :param input: shape: [batch_size, num_labels], model logits
        :param target: shape: [batch_size, num_labels], one-hot
        :return:
        """
        # dtype对齐
        y_true = target.to(input.dtype)
        # shape
        sample_size = y_true.size(0)  # 计算样本数量
        y_pred = input.view(sample_size, -1)
        # loss
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * torch.finfo(y_pred.dtype).max
        y_pred_pos = y_pred - (1 - y_true) * torch.finfo(y_pred.dtype).max

        zeros = torch.zeros_like(y_pred[..., :1])  # 用于生成logsum中的1
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)  # zeros 是公式中的1，e的0 = 1
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()


if __name__ == '__main__':
    shape = (2, 3)
    # y_pred = torch.rand(shape)
    y_true = torch.ones(shape)
    data = [[-0.5, 0.3, 0.4], [0.9, 0.9, 0.9]]
    y_pred = torch.tensor(data, dtype=torch.float)
    loss_fun = MultiLabelCategoricalCrossentropy()
    loss = loss_fun(y_true, y_pred)
    print(y_true)
    print(y_pred)
    """
    neg_loss: tensor([0., 0.])   # 因为y_true都是1，都是正例。
    pos_loss: tensor([1.4011, 0.7974])
    """
