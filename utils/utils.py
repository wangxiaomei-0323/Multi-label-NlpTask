from typing import Union, Sequence, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer


def logits2probabilities(logits: torch.Tensor, is_multilabel: bool) -> torch.Tensor:
    """
    计算logits的sigmoid或softmax分值.
    logits shape: [batch_size, num_labels]
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits should have shape(`n_samples`, `n_classes`),"
            "but shape%s was given" % str(logits.shape)
        )
    if is_multilabel:
        return torch.sigmoid(logits)
    else:
        return torch.softmax(logits, dim=1)


def logits2classesId(
        logits: torch.Tensor,
        is_multilabel: bool,
        thresholds: Union[Sequence[float], float] = 0.5,
) -> Union[List[int], List[List[int]]]:
    if logits.ndim != 2:
        raise ValueError(
            "logits should have shape(`n_samples`, `n_classes`), "
            "but shape%s was given" % str(logits.shape)
        )
    # 选择prob > threshold的作为最终预测的label
    prob = logits2probabilities(logits, is_multilabel)
    if is_multilabel:
        return torch.where(prob > thresholds, 1, 0).tolist()
    else:
        p = np.argmax(prob.tolist(), axis=1)
        return list(map(int, p))


def probabilities2classes(
        probabilities: torch.Tensor,
        is_multilabel: bool,
        thresholds: Union[Sequence[float], float] = 0.5,
) -> Union[List[int], List[List[int]]]:
    """
    根据给定的`threshold`, 将分类问题的`scores`解析为对应的类别.
    probabilities: 输入的`logits`, shape(`n_samples`, `n_classes`)
    is_multilabel: 是否为多标签分类
    threshold: 正例判断的阈值(仅在多标签时有作用)

    :return: 一个长度为`n_samples`的``list``
    如果是多标签分类, 每一个sample对应的结果为一个``list``, list根据prob > threshold选择label
    其中的每个``int``值为这个sample对应的类别.

    如果不是多标签分类, 则每一个sample对应的结果为一个``int``,
    表示这个sample对应的类别.
    """
    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities should have shape(`n_samples`, `n_classes`), "
            "but shape%s was given" % str(probabilities.shape)
        )
    # 选择prob > threshold的作为最终预测的label_id
    if is_multilabel:
        return [np.where(probabilities[i, :] > thresholds)[0].tolist() for i in range(probabilities.shape[0])]
    else:
        label_id = np.argmax(probabilities.tolist(), axis=1)
        return list(map(int, label_id))


def label_binarizer(
        y: Union[Sequence[Sequence[str]], Sequence[str]],
        p: Union[Sequence[Sequence[str]], Sequence[str]],
        classes: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if isinstance(y[0], str):
        y = [[yi] for yi in y]
    if isinstance(p[0], str):
        p = [[pi] for pi in p]
    binarizer = MultiLabelBinarizer(classes=classes)
    return (
        binarizer.fit_transform(y),
        binarizer.transform(p),
        binarizer.classes_.tolist(),
    )


def precision_recall_fscore(
        tp: int, fp: int, fn: int, beta: float = 1
) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    if precision + recall != 0:
        fscore = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    else:
        fscore = 0
    return precision, recall, fscore


def classification_fscore(
        p: Union[Sequence[Sequence[str]], Sequence[str], Sequence[int]],
        y: Union[Sequence[Sequence[str]], Sequence[str], Sequence[int]],
        classes: Union[Sequence[str], Sequence[int]],
        beta: float = 1,
) -> pd.DataFrame:
    """
    y: 标注标签 (batch_size, labels)
    p: 预测标签
    classes: 所有的标签集合, 如果不提供则取y
    beta: 加权值, 默认为1, f_score = (1 + beta**2) * (P * R) / (beta**2 * P + R)
    :return:
    """
    if isinstance(y[0], (float, int)):
        confusion_matrix = zip(
            classes, multilabel_confusion_matrix(y, p, labels=classes)
        )
    else:
        y_one_hot, p_one_hot, classes = label_binarizer(y, p, classes)
        confusion_matrix = list(
            zip(classes, multilabel_confusion_matrix(y_one_hot, p_one_hot))
        )
    records = []
    for class_, arr in confusion_matrix:
        tp, fp, fn, tn = arr[1, 1], arr[0, 1], arr[1, 0], arr[0, 0]
        precision, recall, fscore = precision_recall_fscore(tp, fp, fn, beta)
        support = tp + fn
        records.append((class_, precision, recall, fscore, support, tp, fp, fn, tn))

    columns = [
        "class",
        "precision", "recall", "fscore",
        "support",
        "TP", "FP", "FN", "TN",
    ]
    df = pd.DataFrame(records, columns=columns).sort_values(["support", "TP"], ascending=False)

    support, tp, fp, fn, tn = df[["support", "TP", "FP", "FN", "TN"]].sum(axis=0)
    precision, recall, fscore = precision_recall_fscore(tp, fp, fn, beta)
    df_avg = pd.DataFrame([("avg", precision, recall, fscore, support, tp, fp, fn, tn)], columns=columns)
    return pd.concat([df, df_avg], axis=0)


if __name__ == '__main__':
    y = [["a"], ["a", "b"]]
    p = [["a"], ["a"]]
    classes = ["a", "b"]
    df = classification_fscore(p, y, classes)
    print(df)
