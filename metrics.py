from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from re import A

import torch
import numpy as np
from sklearn.metrics import classification_report
from pycm import ConfusionMatrix

def calc_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    correct = (preds.argmax(dim=1) == labels).sum().item()
    total = labels.numel()
    return correct / total

from sklearn.metrics import f1_score
def calc_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:

    labels_pred = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    cm = ConfusionMatrix(actual_vector=labels, predict_vector=labels_pred)
    f1 = f1_score(labels, labels_pred, average=None)
    f1_average = f1_score(labels, labels_pred, average='macro')
    return f1, f1_average


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))


def compute_metrics_cholec(prediction, label):
    metrics = {}
    report = classification_report(label, prediction)
    metrics['report'] = report
    return metrics


def log_computed_metrics_cholec(metrics):
    report = metrics['report']
    return report