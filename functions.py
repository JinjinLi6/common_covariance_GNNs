import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def compute_f1_score_tpr_fpr(preds, labels):
    TP = ((preds == 1) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    FN = ((preds == 0) & (labels == 1)).sum()
    TN = ((preds == 0) & (labels == 0)).sum()

    tpr = TP / (TP + FN) if TP + FN != 0 else 0.0
    fpr = FP / (FP + TN) if FP + TN != 0 else 0.0

    precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    recall = TP / (TP + FN) if TP + FN != 0 else 0.0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    return f1_score, tpr, fpr

def compute_f1_score_tpr_fpr_full(preds, labels):
    TP = ((preds == 1) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    FN = ((preds == 0) & (labels == 1)).sum()
    TN = ((preds == 0) & (labels == 0)).sum()

    tpr = TP / (TP + FN) if TP + FN != 0 else 0.0
    fpr = FP / (FP + TN) if FP + TN != 0 else 0.0

    precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    recall = TP / (TP + FN) if TP + FN != 0 else 0.0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    return TP, FP, FN, TN, f1_score, tpr, fpr

def compute_auc(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)
    return auc

def seed_torch(seed = 2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
