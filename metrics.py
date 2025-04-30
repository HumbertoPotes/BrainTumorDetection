import torch

def compute_iou(pred_mask, true_mask, threshold=0.5, eps=1e-6):
    probabilities = torch.sigmoid(pred_mask)
    pred_bin = (probabilities > threshold).float()
    true_bin = true_mask.float()

    intersection = (pred_bin * true_bin).sum(dim=(1, 2))
    union = (pred_bin + true_bin - pred_bin * true_bin).sum(dim=(1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def false_positive_penalty_iou(pred_mask, true_mask, threshold=0.5, eps=1e-6):
    probabilities = torch.sigmoid(pred_mask)
    pred_bin = (probabilities > threshold).float()
    true_bin = true_mask.float()

    tp = (pred_bin * true_bin).sum(dim=(1, 2))
    fp = (pred_bin * (1 - true_bin)).sum(dim=(1, 2))

    precision_iou = (tp + eps) / (tp + fp + eps)
    return precision_iou.mean().item()