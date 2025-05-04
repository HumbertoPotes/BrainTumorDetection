from utils import sigmoid_to_binary


def compute_iou(pred_mask, true_mask, threshold=0.5):
    eps = 1e-6  # avoid division by zero
    pred_bin = sigmoid_to_binary(pred_mask, threshold=threshold)

    intersection = (pred_bin * true_mask).sum(dim=(1, 2))
    union = (pred_bin + true_mask - pred_bin * true_mask).sum(dim=(1, 2))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def false_positive_penalty_iou(pred_mask, true_mask, threshold=0.5):
    eps = 1e-6  # avoid division by zero
    pred_bin = sigmoid_to_binary(pred_mask, threshold=threshold)

    tp = (pred_bin * true_mask).sum(dim=(1, 2))
    fp = (pred_bin * (1 - true_mask)).sum(dim=(1, 2))

    precision_iou = (tp + eps) / (tp + fp + eps)
    return precision_iou.mean().item()
