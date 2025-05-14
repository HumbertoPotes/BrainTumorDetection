from utils import sigmoid_to_binary
import torch
from torchvision.ops import masks_to_boxes
from utils import clean_by_distance


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


def bb_compute_iou(boxA, boxB, fpp_iou=False):
    # Create the intersection coordinates
    xA = torch.max(boxA[:, 0], boxB[:, 0])
    yA = torch.max(boxA[:, 1], boxB[:, 1])
    xB = torch.min(boxA[:, 2], boxB[:, 2])
    yB = torch.min(boxA[:, 3], boxB[:, 3])

    # Compute intersection area
    inter_width = torch.clamp(xB - xA, min=0)
    inter_height = torch.clamp(yB - yA, min=0)
    inter_area = inter_width * inter_height

    # Areas of boxes
    areaA = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    areaB = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    # Union area
    union = areaA + areaB - inter_area

    if fpp_iou:
        iou = iou = torch.where(areaA > 0, inter_area / areaA, torch.tensor(0.0, device=areaA.device))
    else:
        iou = iou = torch.where(union > 0, inter_area / union, torch.tensor(0.0, device=union.device))

    return iou


def masks_to_iou(masks_pred, masks_true, threshold=0.5, stdev_multiplier=3.5, beta=0.5):
    iou_mask = clean_by_distance(
        masks_pred, threshold=threshold, stdev_multiplier=stdev_multiplier
    )
    has_mask = iou_mask.flatten(1).any(dim=1).bool()  # (N,)
    valid_masks = iou_mask[has_mask]

    masks_pred_bb = torch.zeros((iou_mask.shape[0], 4), dtype=torch.float32, device=iou_mask.device)
    if valid_masks.numel() > 0:
        masks_pred_bb[has_mask] = masks_to_boxes(valid_masks)
    masks_bb = masks_to_boxes(masks_true)
    val_iou = bb_compute_iou(masks_pred_bb, masks_bb)
    val_fpp_iou = bb_compute_iou(masks_pred_bb, masks_bb, fpp_iou=True)
    val_total_iou = beta * val_iou + (1 - beta) * val_fpp_iou

    return val_total_iou


def accuracy(pred_cat, true_cat, threshold=0.5):
    pred_cat = sigmoid_to_binary(pred_cat, threshold=threshold)
    correct = (pred_cat == true_cat).sum().item()
    total = pred_cat.size(0)
    return correct / total
