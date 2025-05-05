import torch
from numpy import empty, uint8
from matplotlib import pyplot as plt
from cv2 import boundingRect
import matplotlib.patches as patches
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sigmoid_to_binary(pred_mask, threshold=0.5):
    probabilities = torch.sigmoid(pred_mask)
    pred_bin = (probabilities > threshold).float()

    return pred_bin


def nanstd(tensor):
    mask = ~torch.isnan(tensor)
    count = mask.sum(dim=1, keepdim=True)
    mean = torch.nan_to_num(tensor, nan=0.0).sum(dim=1, keepdim=True) / count
    var = (
        torch.nan_to_num(((tensor - mean) ** 2), nan=0.0).sum(dim=1, keepdim=True)
        / count
    )
    return torch.sqrt(var)


def clean_by_distance(pred_mask, threshold=0.5, stdev_multiplier=3.5):

    B, H, W = pred_mask.shape
    device = pred_mask.device

    bin_mask = sigmoid_to_binary(pred_mask, threshold=threshold)

    eps = 1e-6  # Avoid division by zero
    total = (
        bin_mask.sum(dim=(1, 2), keepdim=True) + eps
    )  # Sum over height and width not batch

    # Create coordinate grids
    y_coords = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
    x_coords = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)

    # Compute center of mass for each image
    cy = (bin_mask * y_coords).sum(dim=(1, 2), keepdim=True) / total
    cx = (bin_mask * x_coords).sum(dim=(1, 2), keepdim=True) / total

    # Compute distance from coord grid and centers
    dist_to_center = torch.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

    # Keep only pixels close to the center
    masked_mask = bin_mask.view(B, -1) == 1
    std_devs = nanstd(
        torch.where(
            masked_mask,
            dist_to_center.view(B, -1),
            torch.tensor(float("nan"), device=device),
        )
    ).view(B, 1, 1)
    cleaned = ((bin_mask == 1) & (dist_to_center <= stdev_multiplier * std_devs)).to(
        torch.uint8
    )

    return cleaned


def visualize_comparisons(
    image_np, pred_mask, pred_cat, true_mask, true_cat, batch_size
):
    wsize = batch_size * 6 if batch_size > 1 else 8
    hsize = batch_size * 4 if batch_size > 1 else 6
    fig, axs = plt.subplots(batch_size, 2, figsize=(wsize, hsize))

    if batch_size == 1:
        axs = axs.reshape(1, 2)
    for i in range(batch_size):
        axs[i, 0].imshow(image_np[i], cmap="gray")

        # Plot the predicted bounding box
        # Convert the predicted mask to a binary mask
        pred_binary_mask = (pred_mask[i].cpu().numpy() > 0).astype(uint8)
        # Find the bounding box using OpenCV
        x, y, w, h = boundingRect(pred_binary_mask)
        pred_rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
            label="Predicted",
        )
        axs[i, 0].add_patch(pred_rect)

        # Repeat process for the original mask
        og_binary_mask = (true_mask[i] > 0).astype(uint8)
        orig_x, orig_y, orig_w, orig_h = boundingRect(og_binary_mask)
        orig_rect = patches.Rectangle(
            (orig_x, orig_y),
            orig_w,
            orig_h,
            linewidth=2,
            edgecolor="b",
            facecolor="none",
            label="Original",
        )
        axs[i, 0].add_patch(orig_rect)
        axs[i, 0].legend()
        axs[i, 0].set_axis_off()

        # Visualize the mask instead of the bounding box
        axs[i, 1].imshow(image_np[i], cmap="gray")
        # Overlay the predicted mask on the image
        overlay = empty((640, 640, 3))
        overlay[pred_mask[i].cpu().numpy() > 0] = [1, 0, 0]
        axs[i, 1].imshow(
            overlay, alpha=0.7, label="Predicted Mask"
        )  # Add transparency to the overlay
        # original mask
        overlay = empty((640, 640, 3))
        overlay[true_mask[i] > 0] = [0, 0, 1]
        axs[i, 1].imshow(overlay, alpha=0.5, label="Original Mask")
        axs[i, 1].set_axis_off()

        if i == 0:
            axs[i, 0].set_title("Bounding Box Comparison", fontsize=14)
            axs[i, 1].set_title("Mask Comparison", fontsize=14)

    fig.suptitle(
        "Visualization of the Network's Predictions",
    fontsize=16
    )
    fig.text(0.5, 0.9, 
        f"Original Tumor Category: {true_cat.int().tolist()} \nPredicted Tumor Category: {pred_cat.int().tolist()}",
        ha='center', fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=-0.5)

    return fig, axs

def tensorboard_log(
    logs_dict, folder_path, train_or_val, metric,
):
    # folder_path example: runs/May04_01-37-41_COMPUTER_NAME
    ea = EventAccumulator(folder_path)  # path to a specific run
    ea.Reload()

    scalars = ea.Scalars(f'{train_or_val}/{metric}')
    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]

    if train_or_val == "val":
        train_or_val = "Validation"
    else:
        train_or_val = "Train"

    dict_key = f"{train_or_val} {metric}"
    logs_dict[dict_key] = (steps, values)

    return logs_dict


def visualization_of_logs(
        logs_dict
    ):
    n_graphs = len(logs_dict)
    title_graphs = list(logs_dict.keys())
    hsize = n_graphs * 3
    fig, axs = plt.subplots(n_graphs, 1, figsize=(8, hsize))
    axs = [axs] if n_graphs == 1 else axs

    for i, (key,val) in enumerate(logs_dict.items()):
        steps, values = val
        axs[i].plot(steps, values)

        metric = key.split(" ")[1]

        plt.xlabel("Epoch")
        plt.ylabel(f"{metric}")
        axs[i].set_title(f"{title_graphs[i]}")
        if i < n_graphs - 1:
            axs[i].set_xticklabels([])
            axs[i].set_xticks([])

    return fig, axs
