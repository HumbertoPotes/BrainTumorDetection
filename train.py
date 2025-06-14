import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from dataset import BrainTumorDataset
from model import ConvTumorDetector
from torch.utils.tensorboard import SummaryWriter
from metrics import accuracy, masks_to_iou


def train(
    exp_dir: str = "logs",
    num_epoch: int = 200,
    lr: float = 1e-3,
    batch_size: int = 8,
    augment: bool = True,
    pos_w: float = 0.5, # weight for positive (tumorous) class in BCE loss
    alpha: float = 0.999, # weight for loss function
    beta: float = 0.5, # weight for IoU function
    stdev_multiplier: float = 3.5,
):
    print("Starting training...")
    writer = SummaryWriter()

    # set device to GPU if available or MPS if on MacOS
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # set random seed for reproducibility
    torch.manual_seed(
        2025
    )  # the year the Texas Longhorns win the national championship
    torch.cuda.manual_seed(2025) if device == torch.device("cuda") else None
    torch.backends.mps.manual_seed(2025) if device == torch.device("mps") else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load the dataset from Hugging Face Hub
    dataset = load_dataset("dwb2023/brain-tumor-image-dataset-semantic-segmentation")

    # load the training dataset
    train_dataset = BrainTumorDataset(dataset["train"], augment=augment)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = BrainTumorDataset(dataset["valid"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    net = ConvTumorDetector(in_channels=1, num_classes=1)
    net.to(device)
    optim = torch.optim.AdamW(
        [
            {"params": net.network.parameters(), "lr": lr},
            {"params": net.segmentation_head.parameters(), "lr": lr},
            {"params": net.category_head.parameters(), "lr": lr / 100},
        ]
    )

    print(f"Starting training for {num_epoch} epochs")
    for epoch in range(num_epoch):
        # train pass
        net.train()
        train_seg_losses = []
        train_cat_losses = []
        train_total_losses = []
        train_acum_iou = []
        train_acum_acc = []
        loss_fn_seg = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_w]).to(device)
        )
        loss_fn_cat = torch.nn.BCEWithLogitsLoss()

        for images, masks, categories in train_loader:
            # move images to the device
            images = images.to(device)
            masks = masks.to(device).float()
            categories = categories.to(device).float()

            # forward pass and calculate loss
            masks_pred, categories_pred = net(images)
            loss_seg = loss_fn_seg(masks_pred, masks)
            loss_cat = loss_fn_cat(categories_pred, categories)
            total_loss = (alpha * loss_seg) + ((1 - alpha) * loss_cat)
            train_seg_losses.append(loss_seg.item())
            train_cat_losses.append(loss_cat.item())
            train_total_losses.append(total_loss.item())

            # segmentation metrics
            train_total_iou = masks_to_iou(masks_pred, masks, threshold=0.5, stdev_multiplier=stdev_multiplier, beta=beta)
            train_acum_iou.extend(train_total_iou.tolist())

            # category metrics
            train_acc = accuracy(categories_pred, categories, threshold=0.5)
            train_acum_acc.append(train_acc)

            # backward pass
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        # validation pass
        with torch.inference_mode():
            net.eval()
            val_seg_losses = []
            val_cat_losses = []
            val_total_losses = []
            val_acum_iou = []
            val_acum_acc = []

            for images, masks, categories in val_loader:
                # move images to the device
                images = images.to(device)
                masks = masks.to(device).float()
                categories = categories.to(device).float()

                # forward pass and calculate loss
                masks_pred, categories_pred = net(images)
                loss_seg = loss_fn_seg(masks_pred, masks)
                loss_cat = loss_fn_cat(categories_pred, categories)
                total_loss = (alpha * loss_seg) + ((1 - alpha) * loss_cat)
                val_seg_losses.append(loss_seg.item())
                val_cat_losses.append(loss_cat.item())
                val_total_losses.append(total_loss.item())

                # segmentation metrics
                val_total_iou = masks_to_iou(masks_pred, masks, threshold=0.5, stdev_multiplier=stdev_multiplier, beta=beta)
                val_acum_iou.extend(val_total_iou.tolist())

                # category metrics
                val_acc = accuracy(categories_pred, categories, threshold=0.5)
                val_acum_acc.append(val_acc)

        # log the losses and metrics
        train_total_loss = sum(train_total_losses) / len(train_total_losses)
        train_total_acc = sum(train_acum_acc) / len(train_acum_acc)
        train_total_iou = sum(train_acum_iou) / len(train_acum_iou)
        writer.add_scalar("train/total_loss", train_total_loss, global_step=epoch + 1)
        writer.add_scalar("train/total_IoU", train_total_iou, global_step=epoch + 1)
        writer.add_scalar(
            "train/total_accuracy", train_total_acc, global_step=epoch + 1
        )

        val_total_loss = sum(val_total_losses) / len(val_total_losses)
        val_total_acc = sum(val_acum_acc) / len(val_acum_acc)
        val_total_iou = sum(val_acum_iou) / len(val_acum_iou)
        writer.add_scalar("val/total_loss", val_total_loss, global_step=epoch + 1)
        writer.add_scalar("val/total_IoU", val_total_iou, global_step=epoch + 1)
        writer.add_scalar("val/total_accuracy", val_total_acc, global_step=epoch + 1)

        writer.flush()

        # print(
        #     f"Epoch {epoch+1}/{num_epoch}: \n"
        #     f"Training Loss - Total: {train_total_loss:.4f}, Segmentation: {train_seg_loss:.4f}, Category: {train_cat_loss:.4f} \n"
        #     f"Validation Loss - Total: {val_total_loss:.4f}, Segmentation: {val_seg_loss:.4f}, Category: {val_cat_loss:.4f} "
        # )

        # save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Saving model at epoch {epoch+1}")
            torch.save(net.state_dict(), f"{exp_dir}/model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_epoch", type=int, default=30)
    train(**vars(parser.parse_args()))
