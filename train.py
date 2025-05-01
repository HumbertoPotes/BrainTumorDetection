import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from dataset import BrainTumorDataset
from model import ConvTumorDetector
from torch.utils.tensorboard import SummaryWriter
from metrics import compute_iou, false_positive_penalty_iou

def train(
    exp_dir: str = "logs",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
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
    torch.manual_seed(2025) # the year the Texas Longhorns win the national championship
    torch.cuda.manual_seed(2025) if device == torch.device("cuda") else None
    torch.backends.mps.manual_seed(2025) if device == torch.device("mps") else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load the dataset from Hugging Face Hub
    dataset = load_dataset("dwb2023/brain-tumor-image-dataset-semantic-segmentation")

    # load the training dataset
    train_dataset = BrainTumorDataset(dataset['train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = BrainTumorDataset(dataset['valid'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    net = ConvTumorDetector(in_channels=1, num_classes=1)
    net.to(device)
    optim = torch.optim.AdamW([
        {"params": net.network.parameters(), "lr": lr},
        {"params": net.segmentation_head.parameters(), "lr": lr},
        {"params": net.category_head.parameters(), "lr": lr / 100}
    ])

    # hyperparameters
    pos_w = 0.5
    alpha = 0.9
    beta = 0.5

    print(f"Starting training for {num_epoch} epochs")
    for epoch in range(num_epoch):
        # train pass
        net.train()
        train_seg_losses = []
        train_cat_losses = []
        train_total_losses = []
        loss_fn_seg = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(device))
        loss_fn_cat = torch.nn.BCEWithLogitsLoss()
        

        for images, masks, categories in train_loader:
            # move images to the device
            images = images.to(device)
            masks = masks.to(device)
            categories = categories.to(device)

            # forward pass and calculate loss
            masks_pred, categories_pred = net(images)
            loss_seg = loss_fn_seg(masks_pred, masks.float())
            loss_cat = loss_fn_cat(categories_pred, categories.float())
            total_loss = (alpha * loss_seg) + ((1-alpha) * loss_cat)
            train_seg_losses.append(loss_seg.item())
            train_cat_losses.append(loss_cat.item())
            train_total_losses.append(total_loss.item())

            # calculate metrics
            train_iou = compute_iou(masks_pred, masks)
            train_fpp_iou = false_positive_penalty_iou(masks_pred, masks)
            train_total_iou = beta * train_iou + (1 - beta) * train_fpp_iou

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

            for images, masks, categories in val_loader:
                # move images to the device
                images = images.to(device)
                masks = masks.to(device).float()
                categories = categories.to(device).float()

                # forward pass and calculate loss
                masks_pred, categories_pred = net(images)
                loss_seg = loss_fn_seg(masks_pred, masks)
                loss_cat = loss_fn_cat(categories_pred, categories)
                total_loss = (alpha * loss_seg) + ((1-alpha) * loss_cat)
                val_seg_losses.append(loss_seg.item())
                val_cat_losses.append(loss_cat.item())
                val_total_losses.append(total_loss.item())

                # calculate metrics
                val_iou = compute_iou(masks_pred, masks)
                val_fpp_iou = false_positive_penalty_iou(masks_pred, masks)
                val_total_iou = beta * val_iou + (1 - beta) * val_fpp_iou
                

        # log the losses
        train_seg_loss = sum(train_seg_losses)/len(train_seg_losses)
        train_cat_loss = sum(train_cat_losses)/len(train_cat_losses)
        train_total_loss = sum(train_total_losses)/len(train_total_losses)
        writer.add_scalar("train/total_loss", train_total_loss, global_step=epoch+1)
        writer.add_scalar("train/total_IoU", train_total_iou, global_step=epoch+1)

        val_seg_loss = sum(val_seg_losses)/len(val_seg_losses)
        val_cat_loss = sum(val_cat_losses)/len(val_cat_losses)
        val_total_loss = sum(val_total_losses)/len(val_total_losses)
        writer.add_scalar("val/total_loss", val_total_loss, global_step=epoch+1)
        writer.add_scalar("val/total_IoU", val_total_iou, global_step=epoch+1)

        writer.flush()

        # print(
        #     f"Epoch {epoch+1}/{num_epoch}: \n" 
        #     f"Training Loss - Total: {train_total_loss:.4f}, Segmentation: {train_seg_loss:.4f}, Category: {train_cat_loss:.4f} \n"
        #     f"Validation Loss - Total: {val_total_loss:.4f}, Segmentation: {val_seg_loss:.4f}, Category: {val_cat_loss:.4f} "
        # )
        
        # save the model every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"Saving model at epoch {epoch+1}")
            torch.save(net.state_dict(), f"{exp_dir}/model_epoch_{epoch+1}_w05.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_epoch", type=int, default=30)
    train(**vars(parser.parse_args()))