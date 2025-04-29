import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from dataset import BrainTumorDataset
from model import ConvTumorDetector
from torch.utils.tensorboard import SummaryWriter

def train(
    exp_dir: str = "logs",
    num_epoch: int = 30,
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

    net = ConvTumorDetector(in_channels=1, num_classes=2)
    net.to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Starting training for {num_epoch} epochs")
    for epoch in range(num_epoch):
        # train pass
        net.train()
        train_losses = []

        for images, masks in train_loader:
            # move images to the device
            images = images.to(device)
            masks = masks.to(device)

            # forward pass
            outputs = net(images)
            loss_fn = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks.float())
            train_losses.append(loss_fn.item())

            # backward pass
            optim.zero_grad()
            loss_fn.backward()
            optim.step()

        # validation pass
        with torch.inference_mode():
            net.eval()
            val_losses = []

            for images, masks in val_loader:
                # move images to the device
                images = images.to(device)
                masks = masks.to(device)

                # forward pass
                outputs = net(images)
                loss_fn = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks.float())
                val_losses.append(loss_fn.item())

        train_loss = sum(train_losses)/len(train_losses)
        val_loss = sum(val_losses)/len(val_losses)
        writer.add_scalar("train/loss", train_loss, global_step=epoch+1)
        writer.add_scalar("val/loss", val_loss, global_step=epoch+1)
        # writer.add_scalar("epoch", epoch, epoch)
        writer.flush()
        print(f"Epoch {epoch+1}/{num_epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        if (epoch+1) % 10 == 0:
            print(f"Saving model at epoch {epoch+1}")
            torch.save(net.state_dict(), f"{exp_dir}/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=30)
    train(**vars(parser.parse_args()))