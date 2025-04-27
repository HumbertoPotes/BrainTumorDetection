from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
import torch
from dataset import BrainTumorDataset
from model import ConvTumorDetector
from torch.utils.tensorboard import SummaryWriter

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

    # load the dataset from Hugging Face Hub
    dataset = load_dataset("dwb2023/brain-tumor-image-dataset-semantic-segmentation")

    # load the training dataset
    train_dataset = BrainTumorDataset(dataset['train'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    net = ConvTumorDetector(in_channels=3, num_classes=2)
    net.to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0

    print(f"Starting training for {num_epoch} epochs")
    for epoch in range(num_epoch):

        net.train()
        for images, masks in train_loader:
            # move images to the device
            images = images.to(device)
            masks = masks.to(device)

            # forward pass
            outputs = net(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)


            # backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

            writer.add_scalar("train/loss", loss.item(), global_step=global_step)

            global_step += 1

        writer.add_scalar("epoch", epoch, epoch)

        writer.flush()
        
        if (epoch+1) % 10 == 0:
                print(f"Saving model at epoch {epoch}")
                torch.save(net.state_dict(), f"{exp_dir}/model_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()