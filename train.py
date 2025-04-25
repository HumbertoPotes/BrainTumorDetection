from datasets import load_dataset
from PIL import Image
from dataset import BrainTumorDataset
from torch.utils.data import DataLoader
import torch

def train():
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
    print(dataset)

    # load the training dataset
    train_dataset = BrainTumorDataset(dataset['train'])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for images, masks in train_loader:
        # move images to the device
        images = images.to(device)
        masks = masks.to(device)

        # print the shape of the images tensor
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")

if __name__ == "__main__":
    train()
