from datasets import load_dataset
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

    # load the dataset
    dataset = load_dataset("dwb2023/brain-tumor-image-dataset-semantic-segmentation")
    print(dataset)
    for image in dataset["test"]:
        print(len(image["segmentation"][0]))

if __name__ == "__main__":
    train()
