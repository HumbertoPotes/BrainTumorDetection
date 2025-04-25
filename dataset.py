from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import torch

class BrainTumorDataset(Dataset):
    """
    Custom dataset for loading brain tumor images.
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.image_size = (256, 256)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Args:
            idx (int): Index of the image to retrieve.
        Returns:
            image_tensor (Tensor): Transformed image tensor.
            label_tensor (Tensor): Segmentation mask tensor.
        """
        sample = self.dataset[idx]
        image = sample['image']
        segmentation = sample['segmentation'][0]

        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        pts = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1) # fill the polygon indicating tumor with ones (zeroes indicate no tumor)

        # print(f"Mask area: {mask.sum()}, Truth area: {sample['area']}") # check if the mask area matches the truth area of the tumor

        image_tensor = self.transform(image)
        mask_tensor = torch.from_numpy(np.array(Image.fromarray(mask).resize(self.image_size, Image.NEAREST))).long()

        return image_tensor, mask_tensor

    