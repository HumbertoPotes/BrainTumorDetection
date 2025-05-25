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

    def __init__(self, dataset, augment=False):
        super().__init__()
        self.dataset = dataset
        self.image_size = (640, 640)
        self.to_tensor = transforms.ToTensor()
        self.to_grayscale = transforms.Grayscale(num_output_channels=1)
        self.resize = transforms.Resize(self.image_size)
        self.augment = augment

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
        image = sample["image"]
        segmentation = sample["segmentation"][0]
        category = sample["category_id"]

        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        pts = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(
            mask, [pts], 1
        )  # fill the polygon indicating tumor with ones (zeroes indicate no tumor)

        # Preprocess the image and mask
        mask = Image.fromarray(mask) # convert mask to PIL Image
        mask = self.resize(mask)
        image = self.to_grayscale(image)
        image = self.resize(image)

        # Apply augmentations if specified
        if self.augment:
            if np.random.rand() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if np.random.rand() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            if np.random.rand() > 0.5:
                angle = np.random.randint(-30, 30)
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)
            if np.random.rand() > 0.5:
                brightness_factor = np.random.uniform(0.7, 1.3)
                contrast_factor = np.random.uniform(0.7, 1.3)
                image = transforms.functional.adjust_brightness(image, brightness_factor)
                image = transforms.functional.adjust_contrast(image, contrast_factor)

        # print(f"Mask area: {mask.sum()}, Truth area: {sample['area']}") # check if the mask area matches the truth area of the tumor

        # Final conversion to tensors
        image_tensor = self.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        category_tensor = (
            torch.tensor(category, dtype=torch.long)
        ) - 1  # convert to 0-indexed (0 is tumor, 1 is normal)

        return image_tensor, mask_tensor, category_tensor
