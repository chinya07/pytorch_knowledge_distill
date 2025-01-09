import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random
import numpy as np

class RandomErasing:
    """Randomly erases a rectangular region in an image."""
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        """
        Args:
            p (float): Probability of applying random erasing.
            scale (tuple): Range of proportion of erased area against input image.
            ratio (tuple): Aspect ratio range of the erased area.
            value (int, tuple, or str): Erasing value. If int, uses that value. 
                                        If "random", erases with random noise.
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        """Apply Random Erasing to the input image."""
        if random.uniform(0, 1) > self.p:
            return img
            
        if isinstance(img, torch.Tensor):
            img_h, img_w = img.shape[-2:]
        else:  # PIL Image
            img_w, img_h = img.size  # PIL Image uses (width, height)
            
        area = img_h * img_w

        for _ in range(10):  # Try 10 times to find a valid rectangle
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))

            if h < img_h and w < img_w:
                x1 = random.randint(0, img_h - h)
                y1 = random.randint(0, img_w - w)

                if isinstance(img, torch.Tensor):
                    if isinstance(self.value, str) and self.value == "random":
                        erase_value = torch.rand((img.size(0), h, w)).to(img.device)
                    else:
                        erase_value = torch.tensor(self.value).to(img.device).expand((img.size(0), h, w))
                    img[:, x1:x1 + h, y1:y1 + w] = erase_value
                else:  # PIL Image
                    if isinstance(self.value, str) and self.value == "random":
                        erase_value = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
                    else:
                        erase_value = np.full((h, w, 3), self.value, dtype=np.uint8)
                    img_array = np.array(img)
                    img_array[x1:x1 + h, y1:y1 + w] = erase_value
                    img = F.to_pil_image(img_array)
                return img

        return img  # Return original image if no valid rectangle is found


class ColorJitter:
    """Applies random changes in brightness, contrast, saturation, and hue."""
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        """
        Args:
            brightness (float): How much to jitter brightness.
            contrast (float): How much to jitter contrast.
            saturation (float): How much to jitter saturation.
            hue (float): How much to jitter hue.
        """
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, img):
        """Apply Color Jitter to the input image."""
        return self.jitter(img)

