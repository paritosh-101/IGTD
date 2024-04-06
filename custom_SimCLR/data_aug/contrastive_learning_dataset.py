from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torchvision.datasets import ImageFolder
from torchvision.transforms import Grayscale
from PIL import Image
import torch
import random
import numpy as np


class RandomRowShuffle:
    def __init__(self, image_size=48):
        self.image_size = image_size

    def __call__(self, img):
        # Assuming img is a PIL image, convert it to a tensor
        img_tensor = transforms.ToTensor()(img)

        # Generate a random permutation of row indices
        indices = torch.randperm(self.image_size)

        # Apply the permutation to the rows
        shuffled_img_tensor = img_tensor[:, indices, :]

        # Convert back to PIL image for further processing in the pipeline
        return transforms.ToPILImage()(shuffled_img_tensor)
    
class RandomRowFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        # Assuming img is a PIL image, convert it to a tensor
        img_tensor = transforms.ToTensor()(img)

        # Randomly decide whether to flip the rows or not
        if random.random() < self.p:
            # Flip the rows (vertical flip)
            flipped_img_tensor = torch.flip(img_tensor, [1])
        else:
            flipped_img_tensor = img_tensor

        # Convert back to PIL image for further processing in the pipeline
        return transforms.ToPILImage()(flipped_img_tensor)
    
class RandomMaskPatches:
    def __init__(self, image_size=32, patch_size=3):
        self.image_size = image_size
        self.patch_size = patch_size
        self.total_pixels = image_size * image_size
        self.patches_per_side = image_size - patch_size + 1

    def __call__(self, img):
        # Convert image to numpy array
        img_array = np.array(img)

        # Determine the range of patches to mask
        min_patches = int(np.ceil(0.20 * self.total_pixels / (self.patch_size ** 2)))
        max_patches = int(np.floor(0.50 * self.total_pixels / (self.patch_size ** 2)))

        # Randomly select the number of patches to mask
        num_patches_to_mask = random.randint(min_patches, max_patches)

        # Randomly select patches to mask
        for _ in range(num_patches_to_mask):
            x = random.randint(0, self.patches_per_side - 1)
            y = random.randint(0, self.patches_per_side - 1)
            img_array[y:y + self.patch_size, x:x + self.patch_size] = 0  # Masking with black color

        return Image.fromarray(img_array)
    
class RandomRowMask:
    def __init__(self, min_mask_ratio=0.2, max_mask_ratio=0.5):
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def __call__(self, img):
        # Convert image to numpy array
        img_array = np.array(img)

        # Get the number of rows and columns in the image
        num_rows, num_cols = img_array.shape[:2]

        # Create a copy of the image array to store the masked image
        masked_img_array = np.copy(img_array)

        # Iterate over each row
        for row in range(num_rows):
            # Randomly determine the masking ratio for the current row
            mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)

            # Calculate the number of pixels to mask in the current row
            num_pixels_to_mask = int(num_cols * mask_ratio)

            # Randomly select the indices of pixels to mask in the current row
            mask_indices = random.sample(range(num_cols), num_pixels_to_mask)

            # Mask the selected pixels in the current row with black color
            masked_img_array[row, mask_indices] = 0

        return Image.fromarray(masked_img_array)
    
class ContrastiveLearningDataset:
    def __init__(self, root_folder, crop_size=224):
        self.root_folder = root_folder
        self.crop_size = crop_size

    # @staticmethod
    # def get_simclr_pipeline_transform(size, s=1):
    #     """Return a set of data augmentation transformations as described in the SimCLR paper."""
    #     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    #     data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.RandomApply([color_jitter], p=0.8),
    #                                           transforms.RandomGrayscale(p=0.2),
    #                                           GaussianBlur(kernel_size=int(0.1 * size)),
    #                                           transforms.ToTensor()])
    #     return data_transforms
    
    # @staticmethod
    def get_simclr_pipeline_transform(self, s=1, is_grayscale=False):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        transformations = [RandomMaskPatches(image_size=32),
                           transforms.RandomResizedCrop(size=self.crop_size),
                           transforms.RandomHorizontalFlip(),
                           GaussianBlur(kernel_size=int(0.1 * self.crop_size))]
        # transformations = [
        #     RandomRowShuffle(image_size=self.crop_size),
        #     RandomRowFlip(p=0.5),
        #     RandomRowMask(min_mask_ratio=0.2, max_mask_ratio=0.5)
        # ]

        if not is_grayscale:
            # Color jitter only for RGB images
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            transformations.append(transforms.RandomApply([color_jitter], p=0.8))
            transformations.append(transforms.RandomGrayscale(p=0.2))

        # Gaussian blur can be applied to both grayscale and RGB
        # transformations.append(GaussianBlur(kernel_size=int(0.1 * self.crop_size)))
        
        transformations.append(transforms.ToTensor())

        return transforms.Compose(transformations)

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(),
                                                              n_views),
                                                          download=True),
                                                          
                            # Custom RGB Dataset
                            'custom_rgb': lambda: ImageFolder(
                                self.root_folder,
                                transform=ContrastiveLearningViewGenerator(
                                    self.get_simclr_pipeline_transform(),  # Adjust size as needed
                                    n_views)
                            ),

                            # Custom Grayscale Dataset
                            'custom_grayscale': lambda: ImageFolder(
                                self.root_folder,
                                transform=transforms.Compose([
                                    transforms.Resize((self.crop_size, self.crop_size)),
                                    Grayscale(num_output_channels=1),
                                    ContrastiveLearningViewGenerator(
                                        self.get_simclr_pipeline_transform(is_grayscale=True),  # Adjust size as needed
                                        n_views)
                                ])
                            )
                        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
