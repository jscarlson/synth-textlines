import torchvision.transforms as T
from utils.colors import color_shift_from_targets, color_shift
import albumentations as A
import numpy as np


TRANSFORM_DICT = {
    "default":
        T.Compose([
            T.ToTensor(),
            T.RandomApply([color_shift], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
            T.RandomApply([T.GaussianBlur(15, sigma=(1, 4))], p=0.5),
            T.RandomInvert(p=0.2),
            T.RandomGrayscale(p=0.2),
            T.ToPILImage(),
        ]),
    "album":
        T.Compose([
            T.ToTensor(),
            T.RandomApply([color_shift], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3)], p=0.5),
            T.RandomApply([T.GaussianBlur(15, sigma=(1, 4))], p=0.5),
            T.RandomInvert(p=0.2),
            T.RandomGrayscale(p=0.2),
            T.ToPILImage(),
            lambda x: A.PixelDropout(dropout_prob=0.01, drop_value=0, p=0.25)(image=np.array(x))["image"],
            lambda x: A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.25)(image=np.array(x))["image"],
            lambda x: A.ImageCompression(quality_lower=0, quality_upper=100, p=0.25)(image=np.array(x))["image"],
            T.ToPILImage(),
        ]),
    "pr": 
        T.Compose([
            T.ToTensor(),
            lambda x: color_shift_from_targets(x, targets=[[234,234,212], [225, 207, 171]]),
            T.RandomApply([T.GaussianBlur(11)], p=0.35),
            T.ToPILImage()
        ])
}
