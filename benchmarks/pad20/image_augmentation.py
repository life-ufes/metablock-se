# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

Image augmentation classes for PAD-UFES-20 dataset
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
np.bool = bool
import torchvision
import cv2

class ImgTrainTransform:

    def __init__(self, size=(224, 224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        drop_prob = np.random.uniform(0.0, 0.05)
        self.aug = A.Compose([
            A.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)}, p=0.25),
            A.Resize(size[0], size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(rotate=(-120, 120), mode=cv2.BORDER_REFLECT, p=0.25),
            A.GaussianBlur(sigma_limit=(0, 3.0) , p=0.25),

            # noise
            A.OneOf([
                A.PixelDropout(dropout_prob=drop_prob, p=1),
                A.CoarseDropout(num_holes_range=(int(0.00125*size[0]*size[1]), int(0.00125*size[0]*size[1])), hole_height_range=(4, 4), hole_width_range=(4, 4), p=1), # PRECISA VERIFICAR
            ], p=0.1),

            A.OneOf([
                A.OneOrOther(
                    first=A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, p=1),
                    second=A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=False, p=1),
                    p=0.5
                ), # brigthness
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=0, p=1),
            ], p=0.25),
            A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
            ToTensorV2(),
        ])

    def __call__(self, img):   
        img = self.aug(image=np.array(img))['image']        
        return img
        


class ImgEvalTransform:

    def __init__(self, size=(224, 224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        self.size = size

    def __call__(self, img):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        return transforms(img)
