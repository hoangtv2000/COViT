"""File to store augmentation options.
"""
import torchio as tio
from torchvision import transforms
from .auto_augment import SVHNPolicy, SubPolicy
import numpy as np


def base_train_transformation(img_size):
    """Base train augmentation.
    """
    return transforms.Compose([
        transforms.RandomAffine(degrees=(-5, 5), translate=(0.10, 0.10)),
        transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def autoaug_transformation(img_size):
    """Augmentation from AutoAug paper, which is disabled coloring and geometric transformations.
    """
    return transforms.Compose([
        transforms.RandomAffine(degrees=(-5, 5), translate=(0.10, 0.10)),
        transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.1)),
        SVHNPolicy(fillcolor=(0, 0, 0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



def base_test_transformation():
    """Base test img preprocessing.
    """
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])



#################################################################
def torchio_train_transformation():
    """ Modified Train preprocessing & augmentation of TorchIO 3D brain segmentation.
    """
    return tio.Compose([
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
            tio.ZNormalization(), # Pixel Normalization
            tio.OneOf({
                tio.RandomAffine(scales=(0.9, 1.1), degrees=(-5, 5), translation=(0.10, 0.10)): 0.75,
                tio.RandomElasticDeformation(): 0.25,
                    }),
            ])


def torchio_test_transformation():
    """ Modified Val and Test preprocessing of TorchIO 3D brain segmentation.
    """
    return tio.Compose([
            tio.ZNormalization(), # Pixel Normalization
            ])
