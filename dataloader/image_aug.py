"""File to store augmentation options.
"""

from torchvision import transforms

def basic_transformation():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
