"""
augmentation.py — Data augmentation pipeline for CIFAR100 (student-modified).

Students: Extend the *training* transform pipeline to improve generalization.
The validation pipeline is fixed — do not modify it.

CIFAR100 images are 32×32. Both pipelines resize to 224×224 to match the
input expected by the pretrained ResNet18 backbone.
"""

import torchvision.transforms as T

# Per-channel mean and std computed on the CIFAR100 training set.
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    """Return the image transform pipeline for CIFAR100.

    Args:
        train: If ``True``, returns the training pipeline (with data
               augmentation). If ``False``, returns the validation pipeline
               (deterministic; do not modify).

    Returns:
        A ``torchvision.transforms.Compose`` object ready to be passed to a
        ``torchvision.datasets.CIFAR100`` dataset.
    """
    if train:
        return T.Compose([
            T.Resize(232),
            T.RandomResizedCrop(224, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ])
    else:
        return T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ])
