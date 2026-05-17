import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY=True

SAMPLES_PER_CLASS = 80 

def _get_fixed_indices(targets, n_per_class: int) -> list[int]:
    # take the first n_per_class samples for each class (sorted by original index)
    targets = np.array(targets)
    n_classes = int(targets.max()) + 1
    out = []
    for c in range(n_classes):
        idx = np.where(targets == c)[0]
        out.extend(idx[:n_per_class].tolist())
    return out


def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,

):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY, # True
        download=True,
        transform=get_transforms(train=True),
    )

    idx = _get_fixed_indices(train_dataset.targets, SAMPLES_PER_CLASS)
    train_dataset = Subset(train_dataset, idx)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train
    )

    return train_dataset, train_loader