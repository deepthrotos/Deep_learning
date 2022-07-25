import torchvision.transforms as T
from torchvision.datasets import CIFAR10


def get_augmentations(train: bool = True) -> T.Compose:
    means = [0.49139968, 0.48215841, 0.44653091]
    stds = [0.24703223, 0.24348513, 0.26158784]

    if train:
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAdjustSharpness(sharpness_factor=2),
                T.ToTensor(),
                T.Normalize(mean=means, std=stds)
            ]
        )

    else:
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=means, std=stds)
            ]
        )

    return transform
