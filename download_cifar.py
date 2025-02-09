import os
from torchvision.datasets import CIFAR10

if __name__ == "__main__":
    CIFAR10(root="data/train", train=True, download=True)
    CIFAR10(root="data/test", train=False, download=True)