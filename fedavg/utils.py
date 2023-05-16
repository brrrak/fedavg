import torch
from torch.utils import data
import torchvision


def get_split_sizes(data_size, n):
    """
    Determines how many samples each client will get
    :param data_size: Size of the complete dataset
    :param n: Number of clients
    :return: List[int]
    """
    assert data_size >= n, "Number of clients cannot exceed dataset size!"
    res = [data_size // n] * (n - 1)  # all except the last element
    last = data_size - sum(res)
    res.append(last)

    return res


def split_mnist_to_loaders(n):
    """
    Splits the MNIST dataset to n dataloaders for training.

    :param n: Number of clients
    :return: List[torch.utils.data.dataloader.DataLoader]
    """
    dataset = torchvision.datasets.MNIST(
        './data/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])
    )

    data_split = data.random_split(dataset,
                                   get_split_sizes(len(dataset), n),
                                   torch.Generator().manual_seed(42))

    train_loaders = [data.DataLoader(
        split,
        batch_size=64,
        shuffle=True,
    ) for split in data_split]

    assert len(train_loaders) == n, "The number of dataloaders and clients don't match!"

    return train_loaders


def get_mnist_test_loader():
    test_loader = data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=64, shuffle=True)

    return test_loader
