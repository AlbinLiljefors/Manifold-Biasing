import torch
from torchvision import datasets, transforms


def load_mnist(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False))


def load_svhn(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.SVHN(data_dir, split='train', download=True, transform=transform)
    test = datasets.SVHN(data_dir, split='test', download=True, transform=transform)
    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False))
