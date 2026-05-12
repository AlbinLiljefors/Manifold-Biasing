import torch
from torchvision import datasets, transforms


MEAN, STD = 0.1307, 0.3081  # MNIST stats, also used for SVHN/USPS transfer


def load_mnist(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return (torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False))


def load_svhn(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    train_ds = datasets.SVHN(data_dir, split='train', download=True, transform=transform)
    test_ds = datasets.SVHN(data_dir, split='test', download=True, transform=transform)
    return (torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False))


def load_mnist_subset(n, seed=42, batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    per_class = n // 10
    rng = torch.Generator().manual_seed(seed)
    targets = train_ds.targets
    if isinstance(targets, list):
        targets = torch.tensor(targets)
    class_indices = {c: (targets == c).nonzero(as_tuple=True)[0] for c in range(10)}

    selected = []
    for c in range(10):
        perm = torch.randperm(len(class_indices[c]), generator=rng)
        selected.append(class_indices[c][perm[:per_class]])
    indices = torch.cat(selected)

    subset = torch.utils.data.Subset(train_ds, indices.tolist())
    return (torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False))
