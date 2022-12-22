import torchvision
from timm.data import transforms
from torch.utils.data import DataLoader

def get_test_dataloader(batch_size=16, num_workers=4, shuffle=True):
    """ return training dataloader
    Args:
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader