import torch
from torch.utils.data import Dataset

class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    Source: https://discuss.pytorch.org/t/apply-different-transform-data-augmentation-to-train-and-validation/63580/2
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return self.map(img), label

    def __len__(self):
        return len(self.dataset)

