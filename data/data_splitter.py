from torch.utils.data import Dataset


class DatasetSplit(Dataset):

    def __init__(self, dataset, index, length):
        super(DatasetSplit, self).__init__()

        self.dataset = dataset
        self.index = index
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        index = (self.index + idx) % len(self.dataset)
        return self.dataset[index]
