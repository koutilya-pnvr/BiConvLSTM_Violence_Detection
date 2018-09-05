from torch.utils.data import Dataset


class DatasetTransform(Dataset):

    def __init__(self, dataset, transform=lambda x: x):
        super(DatasetTransform, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), label
