from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, mparams, condEmb, stepEmb):
        self.mparams = mparams
        self.condEmb = condEmb
        self.stepEmb = stepEmb

    def __getitem__(self, index):
        return self.mparams[index], self.condEmb[index], self.stepEmb[index]

    def __len__(self):
        return len(self.mparams)
