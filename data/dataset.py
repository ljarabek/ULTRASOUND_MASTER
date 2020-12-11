from torch.utils.data import Dataset
import os
import pickle
import numpy as np

class FullSizeVideoDataset(Dataset):
    def __init__(self, folder):
        super(FullSizeVideoDataset).__init__()
        self.flist = [os.path.join(folder, x) for x in os.listdir(folder)]
        self.flist.sort(key=self.key)

    def key(self, s:str):
        i = s[s.rindex("_")+1:]
        return int(i)

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, i: int):
        with open(self.flist[i], "rb") as f:
            data = pickle.load(f)
        data = np.expand_dims(data,0)

        return data
