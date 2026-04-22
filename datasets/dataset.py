from torch.utils.data import Dataset
import numpy as np
import torch


class Dos_Dataset(Dataset):
    def __init__(self, data_dir="./data", split='train', dos_minmax=False, dos_zscore=False,
                 scale_factor=1.0, apply_log=False, smear=0, choice=[], **kwargs) -> None:
        super().__init__()
        self.split = split
        self.smear = smear
        self.data_dir = data_dir + "/" + split + "/"

        self.elements  = self.get_elements()
        self.positions = self.get_positions()
        n = self.positions.shape[0]

        if self.split == 'test_cif':
            self.tgtdos = torch.zeros((n, 64), dtype=torch.float32)
        else:
            self.tgtdos = self.get_tgtdos()

        self.dos_mean = torch.mean(self.tgtdos, dim=1, keepdim=True).float()
        self.dos_std  = torch.std(self.tgtdos,  dim=1, keepdim=True).float()
        self.dos_min  = torch.min(self.tgtdos,  dim=1, keepdim=True).values.float()
        self.dos_max  = torch.max(self.tgtdos,  dim=1, keepdim=True).values.float()

        if scale_factor != 1.0:
            self.tgtdos = self.tgtdos * scale_factor

        if apply_log:
            self.tgtdos = torch.log(self.tgtdos + 1.0e-8)

        if dos_zscore:
            self.tgtdos = (self.tgtdos - self.dos_mean) / self.dos_std

        if dos_minmax:
            self.tgtdos = (self.tgtdos - self.dos_min) / (self.dos_max - self.dos_min)

        if choice:
            cholist = torch.Tensor(choice).int()
            self.elements  = self.elements.index_select(dim=0, index=cholist)
            self.positions = self.positions.index_select(dim=0, index=cholist)
            self.tgtdos    = self.tgtdos.index_select(dim=0, index=cholist)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        index = min(index, self.__len__() - 1)
        return [
            self.elements[index],
            self.positions[index].reshape(-1, 3),
            self.tgtdos[index],
            self.dos_mean[index],
            self.dos_std[index],
            self.dos_max[index],
            self.dos_min[index],
        ]

    def get_elements(self):
        if self.smear == 0:
            filename = self.data_dir + "elements_%s.npy" % self.split
        else:
            filename = self.data_dir + "elements_g%s_%s.npy" % (self.smear, self.split)
        return torch.Tensor(np.load(filename)).long()

    def get_positions(self):
        if self.smear == 0:
            filename = self.data_dir + "positions_%s.npy" % self.split
        else:
            filename = self.data_dir + "positions_g%s_%s.npy" % (self.smear, self.split)
        return torch.Tensor(np.load(filename))

    def get_tgtdos(self):
        if self.smear == 0:
            filename = self.data_dir + "tgtdos_%s.npy" % self.split
        else:
            filename = self.data_dir + "tgtdos_g%s_%s.npy" % (self.smear, self.split)
        return torch.Tensor(np.load(filename))


if __name__ == "__main__":
    test = Dos_Dataset(data_dir="./data/Mat", split="train")
    print(test[15])
