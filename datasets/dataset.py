from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import io
import torch

# from datasets.config import Years

class Dos_Dataset(Dataset):
    def __init__(self, data_dir="./data", split='train', dos_minmax = False, dos_zscore=False, scale_factor=1.0, apply_log=False, smear=0, choice=[],**kwargs) -> None:
        super().__init__()
        self.split = split
        self.smear = smear
        self.data_dir = data_dir+"/"+split+"/"
        
        self.elements  = self.get_elements()  #size (__len__, src_len)
        self.positions = self.get_positions() #size (__len__, src_len*3)
        len,_ = self.positions.shape
        if self.split == 'test_cif':
            self.tgtdos = tensor = torch.zeros((len, 64), dtype=torch.float32)
        else:
            self.tgtdos    = self.get_tgtdos()    #size (__len__, tge_len)
        
        self.dos_mean = torch.mean(self.tgtdos, dim=1, keepdim=True).float()
        self.dos_std = torch.std(self.tgtdos, dim=1, keepdim=True).float()
        self.dos_min = torch.min(self.tgtdos, dim=1, keepdim=True).values.float()
        self.dos_max = torch.max(self.tgtdos, dim=1, keepdim=True).values.float()

        if scale_factor!=1.0:
            self.tgtdos = self.tgtdos * scale_factor  # 放大数据

        if apply_log:
            self.tgtdos = torch.log(self.tgtdos + 1.0e-8)  # 添加一个小的常数以避免 log(0)

        if dos_zscore:
            self.tgtdos = (self.tgtdos - self.dos_mean)/ self.dos_std
        
        if dos_minmax:
            self.tgtdos = (self.tgtdos - self.dos_min) / (self.dos_max - self.dos_min)

        self.choice = choice
        if self.choice:
            cholist = torch.Tensor(self.choice).int()
            self.elements  = self.elements.index_select(dim=0, index=cholist)
            self.positions = self.positions.index_select(dim=0, index=cholist)
            self.tgtdos    = self.tgtdos.index_select(dim=0, index=cholist)
        

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        # type tensor; size [src_len, src_len*3, tgt_len]
        index = min(index, self.__len__())
        array_seq = [self.elements[index], self.positions[index].reshape(-1, 3), self.tgtdos[index], self.dos_mean[index], self.dos_std[index], self.dos_max[index], self.dos_min[index]] 
        return array_seq

    def get_elements(self):
        if self.smear==0:
            filename  = self.data_dir+"elements_%s.npy"%self.split
        else:
            filename  = self.data_dir+"elements_g%s_%s.npy"%(self.smear, self.split)
        elements  = np.load(filename)
        return torch.Tensor(elements).long()

    def get_positions(self):
        if self.smear==0:
            filename  = self.data_dir+"positions_%s.npy"%self.split
        else:
            filename  = self.data_dir+"positions_g%s_%s.npy"%(self.smear, self.split)
        positions  = np.load(filename)
        return torch.Tensor(positions)

    def get_tgtdos(self):
        if self.smear==0:
            filename  = self.data_dir+"tgtdos_%s.npy"%self.split
        else:
            filename  = self.data_dir+"tgtdos_g%s_%s.npy"%(self.smear, self.split)
        tgtdos  = np.load(filename)
        return torch.Tensor(tgtdos)

if __name__ == "__main__":
    test = Dos_Dataset(data_dir="./data/Mat", split="train")
    print(test.__getitem__(15))
