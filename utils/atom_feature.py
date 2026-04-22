import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 
import pandas as pd
from typing import Any, Optional, List


current_dir = os.path.dirname(__file__)
periodic_table_csv = os.path.join(current_dir, 'periodic_table_v2.csv')

class PeriodicTable():
    """Utility class to provide further element type information for crystal graph node embeddings."""
    
    def __init__(self, csv_path=periodic_table_csv,
                 normalize_atomic_mass=True,
                 normalize_atomic_radius=True,
                 normalize_electronegativity=True,
                 imputation_atomic_radius=209.46464646464648, # mean value
                 imputation_electronegativity=1.18): # educated guess (based on neighbour elements)
        self.data = pd.read_csv(csv_path)
        self.data['AtomicMass'] = self.data['AtomicMass'].fillna(imputation_atomic_radius)
        self.data['AtomicRadius'] = self.data['AtomicRadius'].fillna(imputation_atomic_radius)
        self.data['Electronegativity'] = self.data['Electronegativity'].fillna(imputation_electronegativity)

        if normalize_atomic_mass:
            self._normalize_column('AtomicMass')
        if normalize_atomic_radius:
            self._normalize_column('AtomicRadius')
        if normalize_electronegativity:
            self._normalize_column('Electronegativity')
    
    def _normalize_column(self, column):
        self.data[column] = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())

    def get_symbol(self, z: Optional[int] = None):
        if z is None:
            return self.data['Symbol'].to_list()
        else:
            return self.data.loc[z-1]['Symbol']
    
    def get_atomic_mass(self, z: Optional[int] = None): # 1
        if z is None:
            return self.data['AtomicMass'].to_list()
        else:
            return self.data.loc[z-1]['AtomicMass']

    def get_atomic_radius(self, z: Optional[int] = None): # 1
        if z is None:
            return self.data['AtomicRadius'].to_list()
        else:
            return self.data.loc[z-1]['AtomicRadius']
    
    def get_electronegativity(self, z: Optional[int] = None): # 1
        if z is None:
            return self.data['Electronegativity'].to_list()
        else:
            return self.data.loc[z-1]['Electronegativity']

    def atom_feature_map(self):
        self.feature = np.array([
            self.get_atomic_mass(),
            self.get_atomic_radius(),
            self.get_electronegativity()
        ]).T
        # 转换为Tensor并填充空元素（原子序数0~9）
        self.feature = torch.tensor(self.feature, dtype=torch.float32)
        self.feature = F.pad(self.feature, (0, 0, 1, 0), value=0)  # -> [n_elements+1, 27]
        return self.feature
        
class AtomFeatureEncoder(nn.Module):
    def __init__(self, input_dim,  out_dim):
        super(AtomFeatureEncoder, self).__init__()
        self.pt = PeriodicTable()
        self.feature_map = self.pt.atom_feature_map()
        self.proj = nn.Linear(input_dim, out_dim)  # input_dim是原始特征维度
        
    def forward(self, src):
        feature_map = self.feature_map.to(src.device)
        atom_fea = feature_map[src].to(torch.float32)
        atom_fea = self.proj(atom_fea)  # 映射到d_model维

        return atom_fea

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == "__main__":
    a = AtomFeatureEncoder()
    b = a.feature_map
    print("d") 