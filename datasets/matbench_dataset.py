"""
Dataset adapter for MatBench tasks.

Wraps a list of pymatgen Structure objects (and optional scalar targets)
into the tensor format expected by the model.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pymatgen.core import Structure
from typing import List, Optional
import pandas as pd

from utils.structure_utils import structure_to_arrays, MAX_ATOMS


class MatbenchDataset(Dataset):
    """
    Each item is either:
        (elements, positions)                  — inference (no targets)
        (elements, positions, scalar_target)   — training / evaluation
    """

    def __init__(
        self,
        structures: List[Structure],
        targets: Optional[pd.Series] = None,
        max_atoms: int = MAX_ATOMS,
    ):
        self.structures = list(structures)
        self.targets    = targets.values.astype(np.float32) if targets is not None else None
        self.max_atoms  = max_atoms

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        elements, positions = structure_to_arrays(self.structures[idx], self.max_atoms)
        elem_t = torch.from_numpy(elements)
        pos_t  = torch.from_numpy(positions)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return elem_t, pos_t, target
        return elem_t, pos_t
