"""
Utilities for converting pymatgen.Structure objects to model input tensors.

Elements encoding (1-D array, length PAD_LEN = max_atoms + 2):
    idx 0      : token 126  — placeholder row for lattice lengths
    idx 1      : token 127  — placeholder row for lattice angles
    idx 2..n+1 : atomic numbers of each atom (1-indexed)
    remaining  : 0  (padding, creates attention mask)

Positions encoding (2-D array, shape [PAD_LEN, 3]):
    row 0      : (a, b, c) lattice lengths in Angstrom
    row 1      : (alpha, beta, gamma) angles in degrees
    row 2..n+1 : fractional coordinates of each atom (absolute value)
    remaining  : zeros
"""

import numpy as np
import torch
from pymatgen.core import Structure
from typing import List, Tuple

MAX_ATOMS = 126          # maximum atoms per structure (structures are truncated if larger)
PAD_LEN   = MAX_ATOMS + 2  # total sequence length including the 2 lattice rows


def structure_to_arrays(
    structure: Structure,
    max_atoms: int = MAX_ATOMS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a single pymatgen Structure to (elements, positions) numpy arrays.

    Returns:
        elements  : int64 array [PAD_LEN]
        positions : float32 array [PAD_LEN, 3]
    """
    pad_len = max_atoms + 2
    n = min(len(structure), max_atoms)

    elements  = np.zeros(pad_len, dtype=np.int64)
    positions = np.zeros((pad_len, 3), dtype=np.float32)

    # Lattice rows
    elements[0] = 126
    elements[1] = 127
    positions[0] = [structure.lattice.a, structure.lattice.b, structure.lattice.c]
    positions[1] = structure.lattice.angles   # alpha, beta, gamma (degrees)

    # Atom rows
    elements[2:2 + n]  = structure.atomic_numbers[:n]
    positions[2:2 + n] = np.abs(structure.frac_coords[:n])

    return elements, positions


def structures_to_tensors(
    structures: List[Structure],
    max_atoms: int = MAX_ATOMS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch-convert a list of pymatgen Structure objects to tensors.

    Returns:
        elements  : LongTensor  [N, PAD_LEN]
        positions : FloatTensor [N, PAD_LEN, 3]
    """
    pad_len = max_atoms + 2
    n = len(structures)
    elem_arr = np.zeros((n, pad_len), dtype=np.int64)
    pos_arr  = np.zeros((n, pad_len, 3), dtype=np.float32)

    for i, s in enumerate(structures):
        elem_arr[i], pos_arr[i] = structure_to_arrays(s, max_atoms)

    return torch.from_numpy(elem_arr), torch.from_numpy(pos_arr)


def find_last_peak(dos: np.ndarray, freq_grid: np.ndarray) -> float:
    """
    Extract the last local maximum from a 1-D DOS curve.

    Falls back to the global maximum if no local peak is found.

    Args:
        dos       : 1-D array of DOS values (length N)
        freq_grid : 1-D array of corresponding frequencies (length N)

    Returns:
        Frequency of the last peak in the same units as freq_grid.
    """
    # Simple local-maximum detection (ignoring endpoints)
    peaks = [
        i for i in range(1, len(dos) - 1)
        if dos[i] > dos[i - 1] and dos[i] > dos[i + 1]
    ]
    if peaks:
        return float(freq_grid[peaks[-1]])
    # Fallback: global maximum
    return float(freq_grid[np.argmax(dos)])
