%%writefile bridge/datasets/tf_bridge.py
import torch
from torch.utils.data import Dataset
import numpy as np

class TFInitDataset(Dataset):
    """
    Dataset for initial state x0 and perturbation dosage vector d.
    
    - control_x: expression vectors from control cells (TF='D0' or 'D0_confluent')
    - treated_d: dosage vectors for perturbed cells (adata.obsm["Dosage vector"])
    """
    def __init__(self, control_x, treated_d):
        self.control_x = torch.tensor(control_x).float()   # (N_control, G)
        self.treated_d = torch.tensor(treated_d).float()   # (N_treated, n_TF)

    def __len__(self):
        return len(self.treated_d)

    def __getitem__(self, idx):
        # dosage vector for a specific treated cell
        d = self.treated_d[idx]

        # randomly pick a control cell as x0 (baseline)
        j = torch.randint(0, len(self.control_x), (1,)).item()
        x0 = self.control_x[j]

        return x0, d


class TFFinalDataset(Dataset):
    """
    Dataset for final expression state x1 and perturbation dosage vector d.
    
    - treated_x1: expression vectors at final timepoint for perturbed cells
    - treated_d: dosage vectors (same order as treated_x1)
    """
    def __init__(self, treated_x1, treated_d):
        self.x1 = torch.tensor(treated_x1).float()         # (N_treated, G)
        self.d  = torch.tensor(treated_d).float()          # (N_treated, n_TF)

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return self.x1[idx], self.d[idx]
