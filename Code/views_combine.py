import os, numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset

class CombinedCMRDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []  # Each sample is (input_slice, target_slice, view)
        for view in ("sax", "lax3ch", "lax4ch"):
            input_dir = os.path.join(data_root, view, "input")
            target_dir = os.path.join(data_root, view, "target")
            if not os.path.exists(input_dir) or not os.path.exists(target_dir):
                continue
            for in_file in glob(os.path.join(input_dir, "*.npy")):
                base = os.path.basename(in_file)
                gt_file = os.path.join(target_dir, base.replace(".npy", "_gt.npy"))
                if not os.path.exists(gt_file):
                    continue
                inp = np.load(in_file)
                tgt = np.load(gt_file)
                # Flatten
                if inp.ndim == 4:
                    for f in range(inp.shape[0]):
                        for s in range(inp.shape[1]):
                            self.samples.append((inp[f, s], tgt[f, s], view))
                elif inp.ndim == 3:
                    for f in range(inp.shape[0]):
                        self.samples.append((inp[f], tgt[f], view))
                else:
                    self.samples.append((inp, tgt, view))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt, _ = self.samples[idx]
        # Normalize
        inp = (inp - inp.min()) / (inp.max() - inp.min() + 1e-6)
        tgt = (tgt - tgt.min()) / (tgt.max() - tgt.min() + 1e-6)
        return torch.tensor(inp).unsqueeze(0), torch.tensor(tgt).unsqueeze(0)
