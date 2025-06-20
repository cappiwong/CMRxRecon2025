import os, numpy as np
from glob import glob

def load_all_cmr_views(data_root):
    """
    Load all available CMR views (sax, lax3ch, lax4ch),
    flatten them into individual slices, and pair with targets.
    """
    all_pairs = []

    for view in ("sax", "lax3ch", "lax4ch"):
        input_dir = os.path.join(data_root, view, "input")
        target_dir = os.path.join(data_root, view, "target")
        if not os.path.isdir(input_dir) or not os.path.isdir(target_dir):
            print(f"> Missing view: {view}, skipping.")
            continue

        for in_file in sorted(glob(os.path.join(input_dir, "*.npy"))):
            base = os.path.basename(in_file)
            gt_file = os.path.join(target_dir, base.replace(".npy", "_gt.npy"))
            if not os.path.exists(gt_file):
                print(f"> No GT for {in_file}, skipping.")
                continue

            inp = np.load(in_file)
            tgt = np.load(gt_file)

            # Flatten based on ndim
            if inp.ndim == 4:
                for f in range(inp.shape[0]):
                    for s in range(inp.shape[1]):
                        all_pairs.append((inp[f, s], tgt[f, s], view))
            elif inp.ndim == 3:
                for f in range(inp.shape[0]):
                    all_pairs.append((inp[f], tgt[f], view))
            else:
                all_pairs.append((inp, tgt, view))

    print(f"✨ Loaded {len(all_pairs)} slices from {data_root}")
    return all_pairs
