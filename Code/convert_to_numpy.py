# convert_one_image_to_numpy.py

import os
from PIL import Image
import numpy as np

def convert_image_to_npy(img_path, out_path=None, normalize=True):
    """
    Loads a single image, converts to grayscale, optionally normalizes to [0,1],
    and saves as a .npy file.

    Args:
        img_path (str): Path to input image (jpg/png).
        out_path (str, optional): Path to output .npy file. 
            If None, will use same base name as img_path.
        normalize (bool): Whether to scale pixel values to [0,1].
    """
    # 1) Load & convert to grayscale
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # 2) Normalize to [0,1]
    if normalize:
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = arr * 0.0  # constant image case

    # 3) Determine output path
    if out_path is None:
        base, _ = os.path.splitext(img_path)
        out_path = base + ".npy"

    # 4) Save
    np.save(out_path, arr)
    print(f"Saved NumPy array to: {out_path}")
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}, min={arr.min():.3f}, max={arr.max():.3f}")

if __name__ == "__main__":
    # ─────────── CONFIGURE HERE ───────────
    img_path = r"C:\Users\LAPTOP ASUS LAMA C\kepi data E\FILE DATA E\ZHUANTI\CMRxRecon2025\Code\cine_lax_3ch_slice1_time1_mag.png"
    out_path = r"C:\Users\LAPTOP ASUS LAMA C\kepi data E\FILE DATA E\ZHUANTI\CMRxRecon2025\Code"
    # If you leave out_path commented, it will save alongside the image.
    # ──────────────────────────────────────

    # sanity-check input
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Input image not found: {img_path}")

    convert_image_to_npy(img_path,
                         out_path=None,   # or set out_path explicitly
                         normalize=True)
