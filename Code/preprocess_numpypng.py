import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# === Path to FullSample directory ===
rootdir = r"D:\tmp\ChallengeData\ChallengeData\MultiCoil\Cine\TrainingSet"
fullsample_base = os.path.join(rootdir, "FullSample", "Center001", "UIH_30T_umr780")

# === Output directories ===
output_png_dir = "output_image_lax3ch"
output_npy_dir = "output_numpy_lax3ch"
os.makedirs(output_png_dir, exist_ok=True)
os.makedirs(output_npy_dir, exist_ok=True)

# === Loop from P001 to P061 ===
for i in range(1, 62):
    patient_id = f"P{i:03}"  # P001 to P061
    mat_file = os.path.join(fullsample_base, patient_id, "cine_lax_3ch.mat")

    if not os.path.exists(mat_file):
        print(f"Skipping (not found): {mat_file}")
        continue

    try:
        with h5py.File(mat_file, 'r') as hf: 
            print(f"Processing: {mat_file}")

            # Load k-space data
            kspace = hf['kspace']
            kspace_np = kspace['real'][()] + 1j * kspace['imag'][()]

            # IFFT to image domain
            image = np.fft.ifftshift(
                np.fft.ifft2(
                    np.fft.fftshift(kspace_np, axes=(-2, -1)),
                    axes=(-2, -1)
                ),
                axes=(-2, -1)
            )

            # Combine coils with RSS
            image_rss = np.sqrt(np.sum(np.abs(image) ** 2, axis=2))  # [frames, slices, height, width]

            # === Save full numpy array ===
            npy_path = os.path.join(output_npy_dir, f"{patient_id}_cine_lax_3ch.npy")
            np.save(npy_path, image_rss)

            # === Save one sample PNG (frame 0, slice 0) ===
            frame_idx, slice_idx = 0, 0
            output_img = np.abs(image_rss[frame_idx, slice_idx])
            plt.imshow(output_img, cmap='gray')
            plt.title(f"{patient_id} f{frame_idx} s{slice_idx}")
            plt.axis('off')

            png_path = os.path.join(output_png_dir, f"{patient_id}_f{frame_idx}_s{slice_idx}.png")
            plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    except Exception as e:
        print(f"Error processing {mat_file}: {e}")

print("Done. PNGs saved to:", output_png_dir)
print("Numpy arrays saved to:", output_npy_dir)
