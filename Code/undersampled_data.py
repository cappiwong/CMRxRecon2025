import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import h5py

# --------------------------------
# 1) Task R1 masks per view
# --------------------------------
val_masks = {
    'cine_lax_3ch': {'Uniform8','Uniform16','Uniform24','ktRadial16','ktGaussian8','ktGaussian16'},
    'cine_lax_4ch': {'ktRadial8','ktRadial16','ktRadial24','ktGaussian8','ktGaussian16','ktGaussian24'},
    'cine_sax':     {'Uniform16','Uniform24','ktRadial8','ktRadial24','ktGaussian8','ktGaussian16','ktGaussian24'}
}

# --------------------------------
# 2) Dataset: stream central slice+frame
# --------------------------------
class CMRxReconDataset(Dataset):
    def __init__(self, full_dir, mask_dir, gt_dir, cases):
        self.entries = []
        for case in cases:
            mdir = os.path.join(mask_dir, case)
            if not os.path.isdir(mdir): continue
            for fname in sorted(os.listdir(mdir)):
                if not fname.endswith('.mat'): continue
                view, rest = fname.split('_mask_')
                if os.path.splitext(rest)[0] not in val_masks.get(view, ()):
                    continue
                full_path = os.path.join(full_dir, case, f"{view}.mat")
                if not os.path.isfile(full_path):
                    continue
                mask_path = os.path.join(mdir, fname)
                self.entries.append((case, view, full_path, mask_path))
        if not self.entries:
            raise RuntimeError("No valid entries found in dataset.")
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        case, view, full_path, mask_path = self.entries[idx]

        # 1) load mask → (1,Hm,Wm,Tm)
        raw_m = self._load_mat(mask_path)
        mask  = self._reshape_mask(raw_m)
        _, Hm, Wm, Tm = mask.shape

        # 2) load only central slice+frame of full k-space
        arr = self._load_kspace_middle(full_path)

        # 3) to complex64
        if arr.dtype.fields is not None:
            arr = arr['real'] + 1j*arr['imag']
        arr = np.array(arr, dtype=np.complex64)

        # 4) shape to (C,Hk,Wk,1)
        if arr.ndim == 2:
            k_full = arr[np.newaxis,...][...,np.newaxis]
        else:  # (H,W,coils)
            k_full = arr.transpose(2,0,1)[...,np.newaxis]

        C, Hk, Wk, Tk = k_full.shape  # Tk==1

        # 5) unify mask to T=1
        mask = mask[...,0:1]

        # 6) pad/crop mask spatially to (Hk,Wk)
        ph = Hk-Hm; pw = Wk-Wm
        if ph>0 or pw>0:
            ph1, ph2 = max(ph//2,0), max(ph-ph//2,0)
            pw1, pw2 = max(pw//2,0), max(pw-pw//2,0)
            mask = np.pad(mask, ((0,0),(ph1,ph2),(pw1,pw2),(0,0)), mode='constant')
        if ph<0 or pw<0:
            ch = max(-ph//2,0); cw = max(-pw//2,0)
            mask = mask[:, ch:ch+Hk, cw:cw+Wk, :]

        # 7) undersample → IFFT → coil‐combine → zero‐filled image
        k_us   = k_full * mask
        imgs   = np.stack([self._ifft2c(k_us[c]) for c in range(C)], axis=0)
        img_zf = self._coil_combine(imgs)[...,0]  # (Hk,Wk)

        # 8) input & gt
        inp = torch.from_numpy(np.abs(img_zf)).unsqueeze(0).float()
        try:
            raw_gt = self._load_mat(os.path.join(self.gt_dir, case, f"{view}.mat"))
            img_gt = self._reshape_image(raw_gt)
            mid   = img_gt.shape[-1]//2
            tgt   = torch.from_numpy(np.abs(img_gt[0,...,mid])).unsqueeze(0).float()
        except:
            tgt = torch.zeros_like(inp)

        return inp, tgt

    @staticmethod
    def _load_kspace_middle(path):
        """Load only the middle slice & frame from a .mat."""
        try:
            data = sio.loadmat(path)
            for v in data.values():
                if not isinstance(v, str):
                    arr = v
                    break
        except NotImplementedError:
            f = h5py.File(path,'r')
            for v in f.values():
                if not v.name.startswith('_'):
                    ds = v
                    break
            s = ds.shape
            if len(s)==5:
                arr = ds[:,:,:,s[3]//2,s[4]//2]
            elif len(s)==4:
                arr = ds[:,:,:,s[3]//2]
            elif len(s)==3:
                arr = ds[:,:,s[2]//2]
            else:
                raise ValueError(f"Unexpected shape {s}")
            arr = np.array(arr)
            f.close()
        return arr

    @staticmethod
    def _load_mat(path):
        try:
            data = sio.loadmat(path)
            for v in data.values():
                if not isinstance(v, str):
                    return v
        except NotImplementedError:
            with h5py.File(path,'r') as f:
                for v in f.values():
                    if not v.name.startswith('_'):
                        return np.array(v)
        raise RuntimeError(f"Couldn't read {path}")

    @staticmethod
    def _reshape_mask(m):
        m = np.array(m)
        if m.ndim==3:
            m = m.transpose(1,2,0)
            return m[np.newaxis,...]
        if m.ndim==2:
            return m[np.newaxis,...,np.newaxis]
        if m.ndim==4:
            return m
        raise ValueError(f"Unexpected mask shape {m.shape}")

    @staticmethod
    def _ifft2c(x):
        return np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(x,axes=(-2,-1)),axes=(-2,-1)),
            axes=(-2,-1)
        )

    @staticmethod
    def _coil_combine(x):
        return np.sqrt(np.sum(np.abs(x)**2, axis=0))

    @staticmethod
    def _reshape_image(x):
        arr = np.squeeze(np.array(x))
        if arr.ndim==5:
            arr = arr[:,:,:,arr.shape[3]//2,:]
        if arr.ndim==4:
            arr = arr[:,:,0,:]
        if arr.ndim in (2,3):
            return arr[np.newaxis,...]
        return arr[np.newaxis,...]

# --------------------------------
# 3) collate_fn, UNet & main()
# --------------------------------
def collate_fn(batch):
    inps, tgts = zip(*batch)
    H = max(x.shape[1] for x in inps)
    W = max(x.shape[2] for x in inps)
    pad_in, pad_tgt = [], []
    for inp, tgt in batch:
        ph, pw = H-inp.shape[1], W-inp.shape[2]
        t, b   = ph//2, ph-ph//2
        l, r   = pw//2, pw-pw//2
        pad_in.append(F.pad(inp, (l,r,t,b)))
        pad_tgt.append(F.pad(tgt, (l,r,t,b)))
    return torch.stack(pad_in), torch.stack(pad_tgt) 


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )
    def forward(self,x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self,in_ch=1,out_ch=1,feats=(64,128,256,512)):
        super().__init__()

        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)
        c = in_ch
        for f in feats:
            self.downs.append(ConvBlock(c,f)); c=f
        self.bottle = ConvBlock(feats[-1], feats[-1]*2); c=feats[-1]*2
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(c,f,2,2))
            self.ups.append(ConvBlock(c,f))
            c = f
        self.final = nn.Conv2d(feats[0], out_ch, 1)

    def forward(self,x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottle(x)
        for i in range(0,len(self.ups),2):
            x = self.ups[i](x)
            s = skips[-1 - i//2]
            if x.shape[2:] != s.shape[2:]:
                x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([s, x], dim=1)
            x = self.ups[i+1](x)
        return self.final(x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TRAIN_FULL = r"D:\tmp\ChallengeData\ChallengeData\MultiCoil\Cine\TrainingSet\Fullsample\Center001\UIH_30T_umr780"
    TRAIN_MASK = r"D:\tmp\ChallengeData\ChallengeData\MultiCoil\Cine\TrainingSet\Mask_TaskAll\Center001\UIH_30T_umr780"
    TRAIN_GT   = r"D:\tmp\ChallengeData\ChallengeData\MultiCoil\Cine\TrainingSet\ImageShow\Center001\UIH_30T_umr780"
    VAL_US     = r"D:\valtmp\ChallengeData_ValSet-TaskR1\TaskR1\MultiCoil\Cine\ValidationSet\UnderSample_TaskR1\Center001\UIH_30T_umr780"

    val_cases   = ['P007','P012','P016','P022','P023','P025','P038','P039','P045','P048','P052','P059']
    train_cases = [f"P{str(i).zfill(3)}" for i in range(1,62)
                   if f"P{str(i).zfill(3)}" not in val_cases and f"P{str(i).zfill(3)}"!='P047']

    # ————— training —————
    ds      = CMRxReconDataset(TRAIN_FULL, TRAIN_MASK, TRAIN_GT, train_cases)
    loader  = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    model   = UNet().to(device)
    optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn  = nn.L1Loss()

    for ep in range(1,21):
        model.train()
        total = 0.0
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss = loss_fn(out, tgt)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {ep:02d} loss={total/len(loader):.4f}")

    # ————— inference (all slices & frames) —————
    os.makedirs('submissions', exist_ok=True)
    model.eval()
    with torch.no_grad():
        for case in val_cases:
            in_dir  = os.path.join(VAL_US, case)
            out_dir = os.path.join('submissions', case)
            os.makedirs(out_dir, exist_ok=True)

            for fn in sorted(os.listdir(in_dir)):
                if not fn.endswith('.mat'):
                    continue

                # 1) load full undersampled k-space
                raw = CMRxReconDataset._load_mat(os.path.join(in_dir, fn))
                if raw.dtype.fields is not None:
                    raw = raw['real'] + 1j*raw['imag']
                k = np.squeeze(raw)

                # 2) reshape to (C, H, W, S, T)
                if k.ndim == 5:
                    # (H, W, coils, slices, frames)
                    k = k.transpose(2, 0, 1, 3, 4)
                elif k.ndim == 4:
                    # (coils, H, W, T) → add slice dim=1
                    k = k.transpose(0,1,2,3)[..., np.newaxis, :]
                elif k.ndim == 3:
                    # (H, W, T) → coil=1, slice=1
                    k = k[np.newaxis, ..., np.newaxis, :]

                C, H, W, S, T = k.shape

                # 3) zero-fill IFFT + coil combine → img_zf (H, W, S, T)
                imgs   = np.stack([
                    np.fft.ifftshift(
                        np.fft.ifft2(np.fft.fftshift(k[c], axes=(-2,-1)), axes=(-2,-1)),
                        axes=(-2,-1)
                    ) for c in range(C)
                ], axis=0)
                img_zf = np.sqrt(np.sum(np.abs(imgs)**2, axis=0))

                # 4) reconstruct every slice & frame
                recon = np.zeros((H, W, S, T), dtype=np.float32)
                for s in range(S):
                    for t in range(T):
                        inp = torch.from_numpy(np.abs(img_zf[..., s, t])) \
                                   .unsqueeze(0).unsqueeze(0) \
                                   .float().to(device)
                        out = model(inp).squeeze().cpu().numpy()
                        recon[..., s, t] = out

                # 5) save with correct variable name & type
                sio.savemat(
                    os.path.join(out_dir, fn),
                    {'img4ranking': recon.astype(np.single)}
                )

    print("Done.")

if __name__ == '__main__':
    main()
