import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# ------------------------------
# SSIM implementation (no external deps)
# ------------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        math.exp(-(x - window_size//2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    one_d = gaussian(window_size, 1.5).unsqueeze(1)
    two_d = one_d.mm(one_d.t()).unsqueeze(0).unsqueeze(0)
    return two_d.expand(channel, 1, window_size, window_size).contiguous()

def ssim(img1, img2, window_size=11, size_average=True, C1=0.01**2, C2=0.03**2):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return ssim_map.mean() if size_average else ssim_map.mean([1,2,3])

# ------------------------------
# Utility: Center crop
# ------------------------------
def center_crop(enc_feat, dec_feat):
    _, _, h, w = dec_feat.shape
    enc_h, enc_w = enc_feat.shape[2], enc_feat.shape[3]
    start_h = (enc_h - h) // 2
    start_w = (enc_w - w) // 2
    return enc_feat[:, :, start_h:start_h+h, start_w:start_w+w]

# ------------------------------
# Dataset: separate input & target files
# ------------------------------
class MRIDataset(Dataset):
    def __init__(self, input_npy, target_npy):
        # Load input and target volumes: input shape [F, S, H, W]
        self.input_vol = np.load(input_npy)
        tgt = np.load(target_npy)
        # If target is a single 2D slice, tile to match input volume
        if tgt.ndim == 2:
            h, w = tgt.shape
            # reshape to (1,1,h,w)
            tgt = np.expand_dims(np.expand_dims(tgt, 0), 0)
            frames, slices = self.input_vol.shape[:2]
            tgt = np.tile(tgt, (frames, slices, 1, 1))
        self.target_vol = tgt
        assert self.input_vol.shape == self.target_vol.shape, "Input and target shapes must match after tiling"
        self.frames, self.slices = self.input_vol.shape[:2]

    def __len__(self):
        return self.frames * self.slices

    def __getitem__(self, idx):
        f, s = divmod(idx, self.slices)
        inp = self.input_vol[f, s]
        tgt = self.target_vol[f, s]
        # Normalize each to [0,1]
        inp = (inp - inp.min()) / (inp.max() - inp.min())
        tgt = (tgt - tgt.min()) / (tgt.max() - tgt.min())
        inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
        tgt_t = torch.tensor(tgt, dtype=torch.float32).unsqueeze(0)
        return inp_t, tgt_t

# ------------------------------
# Model: TransUNet with residual
# ------------------------------
class TransUNet(nn.Module):
    def __init__(self, vit_patch_size=16, vit_dim=256, vit_depth=2, vit_heads=4):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # ViT bottleneck
        self.vit_patch_size = vit_patch_size
        self.patch_embed = nn.Conv2d(256, vit_dim, vit_patch_size, vit_patch_size)
        layer = nn.TransformerEncoderLayer(d_model=vit_dim, nhead=vit_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=vit_depth)
        self.vit_unpatch = nn.ConvTranspose2d(vit_dim, 256, vit_patch_size, vit_patch_size)
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = nn.Conv2d(256, 128, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = F.relu(self.enc1(x)); p1 = self.pool(e1)
        e2 = F.relu(self.enc2(p1)); p2 = self.pool(e2)
        e3 = F.relu(self.enc3(p2))
        B,C,H,W = e3.shape
        p = self.patch_embed(e3).flatten(2).transpose(1,2)
        p = self.transformer(p)
        p = p.transpose(1,2).reshape(B, -1, H//self.vit_patch_size, W//self.vit_patch_size)
        e3 = self.vit_unpatch(p)
        d2 = F.relu(self.dec2(torch.cat([self.up2(e3), center_crop(e2, self.up2(e3))],1)))
        d1 = F.relu(self.dec1(torch.cat([self.up1(d2), center_crop(e1, self.up1(d2))],1)))
        res = self.final_conv(d1)
        if res.shape[2:] != x.shape[2:]:
            x = center_crop(x, res)
        return x + res

# ------------------------------
# Loss & metric
# ------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha, self.mse = alpha, nn.MSELoss()

    def forward(self, out, tgt):
        mse_val = self.mse(out, tgt)
        ssim_val = ssim(out, tgt)
        loss = self.alpha*mse_val + (1-self.alpha)*(1-ssim_val)
        return loss, mse_val, ssim_val

def compute_psnr(mse_val):
    return 10*torch.log10(1.0/mse_val)

# ------------------------------
# Training & visualization
# ------------------------------
if __name__ == '__main__':
    # Paths: input vs. ground-truth
    input_path  = r"C:\Users\LAPTOP ASUS LAMA C\kepi data E\FILE DATA E\ZHUANTI\CMRxRecon2025\Code\output_numpyflip_lax3ch\P001_cine_lax_3ch.npy"
    target_path = r"C:\Users\LAPTOP ASUS LAMA C\kepi data E\FILE DATA E\ZHUANTI\CMRxRecon2025\Code\cine_lax_3ch_slice1_time1_mag.npy"
    bs, lr, epochs, alpha = 2, 1e-3, 20, 0.5

    ds = MRIDataset(input_path, target_path)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransUNet().to(device)
    criterion = CombinedLoss(alpha)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for ep in range(1, epochs+1):
        model.train(); tot_L=tot_M=tot_S=0
        for inp, tgt in dl:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(inp)
            # Crop target to match output
            _,_,ho,wo = out.shape; _,_,ht,wt = tgt.shape
            ch, cw = (ht-ho)//2, (wt-wo)//2
            tgt_crop = tgt[:,:,ch:ch+ho,cw:cw+wo]
            L,M,S = criterion(out, tgt_crop)
            L.backward(); optimizer.step()
            tot_L+=L.item(); tot_M+=M.item(); tot_S+=S.item()
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        avg_L, avg_M, avg_S = tot_L/len(dl), tot_M/len(dl), tot_S/len(dl)
        print(f"Epoch {ep}/{epochs} | LR {lr_now:.5f} | Loss {avg_L:.4f} | MSE {avg_M:.4f} | SSIM {avg_S:.4f} | PSNR {compute_psnr(torch.tensor(avg_M)):.2f} dB")

    # Visualization
    model.eval()
    with torch.no_grad():
        inp, tgt = next(iter(dl)); inp, tgt = inp.to(device), tgt.to(device)
        out = model(inp)
        _,_,ho,wo = out.shape; _,_,ht,wt = tgt.shape
        ch, cw = (ht-ho)//2, (wt-wo)//2
        tgt_crop = tgt[:,:,ch:ch+ho,cw:cw+wo]
        imgs = [(inp[0,0].cpu().numpy(),'Input'),(tgt_crop[0,0].cpu().numpy(),'Target'),(out[0,0].cpu().numpy(),'Output')]
        plt.figure(figsize=(15,5))
        for i,(img,ttl) in enumerate(imgs,1):
            plt.subplot(1,3,i); plt.title(ttl); plt.imshow(img,cmap='gray'); plt.axis('off')
        plt.show()
