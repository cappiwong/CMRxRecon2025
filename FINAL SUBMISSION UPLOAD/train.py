import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import h5py
from skimage.metrics import structural_similarity as ssim
from einops import rearrange

# --------------------------------
# 1) Task R1 masks per view
# --------------------------------
val_masks = {
    'cine_lax_3ch': {'Uniform8','Uniform16','Uniform24','ktRadial16','ktGaussian8','ktGaussian16'},
    'cine_lax_4ch': {'ktRadial8','ktRadial16','ktRadial24','ktGaussian8','ktGaussian16','ktGaussian24'},
    'cine_sax': {'Uniform16','Uniform24','ktRadial8','ktRadial24','ktGaussian8','ktGaussian16','ktGaussian24'}
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
        mask = self._reshape_mask(raw_m)
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
        else: # (H,W,coils)
            k_full = arr.transpose(2,0,1)[...,np.newaxis]

        C, Hk, Wk, Tk = k_full.shape # Tk==1

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

        # 7) undersample → IFFT → coil­combine → zero­filled image
        k_us = k_full * mask
        imgs = np.stack([self._ifft2c(k_us[c]) for c in range(C)], axis=0)
        img_zf = self._coil_combine(imgs)[...,0] # (Hk,Wk)

        # 8) input & gt
        inp = torch.from_numpy(np.abs(img_zf)).unsqueeze(0).float()
        try:
            raw_gt = self._load_mat(os.path.join(self.gt_dir, case, f"{view}.mat"))
            img_gt = self._reshape_image(raw_gt)
            mid = img_gt.shape[-1]//2
            tgt = torch.from_numpy(np.abs(img_gt[0,...,mid])).unsqueeze(0).float()
        except:
            tgt = torch.zeros_like(inp)

        return inp, tgt, case, view


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
# 3) collate_fn
# --------------------------------
def collate_fn(batch):
    inps, tgts, cases, views = zip(*batch)
    H = max(x.shape[1] for x in inps)
    W = max(x.shape[2] for x in inps)
    pad_in, pad_tgt = [], []
    for inp, tgt in zip(inps, tgts):
        ph, pw = H-inp.shape[1], W-inp.shape[2]
        t, b = ph//2, ph-ph//2
        l, r = pw//2, pw-pw//2
        pad_in.append(F.pad(inp, (l,r,t,b)))
        pad_tgt.append(F.pad(tgt, (l,r,t,b)))
    return torch.stack(pad_in), torch.stack(pad_tgt), cases, views


# ================================
# 4) Vision Transformer Components
# ================================

class PatchEmbedding(nn.Module):
    """Convert 2D feature maps to patch tokens"""
    def __init__(self, in_channels, embed_dim, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x) # (B, embed_dim, H//patch_size, W//patch_size)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, (H, W)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Pre-norm architecture (more stable)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ================================
# 5) U-Net Components
# ================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(16, out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(16, out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# 6) Baseline U-Net (Original)
# ================================

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feats=(64,128,256,512)):
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

    def forward(self, x):
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

class TemporalConsistencyLayer(nn.Module):
    """Enforces temporal smoothness across time frames"""
    def __init__(self, channels):
        super().__init__()
        self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        smoothed = self.temporal_conv(x_flat)
        gate = self.gate(x_flat)
        out = x_flat * gate + smoothed * (1 - gate)
        return out.view(B, C, H, W)


class FrequencyAttention(nn.Module):
    """Attention in frequency domain to reduce aliasing"""
    def __init__(self, channels):
        super().__init__()
        self.freq_weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='ortho')
        h_freq, w_freq = x_freq.shape[2], x_freq.shape[3]
        freq_mask = self.create_freq_mask(H, W, h_freq, w_freq, x.device)
        x_freq = x_freq * freq_mask * self.freq_weight
        x_filtered = torch.fft.irfft2(x_freq, s=(H, W), norm='ortho')
        gate = self.spatial_gate(x)
        out = x * gate + x_filtered * (1 - gate)
        return out
    
    def create_freq_mask(self, H, W, h_freq, w_freq, device):
        y = torch.linspace(-1, 1, h_freq, device=device)
        x = torch.linspace(-1, 1, w_freq, device=device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        freq_dist = torch.sqrt(X**2 + Y**2)
        mask = torch.exp(-freq_dist**2 / 0.5)
        return mask.view(1, 1, h_freq, w_freq)


class EntropyEnhancementLayer(nn.Module):
    """Enhance information content and detail preservation"""
    def __init__(self, channels):
        super().__init__()
        self.detail_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        detail = self.detail_branch(x)
        att = self.attention(x)
        out = x + detail * att
        return out


class EnhancedConvBlock(nn.Module):
    """Conv block with frequency attention and entropy enhancement"""
    def __init__(self, in_ch, out_ch, use_enhancements=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(16, out_ch)
        self.relu1 = nn.ReLU(True)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(16, out_ch)
        self.relu2 = nn.ReLU(True)
        
        self.use_enhancements = use_enhancements
        if use_enhancements:
            self.freq_att = FrequencyAttention(out_ch)
            self.entropy_enh = EntropyEnhancementLayer(out_ch)
            self.temporal_cons = TemporalConsistencyLayer(out_ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.use_enhancements:
            out = self.freq_att(out)
            out = self.entropy_enh(out)
            out = self.temporal_cons(out)
        
        out = self.relu2(out)
        return out

# ================================
# 7) Transformer-UNet
# ================================

class TransformerUNet(nn.Module):
    """U-Net with Vision Transformer Bottleneck"""
    def __init__(self, 
                 in_ch=1, 
                 out_ch=1, 
                 feats=(64, 128, 256, 512),
                 embed_dim=512,
                 num_heads=8,
                 num_transformer_blocks=4,
                 patch_size=1,
                 dropout=0.0):
        super().__init__()
        
        # Encoder
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        c = in_ch
        for f in feats:
            self.downs.append(ConvBlock(c, f))
            c = f
        
        # Transformer Bottleneck
        self.patch_embed = PatchEmbedding(feats[-1], embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        self.proj_back = nn.Linear(embed_dim, feats[-1])
        self.patch_size = patch_size
        
        # Decoder
        self.ups = nn.ModuleList()
        c = feats[-1]
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(c, f, 2, 2))
            self.ups.append(ConvBlock(f * 2, f))
            c = f
        
        self.final = nn.Conv2d(feats[0], out_ch, 1)
        
    def forward(self, x):
        # Encoder
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        
        # Transformer Bottleneck
        B, C, H, W = x.shape
        x_tokens, (H_patch, W_patch) = self.patch_embed(x)
        
        N = x_tokens.shape[1]
        if N <= self.pos_embed.shape[1]:
            x_tokens = x_tokens + self.pos_embed[:, :N, :]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), 
                size=N, 
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x_tokens = x_tokens + pos
        
        for block in self.transformer_blocks:
            x_tokens = block(x_tokens)
        
        x_tokens = self.proj_back(x_tokens)
        x = rearrange(x_tokens, 'b (h w) c -> b c h w', h=H_patch, w=W_patch)
        
        # Decoder
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-1 - i // 2]
            if x.shape[2:] != s.shape[2:]:
                x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)
        
        return self.final(x)

class EnhancedTransformerUNet(nn.Module):
    """Enhanced U-Net with Vision Transformer and temporal/frequency improvements"""
    def __init__(self, 
                 in_ch=1, 
                 out_ch=1, 
                 feats=(64, 128, 256, 512),
                 embed_dim=512,
                 num_heads=8,
                 num_transformer_blocks=6,
                 patch_size=1,
                 dropout=0.1):
        super().__init__()
        
        # Encoder with enhancements at deeper levels
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        c = in_ch
        for idx, f in enumerate(feats):
            use_enh = (idx >= len(feats) - 2)
            self.downs.append(EnhancedConvBlock(c, f, use_enhancements=use_enh))
            c = f
        
        # Enhanced Transformer Bottleneck
        self.patch_embed = PatchEmbedding(feats[-1], embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        self.proj_back = nn.Linear(embed_dim, feats[-1])
        self.patch_size = patch_size
        
        # Additional bottleneck enhancement
        self.bottleneck_enhance = nn.Sequential(
            FrequencyAttention(feats[-1]),
            EntropyEnhancementLayer(feats[-1]),
            TemporalConsistencyLayer(feats[-1])
        )
        
        # Decoder with enhancements
        self.ups = nn.ModuleList()
        c = feats[-1]
        for idx, f in enumerate(reversed(feats)):
            self.ups.append(nn.ConvTranspose2d(c, f, 2, 2))
            use_enh = (idx < 2)
            self.ups.append(EnhancedConvBlock(f * 2, f, use_enhancements=use_enh))
            c = f
        
        # Final layers with refinement
        self.pre_final = nn.Sequential(
            FrequencyAttention(feats[0]),
            TemporalConsistencyLayer(feats[0])
        )
        self.final = nn.Conv2d(feats[0], out_ch, 1)
        
        # Post-processing for temporal consistency
        self.post_process = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        
        # Transformer Bottleneck
        B, C, H, W = x.shape
        x_tokens, (H_patch, W_patch) = self.patch_embed(x)
        
        N = x_tokens.shape[1]
        if N <= self.pos_embed.shape[1]:
            x_tokens = x_tokens + self.pos_embed[:, :N, :]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2), 
                size=N, 
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x_tokens = x_tokens + pos
        
        for block in self.transformer_blocks:
            x_tokens = block(x_tokens)
        
        x_tokens = self.proj_back(x_tokens)
        x = rearrange(x_tokens, 'b (h w) c -> b c h w', h=H_patch, w=W_patch)
        
        # Bottleneck enhancement
        x = self.bottleneck_enhance(x)
        
        # Decoder
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[-1 - i // 2]
            if x.shape[2:] != s.shape[2:]:
                x = F.interpolate(x, size=s.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)
        
        # Final refinement
        x = self.pre_final(x)
        x = self.final(x)
        x = self.post_process(x)
        
        return x

# ================================
# 8) Generate .mat files for Validation
# ================================

def generate_validation_submissions(model, val_us_dir, val_cases, device, output_dir='submissions', batch_size=16):
    """
    Generate .mat files for validation set comparison (OPTIMIZED with batch processing)
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    print(f"\n{'='*60}")
    print("Generating Validation Submissions (.mat files)")
    print(f"{'='*60}")
    
    import time
    total_start = time.time()
    
    with torch.no_grad():
        for case_idx, case in enumerate(val_cases, 1):
            case_start = time.time()
            in_dir = os.path.join(val_us_dir, case)
            out_dir = os.path.join(output_dir, case)
            os.makedirs(out_dir, exist_ok=True)
            
            if not os.path.isdir(in_dir):
                print(f"Warning: {in_dir} not found, skipping {case}")
                continue

            print(f"[{case_idx}/{len(val_cases)}] Processing {case}...", end='', flush=True)
            
            for fn in sorted(os.listdir(in_dir)):
                if not fn.endswith('.mat'):
                    continue

                # Load full undersampled k-space
                raw = CMRxReconDataset._load_mat(os.path.join(in_dir, fn))
                if raw.dtype.fields is not None:
                    raw = raw['real'] + 1j*raw['imag']
                k = np.squeeze(raw)

                # Reshape to (C, H, W, S, T)
                if k.ndim == 5:
                    k = k.transpose(2, 0, 1, 3, 4)
                elif k.ndim == 4:
                    k = k.transpose(0,1,2,3)[..., np.newaxis, :]
                elif k.ndim == 3:
                    k = k[np.newaxis, ..., np.newaxis, :]

                C, H, W, S, T = k.shape

                # Zero-fill IFFT + coil combine
                imgs = np.stack([
                    np.fft.ifftshift(
                        np.fft.ifft2(np.fft.fftshift(k[c], axes=(-2,-1)), axes=(-2,-1)),
                        axes=(-2,-1)
                    ) for c in range(C)
                ], axis=0)
                img_zf = np.sqrt(np.sum(np.abs(imgs)**2, axis=0))

                # OPTIMIZED: Batch process all slices & frames
                recon = np.zeros((H, W, S, T), dtype=np.float32)
                
                # Flatten all (slice, time) pairs
                all_inputs = []
                indices = []
                for s in range(S):
                    for t in range(T):
                        all_inputs.append(np.abs(img_zf[..., s, t]))
                        indices.append((s, t))
                
                # Process in batches
                num_batches = (len(all_inputs) + batch_size - 1) // batch_size
                
                for b in range(num_batches):
                    start_idx = b * batch_size
                    end_idx = min((b + 1) * batch_size, len(all_inputs))
                    
                    # Stack batch
                    batch_inputs = np.stack([all_inputs[i] for i in range(start_idx, end_idx)])
                    batch_tensor = torch.from_numpy(batch_inputs).unsqueeze(1).float().to(device)
                    
                    # Forward pass
                    batch_output = model(batch_tensor).squeeze(1).cpu().numpy()
                    
                    # Store results
                    for i, (s, t) in enumerate(indices[start_idx:end_idx]):
                        recon[..., s, t] = batch_output[i]

                # Save with correct variable name
                sio.savemat(
                    os.path.join(out_dir, fn),
                    {'img4ranking': recon.astype(np.single)}
                )
            
            case_time = time.time() - case_start
            print(f" Done in {case_time:.1f}s")
    
    total_time = time.time() - total_start
    print(f"\n✓ All submissions saved to: {output_dir}/")
    print(f"✓ Total time: {total_time/60:.1f} minutes ({total_time/len(val_cases):.1f}s per case)")
    print(f"{'='*60}")


# ================================
# 9) Compute Metrics from .mat files
# ================================

def compute_validation_metrics(pred_dir, gt_dir, val_cases):
    """
    Compute SSIM and PSNR by comparing .mat prediction files with ground truth
    """
    print(f"\n{'='*60}")
    print("Computing Validation Metrics")
    print(f"{'='*60}")
    
    all_ssim = []
    all_psnr = []
    all_nmse = []
    
    for case in val_cases:
        pred_case_dir = os.path.join(pred_dir, case)
        gt_case_dir = os.path.join(gt_dir, case)
        
        if not os.path.isdir(pred_case_dir) or not os.path.isdir(gt_case_dir):
            print(f"Warning: Missing directory for {case}, skipping...")
            continue
        
        for fn in sorted(os.listdir(pred_case_dir)):
            if not fn.endswith('.mat'):
                continue
            
            # Load prediction
            pred_path = os.path.join(pred_case_dir, fn)
            pred_data = CMRxReconDataset._load_mat(pred_path)
            pred = np.squeeze(np.array(pred_data))
            
            # Load ground truth
            # Remove '_mask_XXX' from filename to get gt filename
            view_name = fn.split('_mask_')[0]
            gt_fn = f"{view_name}.mat"
            gt_path = os.path.join(gt_case_dir, gt_fn)
            
            if not os.path.isfile(gt_path):
                print(f" Warning: GT not found for {case}/{fn}")
                continue
            
            gt_data = CMRxReconDataset._load_mat(gt_path)
            gt = np.squeeze(np.array(gt_data))
            
            # Ensure same shape
            if pred.shape != gt.shape:
                print(f" Warning: Shape mismatch for {case}/{fn}: pred {pred.shape} vs gt {gt.shape}")
                continue
            
            # Compute metrics for each slice and frame
            if pred.ndim == 4: # (H, W, S, T)
                for s in range(pred.shape[2]):
                    for t in range(pred.shape[3]):
                        pred_slice = pred[:, :, s, t]
                        gt_slice = gt[:, :, s, t]
                        
                        # SSIM
                        data_range = gt_slice.max() - gt_slice.min()
                        if data_range > 0:
                            ssim_val = ssim(gt_slice, pred_slice, data_range=data_range)
                            all_ssim.append(ssim_val)
                        
                        # PSNR
                        mse = np.mean((gt_slice - pred_slice) ** 2)
                        if mse > 0:
                            psnr_val = 10 * np.log10((gt_slice.max() ** 2) / mse)
                            all_psnr.append(psnr_val)
                        
                        # NMSE
                        gt_energy = np.sum(gt_slice ** 2)
                        if gt_energy > 0:
                            nmse_val = np.sum((gt_slice - pred_slice) ** 2) / gt_energy
                            all_nmse.append(nmse_val)
            
            elif pred.ndim == 3: # (H, W, T)
                for t in range(pred.shape[2]):
                    pred_frame = pred[:, :, t]
                    gt_frame = gt[:, :, t]
                    
                    data_range = gt_frame.max() - gt_frame.min()
                    if data_range > 0:
                        ssim_val = ssim(gt_frame, pred_frame, data_range=data_range)
                        all_ssim.append(ssim_val)
                    
                    mse = np.mean((gt_frame - pred_frame) ** 2)
                    if mse > 0:
                        psnr_val = 10 * np.log10((gt_frame.max() ** 2) / mse)
                        all_psnr.append(psnr_val)
                    
                    gt_energy = np.sum(gt_frame ** 2)
                    if gt_energy > 0:
                        nmse_val = np.sum((gt_frame - pred_frame) ** 2) / gt_energy
                        all_nmse.append(nmse_val)
            
            elif pred.ndim == 2: # (H, W)
                data_range = gt.max() - gt.min()
                if data_range > 0:
                    ssim_val = ssim(gt, pred, data_range=data_range)
                    all_ssim.append(ssim_val)
                
                mse = np.mean((gt - pred) ** 2)
                if mse > 0:
                    psnr_val = 10 * np.log10((gt.max() ** 2) / mse)
                    all_psnr.append(psnr_val)
                
                gt_energy = np.sum(gt ** 2)
                if gt_energy > 0:
                    nmse_val = np.sum((gt - pred) ** 2) / gt_energy
                    all_nmse.append(nmse_val)
    
    if not all_ssim:
        print("No valid metrics computed!")
        return None
    
    results = {
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'nmse_mean': np.mean(all_nmse),
        'nmse_std': np.std(all_nmse),
        'num_samples': len(all_ssim)
    }
    
    print(f"\n{'='*60}")
    print(f"Validation Results (n={results['num_samples']})")
    print(f"{'='*60}")
    print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"NMSE: {results['nmse_mean']:.4f} ± {results['nmse_std']:.4f}")
    print(f"{'='*60}\n")
    
    return results


# ================================
# 10) Training Function
# ================================

def train_model(model, train_loader, val_loader, device, num_epochs=10, 
                model_name="Model", lr=1e-4):
    """Train a model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    
    for ep in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for inp, tgt, _, _ in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            out = model(inp)
            loss = loss_fn(out, tgt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt, _, _ in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                out = model(inp)
                loss = loss_fn(out, tgt)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {ep:02d}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{model_name.lower().replace(" ", "_")}_best.pth')
    
    return model

class CombinedLoss(nn.Module):
    """Combined loss for better reconstruction quality"""
    def __init__(self, alpha=0.8, beta=0.1, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1 = nn.L1Loss()
        
    def forward(self, pred, target):
        # L1 loss
        l1_loss = self.l1(pred, target)
        
        # SSIM loss
        ssim_loss = 1 - self.compute_ssim(pred, target)
        
        # Frequency domain loss
        freq_loss = self.frequency_loss(pred, target)
        
        total = self.alpha * l1_loss + self.beta * ssim_loss + self.gamma * freq_loss
        return total
    
    def compute_ssim(self, x, y):
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        
        sigma_x = F.avg_pool2d(x**2, 3, 1, 1) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, 3, 1, 1) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
        
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
                   ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        
        return ssim_map.mean()
    
    def frequency_loss(self, pred, target):
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')
        return F.l1_loss(torch.abs(pred_freq), torch.abs(target_freq))

# ================================
# 11) Main Execution
# ================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ===== PATHS =====
    TRAIN_FULL = r"D:\tmp\ChallengeData\ChallengeData\MultiCoil\Cine\TrainingSet\Fullsample\Center001\UIH_30T_umr780"
    TRAIN_MASK = r"D:\tmp\ChallengeData\ChallengeData\MultiCoil\Cine\TrainingSet\Mask_TaskAll\Center001\UIH_30T_umr780"
    TRAIN_GT = r"D:\tmp\ChallengeData\ChallengeData\MultiCoil\Cine\TrainingSet\ImageShow\Center001\UIH_30T_umr780"
    VAL_US = r"D:\valtmp\ChallengeData_ValSet-TaskR1\TaskR1\MultiCoil\Cine\ValidationSet\UnderSample_TaskR1\Center001\UIH_30T_umr780"
    VAL_GT = r"D:\valtmp\ChallengeData_ValSet-TaskR1\TaskR1\MultiCoil\Cine\ValidationSet\GroundTruth\Center001\UIH_30T_umr780" # ADD THIS PATH

    val_cases = ['P007','P012','P016','P022','P023','P025','P038','P039','P045','P048','P052','P059']
    train_cases = [f"P{str(i).zfill(3)}" for i in range(1,62)
                   if f"P{str(i).zfill(3)}" not in val_cases and f"P{str(i).zfill(3)}"!='P047']

    # ===== Split training data for local training/validation =====
    print("\nSplitting training data...")
    local_val_cases = train_cases[-6:]
    local_train_cases = train_cases[:-6]
    
    print(f"Local train cases: {len(local_train_cases)}")
    print(f"Local val cases: {len(local_val_cases)}")
    
    # ===== Create Datasets =====
    print("\nLoading datasets...")
    train_ds = CMRxReconDataset(TRAIN_FULL, TRAIN_MASK, TRAIN_GT, local_train_cases)
    val_ds = CMRxReconDataset(TRAIN_FULL, TRAIN_MASK, TRAIN_GT, local_val_cases)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # ===== Train Baseline U-Net =====
    print("\n" + "="*60)
    print("STEP 1: Training Baseline U-Net")
    print("="*60)
    
    unet = UNet(in_ch=1, out_ch=1, feats=(64, 128, 256, 512))
    unet = train_model(
        unet, 
        train_loader, 
        val_loader, 
        device, 
        num_epochs=1,
        model_name="Baseline U-Net",
        lr=1e-4
    )

    # ===== Train Transformer-UNet =====
    print("\n" + "="*60)
    print("STEP 2: Training Original Transformer-UNet")
    print("="*60)
    
    trans_unet = TransformerUNet(
        in_ch=1,
        out_ch=1,
        feats=(64, 128, 256, 512),
        embed_dim=512,
        num_heads=8,
        num_transformer_blocks=4,
        patch_size=1,
        dropout=0.0
    )
    trans_unet = train_model(
        trans_unet,
        train_loader,
        val_loader,
        device,
        num_epochs=1,
        model_name="Transformer-UNet",
        lr=1e-4
    )
    
    # ===== Train ENHANCED Transformer-UNet =====
    print("\n" + "="*60)
    print("STEP 3: Training ENHANCED Transformer-UNet")
    print("="*60)
    
    enhanced_trans_unet = EnhancedTransformerUNet(
        in_ch=1,
        out_ch=1,
        feats=(64, 128, 256, 512),
        embed_dim=512,
        num_heads=8,
        num_transformer_blocks=6,
        patch_size=1,
        dropout=0.1
    )
    
    # Enhanced training with better optimizer and loss
    enhanced_trans_unet = enhanced_trans_unet.to(device)
    optimizer = torch.optim.AdamW(enhanced_trans_unet.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    loss_fn = CombinedLoss(alpha=0.8, beta=0.1, gamma=0.1).to(device)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for ep in range(1, 6):  # 20 epochs
        # Training
        enhanced_trans_unet.train()
        train_loss = 0.0
        for inp, tgt, _, _ in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            out = enhanced_trans_unet(inp)
            loss = loss_fn(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enhanced_trans_unet.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        enhanced_trans_unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt, _, _ in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                out = enhanced_trans_unet(inp)
                loss = loss_fn(out, tgt)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {ep:02d}/5 - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f}")
        
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(enhanced_trans_unet.state_dict(), 'enhanced_transformer-unet_best.pth')
            print(f"  → New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  → Early stopping at epoch {ep}")
                break
    
    # Load best model
    enhanced_trans_unet.load_state_dict(torch.load('enhanced_transformer-unet_best.pth'))

    # ===== Generate Validation Submissions =====
    print("\n" + "="*60)
    print("STEP 3: Generating Validation Submissions")
    print("="*60)
    
    # Generate .mat files for U-Net
    print("\nGenerating U-Net submissions...")
    generate_validation_submissions(
        unet, 
        VAL_US, 
        val_cases, 
        device, 
        output_dir='submissions_unet'
    )
    
    # Generate .mat files for Transformer-UNet
    print("\nGenerating Transformer-UNet submissions...")
    generate_validation_submissions(
        trans_unet, 
        VAL_US, 
        val_cases, 
        device, 
        output_dir='submissions_transformer_unet'
    )

    print("\nGenerating ENHANCED Transformer-UNet submissions...")
    generate_validation_submissions(
        enhanced_trans_unet, 
        VAL_US, 
        val_cases, 
        device, 
        output_dir='submissions_enhanced_transformer_unet',
        batch_size=16
    )

    # ===== Compute Validation Metrics =====
    print("\n" + "="*60)
    print("STEP 4: Computing Validation Metrics")
    print("="*60)
    
    # Compute metrics for U-Net
    print("\n--- Baseline U-Net Validation Metrics ---")
    unet_results = compute_validation_metrics(
        'submissions_unet',
        VAL_GT,
        val_cases
    )
    
    # Compute metrics for Transformer-UNet
    print("\n--- Transformer-UNet Validation Metrics ---")
    trans_results = compute_validation_metrics(
        'submissions_transformer_unet',
        VAL_GT,
        val_cases
    )

    print("\n--- ENHANCED Transformer-UNet Validation Metrics ---")
    enhanced_results = compute_validation_metrics(
        'submissions_enhanced_transformer_unet',
        VAL_GT,
        val_cases
    )

    # ===== Compare Results =====
    if unet_results and trans_results and enhanced_results:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        ssim_imp = trans_results['ssim_mean'] - unet_results['ssim_mean']
        psnr_imp = trans_results['psnr_mean'] - unet_results['psnr_mean']
        
        ssim_imp_enh = enhanced_results['ssim_mean'] - unet_results['ssim_mean']
        psnr_imp_enh = enhanced_results['psnr_mean'] - unet_results['psnr_mean']
        
        print(f"\nBaseline U-Net:")
        print(f" SSIM: {unet_results['ssim_mean']:.4f} ± {unet_results['ssim_std']:.4f}")
        print(f" PSNR: {unet_results['psnr_mean']:.2f} ± {unet_results['psnr_std']:.2f} dB")
        
        print(f"\nOriginal Transformer-UNet:")
        print(f" SSIM: {trans_results['ssim_mean']:.4f} ± {trans_results['ssim_std']:.4f}")
        print(f" PSNR: {trans_results['psnr_mean']:.2f} ± {trans_results['psnr_std']:.2f} dB")
        print(f" Improvement: SSIM {ssim_imp:+.4f}, PSNR {psnr_imp:+.2f} dB")
        
        print(f"\n*** ENHANCED Transformer-UNet ***:")
        print(f" SSIM: {enhanced_results['ssim_mean']:.4f} ± {enhanced_results['ssim_std']:.4f}")
        print(f" PSNR: {enhanced_results['psnr_mean']:.2f} ± {enhanced_results['psnr_std']:.2f} dB")
        print(f" Improvement: SSIM {ssim_imp_enh:+.4f} ({ssim_imp_enh/unet_results['ssim_mean']*100:+.1f}%)")
        print(f"              PSNR {psnr_imp_enh:+.2f} dB ({psnr_imp_enh/unet_results['psnr_mean']*100:+.1f}%)")
        print(f"{'='*60}\n")
        
        # Save comparison report
        with open('validation_comparison_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("CMRxRecon2025 - Validation Set Comparison\n")
            f.write("="*60 + "\n\n")
            
            f.write("VALIDATION SET RESULTS:\n")
            f.write("-"*60 + "\n\n")
            
            f.write("BASELINE U-NET:\n")
            f.write(f" SSIM: {unet_results['ssim_mean']:.4f} ± {unet_results['ssim_std']:.4f}\n")
            f.write(f" PSNR: {unet_results['psnr_mean']:.2f} ± {unet_results['psnr_std']:.2f} dB\n")
            f.write(f" NMSE: {unet_results['nmse_mean']:.4f} ± {unet_results['nmse_std']:.4f}\n")
            f.write(f" Samples: {unet_results['num_samples']}\n\n")
            
            f.write("TRANSFORMER-UNET:\n")
            f.write(f" SSIM: {trans_results['ssim_mean']:.4f} ± {trans_results['ssim_std']:.4f}\n")
            f.write(f" PSNR: {trans_results['psnr_mean']:.2f} ± {trans_results['psnr_std']:.2f} dB\n")
            f.write(f" NMSE: {trans_results['nmse_mean']:.4f} ± {trans_results['nmse_std']:.4f}\n")
            f.write(f" Samples: {trans_results['num_samples']}\n\n")
            
            f.write("IMPROVEMENTS:\n")
            f.write(f" SSIM: {ssim_imp:+.4f} ({ssim_imp/unet_results['ssim_mean']*100:+.1f}%)\n")
            f.write(f" PSNR: {psnr_imp:+.2f} dB ({psnr_imp/unet_results['psnr_mean']*100:+.1f}%)\n")
            f.write(f" NMSE: {nmse_imp:+.4f} ({nmse_imp/unet_results['nmse_mean']*100:+.1f}% reduction)\n")
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print("\nGenerated files:")
    print(" - baseline_u-net_best.pth (U-Net model weights)")
    print(" - transformer-unet_best.pth (Transformer-UNet model weights)")
    print(" - submissions_unet/ (U-Net predictions as .mat files)")
    print(" - submissions_transformer_unet/ (Transformer-UNet predictions as .mat files)")
    print(" - validation_comparison_report.txt (Detailed comparison)")
    print("\n✓ Ready for challenge submission!")


if __name__ == '__main__':
    main()