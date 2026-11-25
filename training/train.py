"""
StegoV5BCH training script (Google Colab).

Trains the PointNet++ / FiLM steganography network that hides one BCH(63,36)
codeword (63 channel bits) inside a 4096-point cloud.

Environment: written for a Google Colab GPU runtime (CUDA, bfloat16 autocast).
It mounts Google Drive and expects these files in DRIVE_PATH:
  - train4096_HD.pt / test4096_HD.pt : float tensors of shape (N, >=4096, 3)
    holding surface-sampled ModelNet40 point clouds
  - v5_best_32b.pth (optional)        : warm-start weights
  - v5bch_best_63b.pth (if RESUME)    : checkpoint to resume from

The bit-count curriculum (16 -> 34 -> 63 bits) is driven by BIT_STEPS/ACC_GATE.
RESUME_STATE reflects the final run used for the released model; set
RESUME = False to train from scratch.
"""

import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from google.colab import drive
import matplotlib.pyplot as plt


SEED = 42
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

DRIVE_PATH = '/content/drive/MyDrive/pointcloud_stego/ModelNet40'
os.makedirs(DRIVE_PATH, exist_ok=True)


BCH_N      = 63
BCH_K      = 36
BCH_PARITY = BCH_N - BCH_K
BCH_T_IMPL = 2
BCH_T_FULL = 5


BCH_THRESHOLD = 1.0 - (BCH_T_IMPL / BCH_N)
print(f"\nBCH(63,36) config:")
print(f"  Implemented: t≤{BCH_T_IMPL} (brute-force, corrects ≤{BCH_T_IMPL} errors)")
print(f"  Theoretical: t={BCH_T_FULL} (requires Berlekamp-Massey — future work)")
print(f"  Required accuracy: ≥{BCH_THRESHOLD*100:.1f}%")
print(f"  Error tolerance: {BCH_T_IMPL}/{BCH_N} = {BCH_T_IMPL/BCH_N*100:.1f}%")

def poly_mod(dividend, divisor):
    d = list(dividend)
    div = list(divisor)
    if len(d) < len(div):
        return d
    for i in range(len(d) - len(div) + 1):
        if d[i] == 1:
            for j in range(len(div)):
                d[i+j] ^= div[j]
    return d[len(d) - len(div) + 1:]

def bch63_generator():

    return [1,1,1,1,0,1,0,1,0,0,1,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,1,1]

def bch_encode(data_bits_36):
    assert len(data_bits_36) == BCH_K
    gen    = bch63_generator()
    shifted = list(data_bits_36) + [0] * BCH_PARITY
    rem    = poly_mod(shifted, gen)
    while len(rem) < BCH_PARITY:
        rem = [0] + rem
    return list(data_bits_36) + rem[:BCH_PARITY]

def bch_decode_t2(received_63):
    received = list(received_63)
    gen      = bch63_generator()

    rem = poly_mod(received, gen)
    if all(r == 0 for r in rem):
        return received[:BCH_K], True, 0

    for i in range(BCH_N):
        test = list(received); test[i] ^= 1
        if all(r == 0 for r in poly_mod(test, gen)):
            return test[:BCH_K], True, 1

    for i in range(BCH_N):
        for j in range(i+1, BCH_N):
            test = list(received); test[i] ^= 1; test[j] ^= 1
            if all(r == 0 for r in poly_mod(test, gen)):
                return test[:BCH_K], True, 2

    return received[:BCH_K], False, -1


DATA_PER_CHUNK = 32
ID_BITS        = 2
CHUNK_BITS     = DATA_PER_CHUNK + ID_BITS
BCH_PAD        = BCH_K - CHUNK_BITS
BCH_CHANNEL    = BCH_N

N_CHUNKS       = 4
TOTAL_PAYLOAD  = DATA_PER_CHUNK * N_CHUNKS

def encode_chunk(payload_32, chunk_id):
    id_bits    = [(chunk_id >> 1) & 1, chunk_id & 1]
    chunk_bits = payload_32 + id_bits
    padded     = chunk_bits + [0] * BCH_PAD
    return bch_encode(padded)

def decode_chunk(channel_bits_63, expected_id=None):
    decoded_36, ok, n_err = bch_decode_t2(channel_bits_63)
    chunk_bits = decoded_36[:CHUNK_BITS]
    payload_32 = chunk_bits[:DATA_PER_CHUNK]
    id_bits    = chunk_bits[DATA_PER_CHUNK:DATA_PER_CHUNK+ID_BITS]
    chunk_id   = id_bits[0]*2 + id_bits[1]

    if expected_id is not None and chunk_id != expected_id:
        return payload_32, chunk_id, False, n_err

    return payload_32, chunk_id, ok, n_err

def encode_message_128(message_128):
    chunks = []
    for i in range(N_CHUNKS):
        payload = message_128[i*DATA_PER_CHUNK:(i+1)*DATA_PER_CHUNK]
        chunks.append(encode_chunk(payload, chunk_id=i))
    return chunks

def decode_message_128(four_channel_bits):
    message   = [0] * TOTAL_PAYLOAD
    successes = 0
    err_counts = []

    for expected_id, ch_bits in enumerate(four_channel_bits):
        payload, got_id, ok, n_err = decode_chunk(ch_bits,
                                                   expected_id=expected_id)
        err_counts.append(n_err)
        if ok:
            start = expected_id * DATA_PER_CHUNK
            message[start:start+DATA_PER_CHUNK] = payload
            successes += 1

    return message, successes / N_CHUNKS, err_counts

def bch_encode_batch_tensor(data_bits_tensor):
    B   = data_bits_tensor.shape[0]
    out = torch.zeros(B, BCH_CHANNEL)
    for b in range(B):
        bits    = data_bits_tensor[b].int().cpu().tolist()
        padded  = bits + [0] * BCH_PAD
        encoded = bch_encode(padded)
        out[b]  = torch.tensor(encoded, dtype=torch.float32)
    return out

def bch_decode_batch_tensor(channel_logits):
    B   = channel_logits.shape[0]
    out = torch.zeros(B, CHUNK_BITS)
    ok_list = []
    for b in range(B):
        ch    = (channel_logits[b] > 0).int().cpu().tolist()
        dec36, ok, _ = bch_decode_t2(ch)
        out[b] = torch.tensor(dec36[:CHUNK_BITS], dtype=torch.float32)
        ok_list.append(ok)
    return out, ok_list


NUM_POINTS       = 4096
NUM_PATCHES      = 512
POINTS_PER_PATCH = 8
MODEL_BITS       = BCH_CHANNEL

BATCH_SIZE = 128
MAX_EPOCHS = 500

ALPHA_INIT = 0.08
ALPHA_MID  = 0.08
ALPHA_END  = 0.08

BIT_STEPS  = [16, 34, 63]
ACC_GATE   = 0.85
MIN_EPOCHS = 8

LAMBDA_CD  = 100.0
LAMBDA_MSG =  80.0
LAMBDA_REP =   8.0

NOISE_AUG_START  = 99999
NOISE_AUG_SIGMA  = 0.002

LR           = 3e-4
WEIGHT_DECAY = 1e-4

RESUME       = True
RESUME_CKPT  = 'v5bch_best_63b.pth'
RESUME_STATE = {
    'bit_idx'   : 2,
    'alpha'     : 0.08,
    'best_acc'  : 0.905,
    'epochs_at' : 0,
}

scaler = torch.amp.GradScaler('cuda')


class ModelNetDataset(Dataset):
    def __init__(self, file_name, augment=False):
        path = os.path.join(DRIVE_PATH, file_name)
        self.data    = torch.load(path)
        self.augment = augment
        print(f"  {file_name}: {len(self.data):,} shapes")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        pc = self.data[idx, :NUM_POINTS, :].clone()
        pc = pc - pc.mean(0)
        pc = pc / (torch.sqrt((pc**2).sum(1)).max() + 1e-8)
        if self.augment:
            theta = np.random.uniform(0, 2*np.pi)
            c, s  = np.cos(theta), np.sin(theta)
            R     = torch.tensor([[c,0,s],[0,1,0],[-s,0,c]],
                                 dtype=torch.float32)
            pc    = pc @ R.T
            pc    = pc + torch.randn_like(pc) * 0.002
        return pc

print("\nLoading data...")
train_ds = ModelNetDataset('train4096_HD.pt', augment=True)
test_ds  = ModelNetDataset('test4096_HD.pt',  augment=False)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)


def chamfer_loss(p1, p2):
    p1s  = (p1**2).sum(-1, keepdim=True)
    p2s  = (p2**2).sum(-1, keepdim=True)
    dist = torch.clamp(
               p1s + p2s.transpose(1,2) - 2*p1.bmm(p2.transpose(1,2)),
               min=0.0)
    return dist.min(2)[0].mean() + dist.min(1)[0].mean()

def repulsion_loss(pc, h=0.04):
    dist   = torch.cdist(pc, pc)
    dist   = dist + torch.eye(pc.shape[1],
                              device=pc.device).unsqueeze(0) * 1e5
    knn, _ = torch.topk(dist, 8, dim=-1, largest=False)
    return torch.clamp(h - knn, min=0.0).mean()


def farthest_point_sample(xyz, npoint):
    B, N, _  = xyz.shape
    cents    = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist     = torch.ones(B, N, device=xyz.device) * 1e10
    far      = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    bidx     = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        cents[:, i] = far
        c = xyz[bidx, far].view(B, 1, 3)
        d = ((xyz - c)**2).sum(-1)
        m = d < dist; dist[m] = d[m]
        far = dist.max(-1)[1]
    return cents

def index_points(pts, idx):
    B  = pts.shape[0]
    vs = list(idx.shape); vs[1:] = [1]*(len(vs)-1)
    rs = list(idx.shape); rs[0]  = 1
    bi = torch.arange(B, device=pts.device,
                       dtype=torch.long).view(vs).repeat(rs)
    return pts[bi, idx]

def query_ball_point(r, k, xyz, new_xyz):
    B, N, _ = xyz.shape; _, S, _ = new_xyz.shape
    idx  = torch.arange(N, device=xyz.device).view(1,1,N).expand(B,S,N).clone()
    dist = ((new_xyz.unsqueeze(2)-xyz.unsqueeze(1))**2).sum(-1)
    idx[dist > r*r] = N
    idx  = idx.sort(-1)[0][:,:,:k]
    first = idx[:,:,0:1].expand_as(idx)
    idx[idx == N] = first[idx == N]
    return idx

class PointNetSA(nn.Module):
    def __init__(self, npoint, radius, nsample, in_ch, mlp):
        super().__init__()
        self.npoint = npoint; self.radius = radius; self.nsample = nsample
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        last = in_ch
        for out in mlp:
            self.convs.append(nn.Conv2d(last, out, 1))
            self.bns.append(nn.BatchNorm2d(out))
            last = out

    def forward(self, xyz, pts):
        x = xyz.permute(0,2,1)
        p = pts.permute(0,2,1) if pts is not None else None
        idx      = farthest_point_sample(x, self.npoint)
        new_xyz  = index_points(x, idx)
        gidx     = query_ball_point(self.radius, self.nsample, x, new_xyz)
        gxyz     = index_points(x, gidx) - new_xyz.unsqueeze(2)
        if p is not None:
            new_pts = torch.cat([gxyz, index_points(p, gidx)], -1)
        else:
            new_pts = gxyz
        new_pts = new_pts.permute(0,3,2,1)
        for conv, bn in zip(self.convs, self.bns):
            new_pts = F.gelu(bn(conv(new_pts)))
        return new_xyz.permute(0,2,1), new_pts.max(2)[0]


class FiLMModulator(nn.Module):
    def __init__(self, feature_dim=256, msg_dim=MODEL_BITS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim, 256), nn.GELU(),
            nn.Linear(256, feature_dim * 2))

    def forward(self, feat, msg):
        p = self.net(msg)
        return p[:,:256].unsqueeze(-1)*feat + p[:,256:].unsqueeze(-1)

class AlphaBoundedDecoder(nn.Module):
    def __init__(self, feature_dim=256, num_patches=512, pts_per_patch=8):
        super().__init__()
        self.num_patches = num_patches; self.pts_per_patch = pts_per_patch
        self.mlp = nn.Sequential(
            nn.Conv1d(feature_dim,512,1), nn.GELU(),
            nn.Conv1d(512,256,1),         nn.GELU(),
            nn.Conv1d(256,pts_per_patch*3,1))

    def forward(self, features, patch_centroids, alpha):
        B   = features.shape[0]
        raw = self.mlp(features).view(B,3,self.pts_per_patch,self.num_patches)
        bounded = alpha * torch.tanh(raw / (alpha+1e-8))
        cents   = patch_centroids.unsqueeze(2).expand(
                      B,3,self.pts_per_patch,self.num_patches)
        return (cents+bounded).view(B,3,self.pts_per_patch*self.num_patches)

class StegoV5BCH(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1     = PointNetSA(1024, 0.1, 32, 3,   [64,64,128])
        self.sa2     = PointNetSA(512,  0.2, 32, 131, [128,128,256])
        self.film    = FiLMModulator(256, MODEL_BITS)
        self.decoder = AlphaBoundedDecoder(256, NUM_PATCHES, POINTS_PER_PATCH)
        self.msg_decoder = nn.Sequential(
            nn.Conv1d(256,512,1), nn.BatchNorm1d(512), nn.GELU(),
            nn.Conv1d(512,256,1), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256,MODEL_BITS,1))

    def embed(self, pc, bits, alpha):
        full = torch.zeros(pc.shape[0], MODEL_BITS, device=pc.device)
        full[:, :bits.shape[1]] = bits
        xyz = pc.transpose(1,2)
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        mod_feat = self.film(l2_pts, full)
        return self.decoder(mod_feat, l2_xyz, alpha).transpose(1,2)

    def extract(self, stego_pc):
        xyz = stego_pc.transpose(1,2)
        r1_xyz, r1_pts = self.sa1(xyz, None)
        _, r2_pts      = self.sa2(r1_xyz, r1_pts)
        return self.msg_decoder(r2_pts).mean(dim=-1)

    def forward(self, pc, bits, alpha):
        stego  = self.embed(pc, bits, alpha)
        logits = self.extract(stego)
        return stego, logits


def load_v5_32b(model):
    path = os.path.join(DRIVE_PATH, 'v5_best_32b.pth')
    if not os.path.exists(path):
        print("  v5_best_32b.pth not found — training from scratch")
        return model
    old  = torch.load(path, map_location=device)
    m_d  = model.state_dict()
    gold = {k: v for k, v in old.items()
            if k in m_d and v.shape == m_d[k].shape}
    m_d.update(gold); model.load_state_dict(m_d)
    print(f"  ✓ Transferred {len(gold)} layers from v5_best_32b.pth")
    print(f"  ↻ Re-init: film ({32}→{MODEL_BITS}), "
          f"msg_decoder (32→{MODEL_BITS})")
    return model


def train():
    model = StegoV5BCH().to(device)
    model = load_v5_32b(model)

    if RESUME:
        ckpt = os.path.join(DRIVE_PATH, RESUME_CKPT)
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"  ✓ Resumed from {RESUME_CKPT}")
        bit_idx   = RESUME_STATE['bit_idx']
        alpha     = RESUME_STATE['alpha']
        best_acc  = RESUME_STATE['best_acc']
        epochs_at = RESUME_STATE['epochs_at']
    else:
        bit_idx = 0; alpha = ALPHA_INIT
        best_acc = 0.0; epochs_at = 0

    active_bits = BIT_STEPS[bit_idx]

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    print(f"\n{'='*64}")
    print(f"  V5-BCH FIXED — 4 objects × 32b = 128-bit payload")
    print(f"  BCH(63,36) t≤{BCH_T_IMPL} | need ≥{BCH_THRESHOLD*100:.1f}% channel acc")
    print(f"  Fixes: noise aug | ID verify | LAMBDA_MSG={LAMBDA_MSG} | honest BCH")
    print(f"{'='*64}\n")

    for epoch in range(MAX_EPOCHS):
        model.train()
        t_acc = t_cd = t_rep = 0; n = 0
        use_bch = (active_bits == MODEL_BITS)

        noise_active = epoch >= NOISE_AUG_START
        noise_sigma  = NOISE_AUG_SIGMA * min(1.0,
                           (epoch - NOISE_AUG_START) / 20) if noise_active else 0.0

        pbar = tqdm(train_loader,
                    desc=f"Ep{epoch:03d}|{active_bits}b"
                         f"{'(BCH)' if use_bch else ''}|α={alpha:.4f}"
                         f"|σ={noise_sigma:.4f}",
                    leave=False)

        for pc in pbar:
            pc  = pc.to(device, non_blocking=True)
            B   = pc.shape[0]
            optimizer.zero_grad(set_to_none=True)

            if use_bch:
                data    = torch.randint(0, 2, (B, CHUNK_BITS),
                                        dtype=torch.float32)
                target  = bch_encode_batch_tensor(data).to(device)
                n_active = MODEL_BITS
            else:
                data    = torch.randint(0, 2, (B, active_bits),
                                        dtype=torch.float32, device=device)
                target  = data
                n_active = active_bits

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                stego = model.embed(pc, target, alpha)

                if noise_active and noise_sigma > 0:
                    sigma  = random.uniform(0.0, noise_sigma)
                    stego_noisy = stego + torch.randn_like(stego) * sigma
                else:
                    stego_noisy = stego

                logits  = model.extract(stego_noisy)
                loss_cd  = chamfer_loss(pc, stego)
                loss_msg = F.binary_cross_entropy_with_logits(
                               logits[:, :n_active],
                               target[:, :n_active])
                loss_rep = repulsion_loss(stego)
                loss     = (LAMBDA_CD  * loss_cd  +
                            LAMBDA_MSG * loss_msg +
                            LAMBDA_REP * loss_rep)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            if use_bch:
                pred_ch = (logits[:, :MODEL_BITS] > 0).float()
                acc = (pred_ch == target[:, :MODEL_BITS]).float().mean().item()
            else:
                acc = ((logits[:,:n_active]>0).float()==data
                       ).float().mean().item()

            t_acc += acc; t_cd += loss_cd.item()
            t_rep += loss_rep.item(); n += 1
            pbar.set_postfix({'Acc':f'{acc:.3f}',
                              'CD':f'{loss_cd.item():.5f}'})

        scheduler.step(); epochs_at += 1
        avg_acc = t_acc/n; avg_cd = t_cd/n

        bch_tag = ""
        if use_bch:
            if avg_acc >= BCH_THRESHOLD:
                bch_tag = f"  ★ BCH OK — channel acc {avg_acc*100:.1f}%"
            else:
                bch_tag = f"  ch_acc={avg_acc*100:.1f}% need {BCH_THRESHOLD*100:.1f}%"

        print(f"Ep{epoch:03d}|{active_bits:2d}b|α={alpha:.4f}| "
              f"CD:{avg_cd:.5f}|Acc:{avg_acc:.3f}{bch_tag}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            ckpt = os.path.join(DRIVE_PATH, f'v5bch_best_{active_bits}b.pth')
            torch.save(model.state_dict(), ckpt)
            print(f"  ★ {ckpt}  (acc={best_acc:.3f})")

        if epoch % 10 == 0:
            ckpt = os.path.join(DRIVE_PATH, f'v5bch_ep{epoch:03d}.pth')
            torch.save(model.state_dict(), ckpt)
            old = os.path.join(DRIVE_PATH, f'v5bch_ep{epoch-20:03d}.pth')
            if os.path.exists(old): os.remove(old)

        if epochs_at < MIN_EPOCHS: continue

        if avg_acc >= 0.92 and alpha > ALPHA_END:
            alpha = ALPHA_END; best_acc = 0.0; epochs_at = 0
            print(f"\n   α → {ALPHA_END}")
        elif avg_acc >= 0.85 and alpha > ALPHA_MID:
            alpha = ALPHA_MID; best_acc = 0.0; epochs_at = 0
            print(f"\n   α → {ALPHA_MID}")
        elif avg_acc >= ACC_GATE and alpha == ALPHA_END:
            if bit_idx < len(BIT_STEPS)-1:
                bit_idx    += 1
                active_bits = BIT_STEPS[bit_idx]
                alpha       = ALPHA_INIT
                best_acc    = 0.0; epochs_at = 0
                print(f"\n{'─'*50}")
                print(f"   Bits → {active_bits}"
                      f"{' ← BCH PIPELINE ACTIVE' if active_bits==MODEL_BITS else ''}"
                      f" | α → {ALPHA_INIT}")
                print(f"{'─'*50}\n")


@torch.no_grad()
def full_eval(ckpt_name='v5bch_best_63b.pth', eval_alpha=0.05, n_trials=500):
    model = StegoV5BCH().to(device)
    model.load_state_dict(torch.load(
        os.path.join(DRIVE_PATH, ckpt_name), map_location=device))
    model.train()

    test_data = test_ds.data

    def rand_rot(x):
        a=random.uniform(0,2*np.pi); c,s=np.cos(a),np.sin(a)
        R=torch.tensor([[c,0,s],[0,1,0],[-s,0,c]],
                       dtype=torch.float32,device=x.device)
        return x@R.T

    attacks = {
        'Clean'         : lambda x: x,
        'Noise σ=0.005' : lambda x: x+torch.randn_like(x)*0.005,
        'Noise σ=0.01'  : lambda x: x+torch.randn_like(x)*0.01,
        'Rotate 90°'    : lambda x: x@torch.tensor([[0,0,1],[0,1,0],[-1,0,0]],
                              dtype=torch.float32,device=x.device).T,
        'Rotate random' : rand_rot,
        'Scale ×0.9'    : lambda x: x*0.9,
        'Scale ×1.1'    : lambda x: x*1.1,
    }

    print(f"\n{'═'*68}")
    print(f"  DISTRIBUTED BCH EVAL | 4 objects × 32b = 128-bit | α={eval_alpha}")
    print(f"  BCH(63,36) t≤{BCH_T_IMPL} | threshold={BCH_THRESHOLD*100:.1f}% ch acc")
    print(f"{'═'*68}")
    print(f"  {'Attack':<16} {'Ch Acc':>7} {'BCH OK%':>8} "
          f"{'Full 128b':>10} {'CD':>10}")
    print(f"  {'─'*16} {'─'*7} {'─'*8} {'─'*10} {'─'*10}")

    results = {}
    for name, atk in attacks.items():
        ch_l, bch_l, f128_l, cd_l = [], [], [], []

        for trial in range(n_trials):
            msg_128 = [random.randint(0,1) for _ in range(TOTAL_PAYLOAD)]
            chunks  = encode_message_128(msg_128)
            extracted_bits = []

            for cid, ch_bits in enumerate(chunks):
                pc = test_data[(trial*4+cid) % len(test_data)]
                pc = pc - pc.mean(0)
                pc = pc/(torch.sqrt((pc**2).sum(1)).max()+1e-8)
                pc = pc.unsqueeze(0).to(device)

                ch_t = torch.tensor(ch_bits, dtype=torch.float32,
                                    device=device).unsqueeze(0)
                stego   = model.embed(pc, ch_t, eval_alpha)
                stego_a = atk(stego)
                logits  = model.extract(stego_a)

                pred = (logits>0).int().squeeze(0).cpu().tolist()
                ch_acc = sum(p==t for p,t in zip(pred,ch_bits))/BCH_CHANNEL
                ch_l.append(ch_acc)
                extracted_bits.append(pred)
                cd_l.append(chamfer_loss(stego, pc).item())

            dec_msg, succ_rate, err_cts = decode_message_128(extracted_bits)
            bch_l.append(succ_rate)
            f128_l.append(float(dec_msg == msg_128))

        r = dict(ch=np.mean(ch_l), bch=np.mean(bch_l),
                 f128=np.mean(f128_l), cd=np.mean(cd_l))
        results[name] = r
        print(f"  {name:<16} {r['ch']:>7.3f} {r['bch']:>8.3f} "
              f"{r['f128']:>10.3f} {r['cd']:>10.6f}")

    print(f"{'═'*68}")
    c = results['Clean']
    print(f"\n  Channel accuracy   : {c['ch']*100:.1f}%  "
          f"(need ≥{BCH_THRESHOLD*100:.1f}% for t≤2 BCH)")
    print(f"  BCH chunk success  : {c['bch']*100:.1f}%  per chunk")
    print(f"  Full 128b recovery : {c['f128']*100:.1f}%  (all 4 chunks correct)")
    print(f"  Chamfer Distance   : {c['cd']:.6f}")

    np.save(os.path.join(DRIVE_PATH,'v5bch_eval.npy'), results)
    return results


if __name__ == '__main__':
    train()
