"""
Robustness evaluation of the final model (Google Colab).

Embeds random 128-bit messages split across 4 test objects (4 x BCH(63,36)
codewords) and measures, per attack: raw channel accuracy, BCH(t=5) hard
decoding, Chase-5 soft decoding, full 128-bit recovery rate and Chamfer
distance. Results are written to final_eval_v2.npy (see results/).

Expects the Colab/Drive layout described in training/train.py.
"""

import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from google.colab import drive

drive.mount('/content/drive', force_remount=False)
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DRIVE_PATH = '/content/drive/MyDrive/pointcloud_stego/ModelNet40'
print(f"Device: {device}")

NUM_POINTS, NUM_PATCHES, POINTS_PER_PATCH = 4096, 512, 8
MODEL_BITS     = 63
BCH_N, BCH_K   = 63, 36
BCH_PARITY     = BCH_N - BCH_K
DATA_PER_CHUNK = 32
ID_BITS        = 2
CHUNK_BITS     = DATA_PER_CHUNK + ID_BITS
BCH_PAD        = BCH_K - CHUNK_BITS
N_CHUNKS       = 4
TOTAL_PAYLOAD  = DATA_PER_CHUNK * N_CHUNKS
EVAL_ALPHA     = 0.08
CKPT_NAME      = 'v5bch_best_63b.pth'
N_TRIALS       = 500
BATCH_SIZE     = 32

class GF64:
    def __init__(self):
        self.exp = [0]*128; self.log = [0]*64
        x = 1
        for i in range(63):
            self.exp[i] = x; self.log[x] = i
            x <<= 1
            if x & 64: x ^= 0x43
        for i in range(63, 128): self.exp[i] = self.exp[i-63]
    def mul(self, a, b):
        if a==0 or b==0: return 0
        return self.exp[(self.log[a]+self.log[b])%63]
    def div(self, a, b):
        if b==0: raise ZeroDivisionError
        if a==0: return 0
        return self.exp[(self.log[a]-self.log[b])%63]

gf = GF64()

def frobenius_orbit(k, field_order=63):
    orbit, seen, e = [], set(), k % field_order
    while e not in seen:
        seen.add(e); orbit.append(e)
        e = (2*e) % field_order
    return orbit

def compute_min_poly_gf2(k):
    orbit = frobenius_orbit(k)
    poly  = [1]
    for e in orbit:
        root    = gf.exp[e] if e > 0 else 1
        new_p   = [0] * (len(poly)+1)
        for i, c in enumerate(poly):
            new_p[i+1] ^= c
            new_p[i]   ^= gf.mul(c, root)
        poly = new_p
    return [int(bool(c)) for c in poly]

def poly_mul_bin(p1, p2):
    r = [0]*(len(p1)+len(p2)-1)
    for i, a in enumerate(p1):
        for j, b in enumerate(p2):
            r[i+j] ^= (a&b)
    return r

def build_generator():
    g = [1]
    for k in [1, 3, 5, 7, 9]:
        g = poly_mul_bin(g, compute_min_poly_gf2(k))
    assert len(g) == BCH_PARITY+1, f"Generator degree wrong: {len(g)-1} (expected {BCH_PARITY})"
    return g

GEN = build_generator()
print(f"Generator polynomial computed: degree {len(GEN)-1} ✓")

def poly_mod_bin(dividend, divisor):
    d = list(dividend); div = list(divisor)
    if len(d) < len(div): return d
    for i in range(len(d)-len(div)+1):
        if d[i]:
            for j in range(len(div)): d[i+j] ^= div[j]
    return d[len(d)-len(div)+1:]

def bch_encode(data_bits_36):
    shifted = list(data_bits_36) + [0]*BCH_PARITY
    rem     = poly_mod_bin(shifted, GEN)
    while len(rem) < BCH_PARITY: rem = [0]+rem
    return list(data_bits_36) + rem[:BCH_PARITY]

def compute_syndromes(recv, t=5):
    synd = []
    for i in range(1, 2*t+1):
        ai = gf.exp[i % 63]
        s  = 0
        for j in range(62, -1, -1):
            s = gf.mul(s, ai); s ^= int(recv[j])
        synd.append(s)
    return synd

def berlekamp_massey(synd):
    n = len(synd)
    C = [1]+[0]*n; B = [1]+[0]*n
    L, m, b = 0, 1, 1
    for i in range(n):
        d = synd[i]
        for j in range(1, L+1): d ^= gf.mul(C[j], synd[i-j])
        if d == 0: m += 1
        elif 2*L <= i:
            T = list(C); cf = gf.div(d, b)
            for j in range(len(B)):
                if m+j < len(C): C[m+j] ^= gf.mul(cf, B[j])
            L, B, b, m = i+1-L, T, d, 1
        else:
            cf = gf.div(d, b)
            for j in range(len(B)):
                if m+j < len(C): C[m+j] ^= gf.mul(cf, B[j])
            m += 1
    return C[:L+1], L

def chien_search(err_loc, n=63):
    pos = []
    for i in range(n):
        an = 1 if i==0 else gf.exp[(63-i)%63]
        v  = 0; pw = 1
        for c in err_loc: v ^= gf.mul(c, pw); pw = gf.mul(pw, an)
        if v == 0: pos.append(i)
    return pos

def bch_decode_t5(recv):
    recv = list(recv)
    if all(r==0 for r in poly_mod_bin(recv, GEN)):
        return recv[:BCH_K], True, 0
    synd = compute_syndromes(recv, 5)
    if all(s==0 for s in synd): return recv[:BCH_K], True, 0
    try:
        el, ne = berlekamp_massey(synd)
        if ne==0 or ne>5: return recv[:BCH_K], False, -1
        ep = chien_search(el)
        if len(ep)!=ne: return recv[:BCH_K], False, -1
        c = list(recv)
        for p in ep: c[p] ^= 1
        if all(r==0 for r in poly_mod_bin(c, GEN)):
            return c[:BCH_K], True, ne
        return recv[:BCH_K], False, -1
    except: return recv[:BCH_K], False, -1


def self_test():
    passed = 0
    for n_err in range(6):
        msg = [random.randint(0,1) for _ in range(BCH_K)]
        cw  = bch_encode(msg)
        cor = list(cw)
        for p in random.sample(range(BCH_N), n_err): cor[p] ^= 1
        dec, ok, _ = bch_decode_t5(cor)
        if ok and dec == msg: passed += 1
    print(f"BCH t=5 self-test: {passed}/6 {'✓' if passed==6 else '✗'}")
self_test()


def chase_decode(recv, logits_t, k=5):
    conf = torch.abs(logits_t).cpu()
    weak = torch.topk(conf, k, largest=False).indices.tolist()
    for pat in range(2**k):
        test = list(recv)
        for bi in range(k):
            if (pat>>bi)&1: test[weak[bi]] ^= 1
        dec, ok, n = bch_decode_t5(test)
        if ok: return dec, True, n
    return bch_decode_t5(recv)

def encode_chunk(p32, cid):
    return bch_encode(p32+[(cid>>1)&1, cid&1]+[0]*BCH_PAD)

def encode_message_128(msg):
    return [encode_chunk(msg[i*32:(i+1)*32], i) for i in range(N_CHUNKS)]

def decode_message_v2(four_ch, four_logits=None, use_chase=True, k=5):
    msg=[0]*TOTAL_PAYLOAD; succ=0
    for eid, ch in enumerate(four_ch):
        lg = four_logits[eid] if four_logits else None
        dec, ok, _ = (chase_decode(ch, lg, k) if (use_chase and lg is not None)
                      else bch_decode_t5(ch))
        bits = dec[:CHUNK_BITS]
        if bits[32]*2+bits[33] != eid: ok=False
        if ok: msg[eid*32:(eid+1)*32]=bits[:32]; succ+=1
    return msg, succ/N_CHUNKS

def farthest_point_sample(xyz, npoint):
    B,N,_=xyz.shape
    cents=torch.zeros(B,npoint,dtype=torch.long,device=xyz.device)
    dist=torch.ones(B,N,device=xyz.device)*1e10
    far=torch.randint(0,N,(B,),dtype=torch.long,device=xyz.device)
    bidx=torch.arange(B,dtype=torch.long,device=xyz.device)
    for i in range(npoint):
        cents[:,i]=far; c=xyz[bidx,far].view(B,1,3)
        d=((xyz-c)**2).sum(-1); m=d<dist; dist[m]=d[m]; far=dist.max(-1)[1]
    return cents

def index_points(pts, idx):
    B=pts.shape[0]; vs=list(idx.shape); vs[1:]=[1]*(len(vs)-1)
    rs=list(idx.shape); rs[0]=1
    bi=torch.arange(B,device=pts.device,dtype=torch.long).view(vs).repeat(rs)
    return pts[bi,idx]

def query_ball_point(r,k,xyz,new_xyz):
    B,N,_=xyz.shape; _,S,_=new_xyz.shape
    idx=torch.arange(N,device=xyz.device).view(1,1,N).expand(B,S,N).clone()
    dist=((new_xyz.unsqueeze(2)-xyz.unsqueeze(1))**2).sum(-1)
    idx[dist>r*r]=N; idx=idx.sort(-1)[0][:,:,:k]
    first=idx[:,:,0:1].expand_as(idx); idx[idx==N]=first[idx==N]
    return idx

class PointNetSA(nn.Module):
    def __init__(self,npoint,radius,nsample,in_ch,mlp):
        super().__init__()
        self.npoint=npoint; self.radius=radius; self.nsample=nsample
        self.convs=nn.ModuleList(); self.bns=nn.ModuleList(); last=in_ch
        for out in mlp:
            self.convs.append(nn.Conv2d(last,out,1))
            self.bns.append(nn.BatchNorm2d(out)); last=out
    def forward(self,xyz,pts):
        x=xyz.permute(0,2,1); p=pts.permute(0,2,1) if pts is not None else None
        idx=farthest_point_sample(x,self.npoint); new_xyz=index_points(x,idx)
        gidx=query_ball_point(self.radius,self.nsample,x,new_xyz)
        gxyz=index_points(x,gidx)-new_xyz.unsqueeze(2)
        new_pts=(torch.cat([gxyz,index_points(p,gidx)],-1) if p is not None
                 else gxyz).permute(0,3,2,1)
        for conv,bn in zip(self.convs,self.bns): new_pts=F.gelu(bn(conv(new_pts)))
        return new_xyz.permute(0,2,1), new_pts.max(2)[0]

class FiLMModulator(nn.Module):
    def __init__(self, feature_dim=256, msg_dim=MODEL_BITS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(msg_dim, 256), nn.GELU(),
            nn.Linear(256, feature_dim*2))
    def forward(self, feat, msg):
        p=self.net(msg)
        return p[:,:256].unsqueeze(-1)*feat + p[:,256:].unsqueeze(-1)

class AlphaBoundedDecoder(nn.Module):
    def __init__(self, feature_dim=256, num_patches=512, pts_per_patch=8):
        super().__init__()
        self.num_patches=num_patches; self.pts_per_patch=pts_per_patch
        self.mlp=nn.Sequential(
            nn.Conv1d(feature_dim,512,1), nn.GELU(),
            nn.Conv1d(512,256,1),         nn.GELU(),
            nn.Conv1d(256,pts_per_patch*3,1))
    def forward(self, features, patch_centroids, alpha):
        B=features.shape[0]
        raw=self.mlp(features).view(B,3,self.pts_per_patch,self.num_patches)
        bounded=alpha*torch.tanh(raw/(alpha+1e-8))
        cents=patch_centroids.unsqueeze(2).expand(B,3,self.pts_per_patch,self.num_patches)
        return (cents+bounded).view(B,3,self.pts_per_patch*self.num_patches)

class StegoV5BCH(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1=PointNetSA(1024,0.1,32,3,[64,64,128])
        self.sa2=PointNetSA(512,0.2,32,131,[128,128,256])
        self.film=FiLMModulator(256,MODEL_BITS)
        self.decoder=AlphaBoundedDecoder(256,NUM_PATCHES,POINTS_PER_PATCH)
        self.msg_decoder=nn.Sequential(
            nn.Conv1d(256,512,1),nn.BatchNorm1d(512),nn.GELU(),
            nn.Conv1d(512,256,1),nn.BatchNorm1d(256),nn.GELU(),
            nn.Conv1d(256,MODEL_BITS,1))
    def embed(self,pc,bits,alpha):
        full=torch.zeros(pc.shape[0],MODEL_BITS,device=pc.device)
        full[:,:bits.shape[1]]=bits
        xyz=pc.transpose(1,2)
        l1_xyz,l1_pts=self.sa1(xyz,None); l2_xyz,l2_pts=self.sa2(l1_xyz,l1_pts)
        mod=self.film(l2_pts,full)
        return self.decoder(mod,l2_xyz,alpha).transpose(1,2)
    def extract(self,stego):
        xyz=stego.transpose(1,2)
        r1,rp=self.sa1(xyz,None); _,r2=self.sa2(r1,rp)
        return self.msg_decoder(r2).mean(dim=-1)

def chamfer_loss(p1,p2):
    p1s=(p1**2).sum(-1,keepdim=True); p2s=(p2**2).sum(-1,keepdim=True)
    d=torch.clamp(p1s+p2s.transpose(1,2)-2*p1.bmm(p2.transpose(1,2)),min=0)
    return d.min(2)[0].mean()+d.min(1)[0].mean()


model = StegoV5BCH().to(device)
saved = torch.load(os.path.join(DRIVE_PATH,CKPT_NAME), map_location=device)
m_d   = model.state_dict()
ok    = {k:v for k,v in saved.items() if k in m_d and v.shape==m_d[k].shape}
m_d.update(ok); model.load_state_dict(m_d, strict=False)
print(f"✓ Loaded {CKPT_NAME}: {len(ok)}/{len(saved)} layers matched")
model.train()

test_data = torch.load(os.path.join(DRIVE_PATH,'test4096_HD.pt'))
print(f"✓ Test data: {test_data.shape[0]:,} shapes")

def rand_rot(x):
    a=random.uniform(0,2*np.pi); c,s=np.cos(a),np.sin(a)
    R=torch.tensor([[c,0,s],[0,1,0],[-s,0,c]],dtype=torch.float32,device=x.device)
    return x@R.T

attacks = {
    'Clean'         : lambda x: x,
    'Noise σ=0.005' : lambda x: x+torch.randn_like(x)*0.005,
    'Noise σ=0.01'  : lambda x: x+torch.randn_like(x)*0.01,
    'Scale ×0.9'    : lambda x: x*0.9,
    'Scale ×1.1'    : lambda x: x*1.1,
    'Rotate 90°'    : lambda x: x@torch.tensor([[0,0,1],[0,1,0],[-1,0,0]],
                          dtype=torch.float32,device=x.device).T,
    'Rotate random' : rand_rot,
}

print(f"\n{'═'*70}")
print(f"  BCH t=5 + Chase-5 | 4×32b=128-bit | α={EVAL_ALPHA} | n={N_TRIALS}")
print(f"{'═'*70}")
print(f"  {'Attack':<16} {'Ch Acc':>7} {'BCH t5':>8} {'Chase5':>8} {'Full128':>8} {'CD':>10}")
print(f"  {'─'*16} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

results = {}
for atk_name, atk_fn in attacks.items():
    ch_l, t5_l, c5_l, f128_l, cd_l = [], [], [], [], []

    for start in tqdm(range(0,N_TRIALS,BATCH_SIZE), desc=atk_name, leave=False):
        end=min(start+BATCH_SIZE,N_TRIALS); actual=end-start
        pc_list, ch_list, msg_list = [], [], []
        for i in range(actual):
            msg=[random.randint(0,1) for _ in range(TOTAL_PAYLOAD)]
            msg_list.append(msg)
            for cid, ch in enumerate(encode_message_128(msg)):
                pc=test_data[((start+i)*4+cid)%len(test_data)]
                pc=pc-pc.mean(0); pc=pc/(torch.sqrt((pc**2).sum(1)).max()+1e-8)
                pc_list.append(pc); ch_list.append(ch)

        with torch.no_grad():
            pc_t=torch.stack(pc_list).to(device)
            ch_t=torch.tensor(ch_list,dtype=torch.float32,device=device)
            stego=model.embed(pc_t,ch_t,EVAL_ALPHA)
            stego_a=atk_fn(stego)
            logits=model.extract(stego_a)

        pred=( logits>0).int().cpu().tolist(); lc=logits.cpu()

        for i in range(actual):
            tp=pred[i*4:(i+1)*4]; tc=ch_list[i*4:(i+1)*4]
            tl=[lc[i*4+j] for j in range(4)]
            ts=stego[i*4:(i+1)*4].detach(); tpc=pc_t[i*4:(i+1)*4]

            ch_l.append(sum(sum(p==t for p,t in zip(tp[j],tc[j]))
                            for j in range(4))/(4*BCH_N))
            cd_l.append(chamfer_loss(ts,tpc).item())

            _,sr5=decode_message_v2(tp,four_logits=None,use_chase=False)
            t5_l.append(sr5)

            dc5,src5=decode_message_v2(tp,four_logits=tl,use_chase=True,k=5)
            c5_l.append(src5)
            f128_l.append(float(dc5==msg_list[i]))

    r=dict(ch=np.mean(ch_l),t5=np.mean(t5_l),
           c5=np.mean(c5_l),f128=np.mean(f128_l),cd=np.mean(cd_l))
    results[atk_name]=r
    print(f"  {atk_name:<16} {r['ch']:>7.3f} {r['t5']:>8.3f} "
          f"{r['c5']:>8.3f} {r['f128']:>8.3f} {r['cd']:>10.6f}")

print(f"{'═'*70}")
c=results['Clean']
print(f"\n  Channel Acc  : {c['ch']*100:.1f}%")
print(f"  BCH t=5/chunk: {c['t5']*100:.1f}%")
print(f"  Chase-5/chunk: {c['c5']*100:.1f}%")
print(f"  Full 128b    : {c['f128']*100:.1f}%")
print(f"  CD           : {c['cd']:.6f}")

np.save(os.path.join(DRIVE_PATH,'final_eval_v2.npy'), results)
print(f"\n  Saved → final_eval_v2.npy")