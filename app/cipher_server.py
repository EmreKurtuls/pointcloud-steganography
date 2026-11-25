import os, re, glob, random, secrets, hashlib, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, jsonify, request
from Crypto.Cipher import AES

app = Flask(__name__)

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODELNET_DIR = os.environ.get('MODELNET_DIR', os.path.join(ROOT_DIR, 'data', 'ModelNet40'))
DB_PATH      = os.path.join(DATA_DIR, 'listings.json')
OBJECTS_DIR  = os.path.join(DATA_DIR, 'objects')
MODEL_PATH   = os.environ.get('MODEL_PATH', os.path.join(ROOT_DIR, 'models', 'model.pth'))
os.makedirs(OBJECTS_DIR, exist_ok=True)

DEMO_ALPHA     = 0.08
N_FILLERS      = 28
FILLER_SEED    = 42
MODEL_BITS     = 63
BCH_N, BCH_K   = 63, 36
BCH_PARITY     = BCH_N - BCH_K
DATA_PER_CHUNK = 32
ID_BITS        = 2
CHUNK_BITS     = DATA_PER_CHUNK + ID_BITS
BCH_PAD        = BCH_K - CHUNK_BITS
N_CHUNKS       = 4
TOTAL_PAYLOAD  = DATA_PER_CHUNK * N_CHUNKS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_catalog(root):
    catalog = {}
    pattern = re.compile(r'^(.+)_\d{4}\.off$')
    for path in sorted(glob.glob(os.path.join(root,'**','*.off'),recursive=True) +
                       glob.glob(os.path.join(root,'*.off'))):
        m = pattern.match(os.path.basename(path))
        if m: catalog.setdefault(m.group(1),[]).append(path)
    return catalog

CATALOG    = build_catalog(MODELNET_DIR) if os.path.exists(MODELNET_DIR) else {}
CATEGORIES = sorted(CATALOG.keys())
print(f"Catalog: {sum(len(v) for v in CATALOG.values())} objects | {len(CATEGORIES)} categories")

def read_off_mesh(path):
    with open(path,'r') as f: lines=[l.strip() for l in f if l.strip()]
    if lines[0]=='OFF': counts=lines[1].split(); start=2
    else: counts=lines[0][3:].split(); start=1
    nv,nf=int(counts[0]),int(counts[1])
    verts=np.array([[float(x) for x in lines[start+i].split()[:3]] for i in range(nv)],dtype=np.float32)
    faces=[]
    for i in range(nv,nv+nf):
        p=lines[start+i].split(); n=int(p[0]); fv=[int(x) for x in p[1:n+1]]
        for j in range(1,n-1): faces.append([fv[0],fv[j],fv[j+1]])
    return verts,np.array(faces,dtype=np.int32)

def sample_surface(verts,faces,n=4096):
    v0=verts[faces[:,0]]; v1=verts[faces[:,1]]; v2=verts[faces[:,2]]
    areas=np.maximum(np.sqrt((np.cross(v1-v0,v2-v0)**2).sum(-1))/2,1e-10)
    probs=areas/areas.sum(); fi=np.random.choice(len(faces),size=n,p=probs)
    r1=np.random.rand(n,1); r2=np.random.rand(n,1)
    mask=(r1+r2)>1; r1[mask]=1-r1[mask]; r2[mask]=1-r2[mask]
    f=faces[fi]
    return (verts[f[:,0]]*(1-r1-r2)+verts[f[:,1]]*r1+verts[f[:,2]]*r2).astype(np.float32)

def load_pc(path,n=4096):
    try:
        verts,faces=read_off_mesh(path)
        pts=sample_surface(verts,faces,n) if len(faces)>0 else verts
    except: pts=np.random.randn(n,3).astype(np.float32)
    idx=np.random.choice(len(pts),n,replace=len(pts)<n)
    pc=torch.tensor(pts[idx],dtype=torch.float32)
    pc=pc-pc.mean(0); pc=pc/(torch.sqrt((pc**2).sum(1)).max()+1e-8)
    return pc

def farthest_point_sample(xyz,npoint):
    B,N,_=xyz.shape; cents=torch.zeros(B,npoint,dtype=torch.long,device=xyz.device)
    dist=torch.ones(B,N,device=xyz.device)*1e10; far=torch.randint(0,N,(B,),dtype=torch.long,device=xyz.device)
    bidx=torch.arange(B,dtype=torch.long,device=xyz.device)
    for i in range(npoint):
        cents[:,i]=far; c=xyz[bidx,far].view(B,1,3); d=((xyz-c)**2).sum(-1)
        m=d<dist; dist[m]=d[m]; far=dist.max(-1)[1]
    return cents
def index_points(pts,idx):
    B=pts.shape[0]; vs=list(idx.shape); vs[1:]=[1]*(len(vs)-1); rs=list(idx.shape); rs[0]=1
    return pts[torch.arange(B,device=pts.device,dtype=torch.long).view(vs).repeat(rs),idx]
def query_ball_point(r,k,xyz,new_xyz):
    B,N,_=xyz.shape; _,S,_=new_xyz.shape
    idx=torch.arange(N,device=xyz.device).view(1,1,N).expand(B,S,N).clone()
    dist=((new_xyz.unsqueeze(2)-xyz.unsqueeze(1))**2).sum(-1)
    idx[dist>r*r]=N; idx=idx.sort(-1)[0][:,:,:k]
    first=idx[:,:,0:1].expand_as(idx); idx[idx==N]=first[idx==N]; return idx
def chamfer_loss(p1,p2):
    d=torch.clamp((p1**2).sum(-1,keepdim=True)+(p2**2).sum(-1,keepdim=True).transpose(1,2)-2*p1.bmm(p2.transpose(1,2)),min=0)
    return d.min(2)[0].mean()+d.min(1)[0].mean()
class PointNetSA(nn.Module):
    def __init__(self,npoint,radius,nsample,in_ch,mlp):
        super().__init__(); self.npoint=npoint; self.radius=radius; self.nsample=nsample
        self.convs=nn.ModuleList(); self.bns=nn.ModuleList(); last=in_ch
        for out in mlp: self.convs.append(nn.Conv2d(last,out,1)); self.bns.append(nn.BatchNorm2d(out)); last=out
    def forward(self,xyz,pts):
        x=xyz.permute(0,2,1); p=pts.permute(0,2,1) if pts is not None else None
        idx=farthest_point_sample(x,self.npoint); new_xyz=index_points(x,idx)
        gidx=query_ball_point(self.radius,self.nsample,x,new_xyz); gxyz=index_points(x,gidx)-new_xyz.unsqueeze(2)
        new_pts=(torch.cat([gxyz,index_points(p,gidx)],-1) if p is not None else gxyz).permute(0,3,2,1)
        for conv,bn in zip(self.convs,self.bns): new_pts=F.gelu(bn(conv(new_pts)))
        return new_xyz.permute(0,2,1),new_pts.max(2)[0]
class FiLMModulator(nn.Module):
    def __init__(self,fd=256,md=63):
        super().__init__(); self.net=nn.Sequential(nn.Linear(md,256),nn.GELU(),nn.Linear(256,fd*2))
    def forward(self,feat,msg): p=self.net(msg); return p[:,:256].unsqueeze(-1)*feat+p[:,256:].unsqueeze(-1)
class AlphaBoundedDecoder(nn.Module):
    def __init__(self,fd=256,np_=512,pp=8):
        super().__init__(); self.np_=np_; self.pp=pp
        self.mlp=nn.Sequential(nn.Conv1d(fd,512,1),nn.GELU(),nn.Conv1d(512,256,1),nn.GELU(),nn.Conv1d(256,pp*3,1))
    def forward(self,feat,cents,alpha):
        B=feat.shape[0]; raw=self.mlp(feat).view(B,3,self.pp,self.np_)
        return (cents.unsqueeze(2).expand(B,3,self.pp,self.np_)+alpha*torch.tanh(raw/(alpha+1e-8))).view(B,3,self.pp*self.np_)
class StegoV5BCH(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1=PointNetSA(1024,0.1,32,3,[64,64,128]); self.sa2=PointNetSA(512,0.2,32,131,[128,128,256])
        self.film=FiLMModulator(256,63); self.decoder=AlphaBoundedDecoder(256,512,8)
        self.msg_decoder=nn.Sequential(nn.Conv1d(256,512,1),nn.BatchNorm1d(512),nn.GELU(),
                                       nn.Conv1d(512,256,1),nn.BatchNorm1d(256),nn.GELU(),nn.Conv1d(256,63,1))
    def embed(self,pc,bits,alpha):
        full=torch.zeros(pc.shape[0],63,device=pc.device); full[:,:bits.shape[1]]=bits
        xyz=pc.transpose(1,2); l1,lp=self.sa1(xyz,None); l2,lf=self.sa2(l1,lp)
        return self.decoder(self.film(lf,full),l2,alpha).transpose(1,2)
    def extract(self,stego):
        xyz=stego.transpose(1,2); r1,rp=self.sa1(xyz,None); _,r2=self.sa2(r1,rp)
        return self.msg_decoder(r2).mean(dim=-1)

model=StegoV5BCH().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH,map_location=device)); model.train()
    print(f"Model loaded on {device}")
else:
    print(f"WARNING: model.pth not found at {MODEL_PATH}")

class GF64:
    def __init__(self):
        self.exp=[0]*128; self.log=[0]*64; x=1
        for i in range(63):
            self.exp[i]=x; self.log[x]=i; x<<=1
            if x&64: x^=0x43
        for i in range(63,128): self.exp[i]=self.exp[i-63]
    def mul(self,a,b): return 0 if(a==0 or b==0) else self.exp[(self.log[a]+self.log[b])%63]
    def div(self,a,b): return 0 if a==0 else self.exp[(self.log[a]-self.log[b])%63]
gf=GF64()
def poly_mod_bin(d,v):
    d=list(d); v=list(v)
    for i in range(len(d)-len(v)+1):
        if d[i]:
            for j in range(len(v)): d[i+j]^=v[j]
    return d[len(d)-len(v)+1:]
def poly_mul_bin(p1,p2):
    r=[0]*(len(p1)+len(p2)-1)
    for i,a in enumerate(p1):
        for j,b in enumerate(p2): r[i+j]^=(a&b)
    return r
def frobenius_orbit(k,n=63):
    orbit,seen,e=[],set(),k%n
    while e not in seen: seen.add(e); orbit.append(e); e=(2*e)%n
    return orbit
def min_poly_gf2(k):
    poly=[1]
    for e in frobenius_orbit(k):
        r=gf.exp[e] if e>0 else 1; np2=[0]*(len(poly)+1)
        for i,c in enumerate(poly): np2[i+1]^=c; np2[i]^=gf.mul(c,r)
        poly=np2
    return [int(bool(c)) for c in poly]
GEN=[1]
for k in [1,3,5,7,9]: GEN=poly_mul_bin(GEN,min_poly_gf2(k))
def bch_encode(data36):
    s=list(data36)+[0]*BCH_PARITY; rem=poly_mod_bin(s,GEN)
    while len(rem)<BCH_PARITY: rem=[0]+rem
    return list(data36)+rem[:BCH_PARITY]
def berlekamp_massey(synd):
    n=len(synd); C=[1]+[0]*n; B=[1]+[0]*n; L,m,b=0,1,1
    for i in range(n):
        d=synd[i]
        for j in range(1,L+1): d^=gf.mul(C[j],synd[i-j])
        if d==0: m+=1
        elif 2*L<=i:
            T=list(C); cf=gf.div(d,b)
            for j in range(len(B)):
                if m+j<len(C): C[m+j]^=gf.mul(cf,B[j])
            L,B,b,m=i+1-L,T,d,1
        else:
            cf=gf.div(d,b)
            for j in range(len(B)):
                if m+j<len(C): C[m+j]^=gf.mul(cf,B[j])
            m+=1
    return C[:L+1],L
def chien_search(el,n=63):
    pos=[]
    for i in range(n):
        an=1 if i==0 else gf.exp[(63-i)%63]; v=0; pw=1
        for c in el: v^=gf.mul(c,pw); pw=gf.mul(pw,an)
        if v==0: pos.append(i)
    return pos
def compute_syndromes(recv,t=5):
    synd=[]
    for i in range(1,2*t+1):
        ai=gf.exp[i%63]; s=0
        for j in range(62,-1,-1): s=gf.mul(s,ai); s^=int(recv[j])
        synd.append(s)
    return synd
def bch_decode_t5(recv):
    recv=list(recv)
    if all(r==0 for r in poly_mod_bin(recv,GEN)): return recv[:BCH_K],True,0
    synd=compute_syndromes(recv,5)
    if all(s==0 for s in synd): return recv[:BCH_K],True,0
    try:
        el,ne=berlekamp_massey(synd)
        if ne==0 or ne>5: return recv[:BCH_K],False,-1
        ep=chien_search(el)
        if len(ep)!=ne: return recv[:BCH_K],False,-1
        c=list(recv)
        for p in ep: c[p]^=1
        if all(r==0 for r in poly_mod_bin(c,GEN)): return c[:BCH_K],True,ne
        return recv[:BCH_K],False,-1
    except: return recv[:BCH_K],False,-1
def chase_decode(recv,logits_t,k=5):
    conf=torch.abs(logits_t).cpu(); weak=torch.topk(conf,k,largest=False).indices.tolist()
    for pat in range(2**k):
        test=list(recv)
        for bi in range(k):
            if (pat>>bi)&1: test[weak[bi]]^=1
        dec,ok,n=bch_decode_t5(test)
        if ok: return dec,True,n
    return bch_decode_t5(recv)
def encode_chunk(p32,cid):
    return bch_encode(p32+[(cid>>1)&1,cid&1]+[0]*BCH_PAD)
def encode_message_128(msg):
    return [encode_chunk(msg[i*32:(i+1)*32],i) for i in range(N_CHUNKS)]
def decode_message_v2(four_ch,four_logits=None,k=5):
    msg=[0]*TOTAL_PAYLOAD; succ=0
    for eid,ch in enumerate(four_ch):
        lg=four_logits[eid] if four_logits else None
        dec,ok,_=chase_decode(ch,lg,k) if lg is not None else bch_decode_t5(ch)
        bits=dec[:CHUNK_BITS]
        if bits[32]*2+bits[33]!=eid: ok=False
        if ok: msg[eid*32:(eid+1)*32]=bits[:32]; succ+=1
    return msg,succ/N_CHUNKS


def derive_key(trade_id,passphrase):
    return hashlib.pbkdf2_hmac('sha256',passphrase.encode(),trade_id.encode(),10000)[:16]
def encrypt_to_bits(message,trade_id,passphrase):
    key=derive_key(trade_id,passphrase)
    nonce=hashlib.md5(trade_id.encode()).digest()[:8]
    cipher=AES.new(key,AES.MODE_CTR,nonce=nonce)
    msg_bytes=message.encode('utf-8')[:16].ljust(16,b'\x00')
    ct=cipher.encrypt(msg_bytes)
    bits=[]
    for b in ct:
        for j in range(7,-1,-1): bits.append((b>>j)&1)
    return bits
def decrypt_from_bits(bits_128,trade_id,passphrase):
    key=derive_key(trade_id,passphrase)
    nonce=hashlib.md5(trade_id.encode()).digest()[:8]
    cipher=AES.new(key,AES.MODE_CTR,nonce=nonce)
    ct=bytes(sum(bits_128[i+j]<<(7-j) for j in range(8)) for i in range(0,128,8))
    msg_bytes=cipher.decrypt(ct)
    try: return msg_bytes.rstrip(b'\x00').decode('utf-8')
    except: return None
def generate_trade_id():
    """Single ID format for all listings — regular and stego look identical.
    This TRD-... ID is what the receiver enters to decode the message."""
    ts  = time.strftime("%Y%m%d")
    uid = ''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(8))
    return f"TRD-{ts}-{uid}"


random.seed(FILLER_SEED)
if CATALOG:
    _filler_paths  = random.sample([p for ps in CATALOG.values() for p in ps], min(N_FILLERS, sum(len(v) for v in CATALOG.values())))
    FILLER_TENSORS = torch.stack([load_pc(p) for p in _filler_paths]).to(device)
    print(f"Fillers: {FILLER_TENSORS.shape}")
else:
    print("No catalog — using random fillers")
    FILLER_TENSORS = torch.randn(N_FILLERS,4096,3,device=device)
    FILLER_TENSORS = FILLER_TENSORS/(FILLER_TENSORS.norm(dim=-1,keepdim=True)+1e-8)

def get_filler_chs():
    chs=[]
    for _ in range(len(FILLER_TENSORS)):
        rm=[random.randint(0,1) for _ in range(TOTAL_PAYLOAD)]
        chs.append(encode_message_128(rm)[0])
    return torch.tensor(chs,dtype=torch.float32,device=device)


def load_db():
    if not os.path.exists(DB_PATH): return []
    with open(DB_PATH) as f: return json.load(f)
def save_db(data):
    with open(DB_PATH,'w') as f: json.dump(data,f,indent=2)

_active_pcs    = [None]*4
_active_names  = [None]*4
_active_cats   = [None]*4

@app.route('/')
def index():
    return render_template('cipher.html',
                           categories=CATEGORIES or ['airplane','chair','table','car'])

@app.route('/api/objects_in_category')
def objects_in_cat():
    cat = request.args.get('cat','')
    choices = ['🎲 Random'] + [os.path.basename(p).replace('.off','')
                                for p in sorted(CATALOG.get(cat,[]))]
    return jsonify(choices)

@app.route('/api/preview', methods=['POST'])
def preview():
    global _active_pcs, _active_names, _active_cats
    data  = request.json
    cats  = data['categories']
    names = data['objects']
    results = []
    for i,(cat,name) in enumerate(zip(cats,names)):
        if CATALOG and cat in CATALOG:
            if name == '🎲 Random' or not name:
                path = random.choice(CATALOG[cat])
            else:
                path = next((p for p in CATALOG[cat]
                             if os.path.basename(p).replace('.off','')==name),
                            random.choice(CATALOG[cat]))
            pc = load_pc(path)
            fname = os.path.basename(path).replace('.off','')
        else:
            pc = torch.randn(4096,3); pc=pc-pc.mean(0); pc=pc/(pc.norm(dim=1).max()+1e-8)
            fname = f"{cat}_{i}"
        _active_pcs[i]   = pc
        _active_names[i] = fname
        _active_cats[i]  = cat
        results.append({'name': fname, 'points': pc.tolist()})
    return jsonify(results)

@app.route('/api/embed', methods=['POST'])
def embed():
    data       = request.json
    message    = data.get('message','')[:16]
    passphrase = data.get('passphrase','')
    listing_name = data.get('listing_name', 'Prop Bundle')
    price      = float(data.get('price', 19.99))

    if not message: return jsonify({'error':'No message'}), 400
    if not passphrase: return jsonify({'error':'No passphrase'}), 400
    if any(p is None for p in _active_pcs):
        return jsonify({'error':'Preview objects first'}), 400

    trade_id = generate_trade_id()
    msg_bits = encrypt_to_bits(message, trade_id, passphrase)
    chunks   = encode_message_128(msg_bits)

    real_pcs = [p for p in _active_pcs]
    filler_chs = get_filler_chs()
    pc_t  = torch.cat([torch.stack(real_pcs).to(device), FILLER_TENSORS], dim=0)
    ch_t  = torch.cat([torch.tensor(chunks,dtype=torch.float32,device=device),
                       filler_chs], dim=0)
    with torch.no_grad():
        stego_all = model.embed(pc_t, ch_t, DEMO_ALPHA)
    stego_4 = stego_all[:4]

    files = []
    for i in range(4):
        fname = f"stego_{trade_id}_{i}.npy"
        np.save(os.path.join(OBJECTS_DIR, fname), stego_4[i].cpu().numpy())
        files.append(fname)

    listings = load_db()
    bundle = {
        "id":        trade_id,
        "seller":    data.get('seller','cipher_user'),
        "name":      listing_name,
        "category":  "bundle",
        "price":     price,
        "stego":     True,
        "sold":      False,
        "files":     files,
        "trade_id":  trade_id,
        "desc":      f"Bundle of {len(files)} high-fidelity 3D models"
    }
    listings.append(bundle)
    save_db(listings)

    stego_points = [stego_4[i].cpu().tolist() for i in range(4)]
    cds = [chamfer_loss(stego_4[i:i+1], pc_t[i:i+1]).item() for i in range(4)]

    return jsonify({
        'trade_id':     trade_id,
        'stego_points': stego_points,
        'cds':          cds,
        'key_preview':  derive_key(trade_id, passphrase).hex()
    })

@app.route('/api/decode', methods=['POST'])
def decode():
    data       = request.json
    trade_id   = data.get('trade_id','').strip()
    passphrase = data.get('passphrase','').strip()
    if not trade_id or not passphrase:
        return jsonify({'error':'Trade ID and passphrase required'}), 400

    listings = load_db()
    bundle   = next((L for L in listings if L.get('id')==trade_id or L.get('trade_id')==trade_id), None)
    if not bundle:
        return jsonify({'error': f'Trade ID {trade_id} not found'}), 404

    stego_tensors = []
    for fname in bundle['files']:
        path = os.path.join(OBJECTS_DIR, fname)
        if not os.path.exists(path):
            return jsonify({'error': f'Stego file missing: {fname}'}), 404
        stego_tensors.append(torch.tensor(np.load(path), dtype=torch.float32))

    filler_chs = get_filler_chs()
    with torch.no_grad():
        dummy_stego = model.embed(FILLER_TENSORS, filler_chs, DEMO_ALPHA)
        real_stego  = torch.stack(stego_tensors).to(device)
        full_batch  = torch.cat([real_stego, dummy_stego], dim=0)
        logits_all  = model.extract(full_batch)

    logits_4=logits_all[:4]; ex_bits=[]; ex_logits=[]
    for i in range(4):
        lg=logits_4[i]; ex_bits.append((lg>0).int().cpu().tolist()); ex_logits.append(lg.cpu())

    extracted_bits, sr = decode_message_v2(ex_bits, ex_logits)
    plaintext = decrypt_from_bits(extracted_bits, trade_id, passphrase)
    chunks_ok = int(sr*4)

    return jsonify({
        'message':    plaintext,
        'chunks_ok':  chunks_ok,
        'success':    bool(plaintext and plaintext.strip())
    })

@app.route('/api/stego_points/<trade_id>/<int:idx>')
def get_stego_points(trade_id, idx):
    listings = load_db()
    bundle   = next((L for L in listings if L.get('id')==trade_id or L.get('trade_id')==trade_id), None)
    if not bundle or idx >= len(bundle.get('files',[])):
        return jsonify({'error':'not found'}), 404
    path = os.path.join(OBJECTS_DIR, bundle['files'][idx])
    if not os.path.exists(path): return jsonify({'error':'file missing'}), 404
    pts = np.load(path)
    return jsonify({'points': pts.tolist()})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  POLYMORPH.CIPHER  — private cipher tool")
    print(f"  http://localhost:{os.environ.get('PORT', 5001)}")
    print("  ⚠  DO NOT share this URL publicly")
    print("="*50 + "\n")
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', port=int(os.environ.get('PORT', 5001)))