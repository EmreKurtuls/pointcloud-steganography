import os, json, random, secrets, time
from flask import Flask, render_template, jsonify, request
import numpy as np

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

import re, glob

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH     = os.path.join(DATA_DIR, 'listings.json')
HIST_PATH   = os.path.join(DATA_DIR, 'history.json')
OBJECTS_DIR = os.path.join(DATA_DIR, 'objects')
os.makedirs(OBJECTS_DIR, exist_ok=True)

MAX_LISTINGS = 20

def generate_trade_id():
    ts  = time.strftime("%Y%m%d")
    uid = ''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(8))
    return f"TRD-{ts}-{uid}"

MODELNET_DIR = os.environ.get('MODELNET_DIR', os.path.join(ROOT_DIR, 'data', 'ModelNet40'))

def build_catalog(root):
    names = {}; paths = {}
    pattern = re.compile(r'^(.+)_\d{4}\.off$')
    for path in sorted(glob.glob(os.path.join(root,'**','*.off'),recursive=True)+
                       glob.glob(os.path.join(root,'*.off'))):
        m = pattern.match(os.path.basename(path))
        if m:
            cat  = m.group(1)
            fname = os.path.basename(path).replace('.off','')
            names.setdefault(cat,[]).append(fname)
            paths[fname] = path
    return names, paths

if os.path.exists(MODELNET_DIR):
    MODEL_CATALOG, MODEL_PATHS = build_catalog(MODELNET_DIR)
    print(f"Model catalog: {sum(len(v) for v in MODEL_CATALOG.values()):,} objects | {len(MODEL_CATALOG)} categories")
else:
    MODEL_CATALOG, MODEL_PATHS = {}, {}
    print("ModelNet40 not found — using synthetic shapes")

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

def load_real_pc(model_name, n=4096):
    if model_name in MODEL_PATHS:
        try:
            verts,faces = read_off_mesh(MODEL_PATHS[model_name])
            pts = sample_surface(verts,faces,n) if len(faces)>0 else verts
            idx = np.random.choice(len(pts),n,replace=len(pts)<n)
            pts = pts[idx]
            pts = (pts-pts.mean(0))/(np.sqrt((pts**2).sum(1)).max()+1e-8)
            if pts[:,1].mean() < -0.1:
                pts[:,1] = -pts[:,1]
            return pts
        except: pass
    cat  = model_name.rsplit('_',1)[0] if '_' in model_name else 'vase'
    seed = abs(hash(model_name)) % 10000
    return make_shape(cat, seed=seed)

SHAPE_NAMES = {
    'airplane': ['Cessna 172','Boeing 747 Fuselage','Spitfire Mk.IX','WW2 Fighter',
                 'Fokker Triplane','F-16 Falcon','Concorde','Biplane Trainer',
                 'Stealth Bomber','Propeller Scout'],
    'chair':    ['Eames Lounge','Barcelona Chair','Office Ergonomic','Wishbone Chair',
                 'Tulip Chair','Cantilever Chair','Rocking Chair','Bar Stool',
                 'Throne Chair','Folding Chair'],
    'table':    ['Dining Table','Coffee Table','Standing Desk','Pedestal Table',
                 'Trestle Table','Folding Table','Side Table','Glass Top Desk',
                 'Industrial Bench','Console Table'],
    'lamp':     ['Arc Floor Lamp','Pendant Cluster','Table Desk Lamp','Tripod Lamp',
                 'Mushroom Lamp','Edison Bulb Set','Sconce Pair','Gooseneck Lamp'],
    'monitor':  ['4K Ultrawide','Gaming Curved','Studio Reference','Dual-Screen Set',
                 'Portable Display','Touch Panel','LED Video Wall','Broadcast Monitor'],
    'car':      ['Vintage Roadster','Racing Coupe','Sports Fastback','Classic Muscle',
                 'Off-Road Truck','City Hatchback','Luxury Saloon','Formula Chassis'],
    'guitar':   ['Gibson Les Paul','Fender Stratocaster','Acoustic Dreadnought',
                 'Classical Nylon','Bass Guitar','Flying V','Explorer','Semi-Hollow'],
    'vase':     ['Celadon Vase','Tall Cylinder','Raku Bowl','Amphora','Gourd Form',
                 'Geometric Faceted','Asymmetric Studio','Ming Replica'],
    'sofa':     ['L-Shape Sectional','Chesterfield','Mid-Century 3-Seat','Loveseat',
                 'Futon Convertible','Cloud Sofa','Modular System','Vintage Settee'],
}

def make_shape(shape_type, n=4096, seed=0):
    rng = np.random.RandomState(seed)
    v   = seed % 4
    s   = 0.7 + (seed % 7) * 0.1

    if shape_type == 'airplane':
        if v == 0:
            pts = np.vstack([rng.randn(n//2,3)*[1.2*s,0.08*s,0.08*s],
                             rng.randn(n//3,3)*[0.15*s,1.2*s,0.03*s],
                             rng.randn(n//6,3)*[0.06*s,0.18*s,0.28*s]+[0.9*s,0,0]])
        elif v == 1:
            pts = np.vstack([rng.randn(n//2,3)*[0.9*s,0.18*s,0.18*s],
                             rng.randn(n//3,3)*[0.35*s,0.8*s,0.05*s],
                             rng.randn(n//6,3)*[0.12*s,0.22*s,0.18*s]+[0.7*s,0,0]])
        elif v == 2:
            pts = np.vstack([rng.randn(n//3,3)*[0.8*s,0.1*s,0.1*s],
                             rng.randn(n//6,3)*[0.2*s,0.9*s,0.03*s]+[0,0,0.25*s],
                             rng.randn(n//6,3)*[0.2*s,0.9*s,0.03*s]+[0,0,-0.15*s],
                             rng.randn(n//3,3)*[0.05*s,0.05*s,0.6*s]])
        else:
            pts = np.vstack([rng.randn(n//4,3)*[1.0*s,0.06*s,0.06*s],
                             rng.randn(n*3//4,3)*[0.5*s,1.0*s,0.04*s]])
    elif shape_type == 'chair':
        if v == 0:
            seat=rng.rand(n//3,3)*[s,s,0.07]+[-s/2,-s/2,0.5*s]
            back=rng.rand(n//3,3)*[s,0.07,0.8*s]+[-s/2,0.43*s,0]
            legs=np.vstack([rng.rand(n//12,3)*[0.08,0.08,0.5*s]+p
                            for p in [[-s*.44,-s*.44,0],[s*.36,-s*.44,0],[-s*.44,s*.36,0],[s*.36,s*.36,0]]])
            pts=np.vstack([seat,back,legs])
        elif v == 1:
            seat=rng.rand(n//4,3)*[1.3*s,1.1*s,0.1]+[-0.65*s,-0.55*s,0.4*s]
            back=rng.rand(n//4,3)*[1.3*s,0.1,0.9*s]+[-0.65*s,0.5*s,0]
            arms=np.vstack([rng.rand(n//8,3)*[0.12,1.1*s,0.5*s]+[-0.65*s,-0.55*s,0.4*s],
                            rng.rand(n//8,3)*[0.12,1.1*s,0.5*s]+[0.53*s,-0.55*s,0.4*s]])
            legs=np.vstack([rng.rand(n//16,3)*[0.1,0.1,0.4*s]+p
                            for p in [[-0.6*s,-0.5*s,0],[0.5*s,-0.5*s,0],[-0.6*s,0.45*s,0],[0.5*s,0.45*s,0]]])
            pts=np.vstack([seat,back,arms,legs])
        elif v == 2:
            seat=rng.rand(n//3,3)*[0.7*s,0.7*s,0.06]+[-0.35*s,-0.35*s,1.0*s]
            pole=rng.rand(n//3,3)*[0.06,0.06,1.05*s]+[-0.03,-0.03,0]
            base=np.vstack([rng.rand(n//12,3)*[0.7*s,0.08,0.05]+[-0.35*s,-0.04,0],
                            rng.rand(n//12,3)*[0.08,0.7*s,0.05]+[-0.04,-0.35*s,0]])
            pts=np.vstack([seat,pole,base])
        else:
            seat=rng.rand(n//3,3)*[1.5*s,1.1*s,0.12]+[-0.75*s,-0.55*s,0.25*s]
            back=rng.rand(n//3,3)*[1.5*s,0.1,0.55*s]+[-0.75*s,0.5*s,0.25*s]
            pts=np.vstack([seat,back,rng.rand(n//3,3)*[1.5*s,1.1*s,0.25*s]+[-0.75*s,-0.55*s,0]])
    elif shape_type == 'table':
        h = 0.45 + v * 0.12
        if v == 0:
            top=rng.rand(n*2//3,3)*[1.6*s,0.9*s,0.06]+[-0.8*s,-0.45*s,h]
            legs=np.vstack([rng.rand(n//16,3)*[0.07,0.07,h]+p
                            for p in [[-0.75*s,-0.4*s,0],[0.68*s,-0.4*s,0],[-0.75*s,0.38*s,0],[0.68*s,0.38*s,0]]])
            pts=np.vstack([top,legs])
        elif v == 1:
            theta=rng.rand(n//2)*2*np.pi; r=0.7*s
            top_pts=np.column_stack([r*np.cos(theta),r*np.sin(theta),np.ones(n//2)*h])
            pole=rng.rand(n//3,3)*[0.1,0.1,h]+[-0.05,-0.05,0]
            base=rng.rand(n//6,3)*[1.0*s,0.1,0.05]+[-0.5*s,-0.05,0]
            pts=np.vstack([top_pts,pole,base])
        elif v == 2:
            h2=0.22*s
            top=rng.rand(n*2//3,3)*[1.4*s,0.75*s,0.05]+[-0.7*s,-0.38*s,h2]
            legs=np.vstack([rng.rand(n//16,3)*[0.06,0.06,h2]+p
                            for p in [[-0.65*s,-0.33*s,0],[0.6*s,-0.33*s,0],[-0.65*s,0.3*s,0],[0.6*s,0.3*s,0]]])
            pts=np.vstack([top,legs])
        else:
            top=rng.rand(n//2,3)*[1.8*s,0.6*s,0.06]+[-0.9*s,-0.3*s,h]
            trestle=np.vstack([rng.rand(n//8,3)*[0.06,0.6*s,h]+[-0.7*s,-0.3*s,0],
                               rng.rand(n//8,3)*[0.06,0.6*s,h]+[0.65*s,-0.3*s,0]])
            cross=rng.rand(n//4,3)*[1.4*s,0.05,0.05]+[-0.7*s,-0.025,h*0.5]
            pts=np.vstack([top,trestle,cross])
    elif shape_type == 'car':
        if v == 0:
            body=rng.randn(n//2,3)*[1.3*s,0.5*s,0.28*s]
            wheels=np.vstack([rng.randn(n//8,3)*[0.07,0.3*s,0.3*s]+p
                              for p in [[-0.8*s,-0.52*s,-0.28*s],[-0.8*s,0.52*s,-0.28*s],[0.75*s,-0.52*s,-0.28*s],[0.75*s,0.52*s,-0.28*s]]])
            pts=np.vstack([body,wheels])
        elif v == 1:
            body=rng.randn(n//2,3)*[0.9*s,0.6*s,0.45*s]
            cab=rng.randn(n//4,3)*[0.4*s,0.55*s,0.55*s]+[-0.5*s,0,0.2*s]
            wheels=np.vstack([rng.randn(n//8,3)*[0.08,0.35*s,0.35*s]+p
                              for p in [[-0.7*s,-0.62*s,-0.4*s],[-0.7*s,0.62*s,-0.4*s],[0.6*s,-0.62*s,-0.4*s],[0.6*s,0.62*s,-0.4*s]]])
            pts=np.vstack([body,cab,wheels])
        elif v == 2:
            body=rng.randn(n*2//3,3)*[1.4*s,0.52*s,0.18*s]
            wheels=np.vstack([rng.randn(n//8,3)*[0.06,0.28*s,0.28*s]+p
                              for p in [[-0.85*s,-0.54*s,-0.18*s],[-0.85*s,0.54*s,-0.18*s],[0.8*s,-0.54*s,-0.18*s],[0.8*s,0.54*s,-0.18*s]]])
            pts=np.vstack([body,wheels])
        else:
            body=rng.randn(n//2,3)*[1.0*s,0.48*s,0.38*s]
            wheels=np.vstack([rng.randn(n//8,3)*[0.09,0.32*s,0.32*s]+p
                              for p in [[-0.7*s,-0.5*s,-0.35*s],[-0.7*s,0.5*s,-0.35*s],[0.65*s,-0.5*s,-0.35*s],[0.65*s,0.5*s,-0.35*s]]])
            roof=rng.randn(n//4,3)*[0.55*s,0.42*s,0.15*s]+[0,0,0.42*s]
            pts=np.vstack([body,wheels,roof])
    elif shape_type == 'lamp':
        pole=rng.rand(n//4,3)*[0.04,0.04,1.1*s]+[-0.02,-0.02,0]
        shade=rng.randn(n//2,3)*[(0.3+v*0.08)*s,(0.3+v*0.08)*s,0.2*s]+[0,0,1.0*s]
        base=rng.rand(n//4,3)*[0.5*s,0.5*s,0.05]+[-0.25*s,-0.25*s,0]
        pts=np.vstack([pole,shade,base])
    elif shape_type == 'guitar':
        if v < 2:
            pts=np.vstack([rng.randn(n//2,3)*[(0.35+v*0.05)*s,(0.5+v*0.05)*s,0.08],
                           rng.rand(n//3,3)*[0.1,1.2*s,0.06]+[-0.05,0.48*s,0],
                           rng.randn(n//6,3)*[0.15,0.12,0.04]+[0,1.55*s,0]])
        else:
            pts=np.vstack([rng.randn(n//2,3)*[0.3*s,0.6*s,0.1],
                           rng.rand(n//3,3)*[0.08,1.4*s,0.06]+[-0.04,0.55*s,0],
                           rng.randn(n//6,3)*[0.18,0.1,0.04]+[0,1.8*s,0]])
    elif shape_type == 'vase':
        theta=rng.rand(n)*2*np.pi; h=rng.rand(n)
        freq=(2+v); amp=(0.15+v*0.06)
        r=(0.3*s+amp*np.sin(theta*freq))*(1-(0.4+v*0.05)*(h-0.5)**2)
        pts=np.column_stack([r*np.cos(theta),r*np.sin(theta),h*s])
    elif shape_type == 'sofa':
        w=(1.5+v*0.2)*s
        pts=np.vstack([rng.rand(n//3,3)*[w,0.8*s,0.35]+[-w/2,-0.4*s,0],
                       rng.rand(n//3,3)*[w,0.2,0.65*s]+[-w/2,-0.38*s,0.35],
                       rng.rand(n//6,3)*[0.18,0.8*s,0.55*s]+[-w/2,-0.4*s,0],
                       rng.rand(n//6,3)*[0.18,0.8*s,0.55*s]+[w/2-0.18,-0.4*s,0]])
    elif shape_type == 'monitor':
        pts=np.vstack([rng.rand(n*2//3,3)*[(1.2+v*0.15)*s,0.06,0.8*s]+[-(0.6+v*0.075)*s,-0.03,0],
                       rng.rand(n//6,3)*[0.08,0.5*s,0.08]+[-0.04,0,-0.05],
                       rng.rand(n//6,3)*[(0.7+v*0.1)*s,0.08,0.04]+[-(0.35+v*0.05)*s,0.3*s,-0.02]])
    else:
        pts=rng.randn(n,3)

    pts=pts[:n]
    pts=(pts-pts.mean(0))/(np.sqrt((pts**2).sum(1)).max()+1e-8)
    return pts.astype(np.float32)

SELLERS = ["vertex_vault","poly_pioneer","mesh_merchant","wire_wolf",
           "depth_dealer","null_island_art","fractured_form","solid_syndicate"]
PRICES  = {"airplane":(18,45),"chair":(12,35),"table":(22,55),"lamp":(8,20),
           "car":(45,120),"guitar":(15,40),"vase":(6,18),"sofa":(25,65),
           "monitor":(30,80),"night_stand":(10,25),"plant":(5,15),"person":(20,50)}

def make_default_listings():
    entries = []

    if MODEL_CATALOG:
        for cat, names in sorted(MODEL_CATALOG.items()):
            if len(entries) >= MAX_LISTINGS:
                break
            pick = random.choice(names)
            lo, hi = PRICES.get(cat, (10, 40))
            entries.append({
                "id":       generate_trade_id(),
                "seller":   random.choice(SELLERS),
                "name":     pick,
                "category": cat,
                "price":    round(random.uniform(lo, hi), 2),
                "shape":    cat,
                "seed":     abs(hash(pick)) % 10000,
                "stego":    False,
                "sold":     False,
                "desc":     f"High-fidelity {cat} scan · 4096pt"
            })
    else:
        for cat in ['airplane','chair','table','car','vase','sofa']:
            lo, hi = PRICES.get(cat, (10,40))
            entries.append({
                "id": generate_trade_id(),"seller":random.choice(SELLERS),
                "name":f"{cat}_sample","category":cat,
                "price":round(random.uniform(lo,hi),2),
                "shape":cat,"seed":random.randint(0,9999),
                "stego":False,"sold":False,"desc":f"{cat} model"})

    return entries

FAKE_HISTORY = [
    {"buyer":"vertex_vault",   "item":"Spitfire Mk.IX",    "seller":"solid_syndicate","price":29.00,"ts":"2026-05-26 00:14","trade_id":"TRD-20260526-AX9M2BC1"},
    {"buyer":"fractured_form", "item":"Barcelona Chair",   "seller":"vertex_vault",  "price":21.00,"ts":"2026-05-25 23:47","trade_id":"TRD-20260525-KP4X7NQ2"},
    {"buyer":"wire_wolf",      "item":"Racing Coupe",      "seller":"depth_dealer",  "price":89.99,"ts":"2026-05-25 22:31","trade_id":"TRD-20260525-MN8R3CV5"},
    {"buyer":"poly_pioneer",   "item":"Prop Bundle Vol.1", "seller":"cipher_user",   "price":19.99,"ts":"2026-05-25 21:18","trade_id":"TRD-20260525-ZT6J8P3Q"},
    {"buyer":"mesh_merchant",  "item":"Celadon Vase",      "seller":"fractured_form","price":9.50, "ts":"2026-05-25 20:55","trade_id":"TRD-20260525-BW2K1LM7"},
    {"buyer":"null_island_art","item":"Gibson Les Paul",   "seller":"null_island_art","price":22.00,"ts":"2026-05-25 19:42","trade_id":"TRD-20260525-DR5Y9XN4"},
    {"buyer":"depth_dealer",   "item":"Asset Bundle S2",  "seller":"cipher_user",   "price":24.50,"ts":"2026-05-25 18:30","trade_id":"TRD-20260525-QF3T6WA8"},
    {"buyer":"solid_syndicate","item":"Arc Floor Lamp",   "seller":"wire_wolf",     "price":12.75,"ts":"2026-05-25 17:11","trade_id":"TRD-20260525-HJ7U2EB6"},
    {"buyer":"vertex_vault",   "item":"L-Shape Sectional","seller":"poly_pioneer",  "price":37.50,"ts":"2026-05-25 15:59","trade_id":"TRD-20260525-CV4N8GX3"},
    {"buyer":"fractured_form", "item":"Prop Bundle Vol.3","seller":"cipher_user",   "price":32.00,"ts":"2026-05-25 14:22","trade_id":"TRD-20260525-YP9A5MK1"},
]

def load_db():
    if not os.path.exists(DB_PATH):
        defaults = make_default_listings()
        with open(DB_PATH,'w') as f: json.dump(defaults,f,indent=2)
    with open(DB_PATH) as f: return json.load(f)

def reset_db():
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    return load_db()

def save_db(data):
    with open(DB_PATH,'w') as f: json.dump(data,f,indent=2)

def load_history():
    if not os.path.exists(HIST_PATH):
        with open(HIST_PATH,'w') as f: json.dump(FAKE_HISTORY,f,indent=2)
    with open(HIST_PATH) as f: return json.load(f)

def save_history(data):
    with open(HIST_PATH,'w') as f: json.dump(data,f,indent=2)

@app.route('/')
def index():
    return render_template('marketplace.html')

@app.route('/api/listings')
def get_listings():
    listings = [L for L in load_db() if not L.get('sold', False)]
    return jsonify(listings)

@app.route('/api/categories')
def get_categories():
    cats = sorted(MODEL_CATALOG.keys()) if MODEL_CATALOG else sorted(SHAPE_NAMES.keys())
    return jsonify(cats)

@app.route('/api/shape_names')
def shape_names():
    return jsonify(SHAPE_NAMES)

@app.route('/api/category_models')
def category_models():
    cat = request.args.get('cat', '')
    if MODEL_CATALOG and cat in MODEL_CATALOG:
        return jsonify(MODEL_CATALOG[cat])
    return jsonify(SHAPE_NAMES.get(cat, []))

@app.route('/api/bundle_pointclouds', methods=['POST'])
def bundle_pointclouds():
    data    = request.json
    shapes  = data.get('shapes', [])
    results = []
    for i, item in enumerate(shapes[:4]):
        model_name = item.get('model_name','')
        pts = load_real_pc(model_name) if model_name else make_shape(
            item.get('category','vase'),
            seed=abs(hash(model_name))%10000 or (i*137+42))
        results.append({'points': pts.tolist(), 'name': model_name})
    return jsonify(results)

@app.route('/api/listing/<lid>/pointcloud')
def get_pointcloud(lid):
    listings = load_db()
    L = next((x for x in listings if x['id']==lid), None)
    if not L: return jsonify({'error':'not found'}), 404

    if L.get('stego') and L.get('files'):
        all_pts = []
        for fname in L['files']:
            path = os.path.join(OBJECTS_DIR, fname)
            if os.path.exists(path):
                all_pts.append(np.load(path).tolist())
        if all_pts:
            return jsonify({'points': all_pts[0], 'all_points': all_pts,
                            'stego': True, 'bundle': True, 'count': len(all_pts)})

    if L.get('user_bundle') and L.get('bundle_shapes'):
        all_pts = []
        for item in L['bundle_shapes']:
            model_name = item.get('model_name','')
            pts = load_real_pc(model_name) if model_name else make_shape(
                item.get('category','vase'),
                seed=item.get('seed', abs(hash(model_name))%10000))
            all_pts.append(pts.tolist())
        return jsonify({'points': all_pts[0], 'all_points': all_pts,
                        'stego': False, 'bundle': True, 'count': len(all_pts),
                        'names': [i.get('model_name','') for i in L['bundle_shapes']]})

    model_name = L.get('model_name') or L.get('name','')
    if model_name in MODEL_PATHS:
        pts = load_real_pc(model_name)
    else:
        pts = make_shape(L.get('shape','vase'), seed=L.get('seed',0))
    return jsonify({'points': pts.tolist(), 'stego': False, 'bundle': False})

@app.route('/api/buy', methods=['POST'])
def buy_item():
    data     = request.json
    lid      = data.get('listing_id')
    buyer    = data.get('buyer', 'anonymous')
    listings = load_db()
    for L in listings:
        if L['id'] == lid:
            if L.get('sold'):
                return jsonify({'success': False, 'message': 'Already sold'})
            L['sold'] = True
            trade_id = generate_trade_id()
            L['trade_id_purchase'] = generate_trade_id()
            save_db(listings)
            history = load_history()
            history.insert(0, {
                "buyer": buyer, "item": L['name'],
                "seller": L['seller'], "price": L['price'],
                "ts": time.strftime("%Y-%m-%d %H:%M"),
                "trade_id": trade_id,
                "stego": L.get('stego', False)})
            save_history(history)
            return jsonify({'success': True, 'trade_id': trade_id,
                            'name': L['name'], 'price': L['price'],
                            'seller': L['seller']})
    return jsonify({'success': False, 'message': 'Not found'})

@app.route('/api/add_listing', methods=['POST'])
def add_listing():
    data     = request.json
    listings = load_db()
    ts_seed  = int(time.time()) % 1000 + random.randint(0,99)

    if data.get('is_bundle'):
        shapes = data.get('shapes', [])
        entry = {
            "id":       generate_trade_id(),
            "seller":   data.get('seller','anonymous'),
            "name":     data.get('name','Bundle'),
            "category": "bundle",
            "price":    float(data.get('price', 9.99)),
            "stego":    False,
            "sold":     False,
            "desc":     data.get('desc',''),
            "bundle_shapes": shapes,
            "user_bundle": True
        }
    else:
        model_name   = data.get('model_name', data.get('name', 'Untitled'))
        listing_name = data.get('listing_name', '').strip() or model_name
        seed = abs(hash(model_name)) % 10000 or ts_seed
        entry = {
            "id":         generate_trade_id(),
            "seller":     data.get('seller','anonymous'),
            "name":       listing_name,
            "model_name": model_name,
            "category":   data.get('category','vase'),
            "price":      float(data.get('price', 9.99)),
            "shape":      data.get('category','vase'),
            "seed":       seed,
            "stego":      False, "sold": False,
            "desc":       data.get('desc','')
        }
    listings.append(entry)
    save_db(listings)
    return jsonify({'success': True, 'id': entry['id']})

@app.route('/api/history')
def get_history():
    return jsonify(load_history())

@app.route('/api/reset_listings', methods=['POST'])
def reset_listings():
    listings = reset_db()
    return jsonify({'success': True, 'count': len(listings)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  POINT.MARKET  — public marketplace")
    print(f"  http://localhost:{os.environ.get('PORT', 5000)}")
    print("="*50 + "\n")
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1',
            host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', 5000)))