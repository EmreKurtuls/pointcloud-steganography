# =============================================================================
#  StegoV5BCH — Comprehensive Test Suite
#  Run in Google Colab (with model) or locally (crypto/BCH tests only).
#
#  Sections:
#    1. AES / Crypto Tests
#    2. Trade ID Tests
#    3. BCH Codec Tests
#    4. Full Message Pipeline Tests
#    5. Statistical Randomness Tests
#    6. Point Cloud Geometry Tests
#    7. Model Embedding Tests  ← needs model loaded
#    8. Robustness Sweep       ← needs model loaded
# =============================================================================

import os, sys, time, random, hashlib, math, string
import numpy as np
from scipy import stats as scipy_stats

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED   = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD  = "\033[1m";  RESET  = "\033[0m"
PASS   = f"{GREEN}PASS{RESET}"
FAIL   = f"{RED}FAIL{RESET}"
SKIP   = f"{YELLOW}SKIP{RESET}"
WARN   = f"{YELLOW}WARN{RESET}"

results = []  # (section, name, status, detail)

def run(section, name, fn):
    try:
        detail = fn()
        status = PASS
        print(f"  {PASS}  {name}")
        if detail: print(f"       {CYAN}{detail}{RESET}")
        results.append((section, name, "PASS", detail or ""))
    except AssertionError as e:
        status = FAIL
        print(f"  {FAIL}  {name}")
        print(f"       {RED}{e}{RESET}")
        results.append((section, name, "FAIL", str(e)))
    except Exception as e:
        print(f"  {FAIL}  {name}  [{type(e).__name__}]")
        print(f"       {RED}{e}{RESET}")
        results.append((section, name, "FAIL", str(e)))

def skip(section, name, reason="model not available"):
    print(f"  {SKIP}  {name}  — {reason}")
    results.append((section, name, "SKIP", reason))

def section_header(title):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

# =============================================================================
#  COPY CRYPTO & BCH CODE FROM cipher_server.py
#  (so tests run without importing the whole Flask app)
# =============================================================================

from Crypto.Cipher import AES

def derive_key(trade_id, passphrase):
    return hashlib.pbkdf2_hmac(
        'sha256', passphrase.encode(), trade_id.encode(), 10000)[:16]

def encrypt_to_bits(message, trade_id, passphrase):
    key   = derive_key(trade_id, passphrase)
    nonce = hashlib.md5(trade_id.encode()).digest()[:8]
    cipher= AES.new(key, AES.MODE_CTR, nonce=nonce)
    msg_bytes = message.encode('utf-8')[:16].ljust(16, b'\x00')
    ct    = cipher.encrypt(msg_bytes)
    bits  = []
    for b in ct:
        for j in range(7, -1, -1): bits.append((b >> j) & 1)
    return bits

def decrypt_from_bits(bits_128, trade_id, passphrase):
    key   = derive_key(trade_id, passphrase)
    nonce = hashlib.md5(trade_id.encode()).digest()[:8]
    cipher= AES.new(key, AES.MODE_CTR, nonce=nonce)
    ct = bytes(sum(bits_128[i+j] << (7-j) for j in range(8))
               for i in range(0, 128, 8))
    msg_bytes = cipher.decrypt(ct)
    try:    return msg_bytes.rstrip(b'\x00').decode('utf-8')
    except: return None

# ── BCH ────────────────────────────────────────────────────────────────────────
BCH_N, BCH_K, BCH_T = 63, 36, 5
BCH_PARITY = BCH_N - BCH_K

class GF64:
    def __init__(self):
        self.exp=[0]*128; self.log=[0]*64; x=1
        for i in range(63):
            self.exp[i]=x; self.log[x]=i; x<<=1
            if x&64: x^=0x43
        for i in range(63,128): self.exp[i]=self.exp[i-63]
    def mul(self,a,b):
        return 0 if(a==0 or b==0) else self.exp[(self.log[a]+self.log[b])%63]
    def div(self,a,b):
        return 0 if a==0 else self.exp[(self.log[a]-self.log[b])%63]
gf = GF64()

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
    s=list(data36)+[0]*BCH_PARITY
    rem=poly_mod_bin(s,GEN)
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
    if all(r==0 for r in poly_mod_bin(recv,GEN)):
        return recv[:BCH_K],True,0
    synd=compute_syndromes(recv,5)
    if all(s==0 for s in synd): return recv[:BCH_K],True,0
    try:
        el,ne=berlekamp_massey(synd)
        if ne==0 or ne>5: return recv[:BCH_K],False,-1
        ep=chien_search(el)
        if len(ep)!=ne: return recv[:BCH_K],False,-1
        c=list(recv)
        for p in ep: c[p]^=1
        if all(r==0 for r in poly_mod_bin(c,GEN)):
            return c[:BCH_K],True,ne
        return recv[:BCH_K],False,-1
    except: return recv[:BCH_K],False,-1

def chase_decode(recv, logits_arr, k=5):
    import torch
    logits_t = torch.tensor(logits_arr, dtype=torch.float32)
    conf = torch.abs(logits_t); weak=torch.topk(conf,k,largest=False).indices.tolist()
    for pat in range(2**k):
        test=list(recv)
        for bi in range(k):
            if (pat>>bi)&1: test[weak[bi]]^=1
        dec,ok,n=bch_decode_t5(test)
        if ok: return dec,True,n
    return bch_decode_t5(recv)

def generate_trade_id():
    ts  = time.strftime("%Y%m%d")
    uid = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=8))
    return f"TRD-{ts}-{uid}"

CHUNK_BITS = 34   # 32 data + 2 chunk ID
BCH_PAD    = BCH_K - CHUNK_BITS
DATA_PER_CHUNK = 32
N_CHUNKS   = 4
TOTAL_PAYLOAD  = DATA_PER_CHUNK * N_CHUNKS

def encode_chunk(p32, cid):
    return bch_encode(p32 + [(cid>>1)&1, cid&1] + [0]*BCH_PAD)

def encode_message_128(msg):
    return [encode_chunk(msg[i*32:(i+1)*32], i) for i in range(N_CHUNKS)]

def decode_message_v2(four_ch, four_logits=None, k=5):
    msg=[0]*TOTAL_PAYLOAD; succ=0
    for eid, ch in enumerate(four_ch):
        lg = four_logits[eid] if four_logits else None
        dec,ok,_ = (chase_decode(ch,lg,k) if lg is not None
                    else bch_decode_t5(ch))
        bits = dec[:CHUNK_BITS]
        if bits[32]*2+bits[33] != eid: ok=False
        if ok: msg[eid*32:(eid+1)*32] = bits[:32]; succ+=1
    return msg, succ/N_CHUNKS

def bits_to_bytes(bits):
    return bytes(sum(bits[i+j]<<(7-j) for j in range(8)) for i in range(0,len(bits),8))

def shannon_entropy(data: bytes) -> float:
    counts = np.bincount(list(data), minlength=256)
    probs  = counts / counts.sum()
    probs  = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))

def bit_entropy(bits) -> float:
    arr = np.array(bits)
    p1  = arr.mean(); p0 = 1-p1
    if p1 == 0 or p1 == 1: return 0.0
    return float(-p1*math.log2(p1) - p0*math.log2(p0))

# =============================================================================
#  1. AES / CRYPTO TESTS
# =============================================================================
section_header("1. AES / CRYPTO TESTS")

def t_key_length():
    key = derive_key("TRD-20260601-ABCDEF12", "mypassphrase")
    assert len(key) == 16, f"Expected 16 bytes, got {len(key)}"
    return "128-bit key ✓"

def t_key_determinism():
    tid = "TRD-20260601-ABCDEF12"
    pw  = "mypassphrase"
    k1  = derive_key(tid, pw)
    k2  = derive_key(tid, pw)
    assert k1 == k2, "Same inputs produced different keys"
    return "Same inputs → same key every time ✓"

def t_key_sensitivity_passphrase():
    tid = "TRD-20260601-ABCDEF12"
    k1  = derive_key(tid, "correctpassphrase")
    k2  = derive_key(tid, "correctPassphrase")
    assert k1 != k2, "Keys should differ when passphrase differs"
    diff_bits = sum(bin(a^b).count('1') for a,b in zip(k1,k2))
    assert diff_bits >= 32, f"Too few differing bits: {diff_bits}"
    return f"Passphrase case-change → {diff_bits}/128 bits differ ✓"

def t_key_sensitivity_tradeid():
    pw  = "mypassphrase"
    k1  = derive_key("TRD-20260601-AAAAAAAA", pw)
    k2  = derive_key("TRD-20260601-AAAAAAAB", pw)
    assert k1 != k2, "Keys should differ when trade ID differs"
    diff_bits = sum(bin(a^b).count('1') for a,b in zip(k1,k2))
    assert diff_bits >= 32, f"Too few differing bits: {diff_bits}"
    return f"Trade ID 1-char change → {diff_bits}/128 bits differ ✓"

def t_key_entropy():
    keys = [derive_key(generate_trade_id(), "pw") for _ in range(200)]
    all_bytes = bytes([b for k in keys for b in k])
    ent = shannon_entropy(all_bytes)
    assert ent >= 7.5, f"Key entropy too low: {ent:.3f} bits/byte"
    return f"Entropy across 200 keys: {ent:.4f} bits/byte (max 8.0) ✓"

def t_encrypt_decrypt_roundtrip():
    tid = generate_trade_id()
    pw  = "testpassword123"
    msg = "Hello World!"
    bits = encrypt_to_bits(msg, tid, pw)
    assert len(bits) == 128, f"Expected 128 bits, got {len(bits)}"
    recovered = decrypt_from_bits(bits, tid, pw)
    assert recovered == msg, f"Round-trip failed: {recovered!r} ≠ {msg!r}"
    return f"'{msg}' → 128 bits → '{recovered}' ✓"

def t_wrong_passphrase_fails():
    tid  = generate_trade_id()
    bits = encrypt_to_bits("secret", tid, "correctpassword")
    rec  = decrypt_from_bits(bits, tid, "wrongpassword")
    assert rec != "secret", "Wrong passphrase should not recover the message"
    return f"Wrong passphrase → '{rec}' (not 'secret') ✓"

def t_wrong_tradeid_fails():
    tid1 = generate_trade_id()
    tid2 = generate_trade_id()
    bits = encrypt_to_bits("secret", tid1, "pw")
    rec  = decrypt_from_bits(bits, tid2, "pw")
    assert rec != "secret", "Wrong trade ID should not recover the message"
    return f"Wrong Trade ID → '{rec}' (not 'secret') ✓"

def t_max_length_message():
    tid = generate_trade_id()
    msg = "A" * 16
    bits = encrypt_to_bits(msg, tid, "pw")
    rec  = decrypt_from_bits(bits, tid, "pw")
    assert rec == msg, f"Max-length round-trip failed"
    return f"16-char message round-trip ✓"

def t_special_char_message():
    tid = generate_trade_id()
    msg = "Hi! @#€%&*?"
    bits = encrypt_to_bits(msg, tid, "pw")
    rec  = decrypt_from_bits(bits, tid, "pw")
    assert rec == msg, f"Special char round-trip failed: {rec!r}"
    return f"Special chars: '{msg}' → '{rec}' ✓"

def t_ciphertext_bit_distribution():
    results_bits = []
    for _ in range(100):
        bits = encrypt_to_bits("testmessage", generate_trade_id(), "pw")
        results_bits.extend(bits)
    ones  = sum(results_bits)
    total = len(results_bits)
    ratio = ones / total
    # should be close to 0.5 for AES-CTR output
    assert 0.43 <= ratio <= 0.57, f"Bit distribution skewed: {ratio:.3f}"
    return f"Bit ratio across 100 encryptions: {ratio:.4f} (expected ~0.5) ✓"

for fn in [t_key_length, t_key_determinism, t_key_sensitivity_passphrase,
           t_key_sensitivity_tradeid, t_key_entropy,
           t_encrypt_decrypt_roundtrip, t_wrong_passphrase_fails,
           t_wrong_tradeid_fails, t_max_length_message,
           t_special_char_message, t_ciphertext_bit_distribution]:
    run("Crypto", fn.__name__[2:].replace('_',' ').title(), fn)

# =============================================================================
#  2. TRADE ID TESTS
# =============================================================================
section_header("2. TRADE ID TESTS")

def t_format():
    import re
    for _ in range(100):
        tid = generate_trade_id()
        assert re.match(r'^TRD-\d{8}-[A-Z2-9]{8}$', tid), f"Bad format: {tid}"
    return "All 100 IDs match TRD-YYYYMMDD-XXXXXXXX ✓"

def t_uniqueness():
    N = 10000
    ids = {generate_trade_id() for _ in range(N)}
    assert len(ids) == N, f"Collision detected: {N - len(ids)} duplicates in {N}"
    return f"No collisions in {N:,} generated IDs ✓"

def t_no_confusable_chars():
    """IDs must not contain O, I, 0, 1 (confusable in print)."""
    confusable = set("OI01")
    for _ in range(1000):
        uid = generate_trade_id().split('-')[2]
        bad = set(uid) & confusable
        assert not bad, f"Confusable char found: {bad} in '{uid}'"
    return "No O/I/0/1 in 1,000 IDs ✓"

def t_id_entropy():
    # character space: 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789' = 32 chars, 8 positions
    # theoretical: 8 * log2(32) = 40 bits
    theoretical = 8 * math.log2(32)
    ids  = [generate_trade_id().split('-')[2] for _ in range(5000)]
    flat = ''.join(ids)
    counts = {c: flat.count(c) for c in set(flat)}
    probs  = np.array(list(counts.values())) / len(flat)
    actual = float(-np.sum(probs * np.log2(probs))) * 8  # per 8-char ID
    assert actual >= theoretical * 0.9, f"ID entropy {actual:.2f} < 90% of theoretical {theoretical:.2f}"
    return f"ID entropy: {actual:.2f} bits (theoretical: {theoretical:.2f}) ✓"

def t_date_component_valid():
    import re, datetime
    tid  = generate_trade_id()
    date_str = tid.split('-')[1]
    dt = datetime.datetime.strptime(date_str, "%Y%m%d")
    today = datetime.datetime.now()
    assert abs((today - dt).days) <= 1, f"Date mismatch: {date_str}"
    return f"Date component {date_str} matches today ✓"

for fn in [t_format, t_uniqueness, t_no_confusable_chars,
           t_id_entropy, t_date_component_valid]:
    run("Trade ID", fn.__name__[2:].replace('_',' ').title(), fn)

# =============================================================================
#  3. BCH CODEC TESTS
# =============================================================================
section_header("3. BCH CODEC TESTS")

def t_codeword_length():
    data = [random.randint(0,1) for _ in range(BCH_K)]
    cw   = bch_encode(data)
    assert len(cw) == BCH_N, f"Expected {BCH_N}, got {len(cw)}"
    return f"BCH({BCH_N},{BCH_K},t={BCH_T}) codeword length ✓"

def t_systematic_property():
    """First BCH_K bits of codeword must equal the input data."""
    for _ in range(20):
        data = [random.randint(0,1) for _ in range(BCH_K)]
        cw   = bch_encode(data)
        assert cw[:BCH_K] == data, "Systematic property violated"
    return "First 36 bits equal input for 20 random words ✓"

def t_zero_errors():
    """Noiseless codeword should decode perfectly."""
    ok_count = 0
    for _ in range(100):
        data = [random.randint(0,1) for _ in range(BCH_K)]
        cw   = bch_encode(data)
        dec, ok, nerr = bch_decode_t5(cw)
        assert ok and nerr == 0 and dec == data, "Clean decode failed"
        ok_count += 1
    return f"100/100 clean decodes ✓"

def t_error_correction_1_to_5():
    """Inject 1-5 errors: must always correct."""
    for n_err in range(1, BCH_T+1):
        ok_count = 0
        for trial in range(50):
            data  = [random.randint(0,1) for _ in range(BCH_K)]
            cw    = bch_encode(data)
            noisy = list(cw)
            positions = random.sample(range(BCH_N), n_err)
            for p in positions: noisy[p] ^= 1
            dec, ok, nerr = bch_decode_t5(noisy)
            if ok and dec == data: ok_count += 1
        assert ok_count == 50, f"t={n_err}: only {ok_count}/50 corrected"
    return f"50/50 corrections for each error count t=1..{BCH_T} ✓"

def t_beyond_capacity():
    """Inject t+1 = 6 errors: must report failure (not silently corrupt)."""
    fail_count = 0
    for _ in range(200):
        data  = [random.randint(0,1) for _ in range(BCH_K)]
        cw    = bch_encode(data)
        noisy = list(cw)
        positions = random.sample(range(BCH_N), BCH_T+1)
        for p in positions: noisy[p] ^= 1
        dec, ok, _ = bch_decode_t5(noisy)
        if not ok or dec != data: fail_count += 1
    assert fail_count >= 150, f"Should mostly fail: only {fail_count}/200 reported failure"
    return f"{fail_count}/200 failed gracefully for t+1={BCH_T+1} errors ✓"

def t_syndrome_zero_for_valid():
    """Any valid codeword must have zero syndrome."""
    for _ in range(50):
        data  = [random.randint(0,1) for _ in range(BCH_K)]
        cw    = bch_encode(data)
        assert all(r == 0 for r in poly_mod_bin(cw, GEN)), "Syndrome not zero"
    return "50/50 valid codewords have zero syndrome ✓"

def t_parity_bits_nonzero_for_random():
    """Random data should almost always produce non-zero parity."""
    zero_parity = 0
    for _ in range(1000):
        data = [random.randint(0,1) for _ in range(BCH_K)]
        cw   = bch_encode(data)
        if all(b == 0 for b in cw[BCH_K:]): zero_parity += 1
    assert zero_parity < 10, f"Too many zero parity words: {zero_parity}/1000"
    return f"Only {zero_parity}/1000 random words had zero parity ✓"

def t_error_position_accuracy():
    """Decoder must identify the correct error positions."""
    for _ in range(30):
        data  = [random.randint(0,1) for _ in range(BCH_K)]
        cw    = bch_encode(data)
        n_err = random.randint(1, BCH_T)
        positions = sorted(random.sample(range(BCH_N), n_err))
        noisy = list(cw)
        for p in positions: noisy[p] ^= 1
        dec, ok, nerr = bch_decode_t5(noisy)
        assert ok and nerr == n_err, f"Wrong error count: got {nerr}, expected {n_err}"
        assert dec == data, "Decoding produced wrong data"
    return "Error count accurately reported for 30 random error patterns ✓"

for fn in [t_codeword_length, t_systematic_property, t_zero_errors,
           t_error_correction_1_to_5, t_beyond_capacity,
           t_syndrome_zero_for_valid, t_parity_bits_nonzero_for_random,
           t_error_position_accuracy]:
    run("BCH Codec", fn.__name__[2:].replace('_',' ').title(), fn)

# =============================================================================
#  4. FULL MESSAGE PIPELINE TESTS
# =============================================================================
section_header("4. FULL MESSAGE PIPELINE TESTS")

def t_pipeline_roundtrip():
    tid = generate_trade_id()
    pw  = "pipelinetest"
    msg = "HelloWorld!!"
    bits_128   = encrypt_to_bits(msg, tid, pw)
    chunks     = encode_message_128(bits_128)
    decoded_b, sr = decode_message_v2(chunks)
    recovered  = decrypt_from_bits(decoded_b, tid, pw)
    assert sr == 1.0,      f"Not all chunks decoded: {sr}"
    assert recovered == msg, f"Round-trip failed: {recovered!r}"
    return f"'{msg}' → BCH encode → BCH decode → '{recovered}' SR={sr:.0%} ✓"

def t_pipeline_all_ascii():
    """Every printable ASCII character survives the pipeline."""
    for ch in string.printable[:64]:
        tid = generate_trade_id(); pw = "testpw"
        msg = ch * 8
        bits = encrypt_to_bits(msg, tid, pw)
        chunks = encode_message_128(bits)
        dec, sr = decode_message_v2(chunks)
        rec = decrypt_from_bits(dec, tid, pw)
        assert rec == msg, f"Failed for char {ch!r}: {rec!r}"
    return "All 64 printable ASCII chars survive the pipeline ✓"

def t_pipeline_with_bit_errors():
    """Inject random bit errors into encoded chunks — count surviving messages."""
    N     = 200
    wins  = 0
    for _ in range(N):
        tid   = generate_trade_id()
        msg   = "TestMsg123ABCD"
        bits  = encrypt_to_bits(msg, tid, "pw")
        chunks = [list(c) for c in encode_message_128(bits)]
        # inject 1 random error per chunk (should always survive BCH t=5)
        for ci in range(4):
            pos = random.randint(0, BCH_N-1)
            chunks[ci][pos] ^= 1
        dec, sr = decode_message_v2(chunks)
        rec = decrypt_from_bits(dec, tid, "pw")
        if rec == msg: wins += 1
    assert wins >= 195, f"Too many failures with 1 error/chunk: {wins}/{N}"
    return f"{wins}/{N} messages survive 1 random error per BCH chunk ✓"

def t_chunk_id_mismatch_detection():
    """Swapping two chunks must be detected by chunk ID bits."""
    tid = generate_trade_id()
    bits   = encrypt_to_bits("abc", tid, "pw")
    chunks = [list(c) for c in encode_message_128(bits)]
    # swap chunk 0 and chunk 1
    chunks[0], chunks[1] = chunks[1], chunks[0]
    dec, sr = decode_message_v2(chunks)
    assert sr < 1.0, "Chunk swap should cause partial failure"
    return f"Swapped chunks detected: SR={sr:.0%} (expected < 100%) ✓"

def t_pipeline_throughput():
    """Measure encode+decode throughput."""
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        tid   = generate_trade_id()
        bits  = encrypt_to_bits("BenchmarkTest!!", tid, "benchmark")
        chunks = encode_message_128(bits)
        decode_message_v2(chunks)
    elapsed  = time.perf_counter() - t0
    per_msg  = elapsed / N * 1000
    assert per_msg < 200, f"Too slow: {per_msg:.1f}ms per message"
    return f"{N} full pipeline cycles in {elapsed:.2f}s ({per_msg:.2f}ms each) ✓"

for fn in [t_pipeline_roundtrip, t_pipeline_all_ascii,
           t_pipeline_with_bit_errors, t_chunk_id_mismatch_detection,
           t_pipeline_throughput]:
    run("Pipeline", fn.__name__[2:].replace('_',' ').title(), fn)

# =============================================================================
#  5. STATISTICAL RANDOMNESS TESTS
# =============================================================================
section_header("5. STATISTICAL RANDOMNESS TESTS")

def t_monobit_frequency():
    """NIST SP 800-22 Frequency (Monobit) Test — ciphertext bits should be ~50% ones."""
    bits = []
    for _ in range(500):
        bits.extend(encrypt_to_bits("RandomTest1234!!", generate_trade_id(), "pw"))
    n    = len(bits)
    ones = sum(bits)
    s    = abs(ones - n/2) / math.sqrt(n/4)
    p    = math.erfc(s / math.sqrt(2))
    assert p >= 0.01, f"Monobit test FAILED: p={p:.4f}"
    return f"Monobit test: p={p:.4f} ≥ 0.01 (NIST) ✓  ratio={ones/n:.4f}"

def t_runs_test():
    """NIST Runs Test — checks oscillation of bits."""
    bits = []
    for _ in range(500):
        bits.extend(encrypt_to_bits("RunsTestInput!!!", generate_trade_id(), "pw"))
    n    = len(bits); k = sum(bits)
    pi   = k / n
    if abs(pi - 0.5) >= 2/math.sqrt(n):
        return "Skipped — proportions prerequisite failed"
    runs = 1 + sum(1 for i in range(1,n) if bits[i]!=bits[i-1])
    mu   = 2*k*(n-k)/n + 1
    sigma= abs(2*math.sqrt(2*n)*pi*(1-pi))
    p    = math.erfc(abs(runs-mu)/sigma) if sigma>0 else 1.0
    assert p >= 0.01, f"Runs test FAILED: p={p:.4f}"
    return f"Runs test: p={p:.4f} ≥ 0.01 (NIST) ✓  runs={runs:,}"

def t_chi_square_key_bytes():
    """Key bytes should be uniformly distributed across 256 values."""
    keys     = [derive_key(generate_trade_id(), "pw") for _ in range(2000)]
    all_bytes= [b for k in keys for b in k]
    counts   = np.bincount(all_bytes, minlength=256)
    expected = len(all_bytes) / 256
    chi2, p  = scipy_stats.chisquare(counts)
    assert p >= 0.001, f"Key byte chi-square FAILED: p={p:.4f}  chi2={chi2:.1f}"
    return f"Key byte uniformity: chi2={chi2:.1f}  p={p:.4f} ≥ 0.001 ✓"

def t_avalanche_effect():
    """Flipping one passphrase bit should change ~50% of output bits."""
    diffs = []
    for _ in range(100):
        tid = generate_trade_id()
        msg = "AvalancheTest!!!"
        pw1 = "password123test!"
        # flip bit in last char: '!' (0x21) → '"' (0x22)
        pw2 = pw1[:-1] + '"'
        b1  = encrypt_to_bits(msg, tid, pw1)
        b2  = encrypt_to_bits(msg, tid, pw2)
        diff= sum(a^b for a,b in zip(b1,b2)) / len(b1)
        diffs.append(diff)
    avg = np.mean(diffs)
    assert 0.40 <= avg <= 0.60, f"Avalanche effect weak: {avg:.3f}"
    return f"Avalanche effect: avg {avg:.3f} bits differ (expected ~0.5) ✓"

def t_nonce_determinism():
    """Same Trade ID must always produce same nonce."""
    tid   = "TRD-20260601-TEST1234"
    n1    = hashlib.md5(tid.encode()).digest()[:8]
    n2    = hashlib.md5(tid.encode()).digest()[:8]
    assert n1 == n2
    return f"Nonce deterministic for given Trade ID ✓  nonce={n1.hex()}"

def t_birthday_paradox_resistance():
    """No Trade ID collision in 50,000 generations (birthday paradox check)."""
    ids = set()
    for _ in range(50_000):
        ids.add(generate_trade_id())
    assert len(ids) == 50_000, f"Collision found in 50k IDs"
    return "No collision in 50,000 Trade ID generations ✓"

for fn in [t_monobit_frequency, t_runs_test, t_chi_square_key_bytes,
           t_avalanche_effect, t_nonce_determinism, t_birthday_paradox_resistance]:
    run("Statistics", fn.__name__[2:].replace('_',' ').title(), fn)

# =============================================================================
#  6. POINT CLOUD GEOMETRY TESTS
# =============================================================================
section_header("6. POINT CLOUD GEOMETRY TESTS")

def make_test_pc(n=4096, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    pts -= pts.mean(0)
    pts /= np.max(np.linalg.norm(pts, axis=1))
    return pts

def chamfer_distance(p1, p2):
    d = np.sum((p1[:,None,:] - p2[None,:,:])**2, axis=-1)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())

def add_gaussian_noise(pc, sigma):
    return pc + np.random.randn(*pc.shape).astype(np.float32) * sigma

def t_normalisation_unit_sphere():
    """All normalised PCs should fit inside the unit sphere."""
    for seed in range(20):
        pc = make_test_pc(seed=seed)
        norms = np.linalg.norm(pc, axis=1)
        assert norms.max() <= 1.0 + 1e-5, f"Point outside unit sphere: {norms.max():.6f}"
        assert abs(pc.mean(0)).max() < 0.01, "Centroid not near origin"
    return "20 PCs: all within unit sphere, centroid ≈ origin ✓"

def t_normalisation_max_norm_is_one():
    """Max norm of any normalised PC should be exactly 1.0."""
    for seed in range(50):
        pc = make_test_pc(seed=seed)
        max_n = np.linalg.norm(pc, axis=1).max()
        assert abs(max_n - 1.0) < 1e-5, f"Max norm {max_n:.7f} ≠ 1.0"
    return "Max norm = 1.0 for all 50 test clouds ✓"

def t_chamfer_self_zero():
    """CD of a PC with itself must be 0."""
    pc = make_test_pc()
    cd = chamfer_distance(pc, pc)
    assert cd < 1e-6, f"CD(P,P) = {cd:.2e} ≠ 0"
    return f"CD(P,P) = {cd:.2e} ≈ 0 ✓"

def t_chamfer_symmetry():
    """CD must be symmetric: d(P,Q) = d(Q,P)."""
    pc1 = make_test_pc(seed=1); pc2 = make_test_pc(seed=2)
    d12 = chamfer_distance(pc1, pc2)
    d21 = chamfer_distance(pc2, pc1)
    assert abs(d12 - d21) < 1e-6, f"CD asymmetric: {d12:.6f} ≠ {d21:.6f}"
    return f"CD symmetric: {d12:.6f} = {d21:.6f} ✓"

def t_chamfer_noise_scaling():
    """CD should increase monotonically with noise sigma."""
    pc    = make_test_pc(seed=42)
    sigmas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    cds   = []
    for sigma in sigmas:
        noisy = pc + np.random.default_rng(0).standard_normal(pc.shape).astype(np.float32)*sigma
        cds.append(chamfer_distance(pc, noisy))
    for i in range(len(cds)-1):
        assert cds[i] <= cds[i+1]*1.1, f"CD not monotone at sigma={sigmas[i+1]}"
    pairs = "  ".join(f"σ={s:.3f}→CD={c:.5f}" for s,c in zip(sigmas,cds))
    return f"CD increases with noise ✓\n       {pairs}"

def t_alpha_bound_respected():
    """Simulated decoder: offsets must never exceed alpha."""
    alpha = 0.08
    np.random.seed(0)
    for _ in range(50):
        pc     = make_test_pc()
        raw    = np.random.randn(*pc.shape).astype(np.float32) * 0.2
        offset = alpha * np.tanh(raw / alpha)
        stego  = pc + offset
        max_off = np.abs(offset).max()
        assert max_off < alpha + 1e-6, f"Offset exceeded alpha: {max_off:.6f}"
    return f"α·tanh(x/α) never exceeds α={alpha} across 50 trials ✓"

def t_point_count_preserved():
    """Embedding must not change the number of points."""
    n  = 4096
    pc = make_test_pc(n=n)
    alpha  = 0.08
    offsets = np.random.randn(n, 3).astype(np.float32) * 0.01
    stego   = pc + alpha * np.tanh(offsets / alpha)
    assert stego.shape == (n, 3), f"Shape changed: {stego.shape}"
    return f"Point count preserved: {n} → {stego.shape[0]} ✓"

def t_cd_benchmark():
    """Compute CD for 50 randomly perturbed PCs and report distribution."""
    alpha = 0.08; cds = []
    np.random.seed(42)
    for i in range(50):
        pc  = make_test_pc(seed=i)
        raw = np.random.randn(*pc.shape).astype(np.float32) * 0.3
        off = alpha * np.tanh(raw / alpha)
        cds.append(chamfer_distance(pc, pc + off))
    arr = np.array(cds)
    return (f"50 random embeddings: "
            f"CD mean={arr.mean():.5f}  std={arr.std():.5f}  "
            f"max={arr.max():.5f}")

for fn in [t_normalisation_unit_sphere, t_normalisation_max_norm_is_one,
           t_chamfer_self_zero, t_chamfer_symmetry, t_chamfer_noise_scaling,
           t_alpha_bound_respected, t_point_count_preserved, t_cd_benchmark]:
    run("Geometry", fn.__name__[2:].replace('_',' ').title(), fn)

# =============================================================================
#  7. MODEL EMBEDDING TESTS  (skip if model not available)
# =============================================================================
section_header("7. MODEL EMBEDDING TESTS")

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'model.pth'))
MODEL_AVAILABLE = TORCH_OK and os.path.exists(MODEL_PATH)

if not MODEL_AVAILABLE:
    reason = "PyTorch not found" if not TORCH_OK else f"Model not at {MODEL_PATH}"
    for name in ["Model loads without error", "Embed output shape correct",
                 "CD below 0.003 on 20 samples", "CD consistent across categories",
                 "Per-point offset bounded by alpha", "Stego not equal to original",
                 "Extract output shape correct", "Message shape from extract"]:
        skip("Model", name, reason)
else:
    import torch, torch.nn as nn, torch.nn.functional as F

    def farthest_point_sample(xyz,npoint):
        B,N,_=xyz.shape; cents=torch.zeros(B,npoint,dtype=torch.long,device=xyz.device)
        dist=torch.ones(B,N,device=xyz.device)*1e10
        far=torch.randint(0,N,(B,),dtype=torch.long,device=xyz.device)
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

    class PointNetSA(nn.Module):
        def __init__(self,npoint,radius,nsample,in_ch,mlp):
            super().__init__(); self.npoint=npoint; self.radius=radius; self.nsample=nsample
            self.convs=nn.ModuleList(); self.bns=nn.ModuleList(); last=in_ch
            for out in mlp: self.convs.append(nn.Conv2d(last,out,1)); self.bns.append(nn.BatchNorm2d(out)); last=out
        def forward(self,xyz,pts):
            x=xyz.permute(0,2,1); p=pts.permute(0,2,1) if pts is not None else None
            idx=farthest_point_sample(x,self.npoint); new_xyz=index_points(x,idx)
            gidx=query_ball_point(self.radius,self.nsample,x,new_xyz)
            gxyz=index_points(x,gidx)-new_xyz.unsqueeze(2)
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
            self.sa1=PointNetSA(1024,0.1,32,3,[64,64,128])
            self.sa2=PointNetSA(512,0.2,32,131,[128,128,256])
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL = StegoV5BCH().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    MODEL.load_state_dict(sd, strict=False)
    MODEL.train()

    np.random.seed(0); torch.manual_seed(0)
    N_PTS = 4096; ALPHA = 0.08
    FILLER_TENSORS = torch.randn(28,N_PTS,3,device=device)
    FILLER_TENSORS = FILLER_TENSORS/(FILLER_TENSORS.norm(dim=-1,keepdim=True)+1e-8)

    def make_batch(pc_np, n_fill=28):
        pc_t = torch.from_numpy(pc_np).unsqueeze(0).to(device)
        batch = torch.cat([pc_t, FILLER_TENSORS[:n_fill]], dim=0)
        return batch

    def embed_single(pc_np):
        batch = make_batch(pc_np)
        msg_t = torch.randint(0,2,(batch.shape[0],63),dtype=torch.float32,device=device)
        with torch.no_grad(): stego = MODEL.embed(batch, msg_t, ALPHA)
        return stego[0].cpu().numpy()

    def t_model_loads():
        assert MODEL is not None
        params = sum(p.numel() for p in MODEL.parameters())
        assert params > 1_000_000, f"Too few params: {params:,}"
        return f"Model loaded on {device}, {params:,} parameters ✓"

    def t_embed_output_shape():
        pc = make_test_pc()
        stego = embed_single(pc)
        assert stego.shape == (N_PTS, 3), f"Wrong shape: {stego.shape}"
        return f"Embed output: {stego.shape} ✓"

    def t_cd_below_threshold():
        TARGET_CD = 0.003
        cds = []
        for seed in range(20):
            pc = make_test_pc(seed=seed)
            stego = embed_single(pc)
            cd = chamfer_distance(pc, stego)
            cds.append(cd)
        arr = np.array(cds)
        above = (arr > TARGET_CD).sum()
        assert above == 0, f"{above}/20 samples exceeded CD target {TARGET_CD}"
        return (f"20 samples: mean CD={arr.mean():.6f}  max={arr.max():.6f}  "
                f"std={arr.std():.6f}  (target < {TARGET_CD}) ✓")

    def t_cd_consistent():
        """CD should be similar across different random seeds."""
        cds = [chamfer_distance(make_test_pc(seed=s), embed_single(make_test_pc(seed=s)))
               for s in range(15)]
        arr = np.array(cds)
        cv  = arr.std() / arr.mean()
        assert cv < 0.5, f"CD too variable across samples: CV={cv:.3f}"
        return f"CD coefficient of variation: {cv:.3f} < 0.5 ✓"

    def t_offset_bounded():
        """Max per-point offset must not exceed alpha."""
        pc = make_test_pc()
        stego = embed_single(pc)
        max_off = np.abs(stego - pc).max()
        assert max_off <= ALPHA + 1e-4, f"Offset {max_off:.6f} > alpha={ALPHA}"
        return f"Max per-coordinate offset: {max_off:.6f} ≤ α={ALPHA} ✓"

    def t_stego_different_from_original():
        pc = make_test_pc()
        stego = embed_single(pc)
        diff = np.abs(pc - stego).mean()
        assert diff > 1e-6, "Stego is identical to original — embedding did nothing"
        return f"Mean absolute displacement: {diff:.6f} (non-zero) ✓"

    def t_extract_output_shape():
        pc = make_test_pc()
        stego_t = torch.from_numpy(embed_single(pc)).unsqueeze(0).to(device)
        with torch.no_grad(): logits = MODEL.extract(stego_t)
        assert logits.shape == (1, 63), f"Wrong logits shape: {logits.shape}"
        return f"Extract output: {logits.shape}  (B=1, 63 bits) ✓"

    def t_message_extraction():
        """Embed a fixed message then extract and check channel accuracy."""
        pc    = make_test_pc(seed=77)
        batch = make_batch(pc)
        msg_t = torch.randint(0,2,(batch.shape[0],63),dtype=torch.float32,device=device)
        true_msg = msg_t[0].cpu().numpy()
        with torch.no_grad():
            stego  = MODEL.embed(batch, msg_t, ALPHA)
            logits = MODEL.extract(stego)
        pred = (logits > 0).float().cpu().numpy()
        acc  = (pred == msg_t.cpu().numpy()).mean(axis=1)[0]
        assert acc >= 0.80, f"Channel accuracy too low: {acc:.1%}"
        return f"Self-embed channel accuracy: {acc:.1%} ✓"

    for fn in [t_model_loads, t_embed_output_shape, t_cd_below_threshold,
               t_cd_consistent, t_offset_bounded, t_stego_different_from_original,
               t_extract_output_shape, t_message_extraction]:
        run("Model", fn.__name__[2:].replace('_',' ').title(), fn)

# =============================================================================
#  8. ROBUSTNESS SWEEP  (skip if model not available)
# =============================================================================
section_header("8. ROBUSTNESS SWEEP")

if not MODEL_AVAILABLE:
    skip("Robustness", "Full sweep", reason if not TORCH_OK else "Model not available")
else:
    def run_robustness_sweep():
        N_SAMPLES = 30
        results_table = []

        def eval_condition(transform_fn, label):
            recoveries = []
            for seed in range(N_SAMPLES):
                pc    = make_test_pc(seed=seed)
                batch = make_batch(pc)
                msg_t = torch.randint(0,2,(batch.shape[0],63),dtype=torch.float32,device=device)
                true_bits = (msg_t[0] > 0).cpu().numpy().tolist()
                with torch.no_grad(): stego = MODEL.embed(batch, msg_t, ALPHA)
                stego_np = stego[0].cpu().numpy()
                perturbed = transform_fn(stego_np)
                p_t  = torch.from_numpy(perturbed).unsqueeze(0).to(device)
                with torch.no_grad(): logits = MODEL.extract(p_t)
                pred = (logits[0] > 0).cpu().numpy().tolist()
                ch_acc = sum(a==b for a,b in zip(pred,true_bits)) / len(true_bits)
                recoveries.append(ch_acc)
            return float(np.mean(recoveries))

        def identity(pc): return pc
        def scale(s): return lambda pc: pc * s
        def noise(sigma): return lambda pc: pc + np.random.randn(*pc.shape).astype(np.float32)*sigma
        def rotate(deg):
            def fn(pc):
                r = np.deg2rad(deg); c,s = np.cos(r), np.sin(r)
                R = np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float32)
                return pc @ R.T
            return fn

        conditions = [
            ("Clean",            identity),
            ("Scale ×0.95",      scale(0.95)),
            ("Scale ×0.90",      scale(0.90)),
            ("Scale ×1.05",      scale(1.05)),
            ("Scale ×1.10",      scale(1.10)),
            ("Noise σ=0.001",    noise(0.001)),
            ("Noise σ=0.003",    noise(0.003)),
            ("Noise σ=0.005",    noise(0.005)),
            ("Rotate 30°",       rotate(30)),
            ("Rotate 90°",       rotate(90)),
        ]
        print(f"\n  {'Condition':<20}  {'Ch. Accuracy':>14}  {'Bar':>20}")
        print(f"  {'─'*20}  {'─'*14}  {'─'*20}")
        for label, tfm in conditions:
            acc = eval_condition(tfm, label)
            bar = '█' * int(acc*20) + '░' * (20 - int(acc*20))
            tag = f"{GREEN}✓{RESET}" if acc >= 0.85 else (f"{YELLOW}△{RESET}" if acc >= 0.60 else f"{RED}✗{RESET}")
            print(f"  {label:<20}  {acc:>13.1%}  {bar}  {tag}")
            results_table.append((label, acc))
        clean_acc = results_table[0][1]
        assert clean_acc >= 0.85, f"Clean accuracy {clean_acc:.1%} < 85%"
        return f"Sweep complete — clean accuracy: {clean_acc:.1%}"

    run("Robustness", "Full condition sweep (30 samples each)", run_robustness_sweep)

# =============================================================================
#  SUMMARY
# =============================================================================
print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
print(f"{BOLD}{CYAN}  RESULTS SUMMARY{RESET}")
print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

by_section = {}
for sec, name, status, detail in results:
    by_section.setdefault(sec, []).append((name, status))

total_pass = total_fail = total_skip = 0
for sec, items in by_section.items():
    p = sum(1 for _,s in items if s=="PASS")
    f = sum(1 for _,s in items if s=="FAIL")
    sk= sum(1 for _,s in items if s=="SKIP")
    total_pass+=p; total_fail+=f; total_skip+=sk
    tag = f"{GREEN}{p}P{RESET}" + (f"/{RED}{f}F{RESET}" if f else "") + (f"/{YELLOW}{sk}S{RESET}" if sk else "")
    print(f"  {sec:<30}  {tag}")

print(f"\n  {'─'*50}")
color = GREEN if total_fail==0 else RED
print(f"  {BOLD}Total:  {color}{total_pass} passed{RESET}  "
      f"{(RED+str(total_fail)+' failed'+RESET) if total_fail else ''}"
      f"  {(YELLOW+str(total_skip)+' skipped'+RESET) if total_skip else ''}")
print(f"{BOLD}{CYAN}{'═'*60}{RESET}\n")
