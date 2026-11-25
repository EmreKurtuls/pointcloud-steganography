# =============================================================================
#  NIST SP 800-22 Rev 1a — Statistical Test Suite
#  Applied to StegoV5BCH AES-128-CTR keystream and PBKDF2-derived keys
#
#  Reference:  NIST SP 800-22 Rev 1a (April 2010)
#              https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final
#
#  Tests implemented (10 of 15 — the 5 omitted require extremely long bit
#  sequences that are impractical for 128-bit key material):
#    1.  Frequency (Monobit)
#    2.  Frequency within a Block
#    3.  Runs
#    4.  Longest Run of Ones in a Block
#    5.  Binary Matrix Rank
#    6.  Discrete Fourier Transform (Spectral)
#    7.  Non-overlapping Template Matching  (m=9 template)
#    8.  Approximate Entropy
#    9.  Cumulative Sums (Cusum) — forward & backward
#   10.  Serial
#
#  Significance level α = 0.01  (p < 0.01 → FAIL)
#
#  Usage:
#    python nist_tests.py
# =============================================================================

import os, sys, math, hashlib, time, random
import numpy as np
from scipy import stats as scipy_stats
from scipy.special import gammaincc, erfc as sc_erfc
from Crypto.Cipher import AES

# ── colour helpers ─────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; X = "\033[0m"

ALPHA = 0.01          # NIST significance level
results = []

def hdr(title):
    print(f"\n{B}{C}{'═'*64}{X}")
    print(f"{B}{C}  {title}{X}")
    print(f"{B}{C}{'═'*64}{X}")

def log(name, p, note=""):
    ok = p >= ALPHA
    tag = f"{G}PASS{X}" if ok else f"{R}FAIL{X}"
    bar_len = min(40, int(p*40))
    bar = "█"*bar_len + "░"*(40-bar_len)
    p_str = f"p={p:.6f}" if p >= 0.0001 else f"p={p:.2e}"
    print(f"  {tag}  {name:<45}  {p_str}")
    if note: print(f"       {C}{note}{X}")
    results.append({"name": name, "p": p, "pass": ok, "note": note})

# =============================================================================
#  SAMPLE GENERATION
#  We test two sources of bits from the real cipher system:
#   A. AES-CTR keystream — bits produced by the actual encryption
#   B. PBKDF2 key bytes  — derived key material used in the system
# =============================================================================

def derive_key(trade_id, passphrase):
    return hashlib.pbkdf2_hmac(
        'sha256', passphrase.encode(), trade_id.encode(), 10000)[:16]

def encrypt_to_bits(message, trade_id, passphrase):
    """Exact copy of cipher_server.py — AES-128-CTR encryption → 128 bits."""
    key   = derive_key(trade_id, passphrase)
    nonce = hashlib.md5(trade_id.encode()).digest()[:8]
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    msg_bytes = message.encode('utf-8')[:16].ljust(16, b'\x00')
    ct    = cipher.encrypt(msg_bytes)
    bits  = []
    for b in ct:
        for j in range(7, -1, -1): bits.append((b >> j) & 1)
    return bits

def generate_trade_id():
    ts  = time.strftime("%Y%m%d")
    uid = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=8))
    return f"TRD-{ts}-{uid}"

# ── realistic inputs — what real users actually type ──────────────────────────
REAL_PASSPHRASES = [
    "Secret2026!",    "polymorph#99",   "3DSecret$Key",   "StegoPaper1",
    "Pointcloud_42",   "PointCloud!7",   "cipher_mode42",  "mesh4market",
    "BCH_t5_works",   "mypassword123",  "Ankara2026TR",   "deeplearn!ng",
    "ModelNet40ok",   "Flask5001pw",    "embed@extract",  "FiLM_modulate",
    "SA1SA2alpha8",   "AES256secure",   "TradeMarket!1",  "HiddenGeom!9",
]
REAL_MESSAGES = [
    "Hello World!",   "SecretMsg2026",  "Hidden note!", "Stego_works!",
    "BCH corrects",   "Polymorph OK",   "3D embedding", "Alpha=0.08!!",
    "ModelNet OK!",   "Meet at noon",  "Test msg #1!", "Flask server",
    "Cipher mode!",   "Key derivat!",   "NIST SP800!!", "AES-CTR ok!!",
]

def get_real_ciphertext_bits(n_bits: int) -> list:
    """
    Source A — Real system ciphertext.
    Uses the ACTUAL encrypt_to_bits() function with:
      - Real Trade IDs (TRD-YYYYMMDD-XXXXXXXX format)
      - Realistic passphrases users might type
      - Realistic short messages (≤16 chars)
    This is byte-for-byte what the cipher tool sends to the BCH encoder.
    """
    bits = []
    i    = 0
    while len(bits) < n_bits:
        tid  = generate_trade_id()
        pw   = REAL_PASSPHRASES[i % len(REAL_PASSPHRASES)]
        msg  = REAL_MESSAGES[i % len(REAL_MESSAGES)]
        bits.extend(encrypt_to_bits(msg, tid, pw))   # 128 bits per call
        i += 1
    return bits[:n_bits]

def get_real_key_bits(n_bits: int) -> list:
    """
    Source B — Real PBKDF2-derived key bytes.
    Uses the ACTUAL derive_key() call with:
      - Real Trade IDs (TRD-YYYYMMDD-XXXXXXXX format)
      - Realistic passphrases
    These are the 128-bit AES keys that the system actually uses.
    """
    bits = []
    i    = 0
    while len(bits) < n_bits:
        tid = generate_trade_id()
        pw  = REAL_PASSPHRASES[i % len(REAL_PASSPHRASES)]
        key = derive_key(tid, pw)     # 16-byte key from real PBKDF2 call
        for byte in key:
            for j in range(7, -1, -1):
                bits.append((byte >> j) & 1)
        i += 1
    return bits[:n_bits]

# =============================================================================
#  NIST SP 800-22 TEST FUNCTIONS
# =============================================================================

def frequency_monobit(bits):
    """
    NIST Section 2.1 — Frequency (Monobit) Test
    Tests whether the number of ones and zeros are approximately equal.
    """
    n  = len(bits)
    s  = sum(2*b - 1 for b in bits)   # +1 for 1, −1 for 0
    s_obs = abs(s) / math.sqrt(n)
    p = math.erfc(s_obs / math.sqrt(2))
    return p, f"s_obs={s_obs:.4f}  ones={sum(bits)}/{n}  ratio={sum(bits)/n:.4f}"


def block_frequency(bits, M=128):
    """
    NIST Section 2.2 — Frequency Test within a Block
    Tests whether the proportion of ones within M-bit blocks is approximately M/2.
    M=128 is the default block size used in the NIST reference.
    """
    n  = len(bits)
    N  = n // M   # number of blocks
    assert N >= 1
    chi_sq = 0.0
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        pi_i  = sum(block) / M
        chi_sq += (pi_i - 0.5) ** 2
    chi_sq *= 4 * M
    p = gammaincc(N / 2, chi_sq / 2)
    return p, f"N={N} blocks of M={M}  χ²={chi_sq:.4f}"


def runs(bits):
    """
    NIST Section 2.3 — Runs Test
    Tests whether the oscillations between zeros and ones are too fast or too slow.
    """
    n   = len(bits)
    pi  = sum(bits) / n
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0, "Prerequisite failed — proportions too far from 0.5"
    v_obs = 1 + sum(1 for k in range(n-1) if bits[k] != bits[k+1])
    p = math.erfc(abs(v_obs - 2*n*pi*(1-pi)) /
                  (2 * math.sqrt(2*n) * pi * (1-pi)))
    return p, f"v_obs={v_obs}  π={pi:.4f}"


def longest_run_ones(bits):
    """
    NIST Section 2.4 — Test for the Longest Run of Ones in a Block
    Uses M=128 block size for n=20,000 bits.
    """
    n = len(bits)
    M = 128   # block size for 20,000 ≤ n < 1,000,000
    K = 5; N = n // M
    # pi values from NIST Table 1 for M=128
    pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    v  = [0]*6   # frequency counts

    for i in range(N):
        block = bits[i*M:(i+1)*M]
        # find longest run of 1s
        lr = cur = 0
        for b in block:
            if b == 1: cur += 1; lr = max(lr, cur)
            else:      cur = 0
        if   lr <= 4: v[0] += 1
        elif lr == 5: v[1] += 1
        elif lr == 6: v[2] += 1
        elif lr == 7: v[3] += 1
        elif lr == 8: v[4] += 1
        else:         v[5] += 1

    chi_sq = sum((v[i] - N*pi[i])**2 / (N*pi[i]) for i in range(K+1))
    p = gammaincc(K/2, chi_sq/2)
    return p, f"χ²={chi_sq:.4f}  longest runs={v}"


def binary_matrix_rank(bits):
    """
    NIST Section 2.5 — Binary Matrix Rank Test
    Tests the linear dependence among fixed-length substrings.
    Uses 32×32 matrices.
    """
    M = Q = 32
    n = len(bits)
    N = n // (M * Q)   # number of matrices
    if N < 38:
        return None, "Need ≥38 matrices (≥38,912 bits)"

    def gf2_rank(mat):
        """Rank of a binary matrix over GF(2)."""
        m = [list(row) for row in mat]
        rank = 0
        for col in range(Q):
            pivot = None
            for row in range(rank, M):
                if m[row][col]:
                    pivot = row; break
            if pivot is None: continue
            m[rank], m[pivot] = m[pivot], m[rank]
            for row in range(M):
                if row != rank and m[row][col]:
                    m[row] = [m[row][j]^m[rank][j] for j in range(Q)]
            rank += 1
        return rank

    fm = sm = rest = 0
    for i in range(N):
        block = bits[i*M*Q:(i+1)*M*Q]
        mat   = [block[r*Q:(r+1)*Q] for r in range(M)]
        r = gf2_rank(mat)
        if   r == M:     fm   += 1
        elif r == M - 1: sm   += 1
        else:            rest += 1

    p_fm   = 0.2888
    p_sm   = 0.5776
    p_rest = 0.1336
    chi_sq = ((fm   - N*p_fm  )**2/(N*p_fm)   +
              (sm   - N*p_sm  )**2/(N*p_sm)   +
              (rest - N*p_rest)**2/(N*p_rest))
    p = math.exp(-chi_sq/2)
    return p, f"N={N} matrices  FM={fm}  SM={sm}  rest={rest}  χ²={chi_sq:.4f}"


def dft_spectral(bits):
    """
    NIST Section 2.6 — Discrete Fourier Transform (Spectral) Test
    Detects periodic features that would indicate non-randomness.
    """
    n  = len(bits)
    x  = np.array([2*b - 1 for b in bits], dtype=float)
    f  = np.fft.fft(x)
    M  = np.abs(f[:n//2])     # magnitude of first n/2 terms
    T  = math.sqrt(math.log(1/0.05) * n)   # 95% threshold
    N0 = 0.95 * n / 2         # expected number below threshold
    N1 = sum(1 for m in M if m < T)
    d  = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
    p  = math.erfc(abs(d) / math.sqrt(2))
    return p, f"N0={N0:.1f}  N1={N1}  d={d:.4f}  T={T:.4f}"


def non_overlapping_template(bits, template=None):
    """
    NIST Section 2.7 — Non-overlapping Template Matching Test
    Uses a 9-bit aperiodic template (default: 000000001).
    """
    if template is None:
        template = [0,0,0,0,0,0,0,0,1]   # aperiodic length-9 template
    m  = len(template)
    n  = len(bits)
    M  = 1032   # block length recommended for n ≥ 50,000
    N  = n // M
    if N < 8:
        return None, f"Need ≥8 blocks of M={M} (≥{8*M} bits)"

    mu    = (M - m + 1) / 2**m
    sigma2= M * (1/2**m - (2*m-1)/2**(2*m))

    counts = []
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        count = j = 0
        while j <= M - m:
            if block[j:j+m] == template: count += 1; j += m
            else: j += 1
        counts.append(count)

    chi_sq = sum((c - mu)**2 / sigma2 for c in counts)
    p = gammaincc(N/2, chi_sq/2)
    return p, (f"template={''.join(map(str,template))}  "
               f"N={N} blocks  μ={mu:.4f}  σ²={sigma2:.4f}  χ²={chi_sq:.4f}")


def approximate_entropy(bits, m=10):
    """
    NIST Section 2.12 — Approximate Entropy Test
    Compares the frequency of overlapping m-bit patterns against (m+1)-bit patterns.
    """
    n   = len(bits)
    extended = bits + bits[:m+1]   # wrap-around

    def phi(m_val):
        counts = {}
        for i in range(n):
            pat = tuple(extended[i:i+m_val])
            counts[pat] = counts.get(pat, 0) + 1
        return sum(c/n * math.log(c/n) for c in counts.values())

    ap_en  = phi(m) - phi(m+1)
    chi_sq = 2 * n * (math.log(2) - ap_en)
    p = gammaincc(2**(m-1), chi_sq/2)
    return p, f"m={m}  ApEn={ap_en:.6f}  χ²={chi_sq:.4f}"


def cumulative_sums(bits, mode='forward'):
    """
    NIST Section 2.13 — Cumulative Sums Test
    Tests whether the cumulative sum of the sequence is too large or too small.
    """
    n  = len(bits)
    x  = [2*b - 1 for b in (bits if mode == 'forward' else bits[::-1])]
    s  = np.cumsum(x)
    z  = int(np.max(np.abs(s)))

    def norm_cdf(v): return 0.5 * math.erfc(-v / math.sqrt(2))

    p = 1.0
    start = int((-n/z + 1) / 4)
    end   = int(( n/z - 1) / 4)
    sum1  = sum(norm_cdf((4*k+1)*z/math.sqrt(n)) -
                norm_cdf((4*k-1)*z/math.sqrt(n))
                for k in range(start, end+1))
    start = int((-n/z - 3) / 4)
    end   = int(( n/z - 1) / 4)
    sum2  = sum(norm_cdf((4*k+3)*z/math.sqrt(n)) -
                norm_cdf((4*k+1)*z/math.sqrt(n))
                for k in range(start, end+1))
    p = 1 - sum1 + sum2
    return p, f"z={z}  mode={mode}"


def serial_test(bits, m=16):
    """
    NIST Section 2.11 — Serial Test (Ψ² statistic)
    Tests whether all m-bit overlapping patterns occur with equal frequency.
    """
    n = len(bits)
    extended = bits + bits[:m-1]

    def psi_sq(m_val):
        if m_val == 0: return 0.0
        counts = {}
        for i in range(n):
            pat = tuple(extended[i:i+m_val])
            counts[pat] = counts.get(pat, 0) + 1
        return (2**m_val / n) * sum(c**2 for c in counts.values()) - n

    psi_m  = psi_sq(m)
    psi_m1 = psi_sq(m-1)
    psi_m2 = psi_sq(m-2)

    del1 = psi_m  - psi_m1
    del2 = psi_m  - 2*psi_m1 + psi_m2

    p1 = gammaincc(2**(m-2), del1/2)
    p2 = gammaincc(2**(m-3), del2/2)
    p  = min(p1, p2)
    return p, f"m={m}  ψ²_m={psi_m:.2f}  ∇ψ²={del1:.2f}  ∇²ψ²={del2:.2f}  p1={p1:.4f}  p2={p2:.4f}"


# =============================================================================
#  RUN THE SUITE
# =============================================================================

print(f"\n{B}{C}")
print("  NIST SP 800-22 Rev 1a — Statistical Test Suite")
print("  Significance level α = 0.01  (p < 0.01 → FAIL)")
print("  Two test sources — both use REAL system inputs:")
print("  A) AES-128-CTR ciphertext  — real TID + realistic passphrase + real message")
print("  B) PBKDF2-derived key bytes — real TID + realistic passphrase")
print(f"{X}")

TESTS = [
    ("Frequency (Monobit)",              frequency_monobit),
    ("Block Frequency  (M=128)",         lambda b: block_frequency(b, M=128)),
    ("Runs",                             runs),
    ("Longest Run of Ones  (M=128)",     longest_run_ones),
    ("Binary Matrix Rank  (32×32)",      binary_matrix_rank),
    ("DFT / Spectral",                   dft_spectral),
    ("Non-overlapping Template  (m=9)",  non_overlapping_template),
    ("Approximate Entropy  (m=10)",      lambda b: approximate_entropy(b, m=10)),
    ("Cumulative Sums (Forward)",        lambda b: cumulative_sums(b, 'forward')),
    ("Cumulative Sums (Backward)",       lambda b: cumulative_sums(b, 'backward')),
    ("Serial  (m=16)",                   lambda b: serial_test(b, m=16)),
]

N_BITS = 20_000   # NIST minimum recommendation for most tests

for label, (source_fn, source_name) in [
    ("A", (get_real_ciphertext_bits, "AES-128-CTR ciphertext  (real TID + real passphrase + real message)")),
    ("B", (get_real_key_bits,        "PBKDF2-derived keys      (real TID + real passphrase)")),
]:
    hdr(f"Source {label} — {source_name}")
    print(f"  Generating {N_BITS:,} bits from {source_name}...")
    np.random.seed(42); random.seed(42)
    bits = source_fn(N_BITS)
    ones = sum(bits); zeros = N_BITS - ones
    print(f"  Bits: {N_BITS:,}  |  1s: {ones:,} ({ones/N_BITS:.4f})  |  0s: {zeros:,} ({zeros/N_BITS:.4f})\n")

    for name, fn in TESTS:
        try:
            result = fn(bits)
            p, note = result if result[0] is not None else (0.0, result[1])
            if result[0] is None:
                print(f"  {Y}SKIP{X}  {name:<45}  {C}{note}{X}")
                results.append({"name": f"{source_name} — {name}", "p": None,
                                 "pass": None, "note": note})
            else:
                log(f"{source_name} — {name}", p, note)
        except Exception as e:
            print(f"  {R}ERR {X}  {name:<45}  {R}{e}{X}")
            results.append({"name": f"{source_name} — {name}", "p": None,
                             "pass": False, "note": str(e)})

# =============================================================================
#  UNIFORMITY OF p-VALUES
#  If the bit sequence is truly random, p-values from repeated tests should
#  themselves be uniformly distributed on [0,1].
#  We run the monobit test 100 times and check uniformity with K-S test.
# =============================================================================
hdr("P-VALUE UNIFORMITY — Kolmogorov-Smirnov Test")
print("  Running Frequency (Monobit) × 100 independent samples (real TID + passphrase)...")
p_vals = []
for i in range(100):
    b = get_real_ciphertext_bits(1000)
    p, _ = frequency_monobit(b)
    p_vals.append(p)

ks_stat, ks_p = scipy_stats.kstest(p_vals, 'uniform')
print(f"\n  p-value distribution across 100 runs:")
print(f"  Mean p-value:  {np.mean(p_vals):.4f}  (expected ~0.50)")
print(f"  Std  p-value:  {np.std(p_vals):.4f}  (expected ~0.289 for Uniform[0,1])")
print(f"  K-S statistic: {ks_stat:.4f}")
print(f"  K-S p-value:   {ks_p:.4f}")

ks_tag = f"{G}PASS{X}" if ks_p >= 0.01 else f"{R}FAIL{X}"
print(f"\n  {ks_tag}  p-values uniform on [0,1]  (K-S test p={ks_p:.4f})")
results.append({"name": "p-value uniformity (K-S)", "p": ks_p,
                "pass": ks_p >= 0.01, "note": f"ks_stat={ks_stat:.4f}"})

# =============================================================================
#  PROPORTION OF PASSING SEQUENCES
#  NIST recommends: for α=0.01 and 100 samples, the proportion of
#  passing sequences should be in [0.96, 1.0] — i.e. at most 1 failure.
# =============================================================================
hdr("PROPORTION OF PASSING SEQUENCES")
print("  Running full Monobit test on 100 independent 20,000-bit sequences (real TID + passphrase)...")
pass_count = 0
for i in range(100):
    b = get_real_ciphertext_bits(N_BITS)
    p, _ = frequency_monobit(b)
    if p >= ALPHA: pass_count += 1

proportion   = pass_count / 100
# NIST SP 800-22 §4.2.1: p_hat ± 3·sqrt(p_hat(1-p_hat)/m), with p_hat = 1-α
p_hat        = 1 - ALPHA
lower, upper = p_hat - 3*math.sqrt(p_hat*(1-p_hat)/100), 1.0
prop_tag = f"{G}PASS{X}" if proportion >= lower else f"{R}FAIL{X}"
print(f"  {pass_count}/100 sequences passed")
print(f"  Proportion: {proportion:.2f}  (acceptable range: [{lower:.4f}, {upper:.2f}])")
print(f"  {prop_tag}  proportion test")
results.append({"name": "Proportion of passing sequences", "p": proportion,
                "pass": proportion >= lower,
                "note": f"range=[{lower:.4f},{upper:.2f}]"})

# =============================================================================
#  SUMMARY TABLE
# =============================================================================
hdr("SUMMARY")
passed = sum(1 for r in results if r["pass"] is True)
failed = sum(1 for r in results if r["pass"] is False)
skipped= sum(1 for r in results if r["pass"] is None)
total  = passed + failed

print(f"\n  {'Test':<55}  {'p-value':>10}  {'Result':>6}")
print(f"  {'─'*55}  {'─'*10}  {'─'*6}")
for r in results:
    if r["p"] is None:
        print(f"  {r['name']:<55}  {'—':>10}  {Y}SKIP{X}")
    else:
        tag  = f"{G}PASS{X}" if r["pass"] else f"{R}FAIL{X}"
        pstr = f"{r['p']:.4f}" if r['p'] >= 0.001 else f"{r['p']:.2e}"
        print(f"  {r['name']:<55}  {pstr:>10}  {tag}")

print(f"\n  {'─'*74}")
color = G if failed == 0 else R
print(f"  {B}Total: {color}{passed} passed{X}  "
      f"{(R+str(failed)+' failed'+X+' ') if failed else ''}"
      f"{(Y+str(skipped)+' skipped'+X) if skipped else ''}")
print(f"\n  {C}Significance level α = {ALPHA}{X}")
print(f"  {C}A test PASSES if p ≥ α = {ALPHA}{X}")
print(f"  {C}For a truly random source, ~{int((1-ALPHA)*100)}% of tests should pass.{X}")
print(f"{B}{C}{'═'*64}{X}\n")