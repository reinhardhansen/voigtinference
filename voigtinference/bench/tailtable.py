import numpy as np, sys
from mpmath import mp, mpf, mpc, exp, sqrt, pi, inf, quad
sys.path.insert(0, "src")
from voigtinference import voigt_score, voigt_hessian
from voigtinference.core import _kl

def erfcx_ref(u):
    return quad(lambda t: exp(-t * t) / (u + mpc(0, 1) * t), [-inf, 0, inf]) / pi

def ref_score_hess(yt, sg, gm):
    ytb, sgb, gmb = mpf(yt), mpf(sg), mpf(gm)
    w = erfcx_ref(mpc(gmb, ytb) / (sgb * sqrt(2)))
    K, L = w.real, -w.imag
    s2p = sqrt(2 / pi); s2 = sgb * sgb; r = L / K
    smu = (ytb - gmb * L / K) / s2
    ssg = ((ytb**2 - gmb**2 - s2) * K - 2 * gmb * ytb * L + s2p * sgb * gmb) / (sgb**3 * K)
    sgm = (gmb * K + ytb * L - s2p * sgb) / (s2 * K)
    hmm = ssg / sgb - smu * smu; hgg = -ssg / sgb - sgm * sgm
    hmg = (ytb * sgm + gmb * smu - r) / s2 - smu * sgm
    hms = -(smu + gmb * hmg - ytb * hmm) / sgb
    hgs = -(sgm + gmb * hgg - ytb * hmg) / sgb
    hss = -(ssg + gmb * hgs - ytb * hms) / sgb
    return [smu, ssg, sgm], [hmm, hms, hmg, hss, hgs, hgg]

def naive(yt, sg, gm):
    a = np.array([yt]); K, L = _kl(a, sg, gm); K, L = K[0], L[0]
    s2 = sg * sg; r = L / K; SQ = np.sqrt(2 / np.pi)
    smu = (yt - gm * L / K) / s2
    ssg = ((yt * yt - gm * gm - s2) * K - 2 * gm * yt * L + SQ * sg * gm) / (sg**3 * K)
    sgm = (gm * K + yt * L - SQ * sg) / (s2 * K)
    hmm = ssg / sg - smu * smu; hgg = -ssg / sg - sgm * sgm
    hmg = (yt * sgm + gm * smu - r) / s2 - smu * sgm
    hms = -(smu + gm * hmg - yt * hmm) / sg
    hgs = -(sgm + gm * hgg - yt * hmg) / sg
    hss = -(ssg + gm * hgs - yt * hms) / sg
    return np.array([smu, ssg, sgm]), np.array([hmm, hms, hmg, hss, hgs, hgg])

# normwise block error (the certified metric; see examples/certify.jl in the
# Julia package): max component error over the block's largest true component.
def relerr(x, ref):
    scale = max(abs(b) for b in ref)
    return max(float(abs(mpf(float(a)) - b) / scale) for a, b in zip(x, ref))

print("PRECISION SELF-CHECK: is 256-bit enough for a HESSIAN reference?")
print(f"{'mult':>7}{'256-bit vs 512-bit':>24}")
for e in (1, 4, 6, 8):
    yt = 10.0**e * np.sqrt(2)
    mp.dps = 77;  _, ha = ref_score_hess(yt, 1.0, 1.0)
    mp.dps = 154; _, hb = ref_score_hess(yt, 1.0, 1.0)
    print(f"1e{e:<6}{max(float(abs(x-y)/abs(y)) for x,y in zip(ha,hb)):24.2e}")

mp.dps = 154
for sg, gm in ((1.0, 1.0), (1.0, 0.01)):
    sc = np.sqrt(sg**2 + gm**2)
    print(f"\n(sigma, gamma) = ({sg}, {gm})   switches: r_s = 5e-7, r_h = 6.25e-5"
          f"  (r = sigma^2/(ytil^2+gamma^2); normwise errors)")
    print(f"{'|ytil|/scale':>13}{'score pkg':>12}{'score naive':>13}{'HESS pkg':>12}{'HESS naive':>13}")
    for e in range(1, 9):
        yt = 10.0**e * sc
        rs, rh = ref_score_hess(yt, sg, gm)
        ps = voigt_score(yt, 0.0, sg, gm)
        H = voigt_hessian(yt, 0.0, sg, gm)
        ph = np.array([H[0,0], H[0,1], H[0,2], H[1,1], H[1,2], H[2,2]])
        ns, nh = naive(yt, sg, gm)
        print(f"{10.0**e:13.0e}{relerr(ps,rs):12.1e}{relerr(ns,rs):13.1e}{relerr(ph,rh):12.1e}{relerr(nh,rh):13.1e}")
