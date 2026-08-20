# Certification of the double-precision Voigt likelihood kernel against
# high-precision references, across gamma/sigma from 1e-8 to 1e8.
#
# Certifies the branch dispatch: the switches gate on
# the Cauchy-limit expansion parameter r = sigma^2/(ytil^2+gamma^2), and this
# script certifies (a) the worst-case relative error of the dispatched score,
# Hessian, and conditional moments per quantity and per gamma/sigma decade,
# (b) behavior at prevfloat/exact/nextfloat of the switch thresholds,
# (c) nonnegativity of the conditional variance,
# (d) Fisher-information convergence and the large-ratio limit
#     I_sigma_sigma * gamma^4 / sigma^2 -> 1  (exactly: the Cauchy-limit
#     score gives (1/pi) * integral (6u^2-2)^2/(1+u^2)^5 du = pi),
# (e) the recorded extreme-ratio failure cases of earlier revisions.
# A threshold-tuning mode reports the minimax constants.
#
# High-precision reference: erfcx(w) in Complex{BigFloat} via Taylor series of
# erf for |w| <= 8 and the Laplace continued fraction for |w| > 8, with an
# internal overlap consistency check. This reference is uniformly valid in
# both arguments (quadrature-based references fail for small gamma at small
# |ytil|, where the integrand's pole approaches the contour with O(1) weight).
# Working precision scales with the cancellation amplification (~r^-4 for the
# Hessian).
#
# Run:  julia --project=. examples/certify.jl          (certification, ~min)
#       julia --project=. examples/certify.jl tune     (threshold scan)
# Exit code is nonzero on any FAIL.

using VoigtInference, Printf

const V = VoigtInference

# ------------------------------------------------------------------
# high-precision erfcx
# ------------------------------------------------------------------

"Taylor-series erfcx for |w| <= 12: e^{w^2}(1 - erf(w)). Self-guarding: the
erf sum cancels internally through terms of size e^{|w|^2}, and forming
1 - erf(w) cancels a further e^{Re(w^2)} when Re(w^2) > 0, so the working
precision is raised by (|w|^2 + max(Re(w^2), 0))*log2(e) + 64 bits and the
result is rounded back to the caller precision."
function _erfcx_series(w0::Complex{BigFloat})
    p0 = precision(real(w0))
    guard = ceil(Int, (Float64(abs2(w0)) + max(Float64(real(w0 * w0)), 0.0)) * 1.4427) + 64
    val = setprecision(BigFloat, p0 + guard) do
        w = Complex{BigFloat}(w0)
        # erf(w) = (2/sqrt(pi)) * sum_{n>=0} (-1)^n w^{2n+1} / (n! (2n+1))
        s = w
        term = w
        w2 = w * w
        n = 0
        while true
            n += 1
            term *= -w2 / n
            add = term / (2n + 1)
            s += add
            abs(add) < abs(s) * eps(BigFloat) && break
            n > 100_000 && error("erf series did not converge")
        end
        erfw = (2 / sqrt(big(pi))) * s
        exp(w2) * (1 - erfw)
    end
    return Complex{BigFloat}(val)
end

"Laplace continued fraction erfcx for |w| > 8 (Re w > 0):
erfcx(w) = (1/sqrt(pi)) / (w + (1/2)/(w + (2/2)/(w + (3/2)/(w + ...))))."
function _erfcx_cf(w::Complex{BigFloat}; depth::Int = 2000)
    x = w
    for k in depth:-1:1
        x = w + (big(k) / 2) / x
    end
    return (1 / sqrt(big(pi))) / x
end

erfcx_big(w::Complex{BigFloat}) = abs(w) <= 12 ? _erfcx_series(w) : _erfcx_cf(w)

"Overlap self-check: series and continued fraction must agree in the ring
10 <= |w| <= 14; run once at startup."
function _selfcheck()
    setprecision(BigFloat, 512) do
        for (re, im_) in ((10.0, 3.0), (0.5, 11.0), (12.5, 4.0), (3.0, 13.0))
            w = complex(big(re), big(im_))
            a, b = _erfcx_series(w), _erfcx_cf(w)
            rel = abs(a - b) / abs(b)
            # with the self-guarding series, agreement in the ring is limited
            # by the continued fraction's convergence, comfortably beyond the
            # ~1e-20 needed to certify double precision. Require 1e-40.
            rel < big"1e-40" || error("erfcx reference overlap check failed at $w (rel $rel)")
        end
    end
    return true
end
_selfcheck()

"K, L at (ytil, sigma, gamma) in BigFloat."
function KL_big(ỹ::BigFloat, σ::BigFloat, γ::BigFloat)
    w = erfcx_big(complex(γ, ỹ) / (σ * sqrt(big(2))))
    return real(w), -imag(w)          # u = K, v = -L  =>  (K, L) physics signs
end

# exact-branch formulas in BigFloat (same algebra as the Float64 exact branch)
function ref_all(ỹf::Float64, σf::Float64, γf::Float64)
    # precision scaled to the worst amplification (~r^-4, Hessian recursion)
    r = σf^2 / (ỹf^2 + γf^2)
    prec = clamp(320 + ceil(Int, 4 * abs(log2(min(r, 1.0)))), 320, 4096)
    setprecision(BigFloat, prec) do
        ỹ, σ, γ = big(ỹf), big(σf), big(γf)
        K, L = KL_big(ỹ, σ, γ)
        s2p = sqrt(big(2) / big(pi))
        sμ = (ỹ - γ * L / K) / σ^2
        sσ = ((ỹ^2 - γ^2 - σ^2) * K - 2γ * ỹ * L + s2p * σ * γ) / (σ^3 * K)
        sγ = (γ * K + ỹ * L - s2p * σ) / (σ^2 * K)
        rr = L / K
        Hμμ = sσ / σ - sμ^2
        Hγγ = -sσ / σ - sγ^2
        Hμγ = (ỹ * sγ + γ * sμ - rr) / σ^2 - sμ * sγ
        Hμσ = -(sμ + γ * Hμγ - ỹ * Hμμ) / σ
        Hγσ = -(sγ + γ * Hγγ - ỹ * Hμγ) / σ
        Hσσ = -(sσ + γ * Hγσ - ỹ * Hμσ) / σ
        cm = ỹ - γ * L / K
        cv = s2p * σ * γ / K - γ^2 * (1 + (L / K)^2)
        (Float64.((sμ, sσ, sγ)), Float64.((Hμμ, Hμσ, Hμγ, Hσσ, Hγσ, Hγγ)),
         Float64(cm), Float64(cv))
    end
end

# Float64 branch formulas, callable independently of the package gating
function f64_exact(ỹ, σ, γ)
    K, L, _, _ = V._KL(ỹ, 0.0, σ, γ)
    s2p = V._SQRT2OVERPI
    sμ = (ỹ - γ * L / K) / σ^2
    sσ = ((ỹ^2 - γ^2 - σ^2) * K - 2γ * ỹ * L + s2p * σ * γ) / (σ^3 * K)
    sγ = (γ * K + ỹ * L - s2p * σ) / (σ^2 * K)
    rr = L / K
    Hμμ = sσ / σ - sμ^2
    Hγγ = -sσ / σ - sγ^2
    Hμγ = (ỹ * sγ + γ * sμ - rr) / σ^2 - sμ * sγ
    Hμσ = -(sμ + γ * Hμγ - ỹ * Hμμ) / σ
    Hγσ = -(sγ + γ * Hγγ - ỹ * Hμγ) / σ
    Hσσ = -(sσ + γ * Hγσ - ỹ * Hμσ) / σ
    cm = ỹ - γ * L / K
    cv = s2p * σ * γ / K - γ^2 * (1 + (L / K)^2)
    ((sμ, sσ, sγ), (Hμμ, Hμσ, Hμγ, Hσσ, Hγσ, Hγγ), cm, cv)
end

function f64_tail(ỹ, σ, γ)
    s = V._score_tail(ỹ, σ, γ)
    H = V._hessian_tail(ỹ, σ, γ)
    cm = σ^2 * s[1]                              # E[Z|y] = σ² s_μ (Tweedie)
    cv = σ^2 * (1.0 + σ^2 * H[1])                # V(Z|y) = σ²(1 + σ² H_μμ)
    (s, H, cm, cv)
end

# NORMWISE error for vector/matrix blocks:
#
#     err = max_j |x_j - t_j| / max_j |t_j|.
#
# Componentwise relative error is meaningless at zeros of the leading-order
# expansion (e.g. the tail s_gamma vanishes identically at ytil = gamma while
# the truth is the omitted O(r) next-order term; the componentwise ratio is
# then exactly 1 while the absolute error is ~r * blockscale). The normwise
# metric is also the operationally relevant one: Newton steps and Wald
# matrices are perturbed at the level of the block norm. This is the combined
# relative/scale-aware criterion that certification requires; the
# certified claim is "every component is accurate to the stated bound TIMES
# THE BLOCK'S LARGEST COMPONENT".
function block_err(xs, ts)
    scale = max(maximum(abs, ts), floatmin(Float64))
    maximum(abs(x - t) for (x, t) in zip(xs, ts)) / scale
end

# ------------------------------------------------------------------
# evaluation grid
# ------------------------------------------------------------------

# walk from y0 to the first float where pred(y) == target (
# a single nextfloat can land on the same side of the computed mask, so the
# nominal threshold's neighbors were not the first actually-dispatched points)
function first_where(pred, y0, target::Bool)
    y = y0
    step = max(abs(y0), 1.0) * 1e-16
    while pred(y) != target
        y += target ? step : -step
        step *= 2
        isfinite(y) || return NaN
    end
    # bisect back to the boundary float
    lo, hi = target ? (y0, y) : (y, y0)
    for _ in 1:200
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        if pred(mid) == target
            target ? (hi = mid) : (lo = mid)
        else
            target ? (lo = mid) : (hi = mid)
        end
    end
    return target ? hi : lo
end

"y-points for a given (sigma, gamma): center, core, transition, thresholds, tail."
function ypoints(σ, γ)
    sc = sqrt(σ^2 + γ^2)
    ys = Float64[0.0, 0.3γ, γ, 2γ, 5γ, 20γ, 0.5sc, 2sc, 41γ, 100γ]
    for rt in (1.0e-4, 5.0e-4, 1.0e-6, 1.0e-5, 1.0e-3, 1.0e-2)  # shipped + probe crossings
        arg = σ^2 / rt - γ^2
        if arg > 0
            yt = sqrt(arg)
            pred = y -> σ^2 < rt * (y^2 + γ^2)      # the actual dispatch mask
            y_in  = first_where(pred, yt, true)     # first dispatched float
            y_out = first_where(pred, yt, false)    # last exact-branch float
            for yy in (y_out, yt, y_in)
                isfinite(yy) && push!(ys, yy)
            end
        end
    end
    push!(ys, 500sc, 1e4 * sc, 1e8 * sc)
    sort!(unique(filter(y -> isfinite(y) && y >= 0, ys)))
end

const LAMBDAS = [10.0^e for e in -8:0.5:8]   # half-decades: intermediate ratios matter

function certify(; verbose::Bool = true)
    σ = 1.0
    worst = Dict("score" => 0.0, "hessian" => 0.0, "condmean" => 0.0,
                 "condvar" => 0.0)
    worst_at = Dict{String,Tuple{Float64,Float64}}()
    negvar = 0
    for λ in LAMBDAS
        γ = λ
        dec_worst = Dict(k => 0.0 for k in keys(worst))
        for ỹ in ypoints(σ, γ)
            sref, Href, cmref, cvref = ref_all(ỹ, σ, γ)
            s = Tuple(voigt_score(ỹ, 0.0, σ, γ))
            Hm = voigt_hessian(ỹ, 0.0, σ, γ)
            H = (Hm[1, 1], Hm[1, 2], Hm[1, 3], Hm[2, 2], Hm[2, 3], Hm[3, 3])
            cm = voigt_condmean(ỹ, 0.0, σ, γ)
            cv = voigt_condvar(ỹ, 0.0, σ, γ)
            errs = Dict("score" => block_err(s, sref),
                        "hessian" => block_err(H, Href),
                        "condmean" => block_err((cm,), (cmref,)),
                        "condvar" => block_err((cv,), (cvref,)))
            cv < 0 && cvref >= 0 && (negvar += 1)
            for (k, e) in errs
                dec_worst[k] = max(dec_worst[k], e)
                if e > worst[k]
                    worst[k] = e
                    worst_at[k] = (λ, ỹ)
                end
            end
        end
        verbose && @printf("λ = %8.0e:  score %8.1e  hessian %8.1e  condmean %8.1e  condvar %8.1e\n",
                           λ, dec_worst["score"], dec_worst["hessian"],
                           dec_worst["condmean"], dec_worst["condvar"])
    end
    println("\nWorst case over the whole grid:")
    for k in ("score", "hessian", "condmean", "condvar")
        λw, yw = worst_at[k]
        @printf("  %-9s %9.2e   at (γ/σ = %.0e, ỹ = %.4g)\n", k, worst[k], λw, yw)
    end
    println("  negative conditional variances (reference nonnegative): $negvar")
    return worst, negvar
end

# ------------------------------------------------------------------
# extreme-ratio regression cases and Fisher checks
# ------------------------------------------------------------------

function regression_cases()
    ok = true
    println("\nExtreme-ratio regression cases:")
    for (σ, γ, y, what, truth) in (
            (1.0, 1e4, 0.0,    "s_σ",  -1.9999999000e-8),
            (1.0, 1e4, 1e4,    "s_σ",   9.99999965e-9),
            (1.0, 1e4, 1e5,    "s_σ",   5.86217038e-10),
            (1.0, 1e6, 4.1e7,  "s_μ",   4.87515e-8),
            (1.0, 1e6, 4.1e7,  "s_γ",   9.98811e-7))
        s = voigt_score(y, 0.0, σ, γ)
        val = what == "s_σ" ? s[2] : what == "s_μ" ? s[1] : s[3]
        rel = abs(val - truth) / abs(truth)
        pass = rel < 1e-4
        ok &= pass
        @printf("  (σ,γ,y)=(1,%.0e,%.1e) %-4s = %+.6e  (truth %+.6e, rel %.1e) %s\n",
                γ, y, what, val, truth, rel, pass ? "PASS" : "FAIL")
    end
    cv = voigt_condvar(1e8, 0.0, 1.0, 1e6)
    pass = isfinite(cv) && cv >= 0 && abs(cv - 1.0) < 1e-3
    ok &= pass
    @printf("  Var(Z|y) at (1,1e6,100γ) = %.6f  (truth ≈ 1) %s\n", cv,
            pass ? "PASS" : "FAIL")
    return ok
end

function fisher_checks()
    ok = true
    println("\nFisher information, large-ratio limit  I_σσ γ⁴/σ² → 1:")
    for λ in (1e2, 1e3, 1e4)
        vals = Float64[]
        for nodes in (200, 400, 800, 1600)
            ℐ = voigt_fisher(0.0, 1.0, λ; nodes = nodes)
            push!(vals, ℐ[2, 2] * λ^4)
        end
        spread = maximum(vals) - minimum(vals)
        # limit correction is O(σ²/γ²); require agreement at that order
        pass = abs(vals[end] - 1.0) < max(100 / λ^2, 5e-3) && spread < 0.05 * abs(vals[end]) + 1e-6
        ok &= pass
        @printf("  γ/σ = %6.0e: nodes 200..1600 → %s  %s\n", λ,
                join((@sprintf("%.4f", v) for v in vals), ", "),
                pass ? "PASS" : "FAIL")
    end
    return ok
end

# ------------------------------------------------------------------
# threshold tuning mode
# ------------------------------------------------------------------

function tune()
    σ = 1.0
    println("Minimax scan (worst-case over the grid, per candidate threshold):")
    # precompute both-branch errors on the grid
    pts = Tuple{Float64,Float64,Float64}[]      # (γ, ỹ, r)
    E = Dict{Symbol,Vector{NTuple{2,Float64}}}(:score => [], :hess => [])
    for λ in LAMBDAS, ỹ in ypoints(σ, λ)
        γ = λ
        r = σ^2 / (ỹ^2 + γ^2)
        sref, Href, _, _ = ref_all(ỹ, σ, γ)
        se, He, _, _ = f64_exact(ỹ, σ, γ)
        st, Ht, _, _ = f64_tail(ỹ, σ, γ)
        push!(pts, (γ, ỹ, r))
        push!(E[:score], (block_err(se, sref), block_err(st, sref)))
        push!(E[:hess], (block_err(He, Href), block_err(Ht, Href)))
    end
    for (name, cands) in ((:score, [1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2]),
                          (:hess, [1e-5, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2]))
        println("  $name:")
        for rt in cands
            wc = 0.0
            for (i, (_, _, r)) in enumerate(pts)
                e_ex, e_ta = E[name][i]
                wc = max(wc, r < rt ? e_ta : e_ex)
            end
            @printf("    r* = %.3e  →  worst case %.3e\n", rt, wc)
        end
    end
end

# ------------------------------------------------------------------

if "tune" in ARGS
    tune()
else
    worst, negvar = certify()
    ok_regression = regression_cases()
    ok_fisher = fisher_checks()
    # acceptance thresholds track the advertised envelope (recorded worst
    # cases: score 1.41e-10, Hessian 6.35e-7, condmean 1.6e-12, condvar
    # 6.2e-12) with a modest cross-platform margin, so a silent regression
    # of the branches or the exact paths FAILS certification rather than
    # passing under loose provisional targets
    ok = worst["score"] < 1e-8 && worst["hessian"] < 1e-5 &&
         worst["condmean"] < 1e-9 && worst["condvar"] < 1e-9 &&
         negvar == 0 && ok_regression && ok_fisher
    println(ok ? "\nCERTIFY: PASS" : "\nCERTIFY: FAIL")
    exit(ok ? 0 : 1)
end
