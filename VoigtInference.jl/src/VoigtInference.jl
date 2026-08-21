"""
    VoigtInference

Exact likelihood inference for the Voigt profile (Gauss-Cauchy convolution)
from a single Faddeeva-function evaluation per data point.

Model:  Y = μ + Z + X,  Z ~ N(0, σ²),  X ~ Cauchy(0, γ)  (Lorentzian HWHM γ),
so Y ~ 𝒱(μ, σ, γ) with density f(y) = K(x,a) / (σ√(2π)), where
K(x,a) = Re w(z) and L(x,a) = Im w(z) are the absorption and dispersion
parts of the Faddeeva function w(z) = exp(-z²) erfc(-iz), evaluated at
z = x + ia with x = (y-μ)/(σ√2), a = γ/(σ√2).

Because w'(z) = -2z w(z) + 2i/√π, all derivatives of the log-likelihood are
algebraic in (K, L): the score, Hessian, and conditional moments of the
Gaussian component are closed form, and the expected Fisher information
reduces to one-dimensional quadrature of the analytic score.

Reference: P. R. Hansen and C. Tong, "Exact likelihood inference and robust
filtering for Gauss-Cauchy convolution models", arXiv:2605.01665.
"""
module VoigtInference

using SpecialFunctions: erfcx
using Statistics: median, quantile
using LinearAlgebra: Diagonal, Symmetric, SymTridiagonal, eigen, cholesky,
                     diag, dot, norm, isposdef, I

export faddeeva, voigt_pdf, voigt_pdf_score, voigt_logpdf, voigt_loglik,
       voigt_score, voigt_hessian, voigt_fisher, voigt_mle, boundary_lr,
       voigt_condmean, voigt_condvar, rand_voigt

# ------------------------------------------------------------------
# Faddeeva function and the (K, L) pair
# ------------------------------------------------------------------

"""
    faddeeva(z)

Faddeeva function w(z) = exp(-z²) erfc(-iz), computed via the scaled
complementary error function (accurate to close to machine precision for
Im(z) > 0, the regime used throughout this package).
"""
faddeeva(z::Complex) = erfcx(-im * z)
faddeeva(z::Real) = faddeeva(complex(z))

"""
    _KL(y, μ, σ, γ) -> (K, L, x, a)

Absorption and dispersion parts K(x,a), L(x,a) of the Faddeeva function at
x = (y-μ)/(σ√2), a = γ/(σ√2). One complex evaluation; everything else in
this package is algebra on the returned pair.
"""
@inline function _KL(y::Real, μ::Real, σ::Real, γ::Real)
    x = (y - μ) / (σ * sqrt(2))
    a = γ / (σ * sqrt(2))
    w = faddeeva(complex(x, a))
    return real(w), imag(w), x, a
end

const _SQRT2PI = sqrt(2π)
const _SQRT2OVERPI = sqrt(2 / π)

"width contract for the density: finite, nonnegative, not both zero"
@inline function _check(σ, γ)
    (isfinite(σ) && isfinite(γ)) ||
        throw(DomainError((σ, γ), "σ and γ must be finite"))
    σ < 0 && throw(DomainError(σ, "σ must be ≥ 0"))
    γ < 0 && throw(DomainError(γ, "γ must be ≥ 0"))
    (σ == 0 && γ == 0) &&
        throw(DomainError((σ, γ), "σ and γ cannot both be 0 (degenerate model)"))
    return nothing
end

"width contract for the interior likelihood calculus: strictly positive.
The score, Hessian, and conditional-moment formulas are the interior
calculus; the boundary submodels have their own calculus (see voigt_mle)."
@inline function _check_interior(σ, γ)
    (isfinite(σ) && isfinite(γ)) ||
        throw(DomainError((σ, γ), "σ and γ must be finite"))
    (σ > 0 && γ > 0) ||
        throw(DomainError((σ, γ),
              "σ and γ must be > 0 for the interior likelihood calculus"))
    return nothing
end

# ------------------------------------------------------------------
# Density and log-likelihood
# ------------------------------------------------------------------

"""
    voigt_pdf(y, μ, σ, γ)

Voigt density f(y) = K(x,a)/(σ√(2π)). Boundary cases σ=0 (pure Lorentzian)
and γ=0 (pure Gaussian) are handled by their limits.
"""
function voigt_pdf(y::Real, μ::Real, σ::Real, γ::Real)
    _check(σ, γ)
    if σ == 0                       # pure Lorentzian
        return γ / (π * ((y - μ)^2 + γ^2))
    elseif γ == 0                   # pure Gaussian
        return exp(-(y - μ)^2 / (2σ^2)) / (σ * _SQRT2PI)
    end
    K, _, _, _ = _KL(y, μ, σ, γ)
    return K / (σ * _SQRT2PI)
end

"voigt_logpdf(y, μ, σ, γ): log of `voigt_pdf` (interior case σ, γ > 0)."
function voigt_logpdf(y::Real, μ::Real, σ::Real, γ::Real)
    _check_interior(σ, γ)
    K, _, _, _ = _KL(y, μ, σ, γ)
    return log(K) - log(σ) - 0.5 * log(2π)
end

"voigt_loglik(y, μ, σ, γ): sample log-likelihood Σᵢ log f(yᵢ)."
function voigt_loglik(y::AbstractVector, μ::Real, σ::Real, γ::Real)
    _check_interior(σ, γ)
    return sum(voigt_logpdf(yi, μ, σ, γ) for yi in y)
end

# ------------------------------------------------------------------
# Score and Hessian  (Hansen & Tong 2026, in Faddeeva convention)
#
# Cauchy-limit branch gating, via the expansion parameter
#
#     r = σ² / (ỹ² + γ²).
#
# The exact formulas lose digits to cancellation whenever the Gaussian
# component is a small perturbation of the Cauchy component. That happens
# BOTH deep in the tail (|ỹ| large) AND at any ỹ when γ ≫ σ (large
# imaginary Faddeeva argument): at the center the digits lost grow like
# 4 log10(γ/σ), so e.g. s_σ(0) has no correct digits by γ/σ ~ 1e4. In
# exactly that regime the moment expansion
#   f(y) = c(ỹ) + (σ²/2)c″(ỹ) + O(σ⁴ c⁴ᵗʰ),   c = Cauchy(0,γ) density,
# is accurate, with relative error O(r). Gating on r covers both failure
# modes with one criterion; the earlier |ỹ|-based switch missed the
# large-γ/σ center entirely.
#
# The branches below carry three expansion orders and are derivatives of
# one truncated log density, so their truncation error is O(r³) pointwise
# and the likelihood identities hold after integration. The branch is
# therefore MORE accurate than the exact formulas well before cancellation
# becomes visible. The thresholds are the selected optima on the finite
# validation scan, certify.jl tune (worst pointwise error: score ~1e-10 at r_s = 1e-4,
# Hessian ~6e-7 at r_h = 5e-4), and the small exact zones also remove the
# integrated information-identity violation that large exact-branch zones
# caused at γ/σ ~ 50-100. Certified by
# examples/certify.jl over γ/σ ∈ [1e-8, 1e8]; see its output for bounds.
# Validated dynamic range: γ/σ ∈ [1e-8, 1e8], |ỹ| up to 1e8 × the profile
# scale. Far outside it (magnitudes beyond ~1e150) the branch criterion's
# ỹ² + γ² overflows and the derivative routines return non-finite values
# rather than silently dispatching; keep inputs within representable
# squares.
@inline _far_tail(ỹ, σ, γ)      = σ^2 < 1.0e-4 * (ỹ^2 + γ^2)   # score, moments
@inline _far_tail_hess(ỹ, σ, γ) = σ^2 < 5.0e-4 * (ỹ^2 + γ^2)   # Hessian

# Cauchy-limit branches. Every entry is the exact derivative of ONE
# truncated expansion of the log density,
#
#     log f = log c + (σ²/2) A + (σ⁴/8) (B - A²) + σ⁶ (C/48 - AB/16 + A³/24),
#
# with c the Lorentzian density and A = c″/c, B = c⁗/c, C = c⁽⁶⁾/c. Deriving all entries
# from the same truncated log density keeps the dispatched score centered
# (E[s] = 0) and makes the dispatched Hessian its derivative, so the
# integrated information identity E[ss′] = -E[H] holds to the retained
# order. Leading-order formulas are pointwise accurate but violate these
# identities after integration: the leading term cancels under the
# expectation while the truncation error does not (E[σA] = σ³/(2γ⁴) ≠ 0;
# E[A] gives the H_σσ curvature the wrong sign). The ratios w, v, rs2
# lie in [0, 1] on the dispatch region, so
# the evaluation cannot overflow. Expressions match the Python
# implementation token for token (bit identity).
@inline function _score_tail(ỹ, σ, γ)
    den = ỹ * ỹ + γ * γ
    w = (ỹ * ỹ) / den
    v = (γ * γ) / den
    ur = ỹ / den
    gr = γ / den
    rs2 = (σ * σ) / den
    sμ = ur * (2.0 + rs2 * (6.0 * w - 10.0 * v)
                + rs2 * rs2 * (42.0 * w * w - 204.0 * w * v + 74.0 * v * v)
                + rs2 * rs2 * rs2 * (414.0 * w * w * w - 3846.0 * w * w * v
                                     + 4506.0 * w * v * v - 706.0 * v * v * v))
    sσ = (σ / den) * ((6.0 * w - 2.0 * v)
                            + rs2 * (42.0 * w * w - 108.0 * w * v
                                     + 10.0 * v * v)
                            + rs2 * rs2 * (414.0 * w * w * w - 2574.0 * w * w * v
                                           + 1674.0 * w * v * v
                                           - 74.0 * v * v * v))
    sγ = (w - v) / γ + gr * rs2 * ((2.0 * v - 14.0 * w)
                                         + rs2 * (-138.0 * w * w
                                                  + 172.0 * w * v
                                                  - 10.0 * v * v)
                                         + rs2 * rs2 * (-1686.0 * w * w * w
                                                        + 5406.0 * w * w * v
                                                        - 2306.0 * w * v * v
                                                        + 74.0 * v * v * v))
    return sμ, sσ, sγ
end

@inline function _hessian_tail(ỹ, σ, γ)
    den = ỹ * ỹ + γ * γ
    w = (ỹ * ỹ) / den
    v = (γ * γ) / den
    ur = ỹ / den
    gr = γ / den
    rs2 = (σ * σ) / den
    Hμμ = ((2.0 * w - 2.0 * v)
           + rs2 * (18.0 * w * w - 68.0 * w * v + 10.0 * v * v)
           + rs2 * rs2 * (210.0 * w * w * w - 1638.0 * w * w * v
                          + 1278.0 * w * v * v - 74.0 * v * v * v)
           + rs2 * rs2 * rs2 * (2898.0 * w * w * w * w - 37512.0 * w * w * w * v
                                + 68796.0 * w * w * v * v
                                - 22696.0 * w * v * v * v + 706.0 * v * v * v * v)) / den
    Hμσ = (σ / den) * ur * ((12.0 * w - 20.0 * v)
                                + rs2 * (168.0 * w * w - 816.0 * w * v
                                         + 296.0 * v * v)
                                + rs2 * rs2 * (2484.0 * w * w * w
                                               - 23076.0 * w * w * v
                                               + 27036.0 * w * v * v
                                               - 4236.0 * v * v * v))
    Hμγ = ur * gr * (-4.0 + rs2 * (40.0 * v - 56.0 * w)
                     + rs2 * rs2 * (-828.0 * w * w + 1928.0 * w * v
                                    - 444.0 * v * v)
                     + rs2 * rs2 * rs2 * (-13488.0 * w * w * w
                                          + 64176.0 * w * w * v
                                          - 49296.0 * w * v * v
                                          + 5648.0 * v * v * v))
    Hσσ = ((6.0 * w - 2.0 * v)
           + rs2 * (126.0 * w * w - 324.0 * w * v + 30.0 * v * v)
           + rs2 * rs2 * (2070.0 * w * w * w - 12870.0 * w * w * v
                          + 8370.0 * w * v * v - 370.0 * v * v * v)) / den
    Hγσ = (σ / den) * gr * ((4.0 * v - 28.0 * w)
                                + rs2 * (-552.0 * w * w + 688.0 * w * v
                                         - 40.0 * v * v)
                                + rs2 * rs2 * (-10116.0 * w * w * w
                                               + 32436.0 * w * w * v
                                               - 13836.0 * w * v * v
                                               + 444.0 * v * v * v))
    Hγγ = ((v - 4.0 * w)
           + rs2 * (-14.0 * w * w + 76.0 * w * v - 6.0 * v * v)
           + rs2 * rs2 * (-138.0 * w * w * w + 1758.0 * w * w * v
                          - 1254.0 * w * v * v + 50.0 * v * v * v)
           + rs2 * rs2 * rs2 * (-1686.0 * w * w * w * w + 38136.0 * w * w * w * v
                                - 70996.0 * w * w * v * v
                                + 21272.0 * w * v * v * v - 518.0 * v * v * v * v)) / den -
          (w * w) / (γ * γ)
    return Hμμ, Hμσ, Hμγ, Hσσ, Hγσ, Hγγ
end

"""
    voigt_score(y, μ, σ, γ) -> Vector{Float64}

Score ∂ log f(y;θ)/∂θ, θ = (μ, σ, γ):

    s_μ = (ỹ - γ L/K)/σ²
    s_σ = ((ỹ² - γ² - σ²) K - 2 γ ỹ L + √(2/π) σ γ) / (σ³ K)
    s_γ = (γ K + ỹ L - √(2/π) σ) / (σ² K)

with ỹ = y - μ.
"""
function voigt_score(y::Real, μ::Real, σ::Real, γ::Real)
    _check_interior(σ, γ)
    ỹ = y - μ
    if _far_tail(ỹ, σ, γ)
        sμ, sσ, sγ = _score_tail(ỹ, σ, γ)
        return [sμ, sσ, sγ]
    end
    K, L, _, _ = _KL(y, μ, σ, γ)
    sμ = (ỹ - γ * L / K) / σ^2
    sσ = ((ỹ^2 - γ^2 - σ^2) * K - 2γ * ỹ * L + _SQRT2OVERPI * σ * γ) / (σ^3 * K)
    sγ = (γ * K + ỹ * L - _SQRT2OVERPI * σ) / (σ^2 * K)
    return [sμ, sσ, sγ]
end

"""
    voigt_pdf_score(y, μ, σ, γ) -> (pdf, score)

Density and score from ONE Faddeeva evaluation. Bit-identical to calling
`voigt_pdf` and `voigt_score` separately (same (K, L) algebra, same far-tail
branch for the score) at half the special-function cost. This is the natural
primitive for least-squares Jacobians of the lineshape, ∂f/∂θ = f sθ, as
used by `examples/raman.jl`.
"""
function voigt_pdf_score(y::Real, μ::Real, σ::Real, γ::Real)
    _check_interior(σ, γ)
    ỹ = y - μ
    if _far_tail(ỹ, σ, γ)
        K, _, _, _ = _KL(y, μ, σ, γ)
        sμ, sσ, sγ = _score_tail(ỹ, σ, γ)
        return K / (σ * _SQRT2PI), [sμ, sσ, sγ]
    end
    K, L, _, _ = _KL(y, μ, σ, γ)
    sμ = (ỹ - γ * L / K) / σ^2
    sσ = ((ỹ^2 - γ^2 - σ^2) * K - 2γ * ỹ * L + _SQRT2OVERPI * σ * γ) / (σ^3 * K)
    sγ = (γ * K + ỹ * L - _SQRT2OVERPI * σ) / (σ^2 * K)
    return K / (σ * _SQRT2PI), [sμ, sσ, sγ]
end

"""
    voigt_hessian(y, μ, σ, γ) -> Symmetric 3×3 matrix

Hessian ∂² log f(y;θ)/∂θ∂θ′, assembled recursively from the score components
and L/K -- no additional special-function evaluations. Parameter order
(μ, σ, γ).
"""
function voigt_hessian(y::Real, μ::Real, σ::Real, γ::Real)
    _check_interior(σ, γ)
    ỹ = y - μ
    if _far_tail_hess(ỹ, σ, γ)
        Hμμ, Hμσ, Hμγ, Hσσ, Hγσ, Hγγ = _hessian_tail(ỹ, σ, γ)
        return Symmetric([Hμμ Hμσ Hμγ;
                          Hμσ Hσσ Hγσ;
                          Hμγ Hγσ Hγγ])
    end
    K, L, _, _ = _KL(y, μ, σ, γ)
    r = L / K
    sμ = (ỹ - γ * L / K) / σ^2      # same expression as voigt_score, so that
                                    # the Hessian is built from exactly the score
                                    # the package reports
    sσ = ((ỹ^2 - γ^2 - σ^2) * K - 2γ * ỹ * L + _SQRT2OVERPI * σ * γ) / (σ^3 * K)
    sγ = (γ * K + ỹ * L - _SQRT2OVERPI * σ) / (σ^2 * K)

    Hμμ = sσ / σ - sμ^2
    Hγγ = -sσ / σ - sγ^2
    Hμγ = (ỹ * sγ + γ * sμ - r) / σ^2 - sμ * sγ
    Hμσ = -(sμ + γ * Hμγ - ỹ * Hμμ) / σ
    Hγσ = -(sγ + γ * Hγγ - ỹ * Hμγ) / σ
    Hσσ = -(sσ + γ * Hγσ - ỹ * Hμσ) / σ

    return Symmetric([Hμμ Hμσ Hμγ;
                      Hμσ Hσσ Hγσ;
                      Hμγ Hγσ Hγγ])
end

# ------------------------------------------------------------------
# Conditional moments of the Gaussian component (deconvolution)
# ------------------------------------------------------------------

"""
    voigt_condmean(y, μ, σ, γ)

E[Z | Y = y] = ỹ - γ L/K  (Tweedie's formula). Redescending: → 0 as |ỹ| → ∞,
so extreme deviations are attributed to the Lorentzian component. The
conditional mean of the Lorentzian component is E[X | Y = y] = γ L/K.
"""
function voigt_condmean(y::Real, μ::Real, σ::Real, γ::Real)
    _check_interior(σ, γ)
    ỹ = y - μ
    if _far_tail(ỹ, σ, γ)                    # E[Z|y] = σ² s_μ (Tweedie)
        sμ, _, _ = _score_tail(ỹ, σ, γ)
        return σ^2 * sμ
    end
    K, L, _, _ = _KL(y, μ, σ, γ)
    return ỹ - γ * L / K
end

"""
    voigt_condvar(y, μ, σ, γ)

V(Z | Y = y) = √(2/π) σγ/K - γ² (1 + (L/K)²). Equals σ² at the extrema of
the conditional mean; → σ² as |y - μ| → ∞.
"""
function voigt_condvar(y::Real, μ::Real, σ::Real, γ::Real)
    _check_interior(σ, γ)
    ỹ = y - μ
    if _far_tail(ỹ, σ, γ)                    # V(Z|y) = σ²(1 + σ² H_μμ)
        Hμμ = first(_hessian_tail(ỹ, σ, γ))
        return σ^2 * (1.0 + σ^2 * Hμμ)
    end
    K, L, _, _ = _KL(y, μ, σ, γ)
    return _SQRT2OVERPI * σ * γ / K - γ^2 * (1 + (L / K)^2)
end

# ------------------------------------------------------------------
# Fisher information (expected), by Gauss-Legendre quadrature
# ------------------------------------------------------------------

function _gauss_legendre(n::Int)
    β = [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]
    E = eigen(SymTridiagonal(zeros(n), β))
    return E.values, 2 .* abs2.(E.vectors[1, :])
end

# the nodes do not depend on the parameters; the eigendecomposition dominates
# the cost of a voigt_fisher call, so cache them
const _GL_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
const _GL_LOCK = ReentrantLock()
# lock-guarded: `voigt_fisher` may be called from user threads, and get! on a
# Dict is not thread-safe
_gauss_legendre_cached(n::Int) =
    lock(_GL_LOCK) do
        get!(() -> _gauss_legendre(n), _GL_CACHE, n)
    end

"""
    voigt_fisher(μ, σ, γ; nodes=400) -> Symmetric 3×3 matrix

Expected Fisher information ℐ(θ) = E[s s′], computed by Gauss-Legendre
quadrature after the substitution y = μ + (σ+γ) tan(t), t ∈ (-π/2, π/2),
which maps the Cauchy-tailed integrand to a bounded one. By the information
matrix equality ℐ(θ) = -E[H]; by symmetry ℐ is block diagonal
(ℐ_μσ = ℐ_μγ = 0).
"""
function voigt_fisher(μ::Real, σ::Real, γ::Real; nodes::Int = 400)
    (isfinite(μ) && isfinite(σ) && isfinite(γ)) ||
        throw(ArgumentError("μ, σ, γ must be finite"))
    (σ > 0 && γ > 0) ||
        throw(ArgumentError("σ and γ must be > 0 for the Fisher information"))
    nodes ≥ 2 || throw(ArgumentError("nodes must be ≥ 2"))
    t, wq = _gauss_legendre_cached(nodes)
    scale = σ + γ
    ℐ = zeros(3, 3)
    for (ti, wi) in zip(t, wq)
        u = (π / 2) * ti
        y = μ + scale * tan(u)
        jac = (π / 2) * scale / cos(u)^2
        s = voigt_score(y, μ, σ, γ)
        f = voigt_pdf(y, μ, σ, γ)
        c = wi * jac * f
        @inbounds for j in 1:3, k in 1:3
            ℐ[j, k] += c * s[j] * s[k]
        end
    end
    return Symmetric(ℐ)
end

# ------------------------------------------------------------------
# Maximum likelihood estimation
# ------------------------------------------------------------------

# moment-free starting values (medians/quantiles; Voigt has no moments)
function _startvalues(y::AbstractVector)
    μ0 = median(y)
    d = abs.(y .- μ0)
    # tails are Lorentzian: P(|Y-μ| > c) ≈ 2γ/(πc)  =>  γ ≈ 0.05 π c at c = q90(|d|)
    γ0 = max(0.05π * quantile(d, 0.90), 1e-8)
    iqr = quantile(y, 0.75) - quantile(y, 0.25)
    # core is Gaussian-plus-Lorentzian: subtract the Lorentzian IQR (= 2γ)
    σ0 = max((iqr - 2γ0) / 1.349, 0.05 * iqr, 1e-8)
    return μ0, σ0, γ0
end

_avg_loglik(y, η) = voigt_loglik(y, η[1], exp(η[2]), exp(η[3])) / length(y)

"""
    _loglik_grad_hess(y, μ, σ, γ) -> (ll, g, H)

Fused kernel: sample log-likelihood, summed score, and summed Hessian from a
single Faddeeva evaluation per observation (the public functions each
re-evaluate w(z); calling them separately costs ~3.8x the density where this
kernel costs ~1.02x). The expression arrangement matches `voigt_score` and
`voigt_hessian` token for token, so the results are bit-identical to the
public API; do not rearrange algebraically equivalent expressions here (see
the cross-check note in the repository).
"""
function _loglik_grad_hess(y, μ, σ, γ)
    ll = 0.0
    gμ = 0.0; gσ = 0.0; gγ = 0.0
    hμμ = 0.0; hμσ = 0.0; hμγ = 0.0
    hσσ = 0.0; hγσ = 0.0; hγγ = 0.0
    s2 = σ * σ
    den2 = σ * sqrt(2)
    aim = γ / den2
    logconst = log(σ) + 0.5 * log(2π)
    g2 = γ * γ
    # branch gating on r = σ²/(ỹ²+γ²): Cauchy-limit branch where r < threshold
    r_s = 1.0e-4                       # score switch   (matches _far_tail)
    r_h = 5.0e-4                       # Hessian switch (matches _far_tail_hess)
    @inbounds for yi in y
        ỹ = yi - μ
        ỹ2 = ỹ * ỹ
        w = faddeeva(complex(ỹ / den2, aim))
        K = real(w)
        L = imag(w)
        r = L / K
        ll += log(K) - logconst
        sμ = (ỹ - γ * L / K) / s2
        sσ = ((ỹ2 - γ * γ - s2) * K - 2 * γ * ỹ * L + _SQRT2OVERPI * σ * γ) / (σ^3 * K)
        sγ = (γ * K + ỹ * L - _SQRT2OVERPI * σ) / (s2 * K)
        hmm = sσ / σ - sμ * sμ
        hgg = -sσ / σ - sγ * sγ
        hmg = (ỹ * sγ + γ * sμ - r) / s2 - sμ * sγ
        hms = -(sμ + γ * hmg - ỹ * hmm) / σ
        hgs = -(sγ + γ * hgg - ỹ * hmg) / σ
        hss = -(sσ + γ * hgs - ỹ * hms) / σ
        if s2 < r_s * (ỹ2 + g2)                        # Cauchy-limit score
            sμ, sσ, sγ = _score_tail(ỹ, σ, γ)
        end
        if s2 < r_h * (ỹ2 + g2)                        # Cauchy-limit Hessian
            hmm, hms, hmg, hss, hgs, hgg = _hessian_tail(ỹ, σ, γ)
        end
        gμ += sμ; gσ += sσ; gγ += sγ
        hμμ += hmm; hμσ += hms; hμγ += hmg
        hσσ += hss; hγσ += hgs; hγγ += hgg
    end
    g = [gμ, gσ, gγ]
    H = [hμμ hμσ hμγ; hμσ hσσ hγσ; hμγ hγσ hγγ]
    return ll, g, H
end

function _avg_grad_hess(y, η)
    μ, σ, γ = η[1], exp(η[2]), exp(η[3])
    n = length(y)
    _, g, H = _loglik_grad_hess(y, μ, σ, γ)
    g ./= n
    H ./= n
    # chain rule to η = (μ, log σ, log γ)
    D = Diagonal([1.0, σ, γ])
    Hη = D * H * D + Diagonal([0.0, σ * g[2], γ * g[3]])
    gη = D * g
    return gη, Symmetric(Matrix(Hη))
end

# ------------------------------------------------------------------
# Boundary submodels (likelihood-based boundary diagnosis)
# ------------------------------------------------------------------

"Gaussian-submodel MLE (the gamma = 0 boundary): closed form. Returns (mu, sigma, loglik)."
function _gaussian_fit(y)
    n = length(y)
    m = sum(y) / n
    s2 = sum(abs2, yi - m for yi in y) / n
    return m, sqrt(s2), -0.5 * n * (log(2π * s2) + 1)
end

"Cauchy-submodel MLE (the sigma = 0 boundary), by safeguarded Newton in
(mu, log gamma) with the Fisher-scoring direction as fallback. The Cauchy
expected information is diagonal and known, I(mu) = n/(2 gamma^2) and
I(log gamma) = n/2, so the scoring direction is always an ascent direction;
Newton supplies the quadratic tail. Cost is deterministic and small
(typically < 15 iterations). Returns (mu, gamma, loglik, converged)."
function _cauchy_fit(y; maxiter::Int = 100, gtol::Real = 1e-9)
    n = length(y)
    μ = median(y)
    γ = max(quantile(abs.(y .- μ), 0.5), 1e-12)   # median |y-μ| estimates γ
    cll(μv, γv) = sum(log(γv) - log(π) - log((yi - μv)^2 + γv^2) for yi in y)
    ll = cll(μ, γ)
    converged = false
    for _ in 1:maxiter
        g2 = γ^2
        gμ = 0.0; gγ = 0.0; hμμ = 0.0; hμγ = 0.0
        for yi in y
            d = yi - μ
            den = d^2 + g2
            gμ  += 2d / den
            gγ  += 1 / γ - 2γ / den
            hμμ += 2 * (d^2 - g2) / den^2
            hμγ += -4d * γ / den^2
        end
        gs = γ * gγ                                # d/d(log γ)
        if hypot(gμ, gs) < gtol * n
            converged = true
            break
        end
        # observed information in (μ, log γ):  hγγ = -n/γ² - hμμ  (Cauchy identity)
        hγγ = -n / g2 - hμμ
        hss = g2 * hγγ + gs
        hμs = γ * hμγ
        det = hμμ * hss - hμs^2
        use_newton = det > 0 && hμμ < 0            # negative definite
        local dμ, ds
        if use_newton
            dμ = -(hss * gμ - hμs * gs) / det
            ds = -(hμμ * gs - hμs * gμ) / det
            use_newton = dμ * gμ + ds * gs > 0     # ascent check
        end
        if !use_newton                             # Fisher scoring
            dμ = 2g2 * gμ / n
            ds = 2gs / n
        end
        gp0 = dμ * gμ + ds * gs
        t = 1.0
        accepted = false
        stalled = false
        for _ in 1:30
            μn = μ + t * dμ
            γn = γ * exp(t * ds)
            lln = cll(μn, γn)
            if isfinite(lln) && lln ≥ ll + 1e-4 * t * gp0
                accepted = true
                stalled = abs(lln - ll) ≤ 1e-13 * (abs(ll) + 1)
                μ, γ, ll = μn, γn, lln
                break
            end
            t /= 2
        end
        accepted || break                          # numerically stationary
        if stalled
            converged = true                       # no double-precision headroom left
            break
        end
    end
    return μ, γ, ll, converged
end

# ------------------------------------------------------------------
# Newton core with diagnostics
# ------------------------------------------------------------------

"Projected gradient norm: components pushing into an active clamp are zeroed."
function _projnorm(g, η, lb, ub)
    g2 = g[2]; g3 = g[3]
    ((η[2] ≤ lb && g2 < 0) || (η[2] ≥ ub && g2 > 0)) && (g2 = 0.0)
    ((η[3] ≤ lb && g3 < 0) || (η[3] ≥ ub && g3 > 0)) && (g3 = 0.0)
    return sqrt(g[1]^2 + g2^2 + g3^2)
end

function _newton_core(y, η0, lb, ub; maxiter, gtol, verbose)
    η = copy(η0)
    ll = _avg_loglik(y, η)
    isfinite(ll) || return (η, -Inf, false, 0, :nonfinite_start, NaN, NaN)
    converged = false
    reason = :max_iterations
    iter = 0
    g = zeros(3)
    blind = 0                # resolution-limited full steps taken on trust
    while iter < maxiter
        iter += 1
        g, H = _avg_grad_hess(y, η)
        if !(all(isfinite, g) && all(isfinite, H))
            reason = :nonfinite_derivatives
            break
        end
        if _projnorm(g, η, lb, ub) < gtol
            converged = true
            reason = :gradient_converged
            break
        end
        # ascent direction: solve (-H + λI)Δ = g with the smallest ridge λ ≥ 0
        λ = 0.0
        Δ = zeros(3)
        found = false
        while λ ≤ 1e12
            A = -Matrix(H) + λ * Matrix(1.0I, 3, 3)
            if isposdef(Symmetric(A))
                Δ = Symmetric(A) \ g
                if dot(Δ, g) > 0
                    found = true
                    break
                end
            end
            λ = λ == 0 ? 1e-6 : 10λ
        end
        if !found
            reason = :no_ascent_direction
            break
        end
        # projected backtracking line search: the Armijo condition is tested
        # against the EXECUTED (clamped) step p, not the proposed t*Δ
        # Backtracking stops as soon as the predicted
        # improvement 1e-4*t*g'p falls below the double-precision resolution
        # of the average log likelihood: no floating-point value could then
        # pass the Armijo test, so further trials only burn Faddeeva passes.
        # (The stall previously cost up to 40 extra passes
        # and was masked by a hard-coded pgnorm < 1e-4 fallback that
        # silently replaced the requested gtol; both are removed. A stalled
        # fit reports :stalled_near_stationary with converged = false and
        # its projected gradient norm, and the caller decides.)
        # When the Armijo margin 1e-4*t*g'p falls below the resolution of
        # the average log likelihood, the sufficient-decrease comparison is
        # uninformative: near the optimum this happens one step before the
        # gradient tolerance is met. In that regime the full Newton step is
        # taken on trust (it is an ascent direction and cannot move ll by
        # more than its resolution), the projected-gradient check at the top
        # of the next iteration remains the sole arbiter of convergence, and
        # the number of such resolution-limited steps is capped.
        t = 1.0
        improved = false
        stalled = false
        # resolution of the average log likelihood: machine epsilon scaled
        # by its magnitude AND by the summation noise of an n-term total
        # (grows like sqrt(n) in practice; too tight a slack rejects trusted
        # steps at large n through pure rounding jitter)
        eps_ll = eps(Float64) * (abs(ll) + 1.0) * (16.0 + sqrt(length(y)))
        for _ in 1:40
            ηc = η .+ t .* Δ
            ηc[2] = clamp(ηc[2], lb, ub)
            ηc[3] = clamp(ηc[3], lb, ub)
            p = ηc .- η
            if all(iszero, p)
                stalled = true                 # step underflowed entirely
                break
            end
            gp = dot(g, p)
            if gp > 0
                limited = 1e-4 * gp < eps_ll
                if limited && blind ≥ 24
                    stalled = true             # trust budget exhausted
                    break
                end
                llc = _avg_loglik(y, ηc)
                if isfinite(llc) && (llc ≥ ll + 1e-4 * gp ||
                                     (limited && llc ≥ ll - 2.0 * eps_ll))
                    η, ll = ηc, llc
                    improved = true
                    blind += limited ? 1 : 0
                    break
                end
                if limited
                    stalled = true             # even the trusted step failed
                    break
                end
            end
            t /= 2
        end
        if !improved
            reason = stalled ? :stalled_near_stationary : :line_search_failed
            break
        end
        verbose && println("iter $iter  loglik $(length(y)*ll)  |g| $(norm(g))")
    end
    if reason == :max_iterations
        # the final permitted iteration ACCEPTED a step, so the loop exited
        # before evaluating derivatives at the returned point: the stored
        # gradient describes the previous iterate. Evaluate the returned
        # point once and reapply the requested convergence criterion, so
        # the reported diagnostics and the convergence verdict belong to
        # the parameters actually returned.
        g, _ = _avg_grad_hess(y, η)
        if all(isfinite, g) && _projnorm(g, η, lb, ub) < gtol
            converged = true
            reason = :gradient_converged
        end
    end
    return η, ll, converged, iter, reason, norm(g), _projnorm(g, η, lb, ub)
end

"""
    voigt_mle(y; maxiter=200, gtol=1e-8, nodes=400, starts=1, verbose=false)

Newton-based local maximization of the exact Voigt log-likelihood in
(μ, log σ, log γ) coordinates, with analytic, machine-stabilized derivatives,
a ridge safeguard, and a projected backtracking line search. With `starts > 1`
a deterministic set of dispersed starting values is tried and the best local
maximum is returned; without it, the result is a local stationary or
boundary-clamped candidate, not a certified global maximum.

Returns a NamedTuple with fields
  `μ, σ, γ`            : parameter estimates
  `se`                 : asymptotic Fisher-information standard errors,
                         sqrt(diag(I⁻¹/n)) with I the expected information by
                         quadrature; NaN unless I is positive definite AND the
                         estimate is interior
  `se_obs`             : asymptotic standard errors from the observed
                         information -Σᵢ H(yᵢ); same validity rules
  `vcov`               : I(θ̂)⁻¹/n (NaN under the same rules)
  `loglik`             : maximized Voigt log-likelihood
  `converged`          : projected-gradient criterion met
  `termination`        : :gradient_converged | :max_iterations |
                         :stalled_near_stationary | :submodel_dominates |
                         :no_ascent_direction | :line_search_failed |
                         :nonfinite_derivatives | :nonfinite_start
  `sigma_boundary, gamma_boundary, upper_boundary` : active clamps at the
                         optimum (γ→0: the fitted model is numerically
                         Gaussian; σ→0: numerically Cauchy)
  `gradient_norm, projected_gradient_norm` : final average-gradient norms
  `expected_info_posdef, observed_info_posdef` : positive definiteness of the
                         two information estimates
  `loglik_gaussian, loglik_cauchy` : boundary-submodel maximized
                         log-likelihoods (γ=0 closed form; σ=0 by Newton),
                         with the fits themselves in `gaussian_fit` and
                         `cauchy_fit` — compare with `loglik` to diagnose
                         effective boundary solutions
  `starts, iterations`

Standard errors are asymptotic likelihood-based (Wald) quantities from the
interior asymptotic-normality theorem for the iid Voigt MLE in the companion
paper (arXiv:2605.01665); they are deliberately NOT reported at boundaries,
where that theorem does not apply — use the boundary submodels instead. The
optimizer never throws on finite, nondegenerate samples; degenerate input
for which the closed-family MLE does not exist (a value tied in at least
half of the observations) is rejected with an explanatory error.
"""
function voigt_mle(y::AbstractVector; maxiter::Int = 200, gtol::Real = 1e-8,
                   nodes::Int = 400, starts::Int = 1, verbose::Bool = false)
    n = length(y)
    n ≥ 3 || throw(ArgumentError("need at least 3 observations"))
    all(isfinite, y) || throw(ArgumentError("data contain non-finite values"))
    # MLE-existence guard: if one exact value occupies at least HALF the
    # sample, the closed-family likelihood is unbounded or its supremum is
    # not attained, so no finite clamp value may be reported as a maximized
    # likelihood. With the tied value as location and γ → 0, the Cauchy
    # submodel gives ℓ_C = (n - 2m) log γ + O(1): a strict majority
    # (2m > n) sends it to +∞; an exact half (2m = n) can leave a finite
    # zero-width supremum that no positive width attains. Exact ties are a
    # probability-zero event under the continuous model but occur in
    # rounded or digitized data.
    ys = sort(y)
    m = 1; run = 1
    for i in 2:n
        run = ys[i] == ys[i-1] ? run + 1 : 1
        run > m && (m = run)
    end
    2m ≥ n && throw(ArgumentError(
        "closed-family MLE does not exist: one value occurs $m times in " *
        "$n observations (at least half). The likelihood is unbounded for " *
        "a strict majority tie, and the zero-width supremum may be " *
        "nonattained for an exact half; the data are degenerate for this " *
        "model (rounded or truncated input?)"))
    maxiter ≥ 1 || throw(ArgumentError("maxiter must be ≥ 1"))
    gtol > 0 || throw(ArgumentError("gtol must be > 0"))
    nodes ≥ 2 || throw(ArgumentError("nodes must be ≥ 2"))
    1 ≤ starts ≤ 7 || throw(ArgumentError(
        "starts must be an integer in 1..7 (7 deterministic starting values exist)"))
    μ0, σ0, γ0 = _startvalues(y)
    s0 = max(σ0, γ0)
    lb, ub = log(1e-8 * s0), log(1e8 * s0)
    startlist = [[μ0, log(σ0), log(γ0)]]
    if starts > 1
        iqr = quantile(y, 0.75) - quantile(y, 0.25)
        for (fσ, fγ, dμ) in ((4.0, 0.25, 0.0), (0.25, 4.0, 0.0),
                             (1.0, 1.0, 0.5iqr), (1.0, 1.0, -0.5iqr),
                             (8.0, 8.0, 0.0), (0.125, 0.125, 0.0))
            length(startlist) ≥ starts && break
            push!(startlist, [μ0 + dμ, clamp(log(σ0 * fσ), lb, ub),
                              clamp(log(γ0 * fγ), lb, ub)])
        end
    end
    best = nothing
    for η0 in startlist
        res = _newton_core(y, η0, lb, ub; maxiter, gtol, verbose)
        (best === nothing || res[2] > best[2]) && (best = res)
    end

    # boundary submodels (fitted before the flags: they feed the guard)
    μg, σg, llg = _gaussian_fit(y)
    μc, γc, llc, cauchy_conv = _cauchy_fit(y)

    # submodel-domination guard: the Voigt family nests both
    # submodels, so a full-model candidate below either submodel likelihood
    # is a failed local search. Refit from submodel-informed starts; if the
    # deficit persists, report it as the termination reason.
    # tolerance scales with the log-likelihood magnitude: a fit clamped at
    # the tiny-width bound sits legitimately a clamp-residual below the exact
    # submodel likelihood (~1e-3 at n = 1e5), which is not a failed search
    tol_dom = 1e-7 * (1.0 + abs(max(llg, llc)))
    # boundary-aware refinement: ANY candidate within 4 log-likelihood units
    # of the better submodel (not only outright deficits) is refit from two
    # submodel-informed starts (nuisance parameters at the submodel fit,
    # vanishing other width) followed by full joint Newton refinement. This
    # resolves small genuine interior improvements near the boundaries --
    # exactly what the calibrated boundary likelihood-ratio statistics are
    # made of; the optimizer's objective resolution (~eps*|ll|*(16+sqrt n))
    # is orders of magnitude below the calibrated critical values. A deficit
    # persisting beyond tol_dom is reported as the termination reason.
    if n * best[2] < max(llg, llc) + 4.0
        informed = ([μg, clamp(log(σg), lb, ub), clamp(log(1e-3 * σg), lb, ub)],
                    [μc, clamp(log(1e-3 * γc), lb, ub), clamp(log(γc), lb, ub)])
        for η0 in informed
            res = _newton_core(y, η0, lb, ub; maxiter, gtol, verbose)
            res[2] > best[2] && (best = res)
        end
        if n * best[2] < max(llg, llc) - tol_dom
            best = (best[1], best[2], best[3], best[4], :submodel_dominates,
                    best[6], best[7])
        end
    end
    η, ll, converged, iter, reason, gnorm, pgnorm = best
    ll_total = n * ll
    μ̂, σ̂, γ̂ = η[1], exp(η[2]), exp(η[3])
    # boundary flags: clamp active, OR width negligible relative to the other
    # width (the log-width gradient vanishes with the width — like γ for the
    # Lorentzian width and like σ² for the Gaussian width, via τ = σ² — so
    # the optimizer can satisfy the gradient criterion at a tiny width
    # without touching the literal clamp), OR likelihood
    # evidence: a Voigt fit that is not strictly better than a nested
    # boundary submodel is effectively that submodel, whatever the fitted
    # width says
    sigma_boundary = η[2] ≤ lb + 1e-10 || σ̂ < 1e-6 * γ̂ || ll_total ≤ llc + 1e-7
    gamma_boundary = η[3] ≤ lb + 1e-10 || γ̂ < 1e-6 * σ̂ || ll_total ≤ llg + 1e-7
    upper_boundary = η[2] ≥ ub - 1e-10 || η[3] ≥ ub - 1e-10
    at_boundary = sigma_boundary || gamma_boundary || upper_boundary
    # Wald standard errors require a positive-definite information matrix and
    # an interior estimate; otherwise NaN plus diagnostics — a
    # clipped zero would misrepresent an invalid covariance as certainty
    se = fill(NaN, 3)
    vcov = fill(NaN, 3, 3)
    expected_pd = false
    try
        ℐ = Matrix(voigt_fisher(μ̂, σ̂, γ̂; nodes = nodes))
        expected_pd = isposdef(Symmetric(ℐ))
        if expected_pd && !at_boundary
            vcov = inv(Symmetric(ℐ)) / n
            se = sqrt.(diag(vcov))
        end
    catch
    end
    se_obs = fill(NaN, 3)
    observed_pd = false
    try
        Hobs = sum(voigt_hessian(yi, μ̂, σ̂, γ̂) for yi in y)
        M = -Symmetric(Matrix(Hobs))
        observed_pd = isposdef(M)
        if observed_pd && !at_boundary
            se_obs = sqrt.(diag(inv(M)))
        end
    catch
    end
    return (μ = μ̂, σ = σ̂, γ = γ̂, se = se, se_obs = se_obs, vcov = vcov,
            loglik = ll_total, converged = converged, termination = reason,
            sigma_boundary = sigma_boundary, gamma_boundary = gamma_boundary,
            upper_boundary = upper_boundary,
            gradient_norm = gnorm, projected_gradient_norm = pgnorm,
            expected_info_posdef = expected_pd,
            observed_info_posdef = observed_pd,
            loglik_gaussian = llg, loglik_cauchy = llc,
            gaussian_fit = (μ = μg, σ = σg), cauchy_fit = (μ = μc, γ = γc),
            cauchy_fit_converged = cauchy_conv,
            # closed-form submodel standard errors (valid inference for the
            # nested model when the full fit is on that boundary): Gaussian
            # (σg/√n, σg/√(2n)); Cauchy expected info is diagonal with
            # I(μ) = I(γ) = n/(2γ²)
            gaussian_se = (σg / sqrt(n), σg / sqrt(2.0 * n)),
            cauchy_se = (γc * sqrt(2.0 / n), γc * sqrt(2.0 / n)),
            starts = length(startlist), iterations = iter)
end

# ------------------------------------------------------------------
# Boundary likelihood-ratio statistics (closed family)
# ------------------------------------------------------------------

"""
    boundary_lr(res) -> (lr_gaussian, lr_cauchy)

Boundary likelihood-ratio statistics against the CLOSED Voigt family.
The closed family includes the γ = 0 (Gaussian) and σ = 0 (Cauchy)
boundary submodels, so the full-model likelihood is the maximum of the
interior candidate and both submodel likelihoods, and each statistic is
nonnegative by construction:

    LR_sub = max(0, 2 (max(ℓ_interior, ℓ_G, ℓ_C) - ℓ_sub)).

The closed-family maximum is what makes the statistic a likelihood
ratio; the outer clamp only absorbs the last-ulp roundoff of the max.
A raw difference `2(res.loglik - res.loglik_gaussian)` is NOT a valid
LR: near a boundary the interior candidate legitimately sits a
clamp-residual below the exact submodel likelihood, so the raw
difference can be negative by far more than roundoff.

The interior component is the RETURNED multistart interior candidate
(the optimizer is a local method), so this is the computed closed-family
comparison, not a certificate of the global interior supremum.
"""
function boundary_lr(res)
    lls = (res.loglik, res.loglik_gaussian, res.loglik_cauchy)
    all(isfinite, lls) || throw(ArgumentError(
        "boundary_lr requires finite likelihoods, got $lls; a nonfinite " *
        "submodel likelihood indicates degenerate input"))
    llfull = max(lls...)
    return (max(0.0, 2 * (llfull - res.loglik_gaussian)),
            max(0.0, 2 * (llfull - res.loglik_cauchy)))
end

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------

"""
    rand_voigt(rng, n, μ, σ, γ)

Draw `n` iid variates from 𝒱(μ, σ, γ) by exact convolution:
Y = μ + σ N(0,1) + γ tan(π(U - 1/2)).
"""
function rand_voigt(rng, n::Int, μ::Real, σ::Real, γ::Real)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    (isfinite(μ) && isfinite(σ) && isfinite(γ)) ||
        throw(ArgumentError("μ, σ, γ must be finite"))
    σ < 0 && throw(DomainError(σ, "σ must be ≥ 0"))
    γ < 0 && throw(DomainError(γ, "γ must be ≥ 0"))
    σ == 0 && γ == 0 &&
        throw(ArgumentError("σ and γ cannot both be 0 (degenerate model)"))
    return μ .+ σ .* randn(rng, n) .+ γ .* tan.(π .* (rand(rng, n) .- 0.5))
end

end # module
