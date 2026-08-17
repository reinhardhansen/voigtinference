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
algebraic in (K, L): the score, Hessian, Fisher information, and the
conditional moments of the Gaussian component are closed form.

Reference: P. R. Hansen and C. Tong, "Exact likelihood inference and robust
filtering for Gauss-Cauchy convolution models", arXiv:2605.01665.
"""
module VoigtInference

using SpecialFunctions: erfcx
using Statistics: median, quantile
using LinearAlgebra: Diagonal, Symmetric, SymTridiagonal, eigen, cholesky,
                     diag, dot, norm, isposdef, I

export faddeeva, voigt_pdf, voigt_logpdf, voigt_loglik,
       voigt_score, voigt_hessian, voigt_fisher, voigt_mle,
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

# ------------------------------------------------------------------
# Density and log-likelihood
# ------------------------------------------------------------------

"""
    voigt_pdf(y, μ, σ, γ)

Voigt density f(y) = K(x,a)/(σ√(2π)). Boundary cases σ=0 (pure Lorentzian)
and γ=0 (pure Gaussian) are handled by their limits.
"""
function voigt_pdf(y::Real, μ::Real, σ::Real, γ::Real)
    σ < 0 && throw(DomainError(σ, "σ must be ≥ 0"))
    γ < 0 && throw(DomainError(γ, "γ must be ≥ 0"))
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
    K, _, _, _ = _KL(y, μ, σ, γ)
    return log(K) - log(σ) - 0.5 * log(2π)
end

"voigt_loglik(y, μ, σ, γ): sample log-likelihood Σᵢ log f(yᵢ)."
voigt_loglik(y::AbstractVector, μ::Real, σ::Real, γ::Real) =
    sum(voigt_logpdf(yi, μ, σ, γ) for yi in y)

# ------------------------------------------------------------------
# Score and Hessian  (Hansen & Tong 2026, in Faddeeva convention)
#
# Far-tail branch: the formulas below are algebraically exact but suffer
# floating-point cancellation deep in the tail, where e.g. ỹL and √(2/π)σ
# agree to many digits while their difference is the answer; the Hessian
# recursion then amplifies the loss by powers of ỹ. In that regime the
# Voigt density is Cauchy-dominated and the moment expansion
#   f(y) = c(ỹ) + (σ²/2)c″(ỹ) + O(σ⁴/ỹ⁶),   c = Cauchy(0,γ) density,
# yields score/Hessian limits with relative error O((σ²+γ²)/ỹ²). We switch
# where the expansion error (decreasing) and the cancellation error of the
# exact branch (increasing) cross over.
# ------------------------------------------------------------------

# The score and the Hessian need DIFFERENT switches. The Hessian recursion
# forms s_μ - ỹ H_μμ, a difference of two quantities of size 2/ỹ whose value
# is of size 1/ỹ³, so it loses digits far sooner than the score does.
# Minimising worst-case relative error against high-precision ground truth,
# over |ỹ|/√(σ²+γ²) in [1,10⁶] and a spread of (σ,γ), puts the crossovers
# at 500 for the score (worst case 2.0e-5) and 40 for the Hessian (8.6e-3).
# Sharing the score's switch leaves the Hessian with NO correct digits from
# ~100√(σ²+γ²) outward (relative error 1.9 at 100×, 4.0e7 at 1000×).
@inline _far_tail(ỹ, σ, γ)      = ỹ^2 > 2.5e5 * (σ^2 + γ^2)   # score, moments
@inline _far_tail_hess(ỹ, σ, γ) = ỹ^2 > 1.6e3 * (σ^2 + γ^2)   # Hessian

@inline function _score_tail(ỹ, σ, γ)
    den = ỹ^2 + γ^2
    sμ = 2ỹ / den
    sσ = σ * (6ỹ^2 - 2γ^2) / den^2
    sγ = 1 / γ - 2γ / den
    return sμ, sσ, sγ
end

@inline function _hessian_tail(ỹ, σ, γ)
    den = ỹ^2 + γ^2
    q = (6ỹ^2 - 2γ^2) / den^2
    Hμμ = 2 * (ỹ^2 - γ^2) / den^2
    Hγγ = -1 / γ^2 - 2 * (ỹ^2 - γ^2) / den^2
    Hμγ = -4ỹ * γ / den^2
    Hμσ = σ * (-12ỹ / den^2 + 4ỹ * (6ỹ^2 - 2γ^2) / den^3)
    Hγσ = σ * (-4γ / den^2 - 4γ * (6ỹ^2 - 2γ^2) / den^3)
    Hσσ = q
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
    voigt_hessian(y, μ, σ, γ) -> Symmetric 3×3 matrix

Hessian ∂² log f(y;θ)/∂θ∂θ′, assembled recursively from the score components
and L/K -- no additional special-function evaluations. Parameter order
(μ, σ, γ).
"""
function voigt_hessian(y::Real, μ::Real, σ::Real, γ::Real)
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
    ỹ = y - μ
    if _far_tail(ỹ, σ, γ)                    # E[Z|y] = σ² s_μ (Tweedie)
        return σ^2 * 2ỹ / (ỹ^2 + γ^2)
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
    ỹ = y - μ
    if _far_tail(ỹ, σ, γ)                    # V(Z|y) = σ²(1 + σ² H_μμ)
        return σ^2 * (1 + σ^2 * 2 * (ỹ^2 - γ^2) / (ỹ^2 + γ^2)^2)
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
_gauss_legendre_cached(n::Int) = get!(() -> _gauss_legendre(n), _GL_CACHE, n)

"""
    voigt_fisher(μ, σ, γ; nodes=400) -> Symmetric 3×3 matrix

Expected Fisher information ℐ(θ) = E[s s′], computed by Gauss-Legendre
quadrature after the substitution y = μ + (σ+γ) tan(t), t ∈ (-π/2, π/2),
which maps the Cauchy-tailed integrand to a bounded one. By the information
matrix equality ℐ(θ) = -E[H]; by symmetry ℐ is block diagonal
(ℐ_μσ = ℐ_μγ = 0).
"""
function voigt_fisher(μ::Real, σ::Real, γ::Real; nodes::Int = 400)
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
    cut_s = 2.5e5 * (s2 + γ * γ)      # score switch,   |ỹ| > 500 * scale
    cut_h = 1.6e3 * (s2 + γ * γ)      # Hessian switch, |ỹ| >  40 * scale
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
        if ỹ2 > cut_s                                   # Cauchy-limit score
            den = ỹ2 + γ * γ
            sμ = 2 * ỹ / den
            sσ = σ * (6 * ỹ2 - 2 * γ^2) / den^2
            sγ = 1 / γ - 2 * γ / den
        end
        if ỹ2 > cut_h                                   # Cauchy-limit Hessian
            den = ỹ2 + γ * γ
            d = 2 * (ỹ2 - γ^2) / den^2
            q = (6 * ỹ2 - 2 * γ^2) / den^2
            hmm = d
            hgg = -1 / γ^2 - d
            hmg = -4 * ỹ * γ / den^2
            hms = σ * (-12 * ỹ / den^2 + 4 * ỹ * (6 * ỹ2 - 2 * γ^2) / den^3)
            hgs = σ * (-4 * γ / den^2 - 4 * γ * (6 * ỹ2 - 2 * γ^2) / den^3)
            hss = q
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

"""
    voigt_mle(y; maxiter=200, gtol=1e-8, nodes=400, verbose=false)

Exact maximum likelihood estimation of 𝒱(μ, σ, γ) from an iid sample `y`,
by Newton's method with the analytic score and Hessian (in (μ, log σ, log γ)
coordinates, with a ridge safeguard and backtracking line search).

Returns a NamedTuple with fields
  `μ, σ, γ`        : parameter estimates
  `se`             : asymptotic standard errors, sqrt(diag(ℐ⁻¹/n)) (expected information)
  `se_obs`         : standard errors from the observed information -Σᵢ H(yᵢ)
  `vcov`           : ℐ(θ̂)⁻¹/n
  `loglik`         : maximized log-likelihood
  `converged, boundary, iterations`

The optimizer never throws on valid data: if the likelihood pushes a width to
the boundary of the (clamped) parameter space -- e.g. γ→0 when the Cauchy
component is weakly identified in small samples -- estimation stops there,
`boundary` is set, and standard errors that cannot be computed are returned
as NaN.

Standard errors are exact and conventional: the MLE is consistent and
asymptotically normal at rate √n (Hansen & Tong 2026, Thm. 4), despite the
Voigt distribution having no finite moments.
"""
function voigt_mle(y::AbstractVector; maxiter::Int = 200, gtol::Real = 1e-8,
                   nodes::Int = 400, verbose::Bool = false)
    n = length(y)
    n ≥ 3 || throw(ArgumentError("need at least 3 observations"))
    μ0, σ0, γ0 = _startvalues(y)
    # keep the log-widths in a sane range relative to the data scale; the true
    # boundary cases σ→0 / γ→0 are outside the regular Voigt family anyway
    s0 = max(σ0, γ0)
    lb, ub = log(1e-8 * s0), log(1e8 * s0)
    η = [μ0, log(σ0), log(γ0)]
    ll = _avg_loglik(y, η)
    converged = false
    iter = 0
    while iter < maxiter
        iter += 1
        g, H = _avg_grad_hess(y, η)
        all(isfinite, g) && all(isfinite, H) || break     # give up gracefully
        if norm(g) < gtol
            converged = true
            break
        end
        # ascent direction: solve (-H + λI) Δ = g with the smallest ridge λ ≥ 0
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
        found || break                                    # give up gracefully
        # backtracking line search (Armijo), widths clamped to [lb, ub]
        t = 1.0
        improved = false
        for _ in 1:40
            ηnew = η .+ t .* Δ
            ηnew[2] = clamp(ηnew[2], lb, ub)
            ηnew[3] = clamp(ηnew[3], lb, ub)
            llnew = _avg_loglik(y, ηnew)
            if isfinite(llnew) && llnew ≥ ll + 1e-4 * t * dot(g, Δ)
                η = ηnew
                ll = llnew
                improved = true
                break
            end
            t /= 2
        end
        if !improved
            # no ascent step found: at a (near-)boundary or numerical optimum
            norm(g) < 1e-4 && (converged = true)
            break
        end
        verbose && println("iter $iter  loglik $(n*ll)  |g| $(norm(g))")
    end
    μ̂, σ̂, γ̂ = η[1], exp(η[2]), exp(η[3])
    boundary = η[2] ≤ lb + 1e-10 || η[3] ≤ lb + 1e-10
    # standard errors: NaN (rather than an exception) if the information matrix
    # cannot be inverted, e.g. at a boundary
    se = fill(NaN, 3)
    vcov = fill(NaN, 3, 3)
    try
        ℐ = voigt_fisher(μ̂, σ̂, γ̂; nodes = nodes)
        vcov = inv(ℐ) / n
        se = sqrt.(max.(diag(vcov), 0.0))
    catch
    end
    se_obs = fill(NaN, 3)
    try
        Hobs = sum(voigt_hessian(yi, μ̂, σ̂, γ̂) for yi in y)
        se_obs = sqrt.(max.(diag(inv(-Symmetric(Matrix(Hobs)))), 0.0))
    catch
    end
    return (μ = μ̂, σ = σ̂, γ = γ̂, se = se, se_obs = se_obs, vcov = vcov,
            loglik = n * ll, converged = converged, boundary = boundary,
            iterations = iter)
end

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------

"""
    rand_voigt(rng, n, μ, σ, γ)

Draw `n` iid variates from 𝒱(μ, σ, γ) by exact convolution:
Y = μ + σ N(0,1) + γ tan(π(U - 1/2)).
"""
rand_voigt(rng, n::Int, μ::Real, σ::Real, γ::Real) =
    μ .+ σ .* randn(rng, n) .+ γ .* tan.(π .* (rand(rng, n) .- 0.5))

end # module
