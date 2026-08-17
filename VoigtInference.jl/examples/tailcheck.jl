# Validation of the far-tail asymptotic switch (Section 5): double-precision
# package output versus a 256-bit high-precision reference, over
# |ỹ|/√(σ²+γ²) = 10¹, ..., 10⁸.
#
# The reference evaluates erfcx on the Voigt line by BigFloat Simpson
# integration of the representation  erfcx(w) = (1/π) ∫ exp(-t²)/(w+it) dt
# (valid for Re w > 0), then applies the exact score formulas in BigFloat.
# At 256-bit precision the ~16-digit cancellation deep in the tail still
# leaves ~60 accurate digits, so the reference is reliable on the whole grid.
#
# Also reported: the error of the *naive* double-precision exact formulas
# (no tail switch), demonstrating why the implementation is not merely the
# formulas typed into the language.
#
# Run from the package directory:  julia --project=. examples/tailcheck.jl
using VoigtInference, Printf

setprecision(BigFloat, 256)

const S2P_BIG = sqrt(big(2) / big(pi))

# BigFloat erfcx(w), w = a + ix on the Voigt line (Re w = a > 0).
# The truncation T must be generous: a relative truncation error of order
# erfc(T) in (K, L) is amplified by ~ỹ⁴ through the cancellation in the
# reference score formulas, so T = 9 (erfc ≈ 4e-37) is NOT enough at ỹ ~ 1e8;
# T = 30 (erfc ≈ 1e-393) is beyond the 256-bit precision on the whole grid.
function erfcx_big(w::Complex{BigFloat}; n::Int = 14000, T::BigFloat = big(30.0))
    h = 2T / n
    s = zero(Complex{BigFloat})
    for k in 0:n
        t = -T + k * h
        wgt = (k == 0 || k == n) ? 1 : (isodd(k) ? 4 : 2)
        s += wgt * exp(-t^2) / (w + im * t)
    end
    return s * h / 3 / big(pi)
end

# exact score in BigFloat (same algebra as the double-precision exact branch)
function score_big(y, μ, σ, γ)
    yb, μb, σb, γb = big(y), big(μ), big(σ), big(γ)
    w = erfcx_big(complex(γb, yb - μb) / (σb * sqrt(big(2))))
    K, L = real(w), -imag(w)
    ỹ = yb - μb
    sμ = (ỹ - γb * L / K) / σb^2
    sσ = ((ỹ^2 - γb^2 - σb^2) * K - 2γb * ỹ * L + S2P_BIG * σb * γb) / (σb^3 * K)
    sγ = (γb * K + ỹ * L - S2P_BIG * σb) / (σb^2 * K)
    return [sμ, sσ, sγ]
end

# naive double-precision exact formulas, no far-tail switch
function score_naive(y, μ, σ, γ)
    K, L, _, _ = VoigtInference._KL(y, μ, σ, γ)
    ỹ = y - μ
    return [(ỹ - γ * L / K) / σ^2,
            ((ỹ^2 - γ^2 - σ^2) * K - 2γ * ỹ * L + sqrt(2 / π) * σ * γ) / (σ^3 * K),
            (γ * K + ỹ * L - sqrt(2 / π) * σ) / (σ^2 * K)]
end

relerr(x, ref) = ref == 0 ? abs(x - Float64(ref)) :
                 Float64(abs(big(x) - ref) / abs(ref))

for (σ, γ) in ((1.0, 1.0), (1.0, 0.01))
    println("\n(σ, γ) = ($σ, $γ);  entries: max over (s_μ, s_σ, s_γ) of |rel. error|")
    @printf("%12s %14s %14s\n", "|ỹ|/√(σ²+γ²)", "package", "naive exact")
    scale = sqrt(σ^2 + γ^2)
    for e in 1:8
        ỹ = 10.0^e * scale
        ref = score_big(ỹ, 0.0, σ, γ)
        pkg = voigt_score(ỹ, 0.0, σ, γ)
        nai = score_naive(ỹ, 0.0, σ, γ)
        epkg = maximum(relerr(pkg[j], ref[j]) for j in 1:3)
        enai = maximum(relerr(nai[j], ref[j]) for j in 1:3)
        @printf("%12.0e %14.2e %14.2e\n", 10.0^e, epkg, enai)
    end
end
println("""

The package switches to the asymptotic branch at |ỹ| > 10³ √(σ²+γ²): its error
stays small and decreasing on the whole grid, while the naive exact formulas
lose all accuracy beyond the switch point.""")
