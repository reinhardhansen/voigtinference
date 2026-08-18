# Section 6(iii)+(iv): pseudo-Voigt width bias and the redescending
# deconvolution figure. Requires Plots.jl (only this script does):
#   using Pkg; Pkg.add("Plots")
#
# Run from the package directory:  julia --project=. examples/figures.jl
# Figures are written to examples/output/.
using VoigtInference, Random, Printf, Statistics
using Plots

const OUT = joinpath(@__DIR__, "output")
isdir(OUT) || mkdir(OUT)

# ==================================================================
# (iii) Width distortion from pseudo-Voigt approximations
# ==================================================================
# Two approximations are fitted by maximum likelihood to Voigt data, with the
# center estimated (not fixed), and their parameters interpreted as
# convolution widths:
#
# 1. The standard Thompson-Cox-Hastings (TCH) pseudo-Voigt: common FWHM Γ,
#    mixing η, i.e. η Lorentzian(FWHM Γ) + (1-η) Gaussian(FWHM Γ). The fitted
#    (Γ, η) are mapped back to implied component widths by inverting the TCH
#    width/mixing map (the standard practice in powder diffraction):
#        Γ⁵ = ΓG⁵ + 2.69269 ΓG⁴ΓL + 2.42843 ΓG³ΓL² + 4.47163 ΓG²ΓL³
#             + 0.07842 ΓGΓL⁴ + ΓL⁵
#        η  = 1.36603 q - 0.47719 q² + 0.11116 q³,  q = ΓL/Γ
#    giving σ_pV = ΓG/√(8 ln 2), γ_pV = ΓL/2.
# 2. The free Gaussian-Lorentzian mixture with independent widths
#    (1-η) N(μ, σp²) + η Cauchy(μ, γp), which nests the TCH shape and can
#    only fit the density better; its widths are still not convolution widths.
#
# Fits are Nelder-Mead with three dispersed starts; the best is kept and the
# simplex-convergence flag reported.

const _SQ8LN2 = sqrt(8 * log(2))

gauss_logpdf_fwhm(d, Γ) = begin
    σg = Γ / _SQ8LN2
    -d^2 / (2σg^2) - log(σg) - 0.5 * log(2π)
end
cauchy_logpdf_fwhm(d, Γ) = log(Γ / 2) - log(π) - log(d^2 + (Γ / 2)^2)

tch_negll(y, x) = begin                    # x = (μ, log Γ, logit η)
    μv, Γ, η = x[1], exp(x[2]), 1 / (1 + exp(-x[3]))
    -sum(log(η * exp(cauchy_logpdf_fwhm(yi - μv, Γ)) +
             (1 - η) * exp(gauss_logpdf_fwhm(yi - μv, Γ))) for yi in y)
end

pv_negll(y, x) = begin                     # x = (μ, logit η, log σp, log γp)
    μv, η = x[1], 1 / (1 + exp(-x[2]))
    σp, γp = exp(x[3]), exp(x[4])
    -sum(log((1 - η) * exp(-(yi - μv)^2 / (2σp^2)) / (σp * sqrt(2π)) +
             η * γp / (π * ((yi - μv)^2 + γp^2))) for yi in y)
end

"invert η = 1.36603 q - 0.47719 q² + 0.11116 q³ on q ∈ [0, 1] (monotone)"
function tch_q(η)
    lo, hi = 0.0, 1.0
    for _ in 1:200
        mid = (lo + hi) / 2
        v = 1.36603mid - 0.47719mid^2 + 0.11116mid^3
        v < η ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"given (Γ, ΓL), solve the TCH quintic for ΓG ∈ [0, Γ] (monotone)"
function tch_gammaG(Γ, ΓL)
    f(G) = G^5 + 2.69269G^4 * ΓL + 2.42843G^3 * ΓL^2 + 4.47163G^2 * ΓL^3 +
           0.07842G * ΓL^4 + ΓL^5 - Γ^5
    lo, hi = 0.0, Γ
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# compact Nelder-Mead (no dependencies); returns (x, converged)
function neldermead(f, x0; iters = 4000, rtol = 1e-9)
    n = length(x0)
    xs = [copy(x0) for _ in 1:(n + 1)]
    for i in 1:n
        xs[i + 1][i] += 0.25
    end
    fs = [f(x) for x in xs]
    conv = false
    for _ in 1:iters
        o = sortperm(fs); xs, fs = xs[o], fs[o]
        if fs[end] - fs[1] < rtol * (abs(fs[1]) + 1)
            conv = true
            break
        end
        c = sum(xs[1:n]) / n                       # centroid excl. worst
        xr = 2c .- xs[end]; fr = f(xr)             # reflect
        if fr < fs[1]
            xe = 3c .- 2xs[end]; fe = f(xe)        # expand
            (xs[end], fs[end]) = fe < fr ? (xe, fe) : (xr, fr)
        elseif fr < fs[n]
            xs[end], fs[end] = xr, fr
        else
            xc = 0.5 .* (c .+ xs[end]); fc = f(xc) # contract
            if fc < fs[end]
                xs[end], fs[end] = xc, fc
            else
                for i in 2:(n + 1)                 # shrink
                    xs[i] = 0.5 .* (xs[1] .+ xs[i]); fs[i] = f(xs[i])
                end
            end
        end
    end
    o = sortperm(fs)
    return xs[o[1]], conv
end

"multistart Nelder-Mead: dispersed logit-η starts; keep the best"
function fit_multistart(negll, starts)
    best, bconv, bf = nothing, false, Inf
    for x0 in starts
        x, conv = neldermead(negll, x0)
        fx = negll(x)
        if fx < bf
            best, bconv, bf = x, conv, fx
        end
    end
    return best, bconv
end

const NBIG = 200_000
λs = [0.25, 0.5, 1.0, 2.0, 4.0]

if !(@isdefined(relσ_tch) && length(relσ_tch) == length(λs))
    relσ_tch, relγ_tch = Float64[], Float64[]
    relσ_pv, relγ_pv = Float64[], Float64[]
    relσ_ml, relγ_ml = Float64[], Float64[]

    println("Pseudo-Voigt (TCH and free mixture) vs exact MLE, n = $NBIG per design")
    @printf("%6s | %9s %9s | %9s %9s | %9s %9s | %6s %s\n",
            "λ", "TCHσ-1", "TCHγ-1", "freeσ-1", "freeγ-1",
            "σ̂-1", "γ̂-1", "ηTCH", "conv")
    for (i, λ) in enumerate(λs)
        local σ0, γ0, rng, y, r
        σ0, γ0 = 1.0, λ
        rng = MersenneTwister(3026000 + i)          # fixed integer seed
        y = rand_voigt(rng, NBIG, 0.0, σ0, γ0)
        Γ0 = _SQ8LN2 * σ0 + 2γ0                     # rough common-FWHM start

        # TCH fit + inversion
        st = [[0.0, log(Γ0), lg] for lg in (-1.4, 0.0, 1.4)]
        xt, ct = fit_multistart(x -> tch_negll(y, x), st)
        Γ̂, η̂ = exp(xt[2]), 1 / (1 + exp(-xt[3]))
        q = tch_q(η̂)
        ΓL = q * Γ̂
        ΓG = tch_gammaG(Γ̂, ΓL)
        σt, γt = ΓG / _SQ8LN2, ΓL / 2

        # free-mixture fit
        st = [[0.0, lg, log(σ0), log(γ0)] for lg in (-1.4, 0.0, 1.4)]
        xp, cp = fit_multistart(x -> pv_negll(y, x), st)
        σp, γp = exp(xp[3]), exp(xp[4])

        # exact Voigt MLE
        r = voigt_mle(y)
        push!(relσ_tch, σt / σ0 - 1); push!(relγ_tch, γt / γ0 - 1)
        push!(relσ_pv, σp / σ0 - 1); push!(relγ_pv, γp / γ0 - 1)
        push!(relσ_ml, r.σ / σ0 - 1); push!(relγ_ml, r.γ / γ0 - 1)
        @printf("%6.2f | %9.4f %9.4f | %9.4f %9.4f | %9.4f %9.4f | %6.3f %s\n",
                λ, relσ_tch[end], relγ_tch[end], relσ_pv[end], relγ_pv[end],
                relσ_ml[end], relγ_ml[end], η̂,
                (ct && cp && r.converged) ? "yes" : "CHECK")
    end
else
    println("Reusing pseudo-Voigt fit results already in this session.")
end

# |relative error| on a log scale
p1 = plot(λs, 100 .* abs.(relσ_tch); xscale = :log10, yscale = :log10,
          marker = :circle, lw = 2, label = "TCH pseudo-Voigt σ",
          xlabel = "λ = γ₀/σ₀", xticks = (λs, string.(λs)),
          ylabel = "|relative error| of width estimate (%)",
          ylims = (0.01, 1000), legend = :bottomleft, framestyle = :box)
plot!(p1, λs, 100 .* abs.(relγ_tch); marker = :square, lw = 2,
      label = "TCH pseudo-Voigt γ")
plot!(p1, λs, 100 .* abs.(relσ_pv); marker = :circle, lw = 1.5, ls = :dot,
      alpha = 0.7, label = "free mixture σp")
plot!(p1, λs, 100 .* abs.(relγ_pv); marker = :square, lw = 1.5, ls = :dot,
      alpha = 0.7, label = "free mixture γp")
plot!(p1, λs, 100 .* abs.(relσ_ml); marker = :circle, lw = 2, ls = :dash,
      label = "exact MLE σ̂")
plot!(p1, λs, 100 .* abs.(relγ_ml); marker = :square, lw = 2, ls = :dash,
      label = "exact MLE γ̂")
savefig(p1, joinpath(OUT, "pseudovoigt_bias.pdf"))
println("wrote $(joinpath(OUT, "pseudovoigt_bias.pdf"))")

# ==================================================================
# (iv) Redescending deconvolution: E[Z|y] and V(Z|y)
# ==================================================================
# canonical case (μ,σ,γ) = (0,1,1). Reference line: if the Cauchy noise were
# replaced by Gaussian noise of the same scale, E[Z|y] would be linear with
# slope σ²/(σ²+γ²) = 1/2 -- the Kalman-type attribution. The exact conditional
# mean has central slope 1 - V(Z|0)/σ² ≈ 0.475, close to 1/2, but redescends.
μ, σ, γ = 0.0, 1.0, 1.0
kal = σ^2 / (σ^2 + γ^2)
ys = range(-12, 12; length = 1201)
# dimensionless axes: (y-μ)/σ against E[Z|Y]/σ and V(Z|Y)/σ², so the two
# curves share a vertical scale
p2 = plot(ys ./ σ, [voigt_condmean(yi, μ, σ, γ) / σ for yi in ys];
          lw = 2, label = "E[Z | Y=y] / σ", xlabel = "(y - μ)/σ",
          legend = :bottomright, framestyle = :box, ylims = (-1.5, 2))
plot!(p2, ys ./ σ, [voigt_condvar(yi, μ, σ, γ) / σ^2 for yi in ys];
      lw = 2, label = "V(Z | Y=y) / σ²")
hline!(p2, [1.0]; ls = :dot, color = :gray, label = "")
plot!(p2, ys ./ σ, (kal / σ) .* ys; ls = :dash, color = :gray, alpha = 0.7,
      label = "slope σ²/(σ²+γ²)  (Gaussian-noise attribution)")
savefig(p2, joinpath(OUT, "cond_moments.pdf"))
println("wrote $(joinpath(OUT, "cond_moments.pdf"))")
