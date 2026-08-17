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
# (iii) Bias in (σ, γ) from fitting a pseudo-Voigt model to Voigt data
# ==================================================================
# pseudo-Voigt: (1-η) N(0, σp²) + η Cauchy(0, γp), fit by MLE (μ fixed at 0),
# with parameters interpreted as the Gaussian/Lorentzian widths.

pv_logpdf(yi, η, σp, γp) =
    log((1 - η) * exp(-yi^2 / (2σp^2)) / (σp * sqrt(2π)) +
        η * γp / (π * (yi^2 + γp^2)))

pv_negll(y, x) = begin                     # x = (logit η, log σp, log γp)
    η = 1 / (1 + exp(-x[1]))
    σp, γp = exp(x[2]), exp(x[3])
    -sum(pv_logpdf(yi, η, σp, γp) for yi in y)
end

# compact Nelder-Mead (no dependencies)
function neldermead(f, x0; iters = 2000, tol = 1e-10)
    n = length(x0)
    xs = [copy(x0) for _ in 1:(n + 1)]
    for i in 1:n
        xs[i + 1][i] += 0.25
    end
    fs = [f(x) for x in xs]
    for _ in 1:iters
        o = sortperm(fs); xs, fs = xs[o], fs[o]
        fs[end] - fs[1] < tol && break
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
    return xs[o[1]]
end

const NBIG = 200_000
λs = [0.25, 0.5, 1.0, 2.0, 4.0]

# The fits take a few minutes; when re-included in the same session (e.g. to
# adjust plot styling), reuse the existing results instead of refitting.
if !(@isdefined(relσ_pv) && length(relσ_pv) == length(λs))
    relσ_pv, relγ_pv = Float64[], Float64[]
    relσ_ml, relγ_ml = Float64[], Float64[]

    println("Pseudo-Voigt vs exact MLE, n = $NBIG per design")
    @printf("%6s | %10s %10s | %10s %10s | %6s\n",
            "λ", "σp/σ0-1", "γp/γ0-1", "σ̂/σ0-1", "γ̂/γ0-1", "η")
    for λ in λs
        local σ0, γ0, rng, y, r
        σ0, γ0 = 1.0, λ
        rng = MersenneTwister(hash((λ, :pv)) % UInt32)
        y = rand_voigt(rng, NBIG, 0.0, σ0, γ0)
        # pseudo-Voigt MLE
        x = neldermead(x -> pv_negll(y, x), [0.0, log(σ0), log(γ0)])
        η, σp, γp = 1 / (1 + exp(-x[1])), exp(x[2]), exp(x[3])
        # exact Voigt MLE
        r = voigt_mle(y)
        push!(relσ_pv, σp / σ0 - 1); push!(relγ_pv, γp / γ0 - 1)
        push!(relσ_ml, r.σ / σ0 - 1); push!(relγ_ml, r.γ / γ0 - 1)
        @printf("%6.2f | %10.4f %10.4f | %10.4f %10.4f | %6.3f\n",
                λ, relσ_pv[end], relγ_pv[end], relσ_ml[end], relγ_ml[end], η)
    end
else
    println("Reusing pseudo-Voigt fit results already in this session.")
end

# |relative error| on a log scale: pseudo-Voigt errors are 7--330%, exact MLE
# errors are ~0.1--1% (sampling noise at this n), so a linear axis hides the
# MLE curves under zero.
p1 = plot(λs, 100 .* abs.(relσ_pv); xscale = :log10, yscale = :log10,
          marker = :circle, lw = 2, label = "pseudo-Voigt σp",
          xlabel = "λ = γ₀/σ₀", xticks = (λs, string.(λs)),
          ylabel = "|relative error| of width estimate (%)",
          ylims = (0.01, 1000), legend = :bottomleft, framestyle = :box)
plot!(p1, λs, 100 .* abs.(relγ_pv); marker = :square, lw = 2,
      label = "pseudo-Voigt γp")
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
