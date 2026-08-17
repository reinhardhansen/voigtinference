# Section 6(v): fit of a measured Raman spectrum (red ochre) with the exact
# Voigt profile and analytic Jacobian.
#
# Data: the `raman` dataset (2048 points) from the CRAN package `voigt`
# (Cannas & Piras), an excerpt of the red-ochre spectra of Pisu et al. (2025),
# Spectrochim. Acta A 329. Export it once from R:
#
#   install.packages("voigt")
#   library(voigt); data(raman)
#   write.csv(raman, "raman.csv", row.names = FALSE)
#
# and place raman.csv in examples/. We fit the fourth peak (rows 782:1097,
# following the package's documented example) with the intensity model
#
#   I(ν) = b + A · f(ν; μ, σ, γ),
#
# where f is the exact Voigt density, by Levenberg-Marquardt least squares
# with the ANALYTIC Jacobian: ∂f/∂θ = f · s_θ from voigt_score, so the
# Jacobian reuses the same Faddeeva evaluation as the model itself.
# A pseudo-Voigt fit (Gaussian/Lorentzian mixture, same b, A) is reported for
# comparison.
#
# Run from the package directory:  julia --project=. examples/raman.jl
using VoigtInference, Printf, Statistics, DelimitedFiles, LinearAlgebra

const HERE = @__DIR__
csvfile = joinpath(HERE, "raman.csv")
isfile(csvfile) || error("raman.csv not found in examples/ — export it from R first (see header).")

raw = readdlm(csvfile, ',', skipstart = 1)
ν_all, I_all = Float64.(raw[:, 1]), Float64.(raw[:, 2])
ν, I = ν_all[782:1097], I_all[782:1097]
m = length(ν)

# ------------------------------------------------------------------
# model and analytic Jacobian; parameters p = (b, A, μ, log σ, log γ)
# ------------------------------------------------------------------
function model_jac(p, ν)
    b, A, μ = p[1], p[2], p[3]
    σ, γ = exp(p[4]), exp(p[5])
    f = zeros(length(ν))
    J = zeros(length(ν), 5)
    for (i, νi) in enumerate(ν)
        fi = voigt_pdf(νi, μ, σ, γ)
        s = voigt_score(νi, μ, σ, γ)     # ∂ log f / ∂(μ, σ, γ)
        f[i] = b + A * fi
        J[i, 1] = 1.0
        J[i, 2] = fi
        J[i, 3] = A * fi * s[1]
        J[i, 4] = A * fi * s[2] * σ       # chain rule for log σ
        J[i, 5] = A * fi * s[3] * γ       # chain rule for log γ
    end
    return f, J
end

function levmar(resfun, p0; maxiter = 200, tol = 1e-12)
    p = copy(p0)
    r, J = resfun(p)
    cost = dot(r, r)
    λ = 1e-3
    iter = 0
    for outer iter in 1:maxiter
        g = J' * r
        H = J' * J
        # damped normal equations; on a singular system just increase damping
        D = Diagonal(max.(diag(H), 1e-10 * maximum(diag(H)), 1e-300))
        Δ = try
            -(H + λ * D) \ g
        catch
            λ *= 10
            λ > 1e12 && break
            continue
        end
        pn = p .+ Δ
        rn, Jn = resfun(pn)
        cn = dot(rn, rn)
        if isfinite(cn) && cn < cost
            p, r, J, cost = pn, rn, Jn, cn
            λ = max(λ / 3, 1e-12)
            norm(g) < tol * (1 + cost) && break
        else
            λ *= 10
            λ > 1e12 && break
        end
    end
    return p, r, J, cost, iter
end

# starting values from the peak geometry
b0 = minimum(I)
i0 = argmax(I)
μ0 = ν[i0]
halfmax = b0 + (I[i0] - b0) / 2
above = findall(I .> halfmax)
fwhm = ν[above[end]] - ν[above[1]]
σ0 = fwhm / 2 / 2.355        # split the width evenly between components
γ0 = fwhm / 4
A0 = (I[i0] - b0) * fwhm * 1.5

p0 = [b0, A0, μ0, log(σ0), log(γ0)]
resfun(p) = ((f, J) = model_jac(p, ν); (f .- I, J))
p̂, r, J, cost, iter = levmar(resfun, p0)

b̂, Â, μ̂ = p̂[1], p̂[2], p̂[3]
σ̂, γ̂ = exp(p̂[4]), exp(p̂[5])
s2 = cost / (m - 5)
Dg = Diagonal([1.0, 1.0, 1.0, σ̂, γ̂])     # back to (b, A, μ, σ, γ)
V = Dg * inv(Symmetric(J' * J)) * Dg .* s2
se = sqrt.(diag(V))
R2 = 1 - cost / sum((I .- mean(I)) .^ 2)

println("Voigt fit to red-ochre Raman peak (n = $m points, $iter LM iterations)")
@printf("  %-4s %12s %12s\n", "", "estimate", "std.err.")
for (nm, v, s) in zip(("b", "A", "μ", "σ", "γ"), (b̂, Â, μ̂, σ̂, γ̂), se)
    @printf("  %-4s %12.4f %12.4f\n", nm, v, s)
end
@printf("  R² = %.6f,  residual sd = %.3f\n", R2, sqrt(s2))
@printf("  physical widths: w_G = √(8 ln 2) σ = %.4f,  w_L = 2γ = %.4f\n\n",
        sqrt(8 * log(2)) * σ̂, 2γ̂)

# ------------------------------------------------------------------
# robustness: linear background  I = b0 + b1 (ν - ν̄) + A f(ν; μ, σ, γ)
# ------------------------------------------------------------------
ν̄ = sum(ν) / m
function model_jac_lin(p, ν)
    b0, b1, A, μ = p[1], p[2], p[3], p[4]
    σ, γ = exp(p[5]), exp(p[6])
    f = zeros(length(ν))
    J = zeros(length(ν), 6)
    for (i, νi) in enumerate(ν)
        fi = voigt_pdf(νi, μ, σ, γ)
        s = voigt_score(νi, μ, σ, γ)
        f[i] = b0 + b1 * (νi - ν̄) + A * fi
        J[i, 1] = 1.0
        J[i, 2] = νi - ν̄
        J[i, 3] = fi
        J[i, 4] = A * fi * s[1]
        J[i, 5] = A * fi * s[2] * σ
        J[i, 6] = A * fi * s[3] * γ
    end
    return f, J
end
linres(p) = ((f, J) = model_jac_lin(p, ν); (f .- I, J))
pl0 = [b̂, 0.0, Â, μ̂, log(σ̂), log(γ̂)]
p̂l, rl, Jl, costl, iterl = levmar(linres, pl0)
σ̂l, γ̂l = exp(p̂l[5]), exp(p̂l[6])
R2l = 1 - costl / sum((I .- mean(I)) .^ 2)
@printf("linear-background robustness: b1 = %.4f, σ = %.4f (vs %.4f), γ = %.4f (vs %.4f), R² = %.6f\n",
        p̂l[2], σ̂l, σ̂, γ̂l, γ̂, R2l)
@printf("width changes: σ %+.2f%%, γ %+.2f%%\n\n",
        100 * (σ̂l / σ̂ - 1), 100 * (γ̂l / γ̂ - 1))

# ------------------------------------------------------------------
# pseudo-Voigt comparison: I = b + A[(1-η) φ(ν;μ,σp) + η c(ν;μ,γp)]
# (a free Gaussian-Lorentzian mixture: both widths and the weight free)
# ------------------------------------------------------------------
φ(ν, μ, σ) = exp(-(ν - μ)^2 / (2σ^2)) / (σ * sqrt(2π))
c(ν, μ, γ) = γ / (π * ((ν - μ)^2 + γ^2))
function pv_model_jac(p, ν)   # p = (b, A, μ, log σp, log γp, logit η)
    b, A, μ = p[1], p[2], p[3]
    σp, γp, η = exp(p[4]), exp(p[5]), 1 / (1 + exp(-p[6]))
    f = zeros(length(ν)); J = zeros(length(ν), 6)
    for (i, νi) in enumerate(ν)
        g, l = φ(νi, μ, σp), c(νi, μ, γp)
        mix = (1 - η) * g + η * l
        f[i] = b + A * mix
        J[i, 1] = 1.0
        J[i, 2] = mix
        J[i, 3] = A * ((1 - η) * g * (νi - μ) / σp^2 + η * l * 2 * (νi - μ) / ((νi - μ)^2 + γp^2))
        J[i, 4] = A * (1 - η) * g * ((νi - μ)^2 / σp^2 - 1)           # ∂/∂ log σp
        J[i, 5] = A * η * l * (1 - 2γp^2 / ((νi - μ)^2 + γp^2))       # ∂/∂ log γp
        J[i, 6] = A * (l - g) * η * (1 - η)                           # ∂/∂ logit η
    end
    return f, J
end
pv0 = [b̂, Â, μ̂, log(σ̂), log(γ̂), 0.0]
pvres(p) = ((f, J) = pv_model_jac(p, ν); (f .- I, J))
q̂, rp, Jp, costp, iterp = levmar(pvres, pv0)
σp, γp, η = exp(q̂[4]), exp(q̂[5]), 1 / (1 + exp(-q̂[6]))
R2p = 1 - costp / sum((I .- mean(I)) .^ 2)
@printf("pseudo-Voigt fit: σp = %.4f, γp = %.4f, η = %.3f,  R² = %.6f (%d iterations)\n",
        σp, γp, η, R2p, iterp)
@printf("naive width interpretation vs Voigt fit: σp/σ̂ = %.3f, γp/γ̂ = %.3f\n\n", σp / σ̂, γp / γ̂)

# ------------------------------------------------------------------
# figure: data, fits, residuals
# ------------------------------------------------------------------
try
    using Plots
    f̂, _ = model_jac(p̂, ν)
    f̂p, _ = pv_model_jac(q̂, ν)
    l = grid(2, 1, heights = [0.72, 0.28])   # (no @layout: macros cannot be
                                             # used with conditionally loaded packages)
    p1 = plot(ν, I; seriestype = :scatter, ms = 1.5, mc = :black, msw = 0,
              label = "data", framestyle = :box, ylabel = "intensity (a.u.)")
    plot!(p1, ν, f̂; lw = 2, label = "Voigt fit")
    plot!(p1, ν, f̂p; lw = 1.5, ls = :dash, label = "pseudo-Voigt fit")
    p2 = plot(ν, I .- f̂; seriestype = :scatter, ms = 1.5, mc = :black, msw = 0,
              label = "Voigt residuals", framestyle = :box,
              xlabel = "Raman shift (cm⁻¹)", ylabel = "residual (a.u.)")
    hline!(p2, [0.0]; ls = :dot, color = :gray, label = "")
    fig = plot(p1, p2; layout = l, size = (700, 500))
    out = joinpath(HERE, "output", "raman_fit.pdf")
    savefig(fig, out)
    println("wrote $out")
catch e
    println("Plots.jl not available; skipped the figure. ($e)")
end
