# Section 6(i): finite-sample behavior of the Voigt MLE.
# Bias, RMSE, coverage of 95% Fisher-information confidence intervals, and the
# fraction of samples in which a width estimate hits the boundary (γ→0 arises
# when the Cauchy component is weakly identified, i.e. small λ and small n).
#
# Run from the package directory:  julia --project=. examples/montecarlo.jl
# Default REPS=500 keeps runtime moderate; increase for publication numbers:
#   REPS=5000 julia --project=. examples/montecarlo.jl
using VoigtInference, Random, Printf, Statistics

const REPS = parse(Int, get(ENV, "REPS", "500"))
const NS   = (100, 1_000, 10_000, 100_000)
const LAMBDAS = (0.01, 0.1, 1.0)          # λ = γ0/σ0
const MU0, SIGMA0 = 0.0, 1.0

println("Voigt MLE Monte Carlo: REPS = $REPS per design\n")
flush(stdout)   # log updates promptly when stdout is redirected to a file
@printf("%6s %6s | %9s %9s %6s | %9s %9s %6s | %9s %9s %6s | %6s\n",
        "λ", "n", "bias(μ)", "rmse(μ)", "cov", "bias(σ)", "rmse(σ)", "cov",
        "bias(γ)", "rmse(γ)", "cov", "bdry%")

latex_rows = String[]
for λ in LAMBDAS
    local γ0 = λ * SIGMA0
    for n in NS
        local est = fill(NaN, REPS, 3)
        local cov = fill(NaN, REPS, 3)
        local nbdry = 0
        local rng = MersenneTwister(hash((λ, n, 2026)) % UInt32)
        for r in 1:REPS
            local yr = rand_voigt(rng, n, MU0, SIGMA0, γ0)
            local res = voigt_mle(yr)
            est[r, :] .= (res.μ, res.σ, res.γ)
            res.boundary && (nbdry += 1)
            local θ0 = (MU0, SIGMA0, γ0)
            for j in 1:3
                if isfinite(res.se[j]) && res.se[j] > 0
                    cov[r, j] = abs(est[r, j] - θ0[j]) ≤ 1.96 * res.se[j] ? 1.0 : 0.0
                end
            end
        end
        local θ0 = (MU0, SIGMA0, γ0)
        local b  = [mean(est[:, j]) - θ0[j] for j in 1:3]
        local rm = [sqrt(mean((est[:, j] .- θ0[j]) .^ 2)) for j in 1:3]
        local cv = [mean(filter(!isnan, cov[:, j])) for j in 1:3]
        local pb = 100 * nbdry / REPS
        @printf("%6.2f %6d | %9.4f %9.4f %6.3f | %9.4f %9.4f %6.3f | %9.4f %9.4f %6.3f | %6.1f\n",
                λ, n, b[1], rm[1], cv[1], b[2], rm[2], cv[2], b[3], rm[3], cv[3], pb)
        flush(stdout)
        push!(latex_rows,
              @sprintf("%.2f & %d & %.4f & %.4f & %.3f & %.4f & %.4f & %.3f & %.4f & %.4f & %.3f & %.1f \\\\",
                       λ, n, b[1], rm[1], cv[1], b[2], rm[2], cv[2], b[3], rm[3], cv[3], pb))
    end
end

println("""

Notes: 'bdry%' is the share of samples where a width estimate reached the
boundary of the clamped parameter space (γ→0: Cauchy component undetected).
Coverage is computed over samples with finite Fisher standard errors.

LaTeX rows (booktabs; columns λ, n, bias/RMSE/coverage for μ, σ, γ, bdry%):
""")
foreach(println, latex_rows)
