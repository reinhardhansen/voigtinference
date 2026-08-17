# Demo: exact MLE and deconvolution for the Voigt profile.
# Run from the package directory with:
#   julia --project=. examples/demo.jl
using VoigtInference, Random, Printf

rng = MersenneTwister(2026)

# ------------------------------------------------------------------
# 1. Simulate and fit
# ------------------------------------------------------------------
μ0, σ0, γ0 = 0.5, 1.0, 0.3          # center, Gaussian sd, Lorentzian HWHM
n = 5_000
y = rand_voigt(rng, n, μ0, σ0, γ0)

r = voigt_mle(y)

println("Exact Voigt MLE, n = $n  (converged: $(r.converged), $(r.iterations) iterations)")
@printf("  %-3s  %8s  %8s  %10s\n", "", "true", "MLE", "std.err.")
@printf("  μ    %8.4f  %8.4f  %10.4f\n", μ0, r.μ, r.se[1])
@printf("  σ    %8.4f  %8.4f  %10.4f\n", σ0, r.σ, r.se[2])
@printf("  γ    %8.4f  %8.4f  %10.4f\n", γ0, r.γ, r.se[3])
@printf("  log-likelihood: %.2f\n\n", r.loglik)

# ------------------------------------------------------------------
# 2. Deconvolution: attribute observed deviations to the Gaussian
#    versus the Lorentzian component
# ------------------------------------------------------------------
println("Conditional moments of the Gaussian component, E[Z|Y=y] and V(Z|Y=y):")
@printf("  %8s  %10s  %10s\n", "y - μ", "E[Z|y]", "V(Z|y)")
for dy in (0.0, 0.5, 1.0, 2.0, 2.46, 4.0, 10.0, 100.0)
    yv = r.μ + dy
    @printf("  %8.2f  %10.4f  %10.4f\n",
            dy, voigt_condmean(yv, r.μ, r.σ, r.γ), voigt_condvar(yv, r.μ, r.σ, r.γ))
end
println("\nNote the redescending attribution: E[Z|y] peaks and then returns to 0;")
println("extreme deviations are assigned to the Lorentzian (Cauchy) component.")
