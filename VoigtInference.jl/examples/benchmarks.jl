# Section 6(ii): cost of exact inference.
# Per-observation wall-clock time of: density; analytic score; finite-difference
# score; analytic Hessian; finite-difference Hessian (diagonal-cost proxy).
#
# Note on automatic differentiation: plain ForwardDiff does NOT compose through
# the complex erfcx/Faddeeva evaluation (no dispatch for Complex{Dual} special
# functions without custom rules), so AD is not an off-the-shelf alternative
# here. The analytic derivatives are not merely faster -- absent custom AD
# rules, they are the only exact option.
#
# Run from the package directory:  julia --project=. examples/benchmarks.jl
using VoigtInference, Random, Printf

const N = 100_000
rng = MersenneTwister(1)
y = rand_voigt(rng, N, 0.5, 1.0, 0.3)
μ, σ, γ = 0.5, 1.0, 0.3

function fd_score(yi, μ, σ, γ; h = 1e-6)
    return [(voigt_logpdf(yi, μ + h, σ, γ) - voigt_logpdf(yi, μ - h, σ, γ)) / (2h),
            (voigt_logpdf(yi, μ, σ + h, γ) - voigt_logpdf(yi, μ, σ - h, γ)) / (2h),
            (voigt_logpdf(yi, μ, σ, γ + h) - voigt_logpdf(yi, μ, σ, γ - h)) / (2h)]
end

# central-difference Hessian: 2 evals per off-diagonal pair + 3 per diagonal
function fd_hessian(yi, μ, σ, γ; h = 1e-4)
    f(m, s, g) = voigt_logpdf(yi, m, s, g)
    H = zeros(3, 3)
    p = [μ, σ, γ]
    for j in 1:3, k in j:3
        ej = [j == 1, j == 2, j == 3] .* h
        ek = [k == 1, k == 2, k == 3] .* h
        H[j, k] = (f((p .+ ej .+ ek)...) - f((p .+ ej .- ek)...) -
                   f((p .- ej .+ ek)...) + f((p .- ej .- ek)...)) / (4h^2)
        H[k, j] = H[j, k]
    end
    return H
end

# accumulate into s to prevent the compiler eliding the work
function timeit(f, y; warmup = 1000)
    s = 0.0
    for i in 1:warmup
        s += f(y[i])
    end
    t = @elapsed for yi in y
        s += f(yi)
    end
    return t, s
end

tasks = [
    ("log-density"        , yi -> voigt_logpdf(yi, μ, σ, γ)),
    ("analytic score"     , yi -> sum(voigt_score(yi, μ, σ, γ))),
    ("FD score (7 evals)" , yi -> sum(fd_score(yi, μ, σ, γ))),
    ("analytic Hessian"   , yi -> sum(voigt_hessian(yi, μ, σ, γ))),
    ("FD Hessian (24 ev.)", yi -> sum(fd_hessian(yi, μ, σ, γ))),
]

println("Per-observation cost, N = $N (best of 3 runs)\n")
base = 0.0
for (name, f) in tasks
    t = minimum((timeit(f, y)[1] for _ in 1:3))
    global base = name == "log-density" ? t : base
    @printf("%-22s %8.1f ns/obs   ratio to density: %5.2f\n",
            name, 1e9 * t / N, t / base)
end

println("\nMLE end-to-end:")
t = @elapsed r = voigt_mle(y)
@printf("n = %d: %.2f s total, %d Newton iterations (converged: %s)\n",
        N, t, r.iterations, r.converged)
