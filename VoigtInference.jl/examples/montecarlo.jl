# Section 6(i): finite-sample behavior of the Voigt MLE.
#
# Per design (λ = γ0/σ0, n): bias, RMSE, and 95% Wald coverage — both
# UNCONDITIONAL (a sample whose standard errors are suppressed counts as
# non-covering) and CONDITIONAL on valid standard errors — plus the full
# diagnostic accounting: convergence and termination reasons, boundary
# flags for σ and γ separately, positive definiteness of both information
# matrices, the valid-SE share, and the share of samples in which a
# likelihood-ratio test does NOT reject the boundary submodel. Because the
# null places a width on the boundary of the parameter space, the LR
# statistic 2(ll_Voigt − ll_sub) is asymptotically a ½χ²₀ + ½χ²₁ mixture
# (Chernoff 1954; Self & Liang 1987); its 5% critical value is 2.706. The
# same boundary geometry is why the width MLE sits at an interior value of
# order n^(-1/2) (γ) or n^(-1/4) (σ) in about half of all submodel-true
# samples, so the flags alone understate submodel adequacy.
#
# Seeds are fixed integers, a deterministic function of the design indices
# (hash() is not stable across Julia versions).
#
# Run from the package directory:  julia --project=. examples/montecarlo.jl
# Default REPS=500 keeps runtime moderate; publication numbers use
#   REPS=5000 julia --project=. examples/montecarlo.jl
using VoigtInference, Random, Printf, Statistics

const REPS = parse(Int, get(ENV, "REPS", "500"))
const NS   = (100, 1_000, 10_000, 100_000)
const LAMBDAS = (0.01, 0.1, 1.0)          # λ = γ0/σ0
const MU0, SIGMA0 = 0.0, 1.0
const LR_CRIT = 2.706                     # 5%, ½χ²₀ + ½χ²₁ boundary mixture

function run_design(λ, n, seed)
    γ0 = λ * SIGMA0
    θ0 = (MU0, SIGMA0, γ0)
    rng = MersenneTwister(seed)
    est = fill(NaN, REPS, 3)
    covered = falses(REPS, 3)             # unconditional: invalid SE = not covered
    sevalid = falses(REPS, 3)
    conv = 0; sb = 0; gb = 0; ub = 0; pde = 0; pdo = 0
    lr_gauss = 0; lr_cauchy = 0
    terms = Dict{Symbol,Int}()
    for r in 1:REPS
        yr = rand_voigt(rng, n, MU0, SIGMA0, γ0)
        res = voigt_mle(yr)
        est[r, :] .= (res.μ, res.σ, res.γ)
        res.converged            && (conv += 1)
        res.sigma_boundary       && (sb += 1)
        res.gamma_boundary       && (gb += 1)
        res.upper_boundary       && (ub += 1)
        res.expected_info_posdef && (pde += 1)
        res.observed_info_posdef && (pdo += 1)
        terms[res.termination] = get(terms, res.termination, 0) + 1
        2 * (res.loglik - res.loglik_gaussian) ≤ LR_CRIT && (lr_gauss += 1)
        2 * (res.loglik - res.loglik_cauchy)  ≤ LR_CRIT && (lr_cauchy += 1)
        for j in 1:3
            if isfinite(res.se[j]) && res.se[j] > 0
                sevalid[r, j] = true
                covered[r, j] = abs(est[r, j] - θ0[j]) ≤ 1.96 * res.se[j]
            end
        end
    end
    b   = [mean(est[:, j]) - θ0[j] for j in 1:3]
    rm  = [sqrt(mean((est[:, j] .- θ0[j]) .^ 2)) for j in 1:3]
    cvu = [count(covered[:, j]) / REPS for j in 1:3]
    nv  = [count(sevalid[:, j]) for j in 1:3]
    cvc = [nv[j] > 0 ? count(covered[:, j] .& sevalid[:, j]) / nv[j] : NaN for j in 1:3]
    diag = (conv = conv, sb = sb, gb = gb, ub = ub, pde = pde, pdo = pdo,
            se = minimum(nv), lrg = lr_gauss, lrc = lr_cauchy, terms = terms)
    return b, rm, cvu, cvc, diag
end

function main()
    println("Voigt MLE Monte Carlo: REPS = $REPS per design")
    println("Coverage: covU counts suppressed standard errors as non-covering;")
    println("covC conditions on valid standard errors.\n")
    @printf("%5s %6s | %8s %8s %5s %5s | %8s %8s %5s %5s | %8s %8s %5s %5s\n",
            "λ", "n", "bias(μ)", "rmse(μ)", "covU", "covC",
            "bias(σ)", "rmse(σ)", "covU", "covC",
            "bias(γ)", "rmse(γ)", "covU", "covC")
    flush(stdout)

    rowsA = String[]; rowsB = String[]; diaglines = String[]
    pct(x) = 100 * x / REPS
    for (li, λ) in enumerate(LAMBDAS), (ni, n) in enumerate(NS)
        seed = 2026_000 + 100 * li + ni
        b, rm, cvu, cvc, d = run_design(λ, n, seed)
        @printf("%5.2f %6d | %8.4f %8.4f %5.3f %5.3f | %8.4f %8.4f %5.3f %5.3f | %8.4f %8.4f %5.3f %5.3f\n",
                λ, n, b[1], rm[1], cvu[1], cvc[1], b[2], rm[2], cvu[2], cvc[2],
                b[3], rm[3], cvu[3], cvc[3])
        flush(stdout)
        push!(diaglines,
              @sprintf("%5.2f %6d | conv %5.1f%% | bdry σ %5.1f%% γ %5.1f%% up %4.1f%% | PD exp %5.1f%% obs %5.1f%% | SE %5.1f%% | LR keeps: Gauss %5.1f%% Cauchy %5.1f%%%s",
                       λ, n, pct(d.conv), pct(d.sb), pct(d.gb), pct(d.ub),
                       pct(d.pde), pct(d.pdo), pct(d.se), pct(d.lrg), pct(d.lrc),
                       d.conv == REPS ? "" :
                           " | terminations: " * join(("$k=$v" for (k, v) in sort(collect(d.terms))), ", ")))
        push!(rowsA,
              @sprintf("%.2f & %d & %.4f & %.4f & %.3f & %.3f & %.4f & %.4f & %.3f & %.3f & %.4f & %.4f & %.3f & %.3f \\\\",
                       λ, n, b[1], rm[1], cvu[1], cvc[1], b[2], rm[2], cvu[2], cvc[2],
                       b[3], rm[3], cvu[3], cvc[3]))
        push!(rowsB,
              @sprintf("%.2f & %d & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f \\\\",
                       λ, n, pct(d.conv), pct(d.sb), pct(d.gb), pct(d.pde), pct(d.pdo),
                       pct(d.se), pct(d.lrg), pct(d.lrc)))
    end

    println("\nDiagnostics per design:")
    foreach(println, diaglines)

    println("""

Notes: 'bdry σ/γ' are the shares of samples in which the corresponding width
was estimated on (or effectively on) its boundary — the Cauchy or Gaussian
component undetected. 'LR keeps' is the share in which the boundary
likelihood-ratio test at 5% (critical value 2.706, ½χ²₀ + ½χ²₁) does not
reject the submodel; near the boundary this exceeds the flag shares because
the width MLE is interior, of order n^(-1/2) or n^(-1/4), in about half of
submodel-true samples.

LaTeX rows, Table A (λ, n, then bias/RMSE/covU/covC for μ, σ, γ):
""")
    foreach(println, rowsA)
    println("""

LaTeX rows, Table B (λ, n, conv%, bdryσ%, bdryγ%, PDexp%, PDobs%, SE%,
LR-keep Gauss %, LR-keep Cauchy %):
""")
    foreach(println, rowsB)
end

main()
