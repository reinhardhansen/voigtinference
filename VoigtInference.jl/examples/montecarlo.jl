# Section 6(i): finite-sample behavior of the Voigt MLE.
#
# Per design (λ = γ0/σ0, n): bias, RMSE, and 95% Wald coverage — both
# UNCONDITIONAL (a sample whose standard errors are suppressed counts as
# non-covering) and CONDITIONAL on valid standard errors — plus the full
# diagnostic accounting: convergence and termination reasons, boundary
# flags for σ and γ separately, positive definiteness of both information
# matrices, the valid-SE share, and the share of samples in which a
# likelihood-ratio test at the CALIBRATED finite-sample 5% cutoff does NOT
# reject the boundary submodel. The two boundaries differ: in
# τ = σ² the Cauchy-boundary score is square-integrable, so Self-Liang
# one-sided asymptotics apply and the cutoff approaches the ½χ²₀ + ½χ²₁
# value 2.706; at the Gaussian boundary the γ-score has infinite Fisher
# information and no such theory holds. Both statistics are location-scale
# PIVOTAL under their nulls, so the exact finite-sample cutoffs depend only
# on n and are calibrated once per n by simulation (boundary_lr.jl, which
# also prints the true size of the naive 2.706 rule). The same boundary
# geometry is why the width MLE frequently sits at a small interior value
# in submodel-true samples (for σ provably of order n^(-1/4), via τ = σ²;
# for γ no standard rate applies), so the flags alone understate submodel
# adequacy.
#
# Seeds are fixed integers, a deterministic function of the design and
# replication indices (hash() is not stable across Julia versions).
# Replications are embarrassingly parallel and each seeds its own generator,
# so the output is bit-identical for ANY thread count.
#
# Run from the package directory:  julia -t auto --project=. examples/montecarlo.jl
# Default REPS=500 keeps runtime moderate; publication numbers use
#   REPS=5000 CALIB_B=9999 julia -t auto --project=. examples/montecarlo.jl
using VoigtInference, Random, Printf, Statistics
import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)   # replications are the parallel unit

include(joinpath(@__DIR__, "boundary_lr.jl"))

const REPS = parse(Int, get(ENV, "REPS", "500"))
const NS   = (100, 1_000, 10_000, 100_000)
const LAMBDAS = (0.01, 0.1, 1.0)          # λ = γ0/σ0
const MU0, SIGMA0 = 0.0, 1.0
const STARTS = 7                          # robust deterministic multistart

function run_design(λ, n, seed, cutg, cutc)
    γ0 = λ * SIGMA0
    θ0 = (MU0, SIGMA0, γ0)
    est = fill(NaN, REPS, 3)
    covered = fill(false, REPS, 3)        # unconditional: invalid SE = not covered
    sevalid = fill(false, REPS, 3)        # Bool arrays, not BitArrays: threads
    convA = fill(false, REPS); sbA = fill(false, REPS); gbA = fill(false, REPS)
    ubA = fill(false, REPS); pdeA = fill(false, REPS); pdoA = fill(false, REPS)
    lrgA = fill(false, REPS); lrcA = fill(false, REPS)
    termA = Vector{Symbol}(undef, REPS)
    Threads.@threads for r in 1:REPS
        # per-replication seed, deterministic in (design, replication);
        # 1e6 spacing > REPS keeps design blocks disjoint, and the result
        # cannot depend on the thread schedule
        rng = MersenneTwister(1_000_000 * seed + r)
        yr = rand_voigt(rng, n, MU0, SIGMA0, γ0)
        res = voigt_mle(yr; starts = STARTS)
        est[r, :] .= (res.μ, res.σ, res.γ)
        convA[r] = res.converged
        sbA[r]  = res.sigma_boundary
        gbA[r]  = res.gamma_boundary
        ubA[r]  = res.upper_boundary
        pdeA[r] = res.expected_info_posdef
        pdoA[r] = res.observed_info_posdef
        termA[r] = res.termination
        lrgA[r] = 2 * (res.loglik - res.loglik_gaussian) ≤ cutg
        lrcA[r] = 2 * (res.loglik - res.loglik_cauchy)  ≤ cutc
        for j in 1:3
            if isfinite(res.se[j]) && res.se[j] > 0
                sevalid[r, j] = true
                covered[r, j] = abs(est[r, j] - θ0[j]) ≤ 1.96 * res.se[j]
            end
        end
    end
    terms = Dict{Symbol,Int}()
    for r in 1:REPS
        terms[termA[r]] = get(terms, termA[r], 0) + 1
    end
    b   = [mean(est[:, j]) - θ0[j] for j in 1:3]
    rm  = [sqrt(mean((est[:, j] .- θ0[j]) .^ 2)) for j in 1:3]
    cvu = [count(covered[:, j]) / REPS for j in 1:3]
    nv  = [count(sevalid[:, j]) for j in 1:3]
    cvc = [nv[j] > 0 ? count(covered[:, j] .& sevalid[:, j]) / nv[j] : NaN for j in 1:3]
    diag = (conv = count(convA), sb = count(sbA), gb = count(gbA),
            ub = count(ubA), pde = count(pdeA), pdo = count(pdoA),
            se = minimum(nv), lrg = count(lrgA), lrc = count(lrcA),
            terms = terms)
    return b, rm, cvu, cvc, diag
end

function main()
    println("Voigt MLE Monte Carlo: REPS = $REPS per design, starts = $STARTS")
    println("Coverage: covU counts suppressed standard errors as non-covering;")
    println("covC conditions on valid standard errors.\n")
    cutg, cutc = calibrate_boundary_cutoffs(collect(NS); starts = STARTS)
    println()
    @printf("%5s %6s | %8s %8s %5s %5s | %8s %8s %5s %5s | %8s %8s %5s %5s\n",
            "λ", "n", "bias(μ)", "rmse(μ)", "covU", "covC",
            "bias(σ)", "rmse(σ)", "covU", "covC",
            "bias(γ)", "rmse(γ)", "covU", "covC")
    flush(stdout)

    rowsA = String[]; rowsB = String[]; diaglines = String[]
    pct(x) = 100 * x / REPS
    for (li, λ) in enumerate(LAMBDAS), (ni, n) in enumerate(NS)
        seed = 2026_000 + 100 * li + ni
        b, rm, cvu, cvc, d = run_design(λ, n, seed, cutg[n], cutc[n])
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
likelihood-ratio test at the CALIBRATED finite-sample 5% cutoff (printed
above; pivotal, so it depends only on n) does not reject the submodel; near
the boundary this exceeds the flag shares because the width MLE frequently
sits at a small interior value in submodel-true samples. The
Cauchy-side cutoff approaches 2.706 (Self-Liang in τ = σ²); the Gaussian
side is nonregular and its cutoff is a genuine finite-sample quantity.

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
