# Finite-sample calibration of the boundary likelihood-ratio statistics
#
#     LR_G = max(0, 2 (max(ℓ_int, ℓ_G, ℓ_C) - ℓ_G))    (null: γ = 0)
#     LR_C = max(0, 2 (max(ℓ_int, ℓ_G, ℓ_C) - ℓ_C))    (null: σ = 0)
#
# computed by VoigtInference.boundary_lr. The closed Voigt family includes
# both boundary submodels, so the full-model likelihood is the maximum over
# the closed family and both statistics are nonnegative by construction. A
# raw difference against the interior candidate alone is NOT a likelihood
# ratio: near a boundary the interior candidate legitimately sits a clamp
# residual below the exact submodel likelihood, so the raw difference can
# be negative by far more than roundoff.
#
# Both the Voigt family and each submodel are location-scale families, so
# under either null the LR statistic is invariant to the location and scale
# of the data: its null distribution depends ONLY on the sample size n.
# One Monte Carlo calibration per n therefore serves every design and every
# data set of that size (a fitted-null parametric bootstrap collapses to
# this pivotal simulation).
#
# The two boundaries are theoretically different:
#   - Cauchy boundary: in the local parameter tau = sigma^2 >= 0 the score
#     is square-integrable, so Self-Liang one-sided asymptotics apply and
#     the 95% cutoff should approach the 1/2 chi2_0 + 1/2 chi2_1 value
#     2.706 as n grows.
#   - Gaussian boundary: the gamma-score at gamma = 0 grows like
#     exp(ytil^2/(2 sigma^2)) (times an algebraic factor), the Fisher
#     information is infinite, and the regularity conditions behind the
#     2.706 cutoff FAIL. The calibrated cutoff is the honest reference.
#
# Under the Gaussian null the LR distribution has a large atom at exactly
# zero (the closed-family maximum is the Gaussian fit itself). If the atom
# reaches 95%, the calibrated cutoff is 0 and the nonrandomized rule
# "reject iff LR > cutoff" has size AT MOST 5%, not exactly 5%. The
# calibration therefore reports the zero-atom share and the interior
# objective deficits, and validate_boundary_cutoffs measures the achieved
# rejection rate on INDEPENDENT replications.
#
# Seeds: every replication seeds its own generator with a fixed integer,
# so the output is bit-identical for any thread count. The base 18500902
# is Woldemar Voigt's birthday (2 September 1850). Calibration uses blocks
# 1e6*(SEEDBASE+k), validation 1e6*(SEEDBASE+500+k); montecarlo.jl's
# design blocks are 1e6*(SEEDBASE+100*li+ni). All disjoint.
#
# Standalone use:  julia -t auto --project=. examples/boundary_lr.jl [n ...]
# (default n = 100, 1000; B via env CALIB_B, default 999). Publication
# numbers use CALIB_B = 9999, for which the tail-probability standard
# error is about 0.002.
using VoigtInference, Random, Printf, Statistics

const CALIB_B = parse(Int, get(ENV, "CALIB_B", "999"))
const SEEDBASE = 18_500_902        # Woldemar Voigt, 2 September 1850

"95% empirical quantile (type-1: order statistic at ceil(0.95 B))"
_q95(v) = sort(v)[clamp(ceil(Int, 0.95 * length(v)), 1, length(v))]

"""
    _null_replications(k, n; B, starts, seedbase, offset) ->
        (lrg, lrc, defg, defc, termg, termc, csubg, csubc)

Simulate B Gaussian-null and B Cauchy-null samples of size n with
per-replication seeds and fit each with voigt_mle. Returns the
closed-family LR statistics, the interior objective deficits
max(0, max(ℓ_G,ℓ_C) - ℓ_interior) per null, the termination symbols,
and the Cauchy-submodel-fit convergence flags.
"""
function _null_replications(k, n; B, starts, seedbase, offset)
    lrg = Vector{Float64}(undef, B); lrc = Vector{Float64}(undef, B)
    defg = Vector{Float64}(undef, B); defc = Vector{Float64}(undef, B)
    termg = Vector{Symbol}(undef, B); termc = Vector{Symbol}(undef, B)
    csubg = fill(false, B); csubc = fill(false, B)
    Threads.@threads for b in 1:B
        base = 1_000_000 * (seedbase + offset + k)
        yg = randn(MersenneTwister(base + 2b), n)          # Gaussian null
        r = voigt_mle(yg; starts = starts)
        lrg[b] = boundary_lr(r)[1]
        defg[b] = max(0.0, max(r.loglik_gaussian, r.loglik_cauchy) - r.loglik)
        termg[b] = r.termination
        csubg[b] = r.cauchy_fit_converged
        u = rand(MersenneTwister(base + 2b + 1), n)
        yc = tan.(π .* (u .- 0.5))                         # Cauchy null
        r = voigt_mle(yc; starts = starts)
        lrc[b] = boundary_lr(r)[2]
        defc[b] = max(0.0, max(r.loglik_gaussian, r.loglik_cauchy) - r.loglik)
        termc[b] = r.termination
        csubc[b] = r.cauchy_fit_converged
    end
    return lrg, lrc, defg, defc, termg, termc, csubg, csubc
end

_termsummary(t) = join(("$k=$v" for (k, v) in sort(collect(
    Dict(s => count(==(s), t) for s in unique(t))))), ", ")

function _reportline(tag, lr, def, term, csub, B)
    @printf("    %-14s cutoff %6.3f | zero atom %5.1f%% | deficit q95 %8.1e max %8.1e | size of 2.706 rule %5.3f | cauchy-subfit conv %5.1f%%\n",
            tag, _q95(lr), 100 * count(iszero, lr) / B,
            quantile(def, 0.95), maximum(def), sum(lr .> 2.706) / B,
            100 * count(csub) / B)
    println("                   terminations: ", _termsummary(term))
end

"""
    calibrate_boundary_cutoffs(ns; B, starts, seedbase) -> (cutg, cutc)

Dicts mapping n to the calibrated 95% null cutoffs of the closed-family
LR_G and LR_C, with full diagnostics printed per n: the zero-atom share,
interior objective deficits, the true size of the naive 2.706 rule,
termination frequencies, and submodel-fit convergence. Deterministic for
fixed (ns order, B, seedbase), for any thread count.
"""
function calibrate_boundary_cutoffs(ns; B::Int = CALIB_B, starts::Int = 7,
                                    seedbase::Int = SEEDBASE)
    cutg = Dict{Int,Float64}()
    cutc = Dict{Int,Float64}()
    println("Boundary LR calibration (closed family; pivotal: depends only on n); B = $B")
    for (k, n) in enumerate(ns)
        lrg, lrc, defg, defc, termg, termc, csubg, csubc =
            _null_replications(k, n; B, starts, seedbase, offset = 0)
        cutg[n] = _q95(lrg)
        cutc[n] = _q95(lrc)
        @printf("  n = %6d\n", n)
        _reportline("Gaussian null:", lrg, defg, termg, csubg, B)
        _reportline("Cauchy null:", lrc, defc, termc, csubc, B)
        flush(stdout)
    end
    return cutg, cutc
end

"""
    validate_boundary_cutoffs(ns, cutg, cutc; B, starts, seedbase)

Achieved rejection rates of the calibrated cutoffs on INDEPENDENT
replications (disjoint seed block): the honest size check. For a cutoff
equal to the 95% order statistic the target is at most 5%; when the null
distribution has an atom at the cutoff the achieved size is below 5%.
"""
function validate_boundary_cutoffs(ns, cutg, cutc; B::Int = CALIB_B,
                                   starts::Int = 7, seedbase::Int = SEEDBASE)
    println("Validation on independent replications; B = $B")
    for (k, n) in enumerate(ns)
        lrg, lrc, _, _, _, _, _, _ =
            _null_replications(k, n; B, starts, seedbase, offset = 500)
        pg = sum(lrg .> cutg[n]) / B
        pc = sum(lrc .> cutc[n]) / B
        se(p) = sqrt(max(p * (1 - p), 1 / B) / B)
        @printf("  n = %6d | Gaussian null: rejection %5.3f (MC-SE %5.3f) | Cauchy null: rejection %5.3f (MC-SE %5.3f)\n",
                n, pg, se(pg), pc, se(pc))
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    ns = isempty(ARGS) ? [100, 1000] : parse.(Int, ARGS)
    cutg, cutc = calibrate_boundary_cutoffs(ns)
    println()
    validate_boundary_cutoffs(ns, cutg, cutc)
end
