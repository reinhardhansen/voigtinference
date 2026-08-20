# Finite-sample calibration of the boundary likelihood-ratio statistics
#
#     LR_G = 2 (ll_Voigt - ll_Gaussian)    (null: gamma = 0)
#     LR_C = 2 (ll_Voigt - ll_Cauchy)      (null: sigma = 0)
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
#     exp(ytil^2/(2 sigma^2)), the Fisher information is infinite, and the
#     regularity conditions behind the 2.706 cutoff FAIL. The calibrated
#     cutoff is the honest reference; the printed size of the naive 2.706
#     rule quantifies its miscalibration.
#
# Standalone use:  julia -t auto --project=. examples/boundary_lr.jl [n ...]
# (default n = 100, 1000; B via env CALIB_B, default 999). The Monte Carlo
# standard error of an estimated 95% cutoff scales like the density-inverse
# times sqrt(0.05*0.95/B); publication numbers use CALIB_B = 9999, for
# which the tail-probability uncertainty is about 0.002.
#
# Replications are embarrassingly parallel and each one is seeded by its own
# fixed integer, a deterministic function of (n-index, replication), so the
# output is bit-identical for ANY thread count (julia -t 1 and -t 8 agree
# byte for byte).
using VoigtInference, Random, Printf, Statistics

const CALIB_B = parse(Int, get(ENV, "CALIB_B", "999"))

"95% empirical quantile (type-1: order statistic at ceil(0.95 B))"
_q95(v) = sort(v)[clamp(ceil(Int, 0.95 * length(v)), 1, length(v))]

"""
    calibrate_boundary_cutoffs(ns; B, starts, seedbase) -> (cutg, cutc)

Dicts mapping n to the calibrated 95% null cutoffs of LR_G and LR_C.
Prints, per n, the cutoffs and the true size of the naive 2.706 rule.
Deterministic for fixed (ns order, B, seedbase), for any thread count:
every replication seeds its own generator with a fixed integer.
"""
function calibrate_boundary_cutoffs(ns; B::Int = CALIB_B, starts::Int = 7,
                                    seedbase::Int = 7000)
    cutg = Dict{Int,Float64}()
    cutc = Dict{Int,Float64}()
    println("Boundary LR calibration (pivotal: depends only on n); B = $B")
    for (k, n) in enumerate(ns)
        lrg = Vector{Float64}(undef, B)
        lrc = Vector{Float64}(undef, B)
        Threads.@threads for b in 1:B
            # per-replication integer seeds: the 1e6*(seedbase+k) blocks are
            # disjoint across k (2B+1 < 1e6) and from the montecarlo.jl
            # design blocks (~2.0e12), and the result cannot depend on the
            # thread schedule
            base = 1_000_000 * (seedbase + k)
            yg = randn(MersenneTwister(base + 2b), n)        # Gaussian null
            r = voigt_mle(yg; starts = starts)
            lrg[b] = 2 * (r.loglik - r.loglik_gaussian)
            u = rand(MersenneTwister(base + 2b + 1), n)
            yc = tan.(π .* (u .- 0.5))                       # Cauchy null
            r = voigt_mle(yc; starts = starts)
            lrc[b] = 2 * (r.loglik - r.loglik_cauchy)
        end
        cutg[n] = _q95(lrg)
        cutc[n] = _q95(lrc)
        @printf("  n = %6d | Gaussian null: 95%% cutoff %6.3f, size of 2.706 rule %5.3f | Cauchy null: cutoff %6.3f, size %5.3f\n",
                n, cutg[n], mean(lrg .> 2.706), cutc[n], mean(lrc .> 2.706))
        flush(stdout)
    end
    return cutg, cutc
end

if abspath(PROGRAM_FILE) == @__FILE__
    ns = isempty(ARGS) ? [100, 1000] : parse.(Int, ARGS)
    calibrate_boundary_cutoffs(ns)
end
