# Julia side of the Julia-vs-Python benchmark.
#
# Mirrors bench/bench.py layer for layer, on the SAME data files, with the same
# timing protocol (warm-up call, inner-loop count chosen for a ~0.25 s block,
# best of 5 blocks).  The protocol is hand-rolled rather than BenchmarkTools so
# that both languages are measured identically and so that this script needs no
# dependency beyond the package itself.
#
# Run `python bench/gendata.py` first, then, from the VoigtInference.jl package
# directory:
#
#     julia -t 1 --project=. /path/to/bench/bench.jl -o /path/to/bench/results_julia.json

using VoigtInference
using SpecialFunctions: erfcx
import SpecialFunctions               # module reference for pkgversion
using Printf
using Statistics: median

const TARGET_SECONDS = 0.25
const MIN_REPEATS = 5
const SQRT2OVERPI = sqrt(2 / pi)

# ---------------------------------------------------------------- utilities

function read_params(dir)
    p = Dict{String,Any}()
    for line in eachline(joinpath(dir, "params.txt"))
        parts = split(strip(line))
        if isempty(parts)
            continue
        end
        key = parts[1]
        if key == "sizes"
            p[key] = parse.(Int, parts[2:end])
        else
            p[key] = parse(Float64, parts[2])
        end
    end
    return p
end

function read_f64(path)
    return collect(reinterpret(Float64, read(path)))
end

"""Best-of-`repeats` seconds per call of `f`, after a warm-up call."""
function timeit(f; target = TARGET_SECONDS, repeats = MIN_REPEATS)
    f()                                  # warm-up: triggers JIT compilation
    n = 1
    while true
        t0 = time_ns()
        for _ in 1:n
            f()
        end
        dt = (time_ns() - t0) / 1e9
        if dt >= target || n >= 2^20
            break
        end
        n = max(2 * n, ceil(Int, n * target / max(dt, 1e-9)))
    end
    times = Float64[]
    for _ in 1:repeats
        t0 = time_ns()
        for _ in 1:n
            f()
        end
        push!(times, (time_ns() - t0) / 1e9 / n)
    end
    return minimum(times), median(times)
end

# --------------------------------------------------------- measured kernels
#
# Each kernel accumulates into a scalar or array so the compiler cannot elide
# the work.

function bench_primitive!(w, z)
    @inbounds for i in eachindex(z)
        w[i] = erfcx(-im * z[i])
    end
    return w
end

function sum_logdensity(y, mu, sg, gm)
    s = 0.0
    @inbounds for yi in y
        s += voigt_logpdf(yi, mu, sg, gm)
    end
    return s
end

function sum_score(y, mu, sg, gm)
    s = 0.0
    @inbounds for yi in y
        s += sum(voigt_score(yi, mu, sg, gm))
    end
    return s
end

function sum_hessian(y, mu, sg, gm)
    s = 0.0
    @inbounds for yi in y
        s += sum(voigt_hessian(yi, mu, sg, gm))
    end
    return s
end

# finite-difference score: 6 log-density evaluations (matches bench.py)
function fd_score_sum(y, mu, sg, gm, h = 1e-6)
    s = 0.0
    @inbounds for yi in y
        s += (voigt_logpdf(yi, mu + h, sg, gm) - voigt_logpdf(yi, mu - h, sg, gm)) / (2 * h)
        s += (voigt_logpdf(yi, mu, sg + h, gm) - voigt_logpdf(yi, mu, sg - h, gm)) / (2 * h)
        s += (voigt_logpdf(yi, mu, sg, gm + h) - voigt_logpdf(yi, mu, sg, gm - h)) / (2 * h)
    end
    return s
end

# finite-difference Hessian: 6 pairs x 4 evaluations = 24 (matches bench.py)
function fd_hessian_sum(y, mu, sg, gm, h = 1e-4)
    s = 0.0
    p = (mu, sg, gm)
    @inbounds for yi in y
        for j in 1:3
            for k in j:3
                ej = ntuple(i -> i == j ? h : 0.0, 3)
                ek = ntuple(i -> i == k ? h : 0.0, 3)
                a = p .+ ej .+ ek
                b = p .+ ej .- ek
                c = p .- ej .+ ek
                d = p .- ej .- ek
                s += (voigt_logpdf(yi, a[1], a[2], a[3]) -
                      voigt_logpdf(yi, b[1], b[2], b[3]) -
                      voigt_logpdf(yi, c[1], c[2], c[3]) +
                      voigt_logpdf(yi, d[1], d[2], d[3])) / (4 * h * h)
            end
        end
    end
    return s
end

# The separate PUBLIC routines: voigt_logpdf, voigt_score and voigt_hessian
# each make their own Faddeeva evaluation, so this loop costs THREE passes
# per observation. (The Julia package's optimizer does NOT do this: its
# internal kernel is fused like fused_one_eval below; this variant is timed
# as the reference point for what fusion buys.)
function ll_score_hess_public(y, mu, sg, gm)
    ll = 0.0
    g = zeros(3)
    H = zeros(3, 3)
    @inbounds for yi in y
        ll += voigt_logpdf(yi, mu, sg, gm)
        g .+= voigt_score(yi, mu, sg, gm)
        H .+= voigt_hessian(yi, mu, sg, gm)
    end
    return ll, g, H
end

# The like-for-like counterpart of voigtinference.loglik_grad_hess: the
# log-likelihood, score sum and Hessian sum from ONE Faddeeva evaluation per
# observation, mirroring the fused kernel inside VoigtInference.jl's
# optimizer.
#
# The far-tail branch is deliberately omitted here: the benchmark sample has no
# far-tail points, and including the switch would time a branch that never
# fires.  Correctness of the branch is the cross-check's job, not the
# benchmark's.
function fused_one_eval(y, mu, sg, gm)
    ll = 0.0
    gmu = 0.0; gsg = 0.0; ggm = 0.0
    hmumu = 0.0; hmusg = 0.0; hmugm = 0.0
    hsgsg = 0.0; hgmsg = 0.0; hgmgm = 0.0
    s2 = sg * sg
    den2 = sg * sqrt(2)
    aim = gm / den2
    logconst = log(sg) + 0.5 * log(2 * pi)
    g2 = gm * gm
    r_s = 1.0e-4                        # score switch   (matches the package)
    r_h = 5.0e-4                        # Hessian switch (matches the package)
    @inbounds for yi in y
        yt = yi - mu
        yt2 = yt * yt
        wz = erfcx(-im * complex(yt / den2, aim))
        K = real(wz)
        L = imag(wz)
        r = L / K
        ll += log(K) - logconst
        smu = (yt - gm * L / K) / s2
        ssg = ((yt2 - gm * gm - s2) * K - 2 * gm * yt * L + SQRT2OVERPI * sg * gm) / (sg^3 * K)
        sgm = (gm * K + yt * L - SQRT2OVERPI * sg) / (s2 * K)
        hmm = ssg / sg - smu * smu
        hgg = -ssg / sg - sgm * sgm
        hmg = (yt * sgm + gm * smu - r) / s2 - smu * sgm
        hms = -(smu + gm * hmg - yt * hmm) / sg
        hgs = -(sgm + gm * hgg - yt * hmg) / sg
        hss = -(ssg + gm * hgs - yt * hms) / sg
        if s2 < r_s * (yt2 + g2)                        # Cauchy-limit score
            smu, ssg, sgm = VoigtInference._score_tail(yt, sg, gm)
        end
        if s2 < r_h * (yt2 + g2)                        # Cauchy-limit Hessian
            hmm, hms, hmg, hss, hgs, hgg = VoigtInference._hessian_tail(yt, sg, gm)
        end
        gmu += smu; gsg += ssg; ggm += sgm
        hmumu += hmm; hmusg += hms; hmugm += hmg
        hsgsg += hss; hgmsg += hgs; hgmgm += hgg
    end
    g = [gmu, gsg, ggm]
    H = [hmumu hmusg hmugm; hmusg hsgsg hgmsg; hmugm hgmsg hgmgm]
    return ll, g, H
end

# ------------------------------------------------------------------- output

function json_escape(s)
    return replace(string(s), "\\" => "\\\\", "\"" => "\\\"")
end

function main()
    outpath = "results_julia.json"
    datadir = joinpath(@__DIR__, "data")
    args = copy(ARGS)
    i = 1
    while i <= length(args)
        if (args[i] == "-o" || args[i] == "--out") && i < length(args)
            outpath = args[i + 1]
            i += 2
        elseif args[i] == "--data" && i < length(args)
            datadir = args[i + 1]
            i += 2
        else
            i += 1
        end
    end

    if !isfile(joinpath(datadir, "params.txt"))
        error("run `python bench/gendata.py` first (looked in $datadir)")
    end

    p = read_params(datadir)
    mu = p["mu"]
    sg = p["sigma"]
    gm = p["gamma"]
    sizes = p["sizes"]
    nbig = maximum(sizes)
    y = read_f64(joinpath(datadir, "y_$(nbig).f64"))
    if length(y) != nbig
        error("data file length mismatch: got $(length(y)), expected $nbig")
    end

    # sanity: the fused kernel must reproduce the public API
    ll_a, g_a, H_a = ll_score_hess_public(y[1:1000], mu, sg, gm)
    ll_b, g_b, H_b = fused_one_eval(y[1:1000], mu, sg, gm)
    dmax = max(abs(ll_a - ll_b) / abs(ll_a),
               maximum(abs.(g_a .- g_b) ./ max.(abs.(g_a), 1e-300)),
               maximum(abs.(H_a .- H_b) ./ max.(abs.(H_a), 1e-300)))
    @printf("fused kernel agrees with the public API to %.2e\n\n", dmax)
    if !(dmax < 1e-12)
        error("fused kernel disagrees with the public API (max rel diff $dmax)")
    end

    # --- layer 1: the primitive -------------------------------------------
    z = [complex((yi - mu) / (sg * sqrt(2)), gm / (sg * sqrt(2))) for yi in y]
    w = similar(z)
    pb, pm = timeit(() -> bench_primitive!(w, z))
    primitive_ns = 1e9 * pb / nbig
    primitive_ns_med = 1e9 * pm / nbig

    # --- layer 2: per-observation cost ------------------------------------
    names = ["log-density", "analytic score", "analytic Hessian",
             "fused ll+grad+hess", "ll+score+hess (3 evals)",
             "FD score (6 evals)", "FD Hessian (24 evals)"]
    thunks = Function[
        () -> sum_logdensity(y, mu, sg, gm),
        () -> sum_score(y, mu, sg, gm),
        () -> sum_hessian(y, mu, sg, gm),
        () -> fused_one_eval(y, mu, sg, gm),
        () -> ll_score_hess_public(y, mu, sg, gm),
        () -> fd_score_sum(y, mu, sg, gm),
        () -> fd_hessian_sum(y, mu, sg, gm),
    ]
    ns = Float64[]
    ns_med = Float64[]
    ratios = Float64[]
    base = 0.0
    for k in eachindex(thunks)
        b, m = timeit(thunks[k])
        if k == 1
            base = b
        end
        push!(ns, 1e9 * b / nbig)
        push!(ns_med, 1e9 * m / nbig)
        push!(ratios, b / base)
    end

    # --- layer 3: end-to-end MLE ------------------------------------------
    mle_n = Int[]
    mle_sec = Float64[]
    mle_it = Int[]
    mle_conv = Bool[]
    mle_mu = Float64[]
    mle_sig = Float64[]
    mle_gam = Float64[]
    mle_ll = Float64[]
    for n in sizes
        yn = read_f64(joinpath(datadir, "y_$(n).f64"))
        voigt_mle(yn)                     # warm-up
        t0 = time_ns()
        r = voigt_mle(yn)
        dt = (time_ns() - t0) / 1e9
        push!(mle_n, n)
        push!(mle_sec, dt)
        push!(mle_it, r.iterations)
        push!(mle_conv, r.converged)
        push!(mle_mu, r.μ)
        push!(mle_sig, r.σ)
        push!(mle_gam, r.γ)
        push!(mle_ll, r.loglik)
    end

    # --- layer 4: Fisher quadrature ---------------------------------------
    fb, fm = timeit(() -> voigt_fisher(mu, sg, gm; nodes = 400))

    # --- layer 5: throughput ----------------------------------------------
    nmid = sizes[cld(length(sizes), 2)]
    ymid = read_f64(joinpath(datadir, "y_$(nmid).f64"))
    voigt_mle(ymid)
    t0 = time_ns()
    reps = 0
    while (time_ns() - t0) / 1e9 < 2.0
        voigt_mle(ymid)
        reps += 1
    end
    tp_dt = (time_ns() - t0) / 1e9

    # ------------------------------------------------------------ write out
    io = IOBuffer()
    println(io, "{")
    println(io, "  \"language\": \"julia\",")
    sfver = string(pkgversion(SpecialFunctions))
    println(io, "  \"versions\": {\"julia\": \"$(VERSION)\", \"SpecialFunctions\": \"$(sfver)\"},")
    println(io, "  \"platform\": {")
    println(io, "    \"system\": \"$(json_escape(Sys.KERNEL))\",")
    println(io, "    \"machine\": \"$(json_escape(Sys.MACHINE))\",")
    println(io, "    \"processor\": \"$(json_escape(Sys.CPU_NAME))\",")
    println(io, "    \"threads\": $(Threads.nthreads())")
    println(io, "  },")
    println(io, "  \"n_per_obs\": $nbig,")
    println(io, "  \"fused_matches_public_api\": $dmax,")
    println(io, "  \"primitive\": {\"name\": \"SpecialFunctions.erfcx\", \"ns_per_obs\": $primitive_ns, \"ns_per_obs_median\": $primitive_ns_med},")
    println(io, "  \"per_obs\": {")
    for k in eachindex(names)
        comma = k == length(names) ? "" : ","
        println(io, "    \"$(names[k])\": {\"ns_per_obs\": $(ns[k]), \"ns_per_obs_median\": $(ns_med[k]), \"ratio_to_density\": $(ratios[k])}$comma")
    end
    println(io, "  },")
    println(io, "  \"mle\": [")
    for k in eachindex(mle_n)
        comma = k == length(mle_n) ? "" : ","
        spi = mle_sec[k] / max(mle_it[k], 1)
        println(io, "    {\"n\": $(mle_n[k]), \"seconds\": $(mle_sec[k]), \"iterations\": $(mle_it[k]), \"seconds_per_iteration\": $spi, \"converged\": $(mle_conv[k]), \"mu\": $(mle_mu[k]), \"sigma\": $(mle_sig[k]), \"gamma\": $(mle_gam[k]), \"loglik\": $(mle_ll[k])}$comma")
    end
    println(io, "  ],")
    println(io, "  \"fisher\": {\"nodes\": 400, \"seconds\": $fb, \"seconds_median\": $fm},")
    println(io, "  \"throughput\": {\"n\": $(length(ymid)), \"fits\": $reps, \"seconds\": $tp_dt, \"fits_per_sec\": $(reps / tp_dt)}")
    println(io, "}")
    write(outpath, String(take!(io)))

    @printf("%-24s%10.1f ns/obs\n\n", "primitive", primitive_ns)
    @printf("%-24s%10s%9s\n", "task", "ns/obs", "ratio")
    for k in eachindex(names)
        @printf("%-24s%10.1f%9.2f\n", names[k], ns[k], ratios[k])
    end
    println()
    for k in eachindex(mle_n)
        @printf("MLE n = %7d: %7.4f s   %2d iterations\n", mle_n[k], mle_sec[k], mle_it[k])
    end
    @printf("\nFisher (400 nodes): %.2f ms\n", fb * 1e3)
    @printf("throughput at n = %d: %.1f fits/s\n", length(ymid), reps / tp_dt)
    println("\nwrote $outpath")
end

main()
