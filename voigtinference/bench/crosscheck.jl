# Check VoigtInference.jl against the Python reference values.
#
#   python bench/crosscheck.py
#   julia -t 1 --project=. /path/to/bench/crosscheck.jl
#
# Reports the maximum relative difference per quantity.  The grid spans the
# bulk, the neighbourhood of the far-tail switch, and several decades beyond,
# so a disagreement confined to the tail branch shows up rather than being
# averaged away.

using VoigtInference
using Printf

# Agreement expected between the two implementations, per quantity.
#
# All 1e-12, and in practice every entry comes back exactly 0.0: the two
# implementations perform identical floating-point operations in identical
# order, so they agree bit for bit in both branches.  Keep it tight -- slack
# here would hide a regression.
#
# Note that agreement is not accuracy.  V(Z|y) in particular is a difference of
# two terms of size sqrt(2/pi)*sigma*gamma/K returning a result of size sigma^2;
# near |ytil| ~ 400*scale those terms are ~7.0e5 and the answer is ~0.25, so it
# carries only ~10 correct digits no matter how it is computed.  That is a
# property of the formula, bounded in tests/test_accuracy.py against
# high-precision truth -- it is not something this file can or should measure.
const TOL = Dict("pdf" => 1e-12, "logpdf" => 1e-12, "score" => 1e-12,
                 "hessian" => 1e-12, "condmean" => 1e-12, "fisher" => 1e-12,
                 "condvar" => 1e-12)
tol_for(name) = get(TOL, name, 1e-12)

function reldiff(a, b)
    return abs(a - b) / max(abs(b), 1e-300)
end

function main()
    path = isempty(ARGS) ? joinpath(@__DIR__, "reference_python.txt") : ARGS[1]
    if !isfile(path)
        error("reference file not found: $path  (run bench/crosscheck.py first)")
    end

    mu = 0.0
    sg = 0.0
    gm = 0.0
    ncases = 0
    npoints = 0
    mle_rows = Vector{Vector{Float64}}()

    worst_d = Dict{String,Float64}()
    worst_y = Dict{String,Float64}()

    function note(name, d, y)
        if d > get(worst_d, name, -1.0)
            worst_d[name] = d
            worst_y[name] = y
        end
        return nothing
    end

    for line in eachline(path)
        if startswith(line, "#")
            continue
        end
        parts = split(strip(line))
        if isempty(parts)
            continue
        end
        tag = parts[1]
        if tag == "theta"
            mu = parse(Float64, parts[2])
            sg = parse(Float64, parts[3])
            gm = parse(Float64, parts[4])
            ncases += 1
        elseif tag == "fisher"
            v = parse.(Float64, parts[2:10])
            info = voigt_fisher(mu, sg, gm; nodes = 400)
            for i in 1:3
                for j in 1:3
                    ref = v[3 * (i - 1) + j]   # written row-major
                    # the mu off-diagonals are zero by symmetry; compare
                    # absolutely there
                    d = abs(ref) < 1e-10 ? abs(info[i, j] - ref) :
                                           reldiff(info[i, j], ref)
                    note("fisher", d, 0.0)
                end
            end
        elseif tag == "point"
            p = parse.(Float64, parts[2:end])
            npoints += 1
            y = p[1]
            note("pdf", reldiff(voigt_pdf(y, mu, sg, gm), p[2]), y)
            note("logpdf", reldiff(voigt_logpdf(y, mu, sg, gm), p[3]), y)

            sc = voigt_score(y, mu, sg, gm)
            for i in 1:3
                note("score", reldiff(sc[i], p[3 + i]), y)
            end

            H = voigt_hessian(y, mu, sg, gm)
            idx = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
            for k in 1:6
                i, j = idx[k]
                note("hessian", reldiff(H[i, j], p[6 + k]), y)
            end

            note("condmean", reldiff(voigt_condmean(y, mu, sg, gm), p[13]), y)
            note("condvar", reldiff(voigt_condvar(y, mu, sg, gm), p[14]), y)
        elseif tag == "mle"
            push!(mle_rows, parse.(Float64, parts[2:end]))
        end
    end

    @printf("%d theta cases;  %d evaluation points\n\n", ncases, npoints)

    @printf("%-12s%14s%14s%16s\n", "quantity", "max rel diff", "tolerance", "at y")
    ok = true
    for name in ("pdf", "logpdf", "score", "hessian", "condmean", "condvar", "fisher")
        if !haskey(worst_d, name)
            continue
        end
        d = worst_d[name]
        y = worst_y[name]
        tol = tol_for(name)
        flag = d <= tol ? "" : "   <-- ABOVE TOLERANCE"
        if d > tol
            ok = false
        end
        @printf("%-12s%14.2e%14.0e%16.6g%s\n", name, d, tol_for(name), y, flag)
    end

    if !isempty(mle_rows)
        println("\nMLE on the shared benchmark samples:")
        @printf("%8s%14s%14s%14s%14s\n", "n", "d(mu)", "d(sigma)", "d(gamma)", "d(loglik)")
        for row in mle_rows
            n = Int(row[1])
            f = joinpath(@__DIR__, "data", "y_$(n).f64")
            if !isfile(f)
                @printf("%8d   (data file missing, skipped)\n", n)
                continue
            end
            yn = collect(reinterpret(Float64, read(f)))
            r = voigt_mle(yn)
            ds = (reldiff(r.μ, row[2]), reldiff(r.σ, row[3]),
                  reldiff(r.γ, row[4]), reldiff(r.loglik, row[5]))
            @printf("%8d%14.2e%14.2e%14.2e%14.2e\n", n, ds...)
            # optimiser-path agreement enforced too (looser than the
            # evaluation tolerance: paths may differ at the level of the
            # convergence tolerance)
            if maximum(ds) > 1e-11
                ok = false
                println("        ^ ABOVE the 1e-11 MLE-path tolerance")
            end
        end
        println("\n(MLE differences reflect the optimiser path, not the formulas;")
        println(" they should sit at the level of the convergence tolerance.)")
    end

    println()
    if ok
        println("PASS: every quantity is within its tolerance.")
    else
        println("FAIL: at least one quantity exceeds the tolerance.")
    end
    exit(ok ? 0 : 1)
end

main()
