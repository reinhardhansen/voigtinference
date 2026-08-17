using Test, Random, LinearAlgebra
using VoigtInference

const V = VoigtInference

# central finite differences
fd(f, x; h = 1e-5) = (f(x + h) - f(x - h)) / (2h)
fd2(f, x; h = 1e-4) = (f(x + h) - 2f(x) + f(x - h)) / h^2
fdmix(f, x, y; h = 1e-4) =
    (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4h^2)

@testset "density" begin
    μ, σ, γ = 0.3, 1.2, 0.7
    # integrates to one (tan substitution, Gauss-Legendre)
    s = σ + γ
    t, wq = V._gauss_legendre(200)
    mass = sum(wi * (π/2) * s / cos((π/2)*ti)^2 *
               voigt_pdf(μ + s*tan((π/2)*ti), μ, σ, γ) for (ti, wi) in zip(t, wq))
    @test isapprox(mass, 1.0; atol = 1e-8)
    # limits
    @test voigt_pdf(0.5, 0.0, 0.0, 1.0) ≈ 1/(π * (0.25 + 1))          # Cauchy
    @test voigt_pdf(0.5, 0.0, 1.0, 0.0) ≈ exp(-0.125)/sqrt(2π)         # Gaussian
    # symmetry
    @test voigt_pdf(μ + 1.3, μ, σ, γ) ≈ voigt_pdf(μ - 1.3, μ, σ, γ)
end

@testset "score and Hessian vs finite differences" begin
    μ, σ, γ = 0.3, 1.2, 0.7
    for y in (-3.0, 0.0, 0.31, 2.1, 15.0)
        s = voigt_score(y, μ, σ, γ)
        @test s[1] ≈ fd(m -> voigt_logpdf(y, m, σ, γ), μ) atol = 1e-6
        @test s[2] ≈ fd(t -> voigt_logpdf(y, μ, t, γ), σ) atol = 1e-6
        @test s[3] ≈ fd(t -> voigt_logpdf(y, μ, σ, t), γ) atol = 1e-6
        H = voigt_hessian(y, μ, σ, γ)
        @test H[1,1] ≈ fd2(m -> voigt_logpdf(y, m, σ, γ), μ) atol = 1e-4
        @test H[2,2] ≈ fd2(t -> voigt_logpdf(y, μ, t, γ), σ) atol = 1e-4
        @test H[3,3] ≈ fd2(t -> voigt_logpdf(y, μ, σ, t), γ) atol = 1e-4
        @test H[1,2] ≈ fdmix((m, t) -> voigt_logpdf(y, m, t, γ), μ, σ) atol = 1e-4
        @test H[1,3] ≈ fdmix((m, t) -> voigt_logpdf(y, m, σ, t), μ, γ) atol = 1e-4
        @test H[2,3] ≈ fdmix((t, u) -> voigt_logpdf(y, μ, t, u), σ, γ) atol = 1e-4
    end
end

@testset "conditional moments (Tweedie identities)" begin
    μ, σ, γ = 0.3, 1.2, 0.7
    for y in (-2.0, 0.3, 2.1, 8.0)
        # E[Z|y] = -σ² ∂y log f
        @test voigt_condmean(y, μ, σ, γ) ≈
              -σ^2 * fd(t -> voigt_logpdf(t, μ, σ, γ), y) atol = 1e-6
        # V(Z|y) = σ² + σ⁴ ∂²y log f
        @test voigt_condvar(y, μ, σ, γ) ≈
              σ^2 + σ^4 * fd2(t -> voigt_logpdf(t, μ, σ, γ), y) atol = 1e-3
        # E[Z|y] + E[X|y] = ỹ
        K, L, _, _ = V._KL(y, μ, σ, γ)
        @test voigt_condmean(y, μ, σ, γ) + γ * L / K ≈ y - μ
    end
    # redescending: essentially zero far in the tail
    @test abs(voigt_condmean(1e6, μ, σ, γ)) < 1e-2
end

@testset "Fisher information" begin
    μ, σ, γ = 0.0, 1.0, 1.0
    ℐ = voigt_fisher(μ, σ, γ)
    @test isposdef(Matrix(ℐ))
    # block diagonality by symmetry
    @test abs(ℐ[1,2]) < 1e-8
    @test abs(ℐ[1,3]) < 1e-8
    # information equality: ℐ = -E[H] by Monte Carlo
    rng = MersenneTwister(1)
    n = 200_000
    y = rand_voigt(rng, n, μ, σ, γ)
    EH = sum(voigt_hessian(yi, μ, σ, γ) for yi in y) / n
    @test maximum(abs.(Matrix(ℐ) + EH)) < 0.05
end

@testset "MLE recovery" begin
    rng = MersenneTwister(2026)
    μ0, σ0, γ0 = 0.5, 1.0, 0.3
    y = rand_voigt(rng, 5000, μ0, σ0, γ0)
    r = voigt_mle(y)
    @test r.converged
    @test abs(r.μ - μ0) < 5 * r.se[1]
    @test abs(r.σ - σ0) < 5 * r.se[2]
    @test abs(r.γ - γ0) < 5 * r.se[3]
    # score ≈ 0 at the optimum
    g = sum(voigt_score(yi, r.μ, r.σ, r.γ) for yi in y) / length(y)
    @test norm(g) < 1e-6
end
