# Diagnostic: integrated likelihood identities of the dispatched derivatives,
# per width ratio, with naive vs compensated (Kahan) accumulation and two
# quadrature resolutions. Prints the scaled statistics
#   Es_i  = E[s_i] * scale_i           (target 0)
#   M_ij  = E[s_i s_j + H_ij] * scale_i * scale_j   (target 0)
#   eigmin of -E[H] (scaled)           (target > 0)
# with scale = (gamma, gamma^2/sigma, gamma).
#
# Run:  julia --project=. examples/identity_diag.jl
using VoigtInference, Printf, LinearAlgebra

const V = VoigtInference

function kahan_add(s, c, x)
    t = s + x
    if abs(s) >= abs(x)
        c += (s - t) + x
    else
        c += (x - t) + s
    end
    return t, c
end

function moments(γ, σ, n)
    t, wq = V._gauss_legendre(n)
    scale = (γ, γ * γ / σ, γ)
    Es = zeros(3); Ec = zeros(3)
    M = zeros(3, 3); Mc = zeros(3, 3)
    EH = zeros(3, 3); EHc = zeros(3, 3)
    for (ti, wi) in zip(t, wq)
        yi = γ * tan((π / 2) * ti)
        di = γ * (π / 2) / cos((π / 2) * ti)^2
        f = voigt_pdf(yi, 0.0, σ, γ)
        sc = voigt_score(yi, 0.0, σ, γ)
        H = voigt_hessian(yi, 0.0, σ, γ)
        wf = wi * f * di
        for i in 1:3
            Es[i], Ec[i] = kahan_add(Es[i], Ec[i], wf * sc[i] * scale[i])
            for j in i:3
                x = wf * (sc[i] * sc[j] + H[i, j]) * scale[i] * scale[j]
                M[i, j], Mc[i, j] = kahan_add(M[i, j], Mc[i, j], x)
                xh = wf * H[i, j] * scale[i] * scale[j]
                EH[i, j], EHc[i, j] = kahan_add(EH[i, j], EHc[i, j], xh)
            end
        end
    end
    return Es .+ Ec, Symmetric(M .+ Mc), Symmetric(EH .+ EHc)
end

σ = 1.0
println("scaled statistics; naive column uses plain summation at n = 4000")
@printf("%8s %6s | %10s %10s %10s | %10s | %10s | %12s\n",
        "γ/σ", "n", "Es_mu", "Es_si", "Es_ga", "max|M|", "eig-min", "M_ss")
for ratio in (10.0, 50.0, 1.0e2, 1.0e3, 1.0e4, 1.0e6, 1.0e8)
    γ = ratio * σ
    for n in (4000, 12000)
        Es, M, EH = moments(γ, σ, n)
        emin = minimum(eigen(-Matrix(EH)).values)
        @printf("%8.0e %6d | %10.2e %10.2e %10.2e | %10.2e | %10.2e | %12.4e\n",
                ratio, n, Es[1], Es[2], Es[3], maximum(abs, M), emin, M[2, 2])
    end
end
