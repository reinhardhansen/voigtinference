"""voigt_pdf_score must be bit-identical to the separate calls."""
import numpy as np

from voigtinference import voigt_pdf, voigt_pdf_score, voigt_score


def test_fused_pdf_score_identity():
    # exact branch, far tail (large |ytil|), and large gamma/sigma (far via r)
    cases = [
        (0.3, 1.2, 0.7, np.array([-3.0, 0.0, 0.31, 2.1, 15.0, 1e6])),
        (0.0, 1.0, 1e5, np.array([0.0, 1.0, 1e7])),
    ]
    for mu, sigma, gamma, y in cases:
        f, s = voigt_pdf_score(y, mu, sigma, gamma)
        assert np.array_equal(f, voigt_pdf(y, mu, sigma, gamma))
        assert np.array_equal(s, voigt_score(y, mu, sigma, gamma))


def test_fused_pdf_score_scalar():
    f, s = voigt_pdf_score(1.7, 0.3, 1.2, 0.7)
    assert np.isscalar(f) or f.ndim == 0
    assert s.shape == (3,)
    assert f == voigt_pdf(1.7, 0.3, 1.2, 0.7)
    assert np.array_equal(s, voigt_score(1.7, 0.3, 1.2, 0.7))
