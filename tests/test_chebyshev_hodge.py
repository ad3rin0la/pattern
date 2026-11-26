"""Independent matrix-exponential and invariance checks for sparse Hodge heat."""
import warnings

import pytest
import torch

from tope.models.hodge_diffusion import HodgeDiffusionConfig, HodgeHeatDiffusion, hodge_laplacian


def triangle():
    return [torch.tensor([[-1., 0., 1.], [1., -1., 0.], [0., 1., -1.]], dtype=torch.double),
            torch.ones(3, 1, dtype=torch.double)]


def heat(degree=24):
    return HodgeHeatDiffusion(HodgeDiffusionConfig(backend="chebyshev", chebyshev_degree=degree))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("complex_signal", [False, True])
def test_all_ranks_match_matrix_exponential(sparse, complex_signal):
    torch.manual_seed(19)
    boundaries = triangle()
    signals = [torch.randn(n, 3, dtype=torch.double) for n in (3, 3, 1)]
    if complex_signal:
        signals = [x + 1j * torch.randn_like(x) for x in signals]
    times = torch.tensor([0., 0.1, 1., 8.], dtype=torch.double)
    results = heat()(signals, [b.to_sparse() for b in boundaries] if sparse else boundaries, times)
    for rank, result in enumerate(results):
        lap = hodge_laplacian(boundaries, rank).to(signals[rank].dtype)
        expected = torch.stack([torch.linalg.matrix_exp(-t * lap) @ signals[rank] for t in times])
        torch.testing.assert_close(result.reconstruction, expected, atol=2e-12, rtol=2e-12)


def test_harmonic_mass_semigroup_and_orientation():
    b = triangle()[0]
    signals = [torch.randn(3, 2, dtype=torch.double), torch.ones(3, 2, dtype=torch.double)]
    first = heat()(signals, [b.to_sparse()], [0.3])
    second = heat()([r.reconstruction[0] for r in first], [b.to_sparse()], [0.7])
    full = heat()(signals, [b.to_sparse()], [1.0])
    torch.testing.assert_close(full[0].reconstruction[0].sum(0), signals[0].sum(0))
    torch.testing.assert_close(full[1].reconstruction[0], signals[1])
    for a, c in zip(second, full):
        torch.testing.assert_close(a.reconstruction, c.reconstruction)
    signs = torch.tensor([-1., 1., -1.], dtype=torch.double)
    flipped = heat()([signals[0], signals[1] * signs[:, None]], [(b * signs).to_sparse()], [1.])
    torch.testing.assert_close(flipped[0].reconstruction, full[0].reconstruction)
    torch.testing.assert_close(flipped[1].reconstruction, full[1].reconstruction * signs[None, :, None])


def test_signal_and_time_gradients_match_exact():
    b = triangle()[0].to_sparse()
    x = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
    t = torch.tensor([0., 0.2], dtype=torch.double, requires_grad=True)
    actual = heat()([x, x], [b], t)[0].reconstruction
    expected = torch.stack([torch.linalg.matrix_exp(-v * hodge_laplacian([b], 0)) @ x for v in t])
    probe = torch.randn_like(actual)
    got = torch.autograd.grad((actual * probe).sum(), (x, t), retain_graph=True)
    want = torch.autograd.grad((expected * probe).sum(), (x, t))
    for a, e in zip(got, want):
        torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-10)


def test_no_dense_laplacian_or_eigensolve(monkeypatch):
    b = triangle()[0].to_sparse()
    def forbidden(*args, **kwargs):
        raise AssertionError("dense spectral operation used")
    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    heat()([torch.ones(3, 1), torch.ones(3, 1)], [b], [0., 1.])


def test_empty_zero_and_validation():
    b = torch.zeros(3, 0, dtype=torch.double).to_sparse()
    x = torch.randn(3, 2, dtype=torch.double)
    out = heat()([x, torch.empty(0, 2, dtype=torch.double)], [b], [0., 20.])
    torch.testing.assert_close(out[0].reconstruction, x[None].expand(2, -1, -1))
    assert out[1].reconstruction.shape == (2, 0, 2)
    assert heat()([x, x], [triangle()[0]], [])[0].reconstruction.shape == (0, 3, 2)
    with pytest.raises(ValueError, match="chain condition"):
        heat()([x, x, x[:1]], [triangle()[0].to_sparse(), torch.tensor([[1.], [-1.], [1.]], dtype=torch.double).to_sparse()], [1.])
    for times in ([-1.], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="times"):
            heat()([x, x], [triangle()[0]], times)
    with pytest.raises(ValueError, match="max_modes"):
        HodgeDiffusionConfig(backend="chebyshev", max_modes=3)


def test_sparse_chain_validation_does_not_enter_beta_csr_path():
    boundaries = [boundary.to_sparse() for boundary in triangle()]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        heat()(
            [torch.ones(size, 1, dtype=torch.double) for size in (3, 3, 1)],
            boundaries,
            [0.1],
        )
