"""Matrix-free Chebyshev approximation of signed Hodge heat operators.

The polynomial acts on B.T(B x) + B(B.T x), never on a materialized
Laplacian. Sparse COO boundaries give work proportional to their nonzeros.
"""

import math

import torch

from tope.models.hodge_diffusion import (
    PolynomialHodgeDiffusionResult,
    validate_boundary_operators,
)


def _norm_bound(boundary):
    # ||B||_2^2 <= ||B||_1 ||B||_inf, valid for signed/weighted incidences.
    if boundary.is_sparse:
        boundary = boundary.coalesce()
        indices, values = boundary.indices(), boundary.values().abs()
        rows = values.new_zeros(boundary.size(0)).index_add(0, indices[0], values)
        cols = values.new_zeros(boundary.size(1)).index_add(0, indices[1], values)
    else:
        rows, cols = boundary.abs().sum(1), boundary.abs().sum(0)
    if not rows.numel() or not cols.numel():
        return boundary.new_zeros((), layout=torch.strided)
    return rows.max() * cols.max()


def chebyshev_heat(signals, boundaries, times, config):
    """Return heat reconstructions for every rank, including complex fields.

    Split time so each polynomial sees t * lambda_bound <= 4. Degree 24
    is the default; lower degrees trade accuracy for work. The polynomial
    is corrected at lambda=0 to preserve harmonic fields to roundoff.
    """
    # Shape/layout/finite checks remain mandatory even when chain checking is off.
    if config.validate_chain:
        validate_boundary_operators(boundaries, config.chain_tolerance)
    else:
        if not boundaries:
            raise ValueError("at least one boundary operator is required")
        for boundary in boundaries:
            validate_boundary_operators([boundary], config.chain_tolerance)
        for first, second in zip(boundaries, boundaries[1:]):
            if first.size(1) != second.size(0):
                raise ValueError("boundary operators have incompatible cell counts")
            if first.dtype != second.dtype or first.device != second.device:
                raise ValueError("boundary operators must share one device and dtype")
    if len(signals) != len(boundaries) + 1:
        raise ValueError("signals must provide one cochain for every Hodge rank")

    results = []
    for rank, signal in enumerate(signals):
        if not (signal.is_floating_point() or signal.is_complex()):
            raise TypeError("signals must use floating or complex values")
        signal = signal[:, None] if signal.dim() == 1 else signal
        cells = boundaries[0].size(0) if rank == 0 else boundaries[rank - 1].size(1)
        if signal.dim() != 2 or signal.size(0) != cells:
            raise ValueError("signal must have shape (rank_cells, channels)")
        if not torch.isfinite(signal).all():
            raise ValueError("signal must contain only finite values")
        t = torch.as_tensor(times, device=signal.device, dtype=signal.real.dtype).reshape(-1)
        if not torch.isfinite(t).all() or (t < 0).any():
            raise ValueError("diffusion times must be finite and non-negative")
        adjacent = []
        if rank > 0:
            adjacent.append((boundaries[rank - 1], True))
        if rank < len(boundaries):
            adjacent.append((boundaries[rank], False))
        bound = sum(_norm_bound(b).to(signal.real) for b, _ in adjacent).detach()
        if not torch.isfinite(bound):
            raise ValueError("spectral upper bound overflowed; rescale boundaries")
        operators = [(b.to(device=signal.device, dtype=signal.dtype), lower)
                     for b, lower in adjacent]

        def laplacian(x):
            out = torch.zeros_like(x)
            for b, lower in operators:
                bt = b.transpose(0, 1)
                out = out + (bt @ (b @ x) if lower else b @ (bt @ x))
            return out

        if bound.item() == 0 or t.numel() == 0 or signal.numel() == 0:
            reconstruction = signal.unsqueeze(0).expand(t.numel(), -1, -1) + t[:, None, None] * 0
            results.append(PolynomialHodgeDiffusionResult(rank, t, reconstruction, bound, 0))
            continue
        scaled_time = (t.detach().max() * bound).item()
        if not math.isfinite(scaled_time) or scaled_time > 4 * 4096:
            raise ValueError("diffusion requires too many polynomial steps; use the spectral backend")
        steps = max(1, math.ceil(scaled_time / 4))
        degree = config.chebyshev_degree
        samples = 2 * (degree + 1)
        theta = (torch.arange(samples, device=signal.device, dtype=t.dtype) + 0.5) * (math.pi / samples)
        orders = torch.arange(degree + 1, device=signal.device, dtype=t.dtype)
        basis = torch.cos(orders[:, None] * theta[None, :])
        values = torch.exp(-(t[:, None] / steps) * (bound / 2) * (theta.cos()[None, :] + 1))
        coefficients = (2 / samples) * (values @ basis.T)
        # c0 has half weight. Correct p(0) to one without changing the other coefficients.
        weights = torch.ones(degree + 1, device=signal.device, dtype=t.dtype)
        weights[0] = 0.5
        coefficients = coefficients * weights
        signs = torch.where(orders.remainder(2) == 0, 1.0, -1.0)
        correction = 1 - (coefficients * signs).sum(-1)
        coefficients = coefficients + correction[:, None] * (orders == 0)
        x = signal.unsqueeze(0).expand(t.numel(), -1, -1)

        def scaled_apply(v):
            # Pack time and channel axes into columns for sparse matrix products.
            packed = v.permute(1, 0, 2).reshape(cells, -1)
            applied = laplacian(packed).reshape(cells, t.numel(), signal.size(1)).permute(1, 0, 2)
            return (2 / bound) * applied - v

        for _ in range(steps):
            previous = x
            current = scaled_apply(x)
            output = coefficients[:, 0, None, None] * previous + coefficients[:, 1, None, None] * current
            for order in range(2, degree + 1):
                following = 2 * scaled_apply(current) - previous
                output = output + coefficients[:, order, None, None] * following
                previous, current = current, following
            x = output
        results.append(PolynomialHodgeDiffusionResult(rank, t, x, bound, steps))
    return results
