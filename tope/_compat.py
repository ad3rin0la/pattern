"""Compatibility shims for optional native extensions.

`torch_scatter` and `torch_sparse` ship as compiled C++/CUDA wheels that
must be built against a specific torch ABI. They are routinely unavailable
on bleeding-edge torch builds. We fall back to `torch_geometric.utils.scatter`
(pure-Python, dispatches to torch primitives) which covers every call site
in this repo.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    from torch_scatter import (  # type: ignore[import-not-found]
        scatter_add as scatter_add,
        scatter_mean as scatter_mean,
        scatter_softmax as scatter_softmax,
    )
except ImportError:
    from torch_geometric.utils import scatter as _pyg_scatter

    def scatter_add(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        result = _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")
        if out is not None:
            out.copy_(result)
            return out
        return result

    def scatter_mean(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        result = _pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")
        if out is not None:
            out.copy_(result)
            return out
        return result

    def scatter_softmax(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        src_max = _pyg_scatter(src.detach(), index, dim=dim, dim_size=dim_size, reduce="max")
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = _pyg_scatter(out, index, dim=dim, dim_size=dim_size, reduce="sum")
        out_sum = out_sum.index_select(dim, index).clamp_min(1e-16)
        return out / out_sum


try:
    from torch_sparse import SparseTensor as SparseTensor  # type: ignore[import-not-found]
except ImportError:
    SparseTensor = None  # type: ignore[assignment]
