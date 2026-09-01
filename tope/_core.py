"""Optional native kernels and device-native fallbacks for Pattern.

Set ``TOPE_NATIVE_BACKEND=build`` to JIT-compile the C++ hierarchy kernel.
The tensor fallback remains the default and works on every PyTorch device.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Sequence

import torch

_NATIVE: Optional[ModuleType] = None
_LOAD_ATTEMPTED = False


def build_native_backend(verbose: bool = False) -> ModuleType:
    """Compile and load the optional CPU C++ traversal kernel."""
    global _LOAD_ATTEMPTED, _NATIVE
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("csrc") / "electronic_hierarchy.cpp"
    build_directory = Path(
        os.environ.get(
            "TOPE_NATIVE_BUILD_DIR",
            str(Path(tempfile.gettempdir()) / "tope_native" / "tope_core_native"),
        )
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    _NATIVE = load(
        name="tope_core_native",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cflags=["-O3"],
        verbose=verbose,
    )
    _LOAD_ATTEMPTED = True
    return _NATIVE


def _native_backend() -> Optional[ModuleType]:
    global _LOAD_ATTEMPTED, _NATIVE
    if _LOAD_ATTEMPTED:
        return _NATIVE
    _LOAD_ATTEMPTED = True
    mode = os.environ.get("TOPE_NATIVE_BACKEND", "auto").lower()
    if mode in {"0", "false", "off", "none"}:
        return None
    try:
        import tope_core_native

        _NATIVE = tope_core_native
    except ImportError:
        if mode == "build":
            _LOAD_ATTEMPTED = False
            return build_native_backend()
    return _NATIVE


def native_backend_available() -> bool:
    """Return whether the compiled hierarchy backend is loaded."""
    return _native_backend() is not None


def _tensor_hierarchical_candidates(
    query_coords: torch.Tensor,
    centers: Sequence[torch.Tensor],
    incidences: Sequence[torch.Tensor],
    top_coarse: int,
    max_children: int,
) -> List[torch.Tensor]:
    n_ranks = len(centers)
    coarse_k = min(top_coarse, centers[-1].size(0))
    candidates: List[Optional[torch.Tensor]] = [None] * n_ranks
    candidates[-1] = torch.cdist(query_coords, centers[-1]).topk(
        coarse_k, largest=False
    ).indices
    n_queries = query_coords.size(0)

    for rank in range(n_ranks - 2, -1, -1):
        child, parent = incidences[rank]
        selected_parents = candidates[rank + 1]
        assert selected_parents is not None
        selected_edges = (parent[None, :, None] == selected_parents[:, None, :]).any(
            dim=-1
        )
        n_child = centers[rank].size(0)
        hits = torch.zeros(
            (n_queries, n_child), dtype=torch.long, device=query_coords.device
        )
        hits.scatter_add_(1, child.expand(n_queries, -1), selected_edges.long())
        selected_children = hits > 0
        empty = ~selected_children.any(dim=-1)
        selected_children = torch.where(
            empty[:, None], torch.ones_like(selected_children), selected_children
        )

        distance = torch.cdist(query_coords, centers[rank])
        distance = distance.masked_fill(~selected_children, float("inf"))
        retained = selected_children.sum(dim=-1).clamp_max(max_children)
        width = min(max_children, int(retained.max().item()))
        indices = distance.topk(width, largest=False).indices
        valid = torch.arange(width, device=query_coords.device) < retained[:, None]
        candidates[rank] = indices.masked_fill(~valid, -1)

    return [candidate for candidate in candidates if candidate is not None]


def hierarchical_candidates(
    query_coords: torch.Tensor,
    centers: Sequence[torch.Tensor],
    incidences: Sequence[torch.Tensor],
    top_coarse: int,
    max_children: int,
    prefer_native: bool = True,
) -> List[torch.Tensor]:
    """Traverse a cell hierarchy using C++ on CPU or tensors on any device."""
    use_native = (
        prefer_native
        and query_coords.device.type == "cpu"
        and query_coords.dtype in {torch.float32, torch.float64}
    )
    native = _native_backend() if use_native else None
    if native is not None:
        return native.hierarchical_candidates(
            query_coords.contiguous(),
            [value.contiguous() for value in centers],
            [value.contiguous() for value in incidences],
            top_coarse,
            max_children,
        )
    return _tensor_hierarchical_candidates(
        query_coords, centers, incidences, top_coarse, max_children
    )
