"""Pytest fixtures and environment shims for the ToPE test suite."""

from __future__ import annotations

import sys
import types


def _ensure_torch_scatter_stub() -> None:
    """Stub ``torch_scatter`` when the native build is unavailable.

    ``torch-scatter`` ships as a C++/CUDA extension and frequently fails to
    build in minimal CI environments. The import tests only need the symbols
    to exist; runtime behaviour is exercised by separate GPU-marked tests.
    """
    try:
        import torch_scatter  # noqa: F401
        return
    except ImportError:
        pass

    stub = types.ModuleType("torch_scatter")
    for fn_name in (
        "scatter_add",
        "scatter_mean",
        "scatter_max",
        "scatter_min",
        "scatter_softmax",
        "scatter_sum",
        "scatter_mul",
    ):
        setattr(stub, fn_name, lambda *args, **kwargs: None)
    sys.modules["torch_scatter"] = stub


_ensure_torch_scatter_stub()
