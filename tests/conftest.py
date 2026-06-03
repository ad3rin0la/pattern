"""Test-isolation for the by-path module-loading test files.

Several test files (``test_gyro_*``, ``test_data_import_fixes``) load
``tope.*`` submodules by file path while registering lightweight package
*stubs* in ``sys.modules`` — the top-level ``tope`` package cannot be imported
normally here because of an unrelated optional dependency (``torch_scatter``)
that isn't installed.  Those stubs (and the by-path submodules loaded under
``tope.*`` names) would otherwise leak into ``test_imports.py`` — which
exercises the *real* package — and cause spurious, order-dependent failures.

The purge therefore runs exactly once, right before ``test_imports`` executes,
clearing every injected ``tope*`` entry so the real package is re-imported
fresh.  It does so only for that module (not every non-gyro test), because the
other by-path test files rely on their collection-time stubs persisting
through their own (sometimes lazy) imports.
"""

import sys

import pytest

# Package names the by-path test files register as tagged stubs.
_STUB_NAMES = ("tope", "tope.data", "tope.topology", "tope.models")


@pytest.fixture(autouse=True)
def _purge_gyro_test_stubs(request):
    basename = request.module.__name__.rsplit(".", 1)[-1]
    if basename == "test_imports":
        # One-shot: only while a tagged stub is still present.  After the first
        # purge the stubs are gone, so later test_imports tests skip this and
        # keep the partial-import caching that lets some of them pass.
        stub_present = any(
            getattr(sys.modules.get(n), "_gyro_test_stub", False)
            for n in _STUB_NAMES
        )
        if stub_present:
            for name in [
                n for n in list(sys.modules) if n == "tope" or n.startswith("tope.")
            ]:
                del sys.modules[name]
    yield
