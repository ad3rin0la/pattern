"""Test-isolation for the gyro test modules.

``test_gyro_*`` load ``tope.*`` submodules by file path while registering
lightweight package *stubs* in ``sys.modules`` (the top-level ``tope`` package
cannot be imported normally because of a pre-existing, unrelated breakage in
its ``__init__``).  Those stubs — and the by-path submodules loaded under
``tope.*`` names — would otherwise leak into ``test_imports.py`` (which
exercises the real package) and cause spurious, order-dependent failures.

Before the *first* non-gyro test runs, this autouse fixture purges the
injected ``tope*`` entries exactly once (detected via the tag on the stub
packages), so the real package is re-imported fresh.  It deliberately does
*not* iterate ``sys.modules.values()`` (that can trip PEP-562 module
``__getattr__`` hooks) and never purges again afterwards, preserving the
partial-import caching that ``test_imports`` relies on.
"""

import sys

import pytest

# Package names the gyro tests register as stubs.
_STUB_NAMES = ("tope", "tope.topology", "tope.models")


@pytest.fixture(autouse=True)
def _purge_gyro_test_stubs(request):
    basename = request.module.__name__.rsplit(".", 1)[-1]
    if not basename.startswith("test_gyro"):
        # Only act while a tagged stub is actually present (one-shot cleanup).
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
