"""Fixtures of the two-card mesh tests: one 1x2 mesh per test file, closed at the end
of the file. native_module comes from tests/hardware/conftest.py."""

import pytest


@pytest.fixture(scope="module")
def mesh(native_module):
    dev = native_module.mesh([0, 1], shape=(1, 2))
    try:
        yield dev
    finally:
        native_module.close()
