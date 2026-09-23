"""Pytest fixtures for the beets plugin tests in this snapshot.

The plugin test classes use beets' ``io`` fixture (a ``DummyIO`` that
captures stdout and simulates stdin). In the upstream repository it is
provided by the top-level test conftest; it is defined here so the
snapshot's tests are self-contained. Like the upstream fixture, it binds
the helper onto the test instance because beets' ``IOMixin`` reads
``self.io``.
"""

import pytest


@pytest.fixture
def io(request, monkeypatch, capteesys):
    from beets.test._common import DummyIO

    instance = DummyIO(monkeypatch, capteesys)
    # IOMixin.run_with_output accesses ``self.io``.
    if request.instance is not None:
        request.instance.io = instance
    return instance
