"""PPB Vision test suite.

This __init__.py is required: a third-party dependency installs a top-level
``tests`` package into site-packages, and without this file our
``from tests.conftest import ...`` imports would resolve to that package
instead of this directory.
"""
