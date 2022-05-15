import runpy

import pytest


@pytest.fixture
def module_name(pytestconfig):
    modules = [entry for entry in (pytestconfig.rootpath / "src").iterdir() if entry.is_dir()]
    assert len(modules) == 1
    return modules[0].name


def test_import_works(module_name):
    assert __import__(module_name).__version__
