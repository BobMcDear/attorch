"""
Fixture for subset, a switch for running tests on a small subset of shapes.
"""


import pytest
from _pytest.config.argparsing import Parser
from pytest import FixtureRequest


def pytest_addoption(parser: Parser) -> None:
    parser.addoption('--subset',
                     action='store_true',
                     help='Flag to test on a small subset of shapes.')


@pytest.fixture
def subset(request: FixtureRequest) -> bool:
    return request.config.getoption('--subset')
