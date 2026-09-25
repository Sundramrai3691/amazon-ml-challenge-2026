import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-ab",
        action="store_true",
        default=False,
        help="Run the slow real-data A/B retrieval comparison test (S1=500, S2=20k, S3=20k, K=10)",
    )
