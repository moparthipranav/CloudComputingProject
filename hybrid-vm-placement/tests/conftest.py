from pathlib import Path

import pytest

from src.config import deep_update, load_config
from src.problem import build_problem

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config(ROOT / "config" / "config.yaml")


@pytest.fixture(scope="session")
def small_problem(config):
    return build_problem(60, config, seed=7)


@pytest.fixture()
def tiny_config(config) -> dict:
    return deep_update(config, {"simulated_annealing": {"max_iterations": 6000,
                                                        "iterations_per_temperature": 300}})
