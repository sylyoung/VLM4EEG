"""Keep test runtime files under the ignored project outputs directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(ROOT / "outputs/test_runtime")
    Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "outputs/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
