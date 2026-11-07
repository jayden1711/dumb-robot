import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dumbrobot import schema  # noqa: E402
from dumbrobot.assembly import load_script  # noqa: E402
from dumbrobot.body_template import build  # noqa: E402


@pytest.fixture
def template_build():
    return build


@pytest.fixture
def genome():
    return schema.default_genome()


@pytest.fixture
def big_design():
    return load_script(ROOT / "examples" / "hand_made_big_wheels.py")


@pytest.fixture
def small_design():
    return load_script(ROOT / "examples" / "hand_made_small_wheels.py")
