import json
from pathlib import Path

from brepy.cli import main

FIXTURES = Path(__file__).parent / "fixtures"
STAND = str(FIXTURES / "stand" / "stand.py")


def export(tmp_path, *args):
    return main([*args, "--root", str(FIXTURES), "-o", str(tmp_path), STAND])


def test_step_is_the_default_format(tmp_path):
    assert export(tmp_path) == 0
    assert sorted(p.name for p in tmp_path.glob("*.step")) == [
        "stand.step",
        "stand_base.step",
        "stand_tag.step",
        "stand_tag_red.step",
        "stand_tag_white.step",
    ]
    assert json.loads((tmp_path / "preview.json").read_text())["parts"]["stand_base"]


def test_mesh_formats(tmp_path):
    export(tmp_path, "-f", "stl", "-f", "3mf", "--part", "base")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stand_base.3mf", "stand_base.stl"]


def test_png_draws_a_sheet_per_file_and_assembly(tmp_path):
    export(tmp_path, "-f", "png")
    sheets = sorted(p.name for p in (tmp_path / "previews").glob("*.png"))
    assert "stand_base.png" in sheets
    assert "with_peg.png" in sheets
