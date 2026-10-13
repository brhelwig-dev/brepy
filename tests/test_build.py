from pathlib import Path

import pytest
from build123d import Box

from brepy.build import build_files, load_model, output_files, parse_parameters

FIXTURES = Path(__file__).parent / "fixtures"


def stand():
    return load_model(FIXTURES / "stand" / "stand.py", FIXTURES)


def test_a_lone_part_is_named_after_the_script():
    assert list(output_files({"body": Box(1, 1, 1)}, "cube", False)) == ["cube"]


def test_parts_and_shells_get_their_own_files_and_a_combined_one():
    files = build_files(stand(), [], None)
    assert sorted(files) == [
        "stand",
        "stand_base",
        "stand_tag",
        "stand_tag_red",
        "stand_tag_white",
    ]
    assert sorted(files["stand"]) == ["base", "tag_red", "tag_white"]


def test_separate_prints_drop_the_combined_file():
    files = output_files({"a": Box(1, 1, 1), "b": Box(2, 2, 2)}, "pair", True)
    assert sorted(files) == ["pair_a", "pair_b"]


def test_a_part_narrows_to_its_files():
    assert sorted(build_files(stand(), [], "tag")) == [
        "stand_tag",
        "stand_tag_red",
        "stand_tag_white",
    ]


def test_an_unknown_part_is_refused():
    with pytest.raises(SystemExit, match="no part 'lid'"):
        build_files(stand(), [], "lid")


def test_parameters_take_the_annotated_types():
    model = stand()
    assert parse_parameters(model.build, ["width=30", "holed=no"]) == {
        "width": 30.0,
        "holed": False,
    }


def test_an_unknown_parameter_is_refused():
    with pytest.raises(SystemExit, match="not NAME=VALUE"):
        parse_parameters(stand().build, ["depth=3"])


def test_a_script_outside_the_root_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="is not under the root"):
        load_model(FIXTURES / "stand" / "stand.py", tmp_path)
