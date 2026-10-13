import pytest
from build123d import Box, BuildPart, Locations, Part, Pos

from brepy import printability

UP = (0, 0, 1)


def boxes(*placed: tuple[tuple[float, float, float], tuple[float, float, float]]) -> Part:
    with BuildPart() as part:
        for at, size in placed:
            with Locations(at):
                Box(*size)
    assert part.part is not None
    return part.part


def tee() -> Part:
    return boxes(((0, 0, 5), (4, 4, 10)), ((0, 0, 11), (10, 4, 2)))


def test_a_box_on_the_plate_is_fully_supported():
    assert printability.unsupported_area(boxes(((0, 0, 5), (10, 10, 10))), UP) == pytest.approx(0)


def test_a_ledge_hangs_over_the_plate():
    assert printability.unsupported_area(tee(), UP) == pytest.approx(2 * 3 * 4, rel=0.01)


def test_flipping_the_build_direction_supports_the_ledge():
    assert printability.unsupported_area(tee(), (0, 0, -1)) == pytest.approx(0)


def test_section_is_the_plan_at_a_height():
    assert printability.section(Pos(0, 0, 5) * Box(10, 6, 10), 3).area == pytest.approx(60)


def test_island_area_finds_material_printed_on_nothing():
    floating = boxes(((-8, 0, 5), (2, 4, 10)), ((8, 0, 5), (2, 4, 10)), ((0, 0, 3), (4, 4, 2)))
    assert printability.island_area(floating, 0.2, UP) == pytest.approx(16, rel=0.01)
