"""Whole-document fixtures.

vtracer.svg is VTracer output (vtracer.convert_raw_image_to_svg with
path_precision=3) for a synthetic 256x256 Pillow drawing of overlapping
ellipses and polygons. The other inputs are written by hand in the style of
StarVector, Inkscape and a stylesheet-based document. Each *.expected.svg is
the reviewed output.
"""
from pathlib import Path
import re
import unittest

import package_alias  # noqa: F401
from svgoptimize.svg_optimize import optimize_svg
from svgoptimize.svg_path import parse_path, translate_path
from svgoptimize.svg_transform import as_translate, parse_transform
from svgoptimize.svg_values import shortest_color

FIXTURES = Path(__file__).parent / "fixtures"
NAMES = ["vtracer", "starvector", "inkscape", "styled"]


def read(name):
    return (FIXTURES / name).read_bytes()


class FixtureTests(unittest.TestCase):
    def test_expected_output(self):
        for name in NAMES:
            with self.subTest(name):
                self.assertEqual(optimize_svg(read(f"{name}.svg"), False), read(f"{name}.expected.svg"))

    def test_second_pass_changes_nothing(self):
        for name in NAMES:
            with self.subTest(name):
                expected = read(f"{name}.expected.svg")
                self.assertEqual(optimize_svg(expected, False), expected)

    def test_vtracer_geometry_is_exact(self):
        source = read("vtracer.svg").decode()
        result = read("vtracer.expected.svg").decode()
        paths = re.findall(r'<path\b[^>]*>', source)
        self.assertTrue(paths)
        moved = []
        for tag in paths:
            d = re.search(r'\bd="([^"]*)"', tag).group(1)
            transform = re.search(r'\btransform="([^"]*)"', tag).group(1)
            offset = as_translate(parse_transform(transform))
            self.assertIsNotNone(offset)
            moved.append(translate_path(parse_path(d), *offset))
        out_tags = re.findall(r'<path\b[^>]*>', result)
        self.assertTrue(all("transform" not in tag for tag in out_tags))
        written = [parse_path(re.search(r'\bd="([^"]*)"', tag).group(1)) for tag in out_tags]
        self.assertEqual(len(moved), len(written))
        for before, after in zip(moved, written):
            self.assertEqual(len(before), len(after))
            for old, new in zip(before, after):
                # a cubic whose control points lie on its chord is written as a line
                self.assertTrue(old == new or (old[0] == "C" and new == ("L", old[1][-2:])), old)
        fills = [re.search(r'\bfill="([^"]*)"', tag).group(1) for tag in paths]
        out_fills = [re.search(r'\bfill="([^"]*)"', tag).group(1) for tag in out_tags]
        self.assertEqual([shortest_color(fill) for fill in fills], [shortest_color(fill) for fill in out_fills])


if __name__ == "__main__":
    unittest.main()
