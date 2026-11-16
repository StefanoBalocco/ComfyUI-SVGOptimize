import random
import re
import unittest
from decimal import Decimal
from pathlib import Path

import cairosvg

import package_alias  # noqa: F401
from helpers import S, opt
from svgoptimize.svg_path import format_path, parse_path, rewrite_path, translate_path

D = Decimal
FIXTURES = Path(__file__).parent / "fixtures"
ARGUMENTS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def seg(c, *v):
    return (c, tuple(D(str(x)) for x in v))


def random_number(rng):
    text = rng.choice(["", "-", "+"]) + str(rng.randint(0, 60))
    if rng.random() < 0.5:
        text += "." + "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 3)))
    return text


def random_path(rng):
    letters = []
    for index in range(rng.randint(1, 10)):
        letters.append(rng.choice("Mm") if index == 0 else rng.choice("MmLlHhVvCcSsQqTtAaZz"))
    parts = []
    for letter in letters:
        groups = []
        count = ARGUMENTS[letter.upper()]
        for _ in range(rng.randint(1, 3) if count else 0):
            numbers = [random_number(rng) for _ in range(count)]
            if letter in "Aa":
                numbers[3] = rng.choice("01")
                numbers[4] = rng.choice("01")
                if rng.random() < 0.5:
                    numbers[3:5] = ["".join(numbers[3:5])]
            groups.append(rng.choice([" ", ",", " , "]).join(numbers))
        parts.append(letter + rng.choice(["", " "]) + rng.choice([" ", ",", " , "]).join(groups))
    return rng.choice(["", " "]).join(parts)


class ParseTests(unittest.TestCase):
    def test_syntax_variants(self):
        self.assertEqual(parse_path("M10,10L20-5"), [seg("M", 10, 10), seg("L", 20, -5)])
        self.assertEqual(parse_path("m10 10 5 5"), [seg("M", 10, 10), seg("L", 15, 15)])
        self.assertEqual(parse_path("M+1 3.L.5.5"), [seg("M", 1, 3), seg("L", .5, .5)])
        self.assertEqual(parse_path("M0 0a5 5 0 1010 10"), [seg("M", 0, 0), seg("A", 5, 5, 0, 1, 0, 10, 10)])
        self.assertEqual(parse_path("M1e1 0"), [seg("M", 10, 0)])
        self.assertEqual(parse_path(""), [])

    def test_expansion(self):
        self.assertEqual(parse_path("M0 0H10V5h-5v-5zm1 1l1 0"),
            [seg("M", 0, 0), seg("L", 10, 0), seg("L", 10, 5), seg("L", 5, 5), seg("L", 5, 0), seg("Z"),
             seg("M", 1, 1), seg("L", 2, 1)])
        self.assertEqual(parse_path("M0 0C1 1 2 2 3 3S5 5 6 6")[2], seg("C", 4, 4, 5, 5, 6, 6))
        self.assertEqual(parse_path("M0 0Q1 1 2 0T4 0")[2], seg("Q", 3, -1, 4, 0))
        self.assertEqual(parse_path("M0 0T4 0")[1], seg("Q", 0, 0, 4, 0))

    def test_errors(self):
        for bad in ("M0 0 L", "L0 0", "M0 0 X1 1", "M0 0a5 5 0 2 0 1 1", "M0 0,L1 1", "M0 0 1", "M0 0\u017f1 1 2 2", "M\u0663 0"):
            self.assertIsNone(parse_path(bad), bad)

    def test_inexact_relative_move(self):
        self.assertIsNone(parse_path("M1e999 0m1e-999 0"))


class FormatTests(unittest.TestCase):
    def check(self, d, expected):
        self.assertEqual(format_path(parse_path(d)), expected)

    def test_shortest(self):
        self.check("M10 10L20 10L20 20L10 20Z", "M10 10h10v10H10z")
        self.check("M0 0L1 1L2 3", "M0 0l1 1 1 2")
        self.check("M0 0C0 1 2 2 3 3C4 4 5 6 6 6", "M0 0c0 1 2 2 3 3s2 3 3 3")
        self.check("M10 10A5 5 0 0 1 20 10", "M10 10a5 5 0 0 1 10 0")
        self.check("M0.5,0.5 L-0.5,-0.5", "M.5.5l-1-1")
        self.check("M1 1L1 1", "M1 1h0")
        self.check("M0 0h10v10zM0 0h5", "M0 0h10v10zm0 0h5")

    def test_quadratic_shorthand(self):
        self.check("M0 0Q1 1 2 0Q3 -1 4 0", "M0 0q1 1 2 0t2 0")

    def test_straight_curves_become_lines(self):
        self.check("M0 0C1 1 2 2 3 3", "M0 0l3 3")
        self.check("M0 0C0 0 3 3 3 3", "M0 0l3 3")
        self.check("M5 5C4 5 3 5 0 5", "M5 5H0")
        self.check("M0 0Q1 1 2 2", "M0 0l2 2")
        self.check("M0 0Q0 0 2 0", "M0 0h2")
        self.check("M0 0C.5 .5 2 2 3 3z", "M0 0l3 3z")

    def test_curves_that_are_not_monotonic_along_the_chord_stay(self):
        for d in ("M0 0C2 2 1 1 3 3", "M0 0C-1 -1 2 2 3 3", "M0 0C1 1 5 5 3 3", "M0 0Q4 4 2 2", "M0 0Q-1 -1 2 2", "M0 0C1 0 2 2 3 3", "M0 0Q1 0 2 2"):
            self.assertEqual(parse_path(format_path(parse_path(d))), parse_path(d), d)

    def test_zero_length_chords_stay(self):
        for d in ("M1 1C1 1 1 1 1 1", "M1 1C2 2 3 3 1 1", "M1 1Q2 2 1 1"):
            self.assertEqual(parse_path(format_path(parse_path(d))), parse_path(d), d)

    def test_shorthand_after_a_straight_curve_keeps_its_points(self):
        for d, points in (("M0 0C1 1 2 2 3 3S4 8 9 9", ("C", 4, 4, 4, 8, 9, 9)), ("M0 0Q1 1 2 2T4 0", ("Q", 3, 3, 4, 0))):
            segments = parse_path(format_path(parse_path(d)))
            self.assertEqual(segments[1], seg("L", *segments[1][1]))
            self.assertEqual(segments[2], seg(*points), d)

    def test_empty(self):
        self.assertEqual(format_path([]), "")

    def test_moveto_and_closepath_keep_their_letters(self):
        for d in ("M0 0M5 5L6 6", "M0 0zz", "M0 0z M5 5 z L1 1"):
            segments = parse_path(d)
            self.assertEqual(parse_path(format_path(segments)), segments, d)
        self.assertEqual(format_path(parse_path("M0 0zz")), "M0 0zz")

    def test_inexact(self):
        self.assertIsNone(format_path(parse_path("M1e-999 0L1e999 0")))

    def test_translate(self):
        self.assertEqual(translate_path(parse_path("M0 0L10 0A5 5 0 0 1 20 0z"), D(1), D(2)),
            [seg("M", 1, 2), seg("L", 11, 2), seg("A", 5, 5, 0, 0, 1, 21, 2), seg("Z")])

    def test_translate_inexact(self):
        self.assertIsNone(translate_path(parse_path("M1e999 0"), D("1e-999"), D(0)))

    def test_random_paths_cover_the_cases(self):
        paths = [random_path(random.Random(seed)) for seed in range(500)]
        self.assertTrue(any(re.search(r"[Mm][^A-Za-z]*[Mm]", d) for d in paths))
        self.assertTrue(any(re.search(r"[Zz]\s*[Mm]", d) for d in paths))
        self.assertTrue(any(re.search(r"[Zz]\s*[LlHhVvCcSsQqTtAa]", d) for d in paths))
        for letter in "MmLlHhVvCcSsQqTtAaZz":
            self.assertTrue(any(letter in d for d in paths), letter)

    def test_round_trip_is_exact_and_idempotent(self):
        rng = random.Random(0)
        for _ in range(500):
            d = random_path(rng)
            segments = parse_path(d)
            text = format_path(segments)
            parsed = parse_path(text)
            self.assertEqual(len(parsed), len(segments), d)
            for before, after in zip(segments, parsed):
                self.assertTrue(before == after or (before[0] in "CQ" and after == ("L", before[1][-2:])), d)
            self.assertEqual(format_path(parsed), text, d)

    def test_rewrite_path_remembers(self):
        written = {}
        self.assertEqual(rewrite_path("M 10 10 L 20 10", written), "M10 10h10")
        self.assertEqual(written, {"M 10 10 L 20 10": "M10 10h10", "M10 10h10": "M10 10h10"})
        self.assertIsNone(rewrite_path("M0 0 X", written))
        self.assertEqual(rewrite_path("X", {"X": "Y"}), "Y")

    def test_writing_is_idempotent(self):
        for name in ("vtracer", "starvector", "inkscape", "styled"):
            for d in re.findall(r'\bd="([^"]*)"', (FIXTURES / f"{name}.svg").read_text()):
                segments = parse_path(d)
                if segments is not None:
                    text = format_path(segments)
                    self.assertEqual(format_path(parse_path(text)), text, d)


class StraightCurveRenderTests(unittest.TestCase):
    def test_shorthand_after_a_converted_curve_draws_the_same(self):
        for d in ("M10 10C20 20 30 30 40 40S60 80 90 20", "M10 10Q20 20 30 30T50 10T70 40", "M10 10C20 20 30 30 40 40S60 40 70 70S80 80 90 90"):
            doc = f'<svg {S} width="100" height="100"><path d="{d}" fill="none" stroke="#000" stroke-width="3"/></svg>'
            out = opt(doc)
            render = lambda text: cairosvg.svg2png(bytestring=text.encode(), background_color="white")
            self.assertEqual(render(out), render(doc), d)
            self.assertEqual(parse_path(re.search(r'\bd="([^"]*)"', out).group(1))[1][0], "L")


if __name__ == "__main__":
    unittest.main()
