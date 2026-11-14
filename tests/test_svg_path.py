import random
import re
import unittest
from decimal import Decimal

import package_alias  # noqa: F401
from svgoptimize.svg_path import format_path, parse_path, translate_path

D = Decimal
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
        self.check("M0 0C1 1 2 2 3 3C4 4 5 5 6 6", "M0 0c1 1 2 2 3 3s2 2 3 3")
        self.check("M10 10A5 5 0 0 1 20 10", "M10 10a5 5 0 0 1 10 0")
        self.check("M0.5,0.5 L-0.5,-0.5", "M.5.5l-1-1")
        self.check("M1 1L1 1", "M1 1h0")
        self.check("M0 0h10v10zM0 0h5", "M0 0h10v10zm0 0h5")

    def test_quadratic_shorthand(self):
        self.check("M0 0Q1 1 2 0Q3 -1 4 0", "M0 0q1 1 2 0t2 0")

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
            self.assertEqual(parse_path(text), segments, d)
            self.assertEqual(format_path(parse_path(text)), text, d)


if __name__ == "__main__":
    unittest.main()
