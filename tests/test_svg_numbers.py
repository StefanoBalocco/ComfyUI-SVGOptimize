import unittest
from decimal import Decimal

import package_alias  # noqa: F401
from svgoptimize.svg_numbers import format_number, format_numbers, join_numbers, parse_number, separator


class FormatNumberTests(unittest.TestCase):
    def test_shortest_text(self):
        cases = {"0.500": ".5", "-0.5": "-.5", "10.0": "10", "-0": "0", "0": "0",
                 "1000": "1e3", "100": "100", "12000": "12e3", "1.5e3": "1500",
                 "0.0001": "1e-4", "0.001": ".001", "0.00015": "15e-5", "123.456": "123.456"}
        for text, expected in cases.items():
            self.assertEqual(format_number(Decimal(text)), expected, text)

    def test_beyond_context_limits(self):
        cases = {"1" * 1001: "1" * 1001, "1e9999999": "1e9999999", "-1.000e-9999999": "-1e-9999999",
                 "1e999999999": "1e999999999", "-1e-999999999": "-1e-999999999"}
        for text, expected in cases.items():
            result = format_number(Decimal(text))
            self.assertEqual(result, expected)
            self.assertEqual(Decimal(result), Decimal(text))


class SeparatorTests(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(separator("1", "-2"), "")
        self.assertEqual(separator(".5", ".5"), "")
        self.assertEqual(separator("1e3", ".5"), "")
        self.assertEqual(separator("1", ".5"), " ")
        self.assertEqual(separator("-5", ".5"), " ")
        self.assertEqual(separator("1", "2"), " ")
        self.assertEqual(separator("", "2"), "")

    def test_join(self):
        self.assertEqual(join_numbers(["10", "-5", ".5", ".5", "3"]), "10-5 .5.5 3")
        self.assertEqual(format_numbers([Decimal("0.5"), Decimal("-1"), Decimal("2")]), ".5-1 2")


class ParseNumberTests(unittest.TestCase):
    def test_grammar(self):
        self.assertEqual(parse_number("+3"), 3)
        self.assertEqual(parse_number("3."), 3)
        self.assertEqual(parse_number("-.5e-2"), Decimal("-0.005"))
        for bad in ("1e", "abc", "", "1 2"):
            self.assertIsNone(parse_number(bad), bad)
        self.assertIsNone(parse_number("٣"))


if __name__ == "__main__":
    unittest.main()
