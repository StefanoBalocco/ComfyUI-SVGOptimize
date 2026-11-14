import unittest
from decimal import Decimal

import package_alias  # noqa: F401
from svgoptimize.svg_transform import as_translate, format_transform, parse_transform

D = Decimal


def fmt(text):
    return format_transform(parse_transform(text))


class TransformTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_transform("translate(10,20) scale(2)"),
                         [("translate", (D(10), D(20))), ("scale", (D(2),))])
        for bad in ("translate()", "foo(1)", "matrix(1 2 3)", "translate(1"):
            self.assertIsNone(parse_transform(bad), bad)
        self.assertEqual(parse_transform(""), [])

    def test_grammar(self):
        expected = [("translate", (D(1),)), ("scale", (D(2),))]
        for text in ("translate(1),scale(2)", "translate(1) , scale(2)", " translate( 1 )scale(2) "):
            self.assertEqual(parse_transform(text), expected, text)
        self.assertEqual(parse_transform("translate(1-2)"), [("translate", (D(1), D(-2)))])
        self.assertEqual(parse_transform("translate(1.5.5)"), [("translate", (D("1.5"), D("0.5")))])
        for bad in ("translate(1,,2)", "translate(1 2,)", "translate(,1)", "translate(1)scale(2),",
                    ",translate(1)", "translate(1),,scale(2)", "Translate(1)", "translate(1) x",
                    "translate(1 2 3)", "rotate(1 2)", "skewX(1 2)", "scale(1 2 3)"):
            self.assertIsNone(parse_transform(bad), bad)

    def test_shorter_forms(self):
        cases = {"translate(10 0)": "translate(10)", "scale(2,2)": "scale(2)",
                 "rotate(30,0,0)": "rotate(30)", "matrix(1,0,0,1,5,6)": "translate(5 6)",
                 "matrix(2 0 0 3 0 0)": "scale(2 3)", "matrix(2 0 0 2 0 0)": "scale(2)",
                 "matrix(1 0 0 1 0 0)": "", "translate(0) scale(1) rotate(0 5 5) skewX(0)": "",
                 "translate(10.50, -3.0)": "translate(10.5-3)", "translate(1)  scale(2)": "translate(1) scale(2)"}
        for text, expected in cases.items():
            self.assertEqual(fmt(text), expected, text)

    def test_identities(self):
        for text in ("translate(0 0)", "scale(1 1)", "rotate(0 3 4)", "skewY(0)", "matrix(1 0 0 1 0 0)"):
            self.assertEqual(fmt(text), "", text)
        self.assertEqual(fmt("translate(0 5) skewX(1) skewY(2)"), "translate(0 5) skewX(1) skewY(2)")

    def test_as_translate(self):
        self.assertEqual(as_translate(parse_transform("translate(5)")), (D(5), D(0)))
        self.assertEqual(as_translate(parse_transform("matrix(1 0 0 1 5 6)")), (D(5), D(6)))
        self.assertEqual(as_translate(parse_transform("translate(5 6)")), (D(5), D(6)))
        self.assertIsNone(as_translate(parse_transform("translate(1) scale(2)")))
        self.assertIsNone(as_translate(parse_transform("scale(2)")))
        self.assertIsNone(as_translate(parse_transform("matrix(1 0 0 2 5 6)")))
        self.assertIsNone(as_translate([]))


if __name__ == "__main__":
    unittest.main()
