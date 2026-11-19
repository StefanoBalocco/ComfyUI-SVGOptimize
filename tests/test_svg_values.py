import unittest
from decimal import Decimal

import package_alias  # noqa: F401
from helpers import S, opt
from svgoptimize.svg_values import (COLOR_KEYWORDS, INHERITED_PROPERTIES, PRESENTATION_ATTRIBUTES, as_translate, format_number,
                                    format_numbers, format_style, format_transform, is_default, join_numbers, parse_number,
                                    parse_style, parse_transform, separator, shortest_color, shortest_length)


class FormatNumberTests(unittest.TestCase):
    def test_shortest_text(self):
        cases = {"0.500": ".5", "-0.5": "-.5", "10.0": "10", "-0": "0", "0": "0",
                 "1000": "1e3", "100": "100", "12000": "12e3", "1.5e3": "1500",
                 "0.0001": "1e-4", "0.001": ".001", "0.00015": "15e-5", "123.456": "123.456",
                 "-1000": "-1e3", "-0.0001": "-1e-4", "0.00012": "12e-5", "0.000123456": "123456e-9",
                 "0.0001234567": ".0001234567", "1234000": "1234e3", "1E+3": "1e3", "5E-1": ".5",
                 "0E+5": "0", "1.000": "1"}
        for text, expected in cases.items():
            self.assertEqual(format_number(Decimal(text)), expected, text)

    def test_beyond_context_limits(self):
        cases = {"1" * 1001: "1" * 1001, "1e9999999": "1e9999999", "-1.000e-9999999": "-1e-9999999",
                 "1e999999999": "1e999999999", "-1e-999999999": "-1e-999999999"}
        for text, expected in cases.items():
            result = format_number(Decimal(text))
            self.assertEqual(result, expected)
            self.assertEqual(Decimal(result), Decimal(text))
        self.assertEqual(format_number(Decimal("1" * 100005)), "1" * 100005)


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


class ColorTests(unittest.TestCase):
    def test_shortest(self):
        cases = {"#FF0000": "red", "#ffffff": "#fff", "rgb(255,255,255)": "#fff", "rgb( 0 ,0, 128 )": "navy",
                 "rgb(100%, 0%, 0%)": "red", "white": "#fff", "navy": "navy", "#F0F0F0": "#f0f0f0",
                 "GREY": "gray", " #F00 ": "red", "RGB(0,0,0)": "#000", "#AbC": "#abc"}
        for text, expected in cases.items():
            self.assertEqual(shortest_color(text), expected, text)

    def test_untouched(self):
        for text in ("rgb(50%,0%,0%)", "rgb(50%,0,0)", "rgb(256,0,0)", "rgb(10%,0,0%)", "currentColor",
                     "url(#a)", "#abcd", "none", "inherit", "rgb(1.0,0,0)", "rgb(101%,0%,0%)", "rgb(1 2 3/2)", "rgb(1 2 3 4)", "rgb(1,2 3)", "rgb(1 2 3/)", "rgba(0,0,0,.5", " rgb(1,2) "):
            self.assertEqual(shortest_color(text), text, text)
        self.assertEqual(shortest_color("rgb(\u0663,0,0)"), "rgb(\u0663,0,0)")
        self.assertEqual(shortest_color("rgb(\u00a0255,0,0)"), "rgb(\u00a0255,0,0)")

    def test_alpha_forms(self):
        cases = {"rgba(255,0,0,.2)": "#f003", "rgb(255 0 0 / 20%)": "#f003", "rgb(0 0 0/.4)": "#0006", "#FF000080": "#ff000080",
                 "#F008": "#f008", "#ff0000ff": "red", "#ff0000": "red", "transparent": "#0000", "rgba(10,20,30,0)": "#0a141e00",
                 "rgba(1,2,3,1)": "#010203", "rgb(100% 0% 0%/10%)": "rgb(255 0 0/.1)"}
        for text, expected in cases.items():
            self.assertEqual(shortest_color(text), expected, text)

    def test_alpha_stays_exact(self):
        cases = {"rgba(255,0,0,.5)": "rgb(255 0 0/.5)", "rgb(255 0 0/50%)": "rgb(255 0 0/.5)", "rgba(204,204,204,.5)": "hsl(0 0% 80%/.5)",
                 "rgb(255 0 0/.3)": "rgb(255 0 0/.3)", "rgb(255,0,0,0.125)": "rgb(255 0 0/.125)"}
        for text, expected in cases.items():
            self.assertEqual(shortest_color(text), expected, text)

    def test_alpha_rounds_to_a_255th(self):
        cases = {"rgba(255,0,0,.5)": "#ff000080", "rgb(255 0 0/.3)": "#ff00004c", "rgb(255 0 0/50%)": "#ff000080", "rgb(0 0 0/.9999)": "#000"}
        for text, expected in cases.items():
            self.assertEqual(shortest_color(text, True), expected, text)

    def test_inexact_percentage_stays(self):
        for value in ("rgb(1e-9999999%,0%,0%)", f"rgb({'1' * 1100}%,0%,0%)"):
            self.assertIn(f'fill="{value}"', opt(f'<svg {S}><path fill="{value}" d="M0 0h1"/></svg>'))
            self.assertIn(f'fill:{value}', opt(f'<svg {S}><path style="fill:{value};-x-y:1" d="M0 0h1"/></svg>'))

    def test_keyword_table(self):
        self.assertEqual(len(COLOR_KEYWORDS), 147)
        self.assertEqual(COLOR_KEYWORDS["navy"], "#000080")
        self.assertEqual(COLOR_KEYWORDS["grey"], COLOR_KEYWORDS["gray"])


class LengthStyleTests(unittest.TestCase):
    def test_lengths(self):
        cases = {"10px": "10", "10.0%": "10%", "1.50em": "1.5em", "0.5": ".5", "auto": "auto", "10 px": "10 px",
                 "1000em": "1e3em", "0.5ex": ".5ex", " 10px ": "10", "10PX": "10PX", "10px5": "10px5"}
        for text, expected in cases.items():
            self.assertEqual(shortest_length(text), expected, text)

    def test_style(self):
        self.assertEqual(parse_style("fill: red; stroke:#000 ;"), [("fill", "red"), ("stroke", "#000")])
        self.assertEqual(parse_style("fill:url(data:a;b)"), [("fill", "url(data:a;b)")])
        self.assertEqual(parse_style("FILL:red;;"), [("fill", "red")])
        self.assertEqual(parse_style(""), [])
        self.assertEqual(parse_style("display:\xa0none"), [("display", "\xa0none")])
        self.assertEqual(parse_style("\fdisplay:\fnone\f"), [("display", "none")])
        for bad in ("fill:red !important", "/*x*/fill:red", "fill", "fill:red ! IMPORTANT", "fill:", ":red",
                    "fill:red;stroke: ", "font-family:\\41", "fill:url(a", "fill:a)", "font-family:'a;b"):
            self.assertIsNone(parse_style(bad), bad)
        self.assertEqual(parse_style("font-family:'a;b'"), [("font-family", "'a;b'")])
        self.assertEqual(format_style([("fill", "red"), ("stroke", "#000")]), "fill:red;stroke:#000")

    def test_defaults(self):
        self.assertTrue(is_default("fill", "black", "#000"))
        self.assertTrue(is_default("fill", "BLACK", "#000"))
        self.assertTrue(is_default("stroke-width", "1.0", "1"))
        self.assertFalse(is_default("stroke-width", "1px5", "1"))
        self.assertFalse(is_default("stroke-linecap", "Butt", "butt"))
        self.assertNotIn("opacity", INHERITED_PROPERTIES)
        self.assertNotIn("font", PRESENTATION_ATTRIBUTES)


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
