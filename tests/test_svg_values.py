import unittest

import package_alias  # noqa: F401
from helpers import S, opt
from svgoptimize.svg_values import (COLOR_KEYWORDS, INHERITED_PROPERTIES, PRESENTATION_ATTRIBUTES, format_style,
                                    is_default, parse_style, shortest_color, shortest_length)


class ColorTests(unittest.TestCase):
    def test_shortest(self):
        cases = {"#FF0000": "red", "#ffffff": "#fff", "rgb(255,255,255)": "#fff", "rgb( 0 ,0, 128 )": "navy",
                 "rgb(100%, 0%, 0%)": "red", "white": "#fff", "navy": "navy", "#F0F0F0": "#f0f0f0",
                 "GREY": "gray", " #F00 ": "red", "RGB(0,0,0)": "#000", "#AbC": "#abc"}
        for text, expected in cases.items():
            self.assertEqual(shortest_color(text), expected, text)

    def test_untouched(self):
        for text in ("rgb(50%,0%,0%)", "rgb(50%,0,0)", "rgb(256,0,0)", "rgb(10%,0,0%)", "currentColor", "rgba(0,0,0,.5)",
                     "url(#a)", "#abcd", "none", "inherit", "rgb(1.0,0,0)", "rgb(101%,0%,0%)", " rgb(1,2) "):
            self.assertEqual(shortest_color(text), text, text)
        self.assertEqual(shortest_color("rgb(\u0663,0,0)"), "rgb(\u0663,0,0)")
        self.assertEqual(shortest_color("rgb(\u00a0255,0,0)"), "rgb(\u00a0255,0,0)")

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
