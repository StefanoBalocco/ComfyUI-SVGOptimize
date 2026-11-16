"""Inlining of plain stylesheets: the drawing is the same, and what is not plain stays."""
from pathlib import Path
import re
import unittest

import cairosvg

import package_alias  # noqa: F401
from helpers import S, X, opt
from svgoptimize.svg_document import parse_svg, serialize
from svgoptimize.svg_optimize import rewrite
from svgoptimize.svg_stylesheet import inline_stylesheet

FIXTURES = Path(__file__).parent / "fixtures"

CASCADE = f"""<svg {S} width="120" height="60" viewBox="0 0 120 60">
<style>
rect {{ fill: #00f; stroke: #000; stroke-width: 3 }}
.a {{ fill: #f00 }}
#x {{ fill: #0f0 }}
.b {{ stroke: #888; stroke-width: 1 }}
.c {{ stroke: #f80 }}
g {{ fill: #ff0; stroke-linejoin: round }}
g > path, .d {{ stroke: #606; stroke-width: 4 }}
circle.d {{ fill: #0aa }}
</style>
<rect id="x" class="a" x="2" y="2" width="20" height="20"/>
<rect class="b c" x="26" y="2" width="20" height="20"/>
<rect class="c b" x="50" y="2" width="20" height="20"/>
<rect class="a" style="fill:#0ff;stroke-width:6" x="74" y="2" width="20" height="20"/>
<rect class="a" fill="#123" stroke="#fed" x="98" y="2" width="20" height="20"/>
<g><path d="M2 30h20v20z" stroke-width="2"/><path class="b" d="M30 30h20v20z" style="stroke:#0c0"/></g>
<circle class="d" cx="70" cy="40" r="10" stroke-width="1" fill="#abc"/>
<path class="a" d="M90 30h20v20z" stroke="none"/>
</svg>"""


def render(data):
    return cairosvg.svg2png(bytestring=data, output_width=360, output_height=180, background_color="white")


def inlined(text):
    root, preceding = parse_svg(text.encode())
    root = inline_stylesheet(root, preceding)
    return None if root is None else serialize(root, preceding)


class InlineTests(unittest.TestCase):
    def test_fixture_draws_the_same(self):
        data = (FIXTURES / "styled.svg").read_bytes()
        out = inlined(data.decode())
        self.assertEqual(render(out), render(data))
        self.assertNotIn(b"<style", out)
        self.assertNotIn(b"class=", out)
        self.assertEqual(render(opt(data.decode()).encode()), render(data))

    def test_cascade_draws_the_same(self):
        out = inlined(CASCADE)
        self.assertEqual(render(out), render(CASCADE.encode()))
        self.assertNotEqual(render(re.sub(r"<style>.*</style>", "", CASCADE, flags=re.S).encode()), render(out))
        self.assertNotIn(b"<style", out)
        self.assertNotIn(b"class=", out)
        self.assertEqual(render(opt(CASCADE).encode()), render(CASCADE.encode()))

    def test_style_of_the_element_wins(self):
        out = inlined(f'<svg {S}><style>path{{fill:red;marker:none}}</style><path d="M0 0h1" style="fill:blue;marker-start:none"/></svg>')
        self.assertEqual(out, f'<svg {S}><path d="M0 0h1" style="marker:none;fill:blue;marker-start:none"/></svg>'.encode())

    def test_optimized_output_is_stylesheet_free(self):
        out = opt(f'<svg {S}><style>.a{{fill:#FF0000}}</style><path class="a" d="M0 0h1"/></svg>')
        self.assertEqual(out, f'<svg {S}><path d="M0 0h1" fill="red"/></svg>')

    def test_plain_stylesheet_is_inlined(self):
        for style in ("<style>rect{fill:red}</style>", '<style type="text/css">rect{fill:red}</style>', "<style>rect,circle{fill:red}</style><style>g{stroke:red}</style>"):
            with self.subTest(style):
                self.assertIsNotNone(inlined(f'<svg {S}>{style}<rect width="1" height="1"/></svg>'))

    def test_other_stylesheets_stay(self):
        body = '<rect class="a" width="1" height="1"/>'
        for style in ("<style>@media print{.a{fill:red}}</style>", '<style>@import "a.css";.a{fill:red}</style>', "<style>.a{fill:red !important}</style>",
                      "<style>.a:hover{fill:red}</style>", "<style>.a::before{fill:red}</style>", "<style>.a:first-child{fill:red}</style>",
                      "<style>:root{fill:red}</style>", "<style>.a{fill:var(--c)}</style>", "<style>.a{fill:}</style>", "<style>.a[{fill:red}</style>",
                      '<style media="screen">.a{fill:red}</style>', '<style type="text/less">.a{fill:red}</style>',
                      "<style>.a{fill:red}</style><script>1</script>", '<style>.a{fill:red}</style><rect width="1" height="1"><set attributeName="fill" to="red"/></rect>',
                      "<style>.a{fill:red}</style><foreignObject/>"):
            with self.subTest(style):
                self.assertIsNone(inlined(f'<svg {S}>{style}{body}</svg>'))
                self.assertIn("<style", opt(f'<svg {S}>{style}{body}</svg>'))
        self.assertIsNone(inlined(f'<?xml-stylesheet href="a.css"?><svg {S}><style>.a{{fill:red}}</style>{body}</svg>'))
        self.assertIsNone(inlined(f'<?xml-stylesheet href="a.css"?><svg {S}>{body}</svg>'))
        self.assertIsNone(inlined(f'<svg {S}>{body}</svg>'))

    def test_shared_class_never_makes_the_output_longer(self):
        rect = '<rect class="k" x="{}" y="1" width="2" height="2"/>'
        many = f'<svg {S}><style>.k{{fill:#336699;stroke:#cc3300;stroke-width:.5;stroke-linejoin:round}}</style>{"".join(rect.format(2 * i) for i in range(40))}</svg>'
        few = f'<svg {S}><style>.k{{fill:#336699}}</style>{rect.format(1)}</svg>'
        for text in (many, few):
            root, preceding = parse_svg(text.encode())
            plain = len(rewrite(root, preceding))
            root, preceding = parse_svg(text.encode())
            both = len(rewrite(inline_stylesheet(root, preceding), preceding))
            self.assertEqual(len(opt(text)), min(plain, both))
        self.assertIn("<style", opt(many))
        self.assertNotIn("<style", opt(few))


if __name__ == "__main__":
    unittest.main()
