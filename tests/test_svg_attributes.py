import unittest

import package_alias  # noqa: F401
from helpers import S, X, opt, run
from svgoptimize.svg_attributes import clean_values, format_geometry, remove_defaults, styles_to_attributes


class AttributeTests(unittest.TestCase):
    def test_style_to_attributes(self):
        self.assertEqual(run(styles_to_attributes, f'<svg {S}><path d="M0 0h1" style="fill:#FF0000;stroke:blue" fill="green"/></svg>'),
                         f'<svg {S}><path d="M0 0h1" fill="#FF0000" stroke="blue"/></svg>')
        for style in ("fill:red;stroke:blue;stroke-width:2;opacity:.5;stroke-linecap:round", "fill:red;mix-blend-mode:multiply", "fill:red !important"):
            doc = f'<svg {S}><path d="M0 0h1" style="{style}"/></svg>'
            self.assertEqual(run(styles_to_attributes, doc), doc, style)

    def test_css_only_values_stay_in_style(self):
        for doc in (f'<svg {S}><g style="--c:red"><path style="fill:var(--c)" d="M0 0h1"/></g></svg>',
                    f'<svg {S}><path style="fill:VAR(--c)" d="M0 0h1"/></svg>',
                    *(f'<svg {S}><path style="fill:{word}" d="M0 0h1"/></svg>' for word in ("unset", "initial", "revert", "revert-layer"))):
            self.assertEqual(run(styles_to_attributes, doc), doc)
        self.assertEqual(run(styles_to_attributes, f'<svg {S}><path style="fill:inherit" d="M0 0h1"/></svg>'),
                         f'<svg {S}><path d="M0 0h1" fill="inherit"/></svg>')

    def test_values(self):
        self.assertEqual(run(clean_values, f'<svg {S}><circle cx="5.0" r="10px" fill="#FFFFFF" class=" a  b " id="" style="stroke:#FF0000"/></svg>'),
                         f'<svg {S}><circle cx="5" r="10" fill="#fff" class="a b" style="stroke:red"/></svg>')
        doc = f'<svg {S}><path d="M0 0h1" systemLanguage=""/></svg>'
        self.assertEqual(run(clean_values, doc), doc)
        self.assertEqual(run(clean_values, f'<svg {S}><path d="M0 0h1" class="  "/></svg>'), f'<svg {S}><path d="M0 0h1"/></svg>')

    def test_geometry(self):
        self.assertEqual(run(format_geometry, f'<svg {S} viewBox=" 0,0 , 10.0 10 "><path d="M 10 10 L 20 10" transform="translate(0,0)"/>'
                                              '<polygon points="0,0 10,0 10,10" stroke-dasharray="5, 3.0"/><path d="M0 0 X"/></svg>', {}),
                         f'<svg {S} viewBox="0 0 10 10"><path d="M10 10h10"/><polygon points="0 0 10 0 10 10" stroke-dasharray="5 3"/><path d="M0 0 X"/></svg>')

    def test_identity_transform_kept_with_href(self):
        for before, after in (('<linearGradient id="b" href="#a" gradientTransform="scale(1)"/>', '<linearGradient id="b" href="#a" gradientTransform="scale(1)"/>'),
                              ('<pattern id="b" href="#a" patternTransform="translate(0 0)"/>', '<pattern id="b" href="#a" patternTransform="scale(1)"/>'),
                              ('<linearGradient id="b" href="#a" gradientTransform=" "/>', '<linearGradient id="b" href="#a" gradientTransform="scale(1)"/>'),
                              ('<linearGradient id="b" href="#a" gradientTransform="scale(2)"/>', '<linearGradient id="b" href="#a" gradientTransform="scale(2)"/>'),
                              ('<linearGradient id="b" gradientTransform="scale(1)"/>', '<linearGradient id="b"/>')):
            with self.subTest(before):
                self.assertEqual(run(format_geometry, f'<svg {S}>{before}</svg>', {}), f'<svg {S}>{after}</svg>')
        doc = f'<svg {S} {X}><linearGradient id="b" xlink:href="#a" gradientTransform="rotate(0)"/></svg>'
        self.assertIn('gradientTransform="scale(1)"', run(format_geometry, doc, {}))

    def test_css_invalid_lengths_kept(self):
        doc = f'<svg {S}><path d="M0 0h1" stroke-width="0."/><circle r="5."/></svg>'
        self.assertEqual(run(clean_values, doc), doc)
        self.assertIn('stroke-width="0."', opt(f'<svg {S}><g stroke="#00f"><path d="M2 9h30" stroke-width="0."/></g></svg>'))

    def test_empty_transform_kept_with_href(self):
        doc = (f'<svg {S}><defs><linearGradient id="a" gradientTransform="scale(2)"><stop offset="0"/><stop offset="1" stop-color="red"/></linearGradient>'
               '<linearGradient id="b" href="#a" gradientTransform=""/></defs><path d="M0 0h1" fill="url(#b)"/></svg>')
        self.assertEqual(opt(doc).count("gradientTransform="), 2)

    def test_number_lists(self):
        self.assertEqual(run(format_geometry, f'<svg {S} viewBox="0,-5 10,10"><polygon points="0,-5 .5,.5"/></svg>', {}),
                         f'<svg {S} viewBox="0 -5 10 10"><polygon points="0 -5 .5 .5"/></svg>')
        self.assertEqual(run(format_geometry, f'<svg {S}><polygon points="0-5 1.5.5"/></svg>', {}), f'<svg {S}><polygon points="0 -5 1.5 .5"/></svg>')
        for value in ("5px 3", "5%", "none", "1,,2", "1 2,", "a"):
            doc = f'<svg {S}><path d="M0 0h1" stroke-dasharray="{value}"/></svg>'
            self.assertEqual(run(format_geometry, doc, {}), doc, value)

    def test_defaults(self):
        self.assertEqual(run(remove_defaults, f'<svg {S}><path d="M0 0h1" fill="black" stroke="none" opacity="1"/><rect x="0" width="1" height="1"/></svg>', False),
                         f'<svg {S}><path d="M0 0h1"/><rect width="1" height="1"/></svg>')
        inherited = f'<svg {S}><g fill="red"><path d="M0 0h1" fill="#000"/></g></svg>'
        self.assertEqual(run(remove_defaults, inherited, False), inherited)
        styled = f'<svg {S}><g style="fill:red"><path d="M0 0h1" fill="#000"/></g></svg>'
        self.assertEqual(run(remove_defaults, styled, False), styled)
        unparsed = f'<svg {S}><g style="fill:red !important"><path d="M0 0h1" fill="#000"/></g></svg>'
        self.assertEqual(run(remove_defaults, unparsed, False), unparsed)
        used = f'<svg {S} {X}><defs><path id="p" d="M0 0h1" fill="#000"/></defs><use xlink:href="#p" fill="red"/></svg>'
        self.assertEqual(run(remove_defaults, used, False), used)
        self.assertEqual(run(remove_defaults, f'<svg {S}><g><path d="M0 0h1" fill="#000" opacity="1"/></g></svg>', True),
                         f'<svg {S}><g><path d="M0 0h1" fill="#000"/></g></svg>')

    def test_duplicate_declarations_stay_together(self):
        doc = f'<svg {S}><path style="fill:red;fill:#000;-x-y:1" d="M0 0h1"/></svg>'
        self.assertEqual(run(remove_defaults, doc, False), f'<svg {S}><path style="-x-y:1" d="M0 0h1"/></svg>')
        doc = f'<svg {S}><path style="fill:#000;fill:red;-x-y:1" d="M0 0h1"/></svg>'
        self.assertEqual(run(remove_defaults, doc, False), doc)

    def test_nested_styled_groups(self):
        doc = (f'<svg {S}><g style="stroke-linecap:round"><g style="fill:red"><path d="M0 0h1" fill="#000" stroke-linecap="butt"/>'
               '<path d="M0 0h1" stroke="none" style="stroke-linejoin:miter"/></g></g></svg>')
        self.assertEqual(run(remove_defaults, doc, False),
                         f'<svg {S}><g style="stroke-linecap:round"><g style="fill:red"><path d="M0 0h1" fill="#000" stroke-linecap="butt"/>'
                         '<path d="M0 0h1"/></g></g></svg>')

    def test_order_in_optimize(self):
        self.assertEqual(opt(f'<svg {S}><path d="M0 0h1" style="fill:#FF0000;stroke:none"/></svg>'),
                         f'<svg {S}><path d="M0 0h1" fill="red"/></svg>')
        styled = f'<svg {S}><style>g{{fill:red}}</style><g><path d="M0 0h1" style="fill:#000"/></g></svg>'
        self.assertIn('style="fill:#000"', opt(styled))

    def test_animation_in_optimize(self):
        doc = f'<svg {S}><path d="M10 10L20 10"><animate attributeName="d" to="M10 10L30 10" dur="1s"/></path></svg>'
        self.assertIn('d="M10 10L20 10"', opt(doc))
        doc = f'<svg {S}><g><set attributeName="fill" to="red"/><path d="M0 0h1" fill="#000"/></g></svg>'
        self.assertIn('fill="#000"', opt(doc))

    def test_defaults_inside_style(self):
        self.assertEqual(run(remove_defaults, f'<svg {S}><path d="M0 0h1" style="fill:#000;stroke:none;stroke-linecap:square;fill-opacity:1"/></svg>', False),
                         f'<svg {S}><path d="M0 0h1" style="stroke-linecap:square"/></svg>')
        for doc in (f'<svg {S}><path d="M0 0h1" fill="red" style="fill:#000"/></svg>',
                    f'<svg {S}><g fill="red"><path d="M0 0h1" style="fill:#000"/></g></svg>'):
            self.assertEqual(run(remove_defaults, doc, False), doc)
        doc = f'<svg {S}><path d="M0 0h1" style="opacity:1;fill:#000"/></svg>'
        self.assertEqual(run(remove_defaults, doc, True), doc)
        self.assertEqual(run(remove_defaults, f'<svg {S}><path d="M0 0h1" style="stroke:none"/></svg>', False),
                         f'<svg {S}><path d="M0 0h1"/></svg>')

    def test_rect_radii_and_root_origin(self):
        self.assertEqual(run(remove_defaults, f'<svg {S}><rect width="1" height="1" rx="0" ry="0"/><rect width="1" height="1" rx="0"/></svg>', False),
                         f'<svg {S}><rect width="1" height="1"/><rect width="1" height="1"/></svg>')
        doc = f'<svg {S}><rect width="1" height="1" rx="0" ry="2"/></svg>'
        self.assertEqual(run(remove_defaults, doc, False), doc)
        for doc, stylesheet in ((f'<svg {S}><rect width="1" height="1" rx="0" ry="0" style="ry:5px"/></svg>', False),
                                (f'<svg {S}><rect width="1" height="1" rx="0" ry="0"/></svg>', True)):
            self.assertEqual(run(remove_defaults, doc, stylesheet), doc)
        self.assertEqual(run(remove_defaults, f'<svg {S} x="0" y="0"/>', False), f'<svg {S}/>')

    def test_inkscape_style_end_to_end(self):
        out = opt(f'<svg {S}><rect style="fill:#f39c12;stroke:none;fill-rule:evenodd;stroke-linecap:square;stroke-width:1pt;stroke-opacity:1;fill-opacity:1"'
                  ' width="7.5" x="27.042373" y="10.5" height="31" ry="0" rx="0"/></svg>')
        for text in ("style=", 'stroke="none"', "opacity"):
            self.assertNotIn(text, out)
        self.assertIn('fill="#f39c12"', out)
        self.assertIn("<path", out)
        self.assertEqual(opt(out), out)
        self.assertEqual(out, f'<svg {S}><path fill="#f39c12" fill-rule="evenodd" stroke-linecap="square" stroke-width="1pt" d="M27.042373 10.5h7.5v31h-7.5z"/></svg>')

    def test_style_defaults_and_attributes_settle(self):
        out = opt(f'<svg {S}><path d="M0 0h1" style="fill:#f39c12;stroke:none;fill-rule:evenodd;stroke-linecap:square;stroke-width:1pt;stroke-opacity:1;fill-opacity:1"/></svg>')
        self.assertNotIn("style=", out)
        self.assertEqual(opt(out), out)


if __name__ == "__main__":
    unittest.main()
