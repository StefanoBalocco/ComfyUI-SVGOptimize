import unittest

from helpers import S, X, opt, run
from svgoptimize.svg_shapes import apply_translates, convert_shapes


class ShapeTests(unittest.TestCase):
    def test_conversions(self):
        self.assertEqual(run(convert_shapes, f'<svg {S}><line x1="0" y1="0" x2="10" y2="10" stroke="red"/></svg>'),
                         f'<svg {S}><path stroke="red" d="M0 0l10 10"/></svg>')
        self.assertEqual(run(convert_shapes, f'<svg {S}><rect x="10" y="10" width="20" height="30"/></svg>'),
                         f'<svg {S}><path d="M10 10h20v30H10z"/></svg>')
        self.assertEqual(run(convert_shapes, f'<svg {S}><polygon points="0 0 10 0 10 10"/></svg>'),
                         f'<svg {S}><path d="M0 0h10v10z"/></svg>')
        self.assertEqual(run(convert_shapes, f'<svg {S}><ellipse cx="5" cy="5" rx="3" ry="3.0"/></svg>'),
                         f'<svg {S}><circle cx="5" cy="5" r="3"/></svg>')

    def test_polyline_and_missing_coordinates(self):
        self.assertEqual(run(convert_shapes, f'<svg {S}><polyline points="0,0 10,0 10,10" fill="none"/></svg>'),
                         f'<svg {S}><path fill="none" d="M0 0h10v10"/></svg>')
        self.assertEqual(run(convert_shapes, f'<svg {S}><rect width="20" height="30"/></svg>'),
                         f'<svg {S}><path d="M0 0h20v30H0z"/></svg>')

    def test_kept(self):
        for doc in (f'<svg {S}><rect x="1" y="1" width="5" height="5" rx="1"/></svg>',
                    f'<svg {S}><rect x="1" y="1" width="5" height="5" style="ry:1"/></svg>',
                    f'<svg {S}><rect width="50%" height="5"/></svg>',
                    f'<svg {S}><rect width="0" height="5"/></svg>',
                    f'<svg {S}><polyline points="1 2 3"/></svg>',
                    f'<svg {S}><ellipse cx="5" cy="5" rx="3" ry="4"/></svg>',
                    f'<svg {S}><rect x="1" y="1" width="5" height="5"/><path d="M0 0h1" marker-end="url(#m)"/></svg>',
                    f'<svg {S}><rect x="1" y="1" width="5" height="5"/><path d="M0 0h1" style="marker:url(#m)"/></svg>'):
            self.assertEqual(run(convert_shapes, doc), doc)
        self.assertIn("<rect", opt(f'<svg {S}><style>rect{{fill:red}}</style><rect x="1" y="1" width="5" height="5"/></svg>'))
        self.assertIn("<rect", opt(f'<svg {S}><rect x="1" y="1" width="5" height="5"><animate attributeName="width" to="9"/></rect></svg>'))

    def test_style_geometry_keeps_shapes(self):
        doc = f'<svg {S}><rect x="1" y="1" width="50" height="50" style="width:5px"/></svg>'
        self.assertEqual(run(convert_shapes, doc), doc)
        doc = f'<svg {S}><circle cx="1" cy="2" r="3" style="cx:5px" transform="translate(10)"/></svg>'
        self.assertEqual(run(apply_translates, doc), doc)
        doc = f'<svg {S}><path d="M0 0h1" style="d:none" transform="translate(5)"/></svg>'
        self.assertEqual(run(apply_translates, doc), doc)

    def test_style_d_and_transform_keep_shapes(self):
        doc = f'<svg {S}><polyline points="0 0 1 1 2 2" style="d:none"/></svg>'
        self.assertEqual(run(convert_shapes, doc), doc)
        doc = f'<svg {S}><path d="M0 0h1" style="transform:translate(5px)" transform="translate(5)"/></svg>'
        self.assertEqual(run(apply_translates, doc), doc)

    def test_cost_increase_restores(self):
        doc = f'<svg {S}><polyline points="0 0 1 1 2 2 3 3 4 4 5 5" stroke="red" transform="translate(0.123456789012)"/></svg>'
        self.assertEqual(run(apply_translates, doc), doc)

    def test_default_namespace_prefix(self):
        doc = (f'<svg xmlns:svg="http://www.w3.org/2000/svg" {S}><rect x="1" y="1" width="50" height="50"/>'
               '<rect x="10.25" y="20.5" width="100.125" height="200.75" rx="1"/></svg>')
        once = opt(doc)
        self.assertNotIn("svg:", once)
        self.assertEqual(opt(once), once)

    def test_translate(self):
        self.assertEqual(run(apply_translates, f'<svg {S}><path d="M0 0C1 1 2 2 3 3Z" fill="#FDEDE8" transform="translate(262,667)"/></svg>'),
                         f'<svg {S}><path d="M262 667c1 1 2 2 3 3z" fill="#FDEDE8"/></svg>')
        self.assertEqual(run(apply_translates, f'<svg {S}><circle cx="1" cy="2" r="3" transform="translate(10)"/></svg>'),
                         f'<svg {S}><circle cx="11" cy="2" r="3"/></svg>')
        self.assertEqual(run(apply_translates, f'<svg {S}><rect width="5" height="5" transform="translate(5)"/></svg>'),
                         f'<svg {S}><rect width="5" height="5" x="5"/></svg>')
        self.assertEqual(run(apply_translates, f'<svg {S}><polygon points="0 0 10 0 10 10" transform="translate(1 2)"/></svg>'),
                         f'<svg {S}><polygon points="1 2 11 2 11 12"/></svg>')
        for doc in (f'<svg {S}><g fill="url(#g)"><path d="M0 0h1" transform="translate(5)"/></g></svg>',
                    f'<svg {S}><path d="M0 0h1" style="fill:url(#g)" transform="translate(5)"/></svg>',
                    f'<svg {S}><path d="M0 0h1" transform="translate(5) scale(2)"/></svg>',
                    f'<svg {S}><rect width="5" height="5" x="10%" transform="translate(5)"/></svg>'):
            self.assertEqual(run(apply_translates, doc), doc)

    def test_translate_kept_under_use_with_user_space_paint(self):
        paint = '<linearGradient id="g" gradientUnits="userSpaceOnUse" x2="10"><stop offset="0"/><stop offset="1" stop-color="red"/></linearGradient>'
        for target in ('<path id="p" transform="translate(50 50)" d="M0 0h10v10H0z"/>',
                       '<g id="p"><path transform="translate(50 50)" d="M0 0h10v10H0z"/></g>',
                       '<symbol id="p"><path transform="translate(50 50)" d="M0 0h10v10H0z"/></symbol>'):
            doc = f'<svg {S} {X}><defs>{paint}{target}</defs><use xlink:href="#p" fill="url(#g)"/></svg>'
            self.assertEqual(run(apply_translates, doc), doc)
            self.assertIn('transform="translate(50 50)"', opt(doc))

    def test_url_function_name_is_case_insensitive(self):
        doc = f'<svg {S}><path d="M0 0h1" fill="URL(#g)" transform="translate(5)"/></svg>'
        self.assertEqual(run(apply_translates, doc), doc)

    def test_animation_keeps_translate(self):
        self.assertIn('transform="translate(5)"', opt(f'<svg {S}><path d="M0 0h1" transform="translate(5)"><animate attributeName="fill" to="red"/></path></svg>'))

    def test_optimize_bakes_translate(self):
        self.assertIn('d="M5 0h1"', opt(f'<svg {S}><path d="M0 0h1" transform="translate(5)"/></svg>'))


if __name__ == "__main__":
    unittest.main()
