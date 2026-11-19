import unittest

from helpers import S, opt, run
from svgoptimize.svg_viewport import crop_viewport

BIG = "M-2-2l515.1-.3-.3 515.5-514.7-.1z"
FULL = "M-1-1H513V513H-1z"
RECT = "M0 0h512v512H0z"
BOX = 'width="512" height="512" viewBox="0 0 512 512"'


def crop(doc):
    return run(crop_viewport, doc, {})


def svg(path, attrs=BOX, inner="{}"):
    return f'<svg {S} {attrs}>{inner.replace("{}", path)}</svg>'


class CropTests(unittest.TestCase):
    def test_cropped(self):
        self.assertEqual(crop(svg(f'<path d="{BIG}" fill="#cab6d6"/>')), svg(f'<path d="{RECT}" fill="#cab6d6"/>'))
        frame = "M0 0C84.48 0 168.96 0 256 0C256 84.48 256 168.96 256 256C171.52 256 87.04 256 0 256C0 171.52 0 87.04 0 0Z"
        self.assertEqual(crop(svg(f'<path d="{frame}"/>', 'width="256" height="256"')),
                         svg('<path d="M0 0h256v256H0z"/>', 'width="256" height="256"'))
        attrs = 'viewBox="10 20 100 50" width="200" height="100"'
        self.assertEqual(crop(svg('<path d="M-100-100H300V200H-100z"/>', attrs)), svg('<path d="M10 20h100v50H10z"/>', attrs))
        self.assertEqual(crop(svg(f'<path d="{BIG}"/>', inner='<g fill="#123">{}</g>')), svg(f'<path d="{RECT}"/>', inner='<g fill="#123">{}</g>'))

    def test_general_cover(self):
        for d in ("M-5 3L3-5H509L517 3V509L509 517H3L-5 509z", "M-1-1H256V513H-1zM256-1H513V513H256z", "M-2-2C100 3 200-10 514-2L514 514H-2z"):
            for rule in ("", ' fill-rule="evenodd"'):
                self.assertEqual(crop(svg(f'<path d="{d}"{rule}/>')), svg(f'<path d="{RECT}"{rule}/>'), d)

    def test_units(self):
        attrs = 'width="210mm" height="297mm" viewBox="0 0 210 297"'
        self.assertEqual(crop(svg('<path d="M-1-1H211V298H-1z"/>', attrs)), svg('<path d="M0 0h210v297H0z"/>', attrs))
        for attrs in ('width="210mm" height="297cm" viewBox="0 0 210 297"', 'width="210mm" height="297mm"',
                      'width="100%" height="100%" viewBox="0 0 210 297"'):
            doc = svg('<path d="M-1-1H211V298H-1z"/>', attrs)
            self.assertEqual(crop(doc), doc)

    def test_kept(self):
        big = f'<path d="{BIG}"'
        for doc in (svg(big + ' stroke="red"/>'),
                    svg(big + '/>', inner='<g stroke="red">{}</g>'),
                    svg(big + ' transform="scale(2)"/>'),
                    svg(big + '/>', inner='<g transform="scale(2)">{}</g>'),
                    svg(big + ' style="transform:scale(2)"/>'),
                    svg(big + ' filter="blur(1px)"/>'),
                    svg(big + '/>', BOX + ' filter="blur(1px)"'),
                    svg(big + '/>', BOX + ' transform="scale(2)"'),
                    svg(big + ' fill="url(#g)"/>'),
                    svg(big + ' id="a"/>'),
                    svg(big + '/>', inner='<g id="a">{}</g>'),
                    svg(big + '/>', 'width="100%" height="512" viewBox="0 0 512 512"'),
                    svg(big + '/>', 'width="512" height="256" viewBox="0 0 512 512"'),
                    svg(big + '/>', 'viewBox="0 0 512 512"'),
                    svg(big + '/>', BOX + ' overflow="visible"'),
                    svg(big + '/>', inner='<a>{}</a>'),
                    svg(big + '/>', inner='<defs><clipPath id="c">{}</clipPath></defs>'),
                    svg('<path d="M-1-1H513V513"/>'),
                    svg(f'<path d="{FULL}{FULL}" fill-rule="evenodd"/>'),
                    svg(f'<path d="{FULL}{FULL}"/>', inner='<g fill-rule="evenodd">{}</g>'),
                    svg('<path d="M-1-1H513V513H-1zM100 100H400V400H100z" fill-rule="evenodd"/>'),
                    svg('<path d="M-1-1H513V513H-1zM100 100H400V400H100z"/>', inner='<g fill-rule="evenodd">{}</g>'),
                    svg('<path d="M-2 50Q-2-2 50-2H514V514H-2z"/>'),
                    svg('<path d="M-1-1H513V513A5 5 0 0 0 -1 513z"/>')):
            self.assertEqual(crop(doc), doc)
        self.assertEqual(crop(svg(f'<path d="{FULL}{FULL}"/>')), svg(f'<path d="{RECT}"/>'))

    def test_opposite_hole_is_kept(self):
        hole = "M-1-1H513V513H-1zM100 100V400H400V100z"
        doc = svg(f'<path d="{hole}"/>')
        self.assertEqual(crop(doc), doc)

    def test_stylesheet_keeps_geometry(self):
        doc = svg(f'<style>path{{stroke:red}}@media print{{}}</style><path d="{BIG}"/><path d="M10 10h5v5z"/>')
        self.assertIn(BIG, opt(doc))
        doc = svg(f'<path d="M10 10h5v5z"/><style>@media print{{}}</style><path d="{BIG}"/>')
        self.assertIn("M10 10", opt(doc))


class HiddenTests(unittest.TestCase):
    def test_removed(self):
        small = '<path d="M10 10h5v5z"/>'
        cover = f'<path d="{BIG}" fill="#cab6d6"/>'
        last = '<path d="M20 20h5v5z"/>'
        self.assertEqual(crop(svg(small + cover + last)), svg(f'<path d="{RECT}" fill="#cab6d6"/>' + last))
        self.assertEqual(crop(svg(small + f'<g>{cover}</g>' + last)), svg(f'<g><path d="{RECT}" fill="#cab6d6"/></g>' + last))
        self.assertEqual(crop(svg(small + f'<g fill="red">{small}</g><g>{small}<g>{cover}</g></g>')),
                         svg(f'<g><g><path d="{RECT}" fill="#cab6d6"/></g></g>'))
        self.assertEqual(crop(svg(small + f'<path d="{BIG}"/>')), svg(f'<path d="{RECT}"/>'))

    def test_kept(self):
        small = '<path d="M10 10h5v5z"/>'
        for cover in (f'<path d="{BIG}" fill="#cab6d6" fill-opacity=".5"/>', f'<path d="{BIG}" fill="#cab6d680"/>',
                      f'<path d="{BIG}" opacity=".5"/>', f'<g opacity=".5"><path d="{BIG}"/></g>',
                      f'<path d="{BIG}" mix-blend-mode="multiply"/>', f'<path d="{BIG}" display="none"/>',
                      f'<path d="{BIG}" visibility="hidden"/>', f'<g visibility="hidden"><path d="{BIG}"/></g>',
                      f'<path d="{BIG}" systemLanguage="en"/>', f'<path d="{BIG}" fill="none"/>'):
            doc = svg(small + cover)
            self.assertIn(small, crop(doc), cover)
        doc = svg(f'<defs><linearGradient id="g"/></defs><path d="{BIG}"/>')
        self.assertIn("<defs>", crop(doc))
        doc = svg(f'<path id="a" d="M10 10h5v5z"/><path d="{BIG}"/><use href="#a"/>')
        self.assertIn('id="a"', crop(doc))
        doc = svg(f'<path d="M10 10h5v5z"/><path d="{BIG}"/><foreignObject width="1" height="1"/>')
        self.assertIn(small, crop(doc))

    def test_stylesheet_keeps_siblings(self):
        doc = svg(f'<style>@media print{{}}</style><path d="M10 10h5v5z"/><path d="{BIG}"/>')
        self.assertIn("M10 10", opt(doc))


class OutsideTests(unittest.TestCase):
    def test_removed(self):
        for shape in ('<path d="M-30 10h20v20z"/>', '<path d="M-30 10h30v20z"/>', '<circle cx="-20" cy="10" r="5"/>',
                      '<rect x="-20" width="19" height="9" stroke="red" stroke-width="1" stroke-miterlimit="1"/>'):
            self.assertEqual(crop(svg(shape + '<path d="M1 1h5v5z"/>')), svg('<path d="M1 1h5v5z"/>'), shape)

    def test_kept(self):
        for shape in ('<path d="M-30 10h40v20z"/>', '<rect x="-5" width="4" height="9" stroke="red" stroke-width="4"/>',
                      '<path d="M-30 10h20v20z" marker-end="url(#m)"/>', '<path d="M-30 10h20v20z" transform="translate(2)"/>',
                      '<path d="M-30 10h20v20z" id="a"/>', '<path d="M-30 10h20v20z" style="transform:scale(2)"/>'):
            doc = svg(shape)
            self.assertEqual(crop(doc), doc, shape)
        doc = svg('<path d="M-30 10h20v20z"/>', inner='<g filter="blur(1px)">{}</g>')
        self.assertEqual(crop(doc), doc)
        doc = svg('<path d="M-30 10h20v20z"/>', 'width="100%" height="100%" viewBox="0 0 512 512"')
        self.assertEqual(crop(doc), doc)


class CoveredTests(unittest.TestCase):
    SQUARE = '<path d="M0 0h100v100H0z" fill="#123"/>'
    RECT = '<rect x="10" y="10" width="20" height="20"/>'

    def test_removed(self):
        for doc in (self.RECT + self.SQUARE, '<rect x="-10" y="10" width="40" height="20"/>' + self.SQUARE,
                    '<rect x="0" y="10" width="20" height="20"/>' + self.SQUARE):
            self.assertEqual(crop(svg(doc)), svg(self.SQUARE), doc)
        self.assertEqual(crop(svg(self.RECT + self.SQUARE, 'viewBox="0 0 512 512"')), svg(self.SQUARE, 'viewBox="0 0 512 512"'))

    def test_kept(self):
        for doc in (self.RECT + '<path d="M0 0h100v100H0z" fill="#123" opacity=".5"/>',
                    self.RECT + '<path d="M0 0h100v100H0zM5 5v50h50V5z" fill="#123"/>',
                    self.SQUARE + self.RECT,
                    '<rect x="10" y="10" width="95" height="20" stroke="red" stroke-width="20"/>' + self.SQUARE):
            doc = svg(doc, 'width="512" height="512" viewBox="0 0 512 512"')
            self.assertIn("<rect", crop(doc), doc)
        doc = svg(self.RECT + '<path d="M0 0h100v100H0zM5 5v50h50V5z" fill="#123"/>')
        self.assertIn("<rect", crop(doc))


class UnionTests(unittest.TestCase):
    UPPER = '<path d="M256-1H513V513H256z" fill="#123"/>'
    LOWER = '<path d="M-1-1H256V513H-1z" fill="#123"/>'
    CURVED = '<path d="M-1-1H300C310 100 290 200 300 513H-1z" fill="#123"/>'

    def test_rewritten(self):
        self.assertEqual(crop(svg(self.LOWER + self.UPPER)), svg(f'<path d="{RECT}" fill="#123"/>' + self.UPPER))
        self.assertEqual(crop(svg(self.CURVED + '<path d="M250-1H513V513H250z" fill="#123"/>')),
                         svg(f'<path d="{RECT}" fill="#123"/><path d="M250-1H513V513H250z" fill="#123"/>'))

    def test_kept(self):
        for doc in (self.CURVED + '<path d="M250-1H513V513H250z" fill="#123" opacity=".5"/>',
                    self.CURVED + '<path d="M320-1H513V513H320z" fill="#123"/>',
                    self.LOWER + '<path d="M256-1H513V513H256z" fill="#123" fill-opacity=".5"/>'):
            self.assertEqual(crop(svg(doc)), svg(doc), doc)

    def test_the_path_on_top_keeps_its_shape(self):
        self.assertEqual(crop(svg('<path d="M250-1H513V513H250z" fill="#123"/>' + self.CURVED)), svg(f'<path d="{RECT}" fill="#123"/>' + self.CURVED))

    def test_removed_under_two_paths(self):
        rect = '<rect x="10" y="10" width="20" height="20"/>'
        halves = '<path d="M0 0h20v100H0z"/><path d="M20 0h80v100H20z"/>'
        self.assertEqual(crop(svg(rect + halves)), svg(halves))
        gap = '<path d="M0 0h20v100H0z"/><path d="M25 0h75v100H25z"/>'
        self.assertIn("<rect", crop(svg(rect + gap)))

    def test_union_cover_hides_what_is_before(self):
        self.assertEqual(crop(svg('<rect x="10" y="10" width="20" height="20"/>' + self.LOWER + self.UPPER)),
                         svg(f'<path d="{RECT}" fill="#123"/>' + self.UPPER))

    def test_empty_group_goes_in_the_next_round(self):
        doc = svg('<g opacity=".5"><rect x="-50" width="10" height="10"/></g><path d="M1 1h5v5z"/>')
        self.assertNotIn("<g", opt(doc))
        self.assertIn("<g", crop(doc))


if __name__ == "__main__":
    unittest.main()
