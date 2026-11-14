import unittest

from helpers import S, X, opt, run
from svgoptimize.svg_structure import clean_ids, collapse_groups


class IdTests(unittest.TestCase):
    def test_unused_and_renamed(self):
        doc = (f'<svg {S} {X}><defs><linearGradient id="unused"/><linearGradient id="grad"><stop offset="1"/></linearGradient></defs>'
               '<path id="shape" d="M0 0h1" fill="url(#grad)"/><g id="empty"/></svg>')
        self.assertEqual(run(clean_ids, doc),
                         f'<svg {S}><defs><linearGradient id="a"><stop offset="1"/></linearGradient></defs><path d="M0 0h1" fill="url(#a)"/></svg>')

    def test_all_reference_forms(self):
        doc = (f'<svg {S} {X}><defs><path id="long" d="M0 0h1"/></defs><use xlink:href="#long"/>'
               '<g clip-path="url(&quot;#long&quot;)"><path d="M0 0h1" style="mask:url(\'#long\')"/></g></svg>')
        out = run(clean_ids, doc)
        self.assertNotIn("long", out)
        self.assertEqual(out.count("#a"), 3)

    def test_href_and_aria_lists(self):
        doc = (f'<svg {S}><defs><path id="long" d="M0 0h1"/></defs><use href="#long"/>'
               '<g id="t1"><path d="M0 0h1"/></g><g aria-labelledby="t1  long"><path d="M0 0h1"/></g></svg>')
        self.assertEqual(run(clean_ids, doc), f'<svg {S}><defs><path id="a" d="M0 0h1"/></defs><use href="#a"/><g id="b"><path d="M0 0h1"/></g><g aria-labelledby="b  a"><path d="M0 0h1"/></g></svg>')

    def test_most_referenced_first(self):
        doc = (f'<svg {S}><defs><linearGradient id="one"/><linearGradient id="two"/></defs>'
               '<path d="M0 0h1" fill="url(#one)"/><path d="M0 0h1" fill="url(#two)" stroke="url(#two)"/></svg>')
        out = run(clean_ids, doc)
        self.assertIn('fill="url(#a)" stroke="url(#a)"', out)
        self.assertIn('fill="url(#b)"', out)

    def test_skipped_with_style_script_animation(self):
        for extra in ('<style>#k{fill:red}</style>', '<script/>', '<animate/>'):
            self.assertIn('id="k"', opt(f'<svg {S}>{extra}<path id="k" d="M0 0h1"/></svg>'), extra)

    def test_fonts_kept(self):
        doc = f'<svg {S}><defs><font><font-face font-family="F"/></font></defs><text font-family="F">a</text></svg>'
        self.assertEqual(run(clean_ids, doc), doc)

    def test_foreign_object_keeps_ids(self):
        doc = f'<svg {S}><foreignObject><p id="x"/></foreignObject><path id="k" d="M0 0h1"/></svg>'
        self.assertEqual(run(clean_ids, doc), doc)

    def test_dangling_reference_kept(self):
        doc = f'<svg {S} {X}><use xlink:href="#a"/><defs><path id="long" d="M0 0h1"/></defs><use xlink:href="#long"/></svg>'
        self.assertEqual(run(clean_ids, doc), f'<svg {S} {X}><use xlink:href="#a"/><defs><path id="b" d="M0 0h1"/></defs><use xlink:href="#b"/></svg>')

    def test_titles_kept_in_defs(self):
        doc = f'<svg {S}><defs><title>t</title><desc>d</desc></defs></svg>'
        self.assertEqual(run(clean_ids, doc), doc)

    def test_removal_cascades(self):
        doc = (f'<svg {S} {X}><defs><linearGradient id="base"/><linearGradient id="top" xlink:href="#base"/></defs>'
               '<path d="M0 0h1"/></svg>')
        self.assertEqual(run(clean_ids, doc), f'<svg {S}><path d="M0 0h1"/></svg>')

    def test_descendant_with_referenced_id_keeps_container(self):
        doc = f'<svg {S}><defs><g><path id="p" d="M0 0h1"/></g></defs><path d="M0 0h1" fill="url(#p)"/></svg>'
        self.assertEqual(run(clean_ids, doc), f'<svg {S}><defs><g><path id="a" d="M0 0h1"/></g></defs><path d="M0 0h1" fill="url(#a)"/></svg>')

    def test_two_letter_names(self):
        count = 54
        uses = "".join(f'<path d="M0 0h1" fill="url(#i{n})"/>' for n in range(count))
        defs = "".join(f'<linearGradient id="i{n}"/>' for n in range(count))
        out = run(clean_ids, f'<svg {S}><defs>{defs}</defs>{uses}</svg>')
        self.assertIn('id="Z"', out)
        self.assertIn('id="aa"', out)
        self.assertIn('id="ab"', out)


class GroupTests(unittest.TestCase):
    def test_collapse(self):
        self.assertEqual(run(collapse_groups, f'<svg {S}><g><g fill="red" transform="translate(1)"><path d="M0 0h1" stroke="blue" transform="scale(2)"/></g></g></svg>'),
                         f'<svg {S}><path d="M0 0h1" stroke="blue" transform="translate(1) scale(2)" fill="red"/></svg>')
        doc = f'<svg {S}><g fill="red"><path d="M0 0h1" fill="blue"/></g></svg>'
        self.assertEqual(run(collapse_groups, doc), doc)

    def test_common_attributes(self):
        self.assertEqual(run(collapse_groups, f'<svg {S}><g><title>t</title><path d="M0 0h1" fill="red"/><path d="M0 0h2" fill="red"/></g></svg>'),
                         f'<svg {S}><g fill="red"><title>t</title><path d="M0 0h1"/><path d="M0 0h2"/></g></svg>')

    def test_kept(self):
        for doc in (f'<svg {S}><switch><g><path d="M0 0h1"/></g><path d="M0 0h2"/></switch></svg>',
                    f'<svg {S}><g opacity=".5"><path d="M0 0h1"/></g></svg>',
                    f'<svg {S}><g fill="red"><path id="p" d="M0 0h1"/></g></svg>',
                    f'<svg {S}><g stroke="blue"><path id="p" d="M0 0h1" fill="red"/><path d="M0 0h2" fill="red"/></g></svg>',
                    f'<svg {S}><g transform="translate(1)"><linearGradient/></g></svg>',
                    f'<svg {S}><g style="fill:blue"><path d="M0 0h1" fill="red"/><path d="M0 0h2" fill="red"/></g></svg>',
                    f'<svg {S}><g transform="translate(1)"><path d="M0 0h1" style="transform:scale(2)"/></g></svg>'):
            self.assertEqual(run(collapse_groups, doc), doc)

    def test_kept_with_stylesheet(self):
        self.assertIn("<g", opt(f'<svg {S}><style>path{{fill:red}}</style><g fill="red"><path d="M0 0h1"/></g></svg>'))

    def test_child_setting_the_property_keeps_group(self):
        for doc in (f'<svg {S}><g fill="red"><path style="fill:inherit;stroke:#000;stroke-width:2;stroke-linecap:round;stroke-linejoin:round" d="M0 0h10v10H0z"/></g></svg>',
                    f'<svg {S}><g fill="red"><path d="M0 0h1" fill="inherit"/></g></svg>',
                    f'<svg {S}><g font-size="20"><text font-size="2em">Hi</text></g></svg>',
                    f'<svg {S}><g font-weight="700"><text font-weight="bolder">Hi</text></g></svg>',
                    f'<svg {S}><g stroke="url(#a)"><circle r="3" stroke="inherit"/><path d="M0 0h1" stroke="inherit"/></g></svg>',
                    f'<svg {S}><g font-size="20"><text font-size="2em">a</text><text font-size="2em">b</text></g></svg>'):
            self.assertEqual(run(collapse_groups, doc), doc)
        self.assertEqual(run(collapse_groups, f'<svg {S}><g><path fill="red" style="fill:inherit;-x-y:1" d="M0 0h1"/><path fill="red" d="M0 0h2"/></g></svg>'),
                         f'<svg {S}><path fill="red" style="fill:inherit;-x-y:1" d="M0 0h1"/><path fill="red" d="M0 0h2"/></svg>')

    def test_unparsed_child_style_blocks_merge_and_hoist(self):
        doc = f'<svg {S}><g fill="red"><path d="M0 0h1" style="stroke:red !important"/></g></svg>'
        self.assertEqual(run(collapse_groups, doc), doc)
        self.assertEqual(run(collapse_groups, f'<svg {S}><g><path d="M0 0h1" fill="red" style="stroke:red !important"/><path d="M0 0h2" fill="red"/></g></svg>'),
                         f'<svg {S}><path d="M0 0h1" fill="red" style="stroke:red !important"/><path d="M0 0h2" fill="red"/></svg>')

    def test_group_with_metadata_children_kept(self):
        doc = f'<svg {S}><g><title>Leaf</title><path d="M0 0h9v9H0z"/></g><title>Doc</title></svg>'
        self.assertEqual(run(collapse_groups, doc), doc)
        self.assertEqual(opt(doc, True), doc)

    def test_clip_path_and_switch_children_kept(self):
        for doc in (f'<svg {S}><clipPath id="c"><g transform="translate(5)"><rect width="5" height="5"/></g></clipPath></svg>',
                    f'<svg {S}><switch><g systemLanguage="fr"/><path d="M0 0h1"/></switch></svg>'):
            self.assertEqual(run(collapse_groups, doc), doc)
        doc = f'<svg {S}><defs><linearGradient id="g"/></defs><switch><g systemLanguage="fr"/><path d="M0 0h1" fill="url(#g)"/></switch></svg>'
        self.assertIn("<g systemLanguage", opt(doc))

    def test_function_names_are_case_insensitive(self):
        doc = f'<svg {S}><defs><linearGradient id="g"/></defs><path d="M0 0h1" fill="URL(#g)"/></svg>'
        self.assertEqual(run(clean_ids, doc), f'<svg {S}><defs><linearGradient id="a"/></defs><path d="M0 0h1" fill="url(#a)"/></svg>')

    def test_unparsed_transform_not_merged(self):
        doc = f'<svg {S}><g transform="bogus"><path transform="translate(1)" d="M0 0h1"/></g></svg>'
        self.assertEqual(run(collapse_groups, doc), doc)
        doc = f'<svg {S}><g transform="translate(1)"><path transform="bogus" d="M0 0h1"/></g></svg>'
        self.assertEqual(run(collapse_groups, doc), doc)

    def test_text_and_tails_keep_order(self):
        self.assertEqual(run(collapse_groups, f'<svg {S}>a<g>b<path d="M0 0h1"/>c<path d="M0 0h2"/>d</g>e</svg>'),
                         f'<svg {S}>ab<path d="M0 0h1"/>c<path d="M0 0h2"/>de</svg>')

    def test_repeated_passes(self):
        self.assertEqual(opt(f'<svg {S}><g stroke="none"><path d="M0 0h1"/><path d="M0 0h2"/></g></svg>'),
                         f'<svg {S}><path d="M0 0h1"/><path d="M0 0h2"/></svg>')


if __name__ == "__main__":
    unittest.main()
