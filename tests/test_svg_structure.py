import unittest

from helpers import S, X, opt, run
from svgoptimize.svg_structure import clean_ids, collapse_groups, paint_one_stop_gradients, remove_invisible


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
        for extra in ('<style>#k{fill:red}@media print{}</style>', '<script/>', '<animate/>'):
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
        self.assertIn("<g", opt(f'<svg {S}><style>path{{fill:red}}@media print{{}}</style><g fill="red"><path d="M0 0h1"/></g></svg>'))

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

    def test_switch_children_kept(self):
        for body in ('<switch><defs/><path d="M0 0h1"/></switch>', '<switch><title>t</title><path d="M0 0h1"/></switch>',
                     '<switch><linearGradient/><path d="M0 0h1"/></switch>', '<switch><g/><path d="M0 0h1"/></switch>'):
            doc = f'<svg {S}>{body}</svg>'
            with self.subTest(body):
                self.assertEqual(run(clean_ids, doc), doc)

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


class InvisibleTests(unittest.TestCase):
    def test_hidden_removed(self):
        self.assertEqual(run(remove_invisible, f'<svg {S}><g style="display:none"><path d="M0 0h9"/></g><path display="none" d="M0 0h1"/><path d="M0 0h2"/></svg>'),
                         f'<svg {S}><path d="M0 0h2"/></svg>')

    def test_hidden_value_is_trimmed(self):
        self.assertEqual(run(remove_invisible, f'<svg {S}><path style="display: none " d="M0 0h1"/></svg>'), f'<svg {S}/>')

    def test_zero_size_removed(self):
        shapes = ('<rect width="0" height="5"/><rect width="5" height="0px"/><circle r="0"/><ellipse rx="0" ry="3"/>'
                  '<path d=""/><path d="  "/><path/><polyline points=""/>')
        self.assertEqual(run(remove_invisible, f'<svg {S}>{shapes}<path d="M0 0h2"/></svg>'), f'<svg {S}><path d="M0 0h2"/></svg>')

    def test_referenced_only_from_inside_removed(self):
        self.assertEqual(run(remove_invisible, f'<svg {S}><g display="none"><linearGradient id="g"/><path fill="url(#g)" d="M0 0h1"/></g><path d="M0 0h2"/></svg>'),
                         f'<svg {S}><path d="M0 0h2"/></svg>')

    def test_kept(self):
        for body in ('<g display="none"><path id="p" d="M0 0h1"/></g><use href="#p"/>',
                     '<g display="none"><linearGradient id="g"/></g><path fill="url(#g)" d="M0 0h1"/>',
                     '<switch><path display="none" d="M0 0h1"/><path d="M0 0h2"/></switch>',
                     '<switch><rect width="0" height="1"/><path d="M0 0h2"/></switch>',
                     '<text>a<tspan display="none">b</tspan></text>',
                     '<g display="none"><title>t</title><path d="M0 0h1"/></g>',
                     '<filter id="f"/><g filter="url(#f)"><rect width="0" height="1"/><path display="none" d="M0 0h1"/><path d="M0 0h2"/></g>',
                     '<path display="none" style="display:inline" d="M0 0h1"/>',
                     '<path display="none" style="/*x*/fill:red" d="M0 0h1"/>',
                     '<rect width="0" height="5" style="width:5px"/>',
                     '<line x1="1" y1="1" x2="1" y2="1" stroke="#000" stroke-linecap="round"/>',
                     '<linearGradient id="g" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="red"/><stop display="none" offset="1" stop-color="blue"/></linearGradient><path fill="url(#g)" d="M0 0h1"/>',
                     '<filter id="f"><feFlood display="none"/></filter><path filter="url(#f)" d="M0 0h1"/>',
                     '<marker id="m" display="none"><path d="M0 0h1"/></marker><path marker-end="url(#m)" d="M0 0h1"/>',
                     '<mask id="m"/><g mask="url(#m)"><rect width="0" height="1"/><path d="M0 0h2"/></g>',
                     '<clipPath id="c"/><g style="clip-path:url(#c)"><path display="none" d="M0 0h1"/><path d="M0 0h2"/></g>',
                     '<foreignObject width="9" height="9"/><path display="none" d="M0 0h1"/>',
                     '<defs><path id="p" d="M2 2h10v10H2z" fill="red"/></defs><use href="#p" style="display:none"/>',
                     '<image display="none" width="1" height="1" href="a.png"/>',
                     '<path display="None" d="M0 0h1"/>',
                     '<path display=" none" d="M0 0h1"/>',
                     '<path style="display: NONE " d="M0 0h1"/>',
                     '<path style="display:\xa0none" d="M0 0h1"/>',
                     '<marker id="m"/><g marker-end="url(#m)"><path display="none" d="M2 2h10"/><path d="M0 0h1"/></g>',
                     '<marker id="m"/><g style="marker:url(#m)"><polygon display="none" points="0 0 1 1"/><path d="M0 0h1"/></g>',
                     '<defs><filter id="f"><feFlood flood-color="red"/></filter><g id="a"><rect width="10" height="10"/><rect x="30" y="30" width="0" height="5"/></g></defs><use href="#a" filter="url(#f)"/>',
                     '<path stroke-width="0foo" d="M0 0h1"/>',
                     '<ellipse rx="0foo" ry="6"/>',
                     '<rect width=" 0" height="5"/>',
                     '<polygon transform="translate(10 10)" stroke="red" stroke-width="6" stroke-linecap="round"/>'):
            doc = f'<svg {S}>{body}</svg>'
            with self.subTest(doc):
                self.assertEqual(run(remove_invisible, doc), doc)
        doc = f'<svg {S} display="none"><path d="M0 0h1"/></svg>'
        self.assertEqual(run(remove_invisible, doc), doc)

    def test_hidden_rect_removed_with_markers(self):
        self.assertEqual(run(remove_invisible, f'<svg {S}><g marker-end="url(#m)"><path display="none" d="M0 0h1"/><rect display="none" width="1" height="1"/></g></svg>'),
                         f'<svg {S}><g marker-end="url(#m)"><path display="none" d="M0 0h1"/></g></svg>')

    def test_zero_lengths_removed(self):
        for width in ("0", "0.0", ".0", "00", "-0", "0px", "0%", "+0em"):
            with self.subTest(width):
                self.assertEqual(run(remove_invisible, f'<svg {S}><rect width="{width}" height="5"/></svg>'), f'<svg {S}/>')

    def test_use_target_subtree_kept(self):
        self.assertIn('width="0"', opt(f'<svg {S}><defs><g id="a"><rect width="9" height="9"/><rect width="0" height="5"/></g></defs><use href="#a" filter="url(#f)"/></svg>'))

    def test_in_optimizer(self):
        self.assertEqual(opt(f'<svg {S}><defs><linearGradient id="g"><stop offset="0"/><stop offset="1" stop-color="red"/></linearGradient></defs>'
                             '<g id="layer2" style="display:none"><path fill="url(#g)" d="M0 0h9"/></g><path d="M0 0h2"/></svg>'),
                         f'<svg {S}><path d="M0 0h2"/></svg>')
        for body in ('<style>path{display:inline}@media print{}</style><path display="none" d="M0 0h1"/>',
                     '<path display="none" d="M0 0h1"><set attributeName="display" to="inline"/></path>',
                     '<script>//</script><path display="none" d="M0 0h1"/>'):
            with self.subTest(body):
                self.assertIn('display="none"', opt(f'<svg {S}>{body}</svg>'))


class OneStopGradientTests(unittest.TestCase):
    CHAIN = ('<linearGradient id="swatch"><stop style="stop-color:#00ff00;stop-opacity:1" offset="0"/></linearGradient>'
             '<linearGradient id="ref" xlink:href="#swatch" gradientUnits="userSpaceOnUse" x2="10" gradientTransform="matrix(2 0 0 2 5 5)"/>')

    def test_attributes(self):
        doc = (f'<svg {S}><defs><linearGradient id="g" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#ff0000"/></linearGradient></defs>'
               '<path d="M0 0h1" fill="url(#g)" stroke="url(\'#g\') none"/></svg>')
        self.assertEqual(run(paint_one_stop_gradients, doc), doc.replace('fill="url(#g)" stroke="url(\'#g\') none"', 'fill="red" stroke="red"'))

    def test_fallback(self):
        gradient = '<linearGradient id="g" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#ff0000"/></linearGradient>'
        doc = f'<svg {S}>{gradient}<path d="M0 0h1" fill="url(#g) #00f" stroke="url(#g) currentColor"/></svg>'
        self.assertEqual(run(paint_one_stop_gradients, doc), doc.replace('fill="url(#g) #00f" stroke="url(#g) currentColor"', 'fill="red" stroke="red"'))
        for paint in ('fill="url(#g) junk"', 'style="fill:url(#g) 12px"'):
            with self.subTest(paint):
                doc = f'<svg {S}>{gradient}<path d="M0 0h1" {paint}/></svg>'
                self.assertEqual(run(paint_one_stop_gradients, doc), doc)

    def test_chain_and_style(self):
        doc = f'<svg {S} {X}>{self.CHAIN}<path style="fill:url(#ref);stroke:none" d="M0 0h1"/></svg>'
        self.assertEqual(run(paint_one_stop_gradients, doc), doc.replace("fill:url(#ref)", "fill:#0f0"))

    def test_units_through_href(self):
        doc = (f'<svg {S}><linearGradient id="a" gradientUnits="userSpaceOnUse"><stop offset="0"/></linearGradient>'
               '<linearGradient id="b" href="#a"/><path fill="url(#b)" d="M0 0h1"/></svg>')
        self.assertIn('<path fill="#000" d="M0 0h1"/>', run(paint_one_stop_gradients, doc))

    def test_href_through_radial_gradient(self):
        doc = (f'<svg {S}><radialGradient id="a" gradientUnits="userSpaceOnUse"><stop offset="0"/></radialGradient>'
               '<linearGradient id="b" href="#a"/><path fill="url(#b)" d="M0 0h1"/></svg>')
        self.assertIn('<path fill="#000" d="M0 0h1"/>', run(paint_one_stop_gradients, doc))

    def test_conflicting_hrefs_unchanged(self):
        doc = (f'<svg {S} {X}><linearGradient id="a"><stop offset="0"/></linearGradient><linearGradient id="b"><stop offset="0"/></linearGradient>'
               '<linearGradient id="g" gradientUnits="userSpaceOnUse" href="#a" xlink:href="#b"/><path fill="url(#g)" d="M0 0h1"/></svg>')
        self.assertEqual(run(paint_one_stop_gradients, doc), doc)

    def test_foreign_object_unchanged(self):
        doc = (f'<svg {S}><linearGradient id="g" gradientUnits="userSpaceOnUse"><stop offset="0"/></linearGradient>'
               '<foreignObject width="9" height="9"/><path fill="url(#g)" d="M0 0h1"/></svg>')
        self.assertEqual(run(paint_one_stop_gradients, doc), doc)

    def test_unchanged(self):
        units = ' gradientUnits="userSpaceOnUse"'
        stop = '<stop offset="0"/>'
        for gradient, fill in (
            (f'<linearGradient id="g">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units}>{stop}{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units}><stop offset="0" stop-color="currentColor"/></linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units}><stop offset="0" stop-opacity=".5"/></linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} gradientTransform="scale(0)">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} gradientTransform="matrix(1 2 2 4 0 0)">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} gradientTransform="skewX(90)">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} style="transform:scale(0)">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} style="/*x*/">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units}><stop offset="0" style="/*x*/"/></linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} gradientTransform="skewY(-90)">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} gradientTransform="scale(1 0)">{stop}</linearGradient>', "url(#g)"),
            ('<linearGradient id="g" href="#h"/><linearGradient id="h" href="#g"/>', "url(#g)"),
            ('<linearGradient id="g" href="#missing"/>', "url(#g)"),
            (f'<linearGradient id="g"{units}>{stop}</linearGradient>', "url(other.svg#g)"),
            (f'<radialGradient id="g"{units}>{stop}</radialGradient>', "url(#g)"),
            (f'<linearGradient id="g" gradientUnits=" userSpaceOnUse">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units}>{stop}<title>t</title></linearGradient>', "url(#g)"),
            (f'<linearGradient id="a">{stop}<desc/></linearGradient><linearGradient id="g"{units} href="#a"/>', "url(#g)"),
            (f'<linearGradient id="g"{units} gradientTransform="scale(1e-400)">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} gradientTransform="scale(1e400)">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} gradientTransform="scale(1e-200) scale(1e-200)">{stop}</linearGradient>', "url(#g)"),
            (f'<linearGradient id="g"{units} gradientTransform="matrix(1 1 1 1.0000000000000001 0 0)">{stop}</linearGradient>', "url(#g)"),
        ):
            with self.subTest(gradient=gradient, fill=fill):
                doc = f'<svg {S}>{gradient}<path fill="{fill}" d="M0 0h1"/></svg>'
                self.assertEqual(run(paint_one_stop_gradients, doc), doc)

    def test_stop_color_alpha(self):
        doc = f'<svg {S}><linearGradient id="g" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="rgba(255,0,0,.2)"/></linearGradient><path fill="url(#g)" d="M0 0h1"/></svg>'
        self.assertEqual(opt(doc), f'<svg {S}><path fill="#f003" d="M0 0h1"/></svg>')

    def test_optimized(self):
        doc = f'<svg {S} {X}>{self.CHAIN}<path style="fill:url(#ref);stroke:none" d="M0 0h1"/></svg>'
        self.assertEqual(opt(doc), f'<svg {S}><path d="M0 0h1" fill="#0f0"/></svg>')

    def test_kept_with_stylesheet(self):
        doc = f'<svg {S} {X}><style>path{{}}@media print{{}}</style>{self.CHAIN}<path fill="url(#ref)" d="M0 0h1"/></svg>'
        self.assertIn('fill="url(#', opt(doc))


if __name__ == "__main__":
    unittest.main()
