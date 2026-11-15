import unittest

from lxml import etree

import package_alias  # noqa: F401
from helpers import S, opt, run
from svgoptimize.svg_document import SVG_NS, Facts, clean_document, declared, defined_ids, document_facts, element_cost, elements, local_name, parse_svg, serialize
from svgoptimize.svg_optimize import optimize_svg

SODIPODI = 'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"'


class DocumentTests(unittest.TestCase):
    def test_cleanup_and_minimal_output(self):
        self.assertEqual(opt(f'<?xml version="1.0"?>\n<!DOCTYPE svg>\n<!-- c -->\n<svg {S} version="1.1">\n  <path d="M0 0h1"/>\n</svg>'),
                         f'<svg {S}><path d="M0 0h1"/></svg>')

    def test_stylesheet_instruction_kept(self):
        out = opt(f'<?xml-stylesheet href="a.css"?><svg {S}/>')
        self.assertTrue(out.startswith('<?xml-stylesheet href="a.css"?><svg'))

    def test_editor_data(self):
        doc = (f'<svg {S} xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
               'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" inkscape:version="1" '
               'enable-background="new" data-name="x"><sodipodi:namedview/><path d="M0 0h1" inkscape:label="a"/></svg>')
        self.assertEqual(run(clean_document, doc, False), f'<svg {S}><path d="M0 0h1"/></svg>')
        self.assertEqual(opt(doc), f'<svg {S}><path d="M0 0h1"/></svg>')

    def test_metadata(self):
        doc = f'<svg {S}><title>T</title><desc>D</desc><metadata><![CDATA[{{"a": "<&>"}}]]></metadata></svg>'
        self.assertEqual(run(clean_document, doc, False), f'<svg {S}/>')
        self.assertEqual(run(clean_document, doc, True), f'<svg {S}><title>T</title><desc>D</desc><metadata>{{"a": "&lt;&amp;>"}}</metadata></svg>')
        self.assertEqual(opt(doc), f'<svg {S}/>')
        self.assertEqual(opt(doc, True), f'<svg {S}><title>T</title><desc>D</desc><metadata>{{"a": "&lt;&amp;>"}}</metadata></svg>')

    def test_switch_children_kept(self):
        for body in ('<desc>d</desc>', '<title>t</title>', '<metadata/>'):
            doc = f'<svg {S}><switch>{body}<path d="M0 0h1"/></switch></svg>'
            with self.subTest(body):
                self.assertEqual(run(clean_document, doc, False), doc)

    def test_metadata_namespaces_kept_inside_metadata(self):
        rdf = 'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/"'
        doc = f'<svg {S} {rdf}><metadata><rdf:RDF><dc:title>t</dc:title></rdf:RDF></metadata><dc:title>x</dc:title></svg>'
        self.assertEqual(run(clean_document, doc, True), f'<svg {S} {rdf}><metadata><rdf:RDF><dc:title>t</dc:title></rdf:RDF></metadata></svg>')

    def test_enable_background_in_style(self):
        self.assertEqual(run(clean_document, f'<svg {S}><g style="fill:red;enable-background:new 0 0 1 1"/></svg>', False),
                         f'<svg {S}><g style="fill:red"/></svg>')
        self.assertEqual(run(clean_document, f'<svg {S}><g style="enable-background:new"/></svg>', False), f'<svg {S}><g/></svg>')
        self.assertIn('style="a:b;/*x*/enable-background:new"', run(clean_document, f'<svg {S}><g style="a:b;/*x*/enable-background:new"/></svg>', False))

    def test_xml_space(self):
        self.assertEqual(opt(f'<svg {S} xml:space="preserve"><path d="M0 0h1"/></svg>'), f'<svg {S}><path d="M0 0h1"/></svg>')
        self.assertIn('xml:space="preserve"', opt(f'<svg {S} xml:space="preserve"><text> a  b </text></svg>'))

    def test_entities(self):
        doc = b'<!DOCTYPE svg [<!ENTITY ns "http://www.w3.org/2000/svg">]><svg xmlns="&ns;"/>'
        self.assertEqual(optimize_svg(doc, False), f'<svg {S}/>'.encode())
        with self.assertRaises(ValueError):
            optimize_svg(b'<!DOCTYPE svg [<!ENTITY e SYSTEM "file:///etc/hostname">]><svg xmlns="http://www.w3.org/2000/svg">&e;</svg>', False)
        for bad in (b"not xml", b"<html/>", b'<svg/>'):
            with self.assertRaises(ValueError):
                optimize_svg(bad, False)

    def test_entity_with_markup_rejected(self):
        data = b'<!DOCTYPE svg [<!ENTITY p "<path d=\'M0 0h10\'/>">]><svg xmlns="http://www.w3.org/2000/svg">&p;</svg>'
        with self.assertRaises(ValueError):
            parse_svg(data)

    def test_public_doctype(self):
        doc = ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
               f'"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n<svg {S}><path d="M0 0h1"/></svg>')
        self.assertEqual(opt(doc), f'<svg {S}><path d="M0 0h1"/></svg>')

    def test_foreign_object_untouched(self):
        root, _ = parse_svg(f'<svg {S}><foreignObject><div xmlns="http://www.w3.org/1999/xhtml" data-name="x"><p/></div></foreignObject><g/></svg>'.encode())
        self.assertEqual([local_name(e) for e in elements(root)], ["svg", "foreignObject", "g"])
        self.assertEqual(run(clean_document, f'<svg {S}><foreignObject><div xmlns="http://www.w3.org/1999/xhtml" data-name="x"><p/></div></foreignObject><g/></svg>', False),
                         f'<svg {S}><foreignObject><div xmlns="http://www.w3.org/1999/xhtml" data-name="x"><p/></div></foreignObject><g/></svg>')

    def test_removed_nodes_keep_tails(self):
        self.assertEqual(run(clean_document, f'<svg {S} {SODIPODI}><text>Hello <sodipodi:x/>world</text></svg>', False), f'<svg {S}><text>Hello world</text></svg>')
        self.assertEqual(run(clean_document, f'<svg {S}><text>a<?x y?>b</text></svg>', False), f'<svg {S}><text>ab</text></svg>')
        self.assertEqual(run(clean_document, f'<svg {S}><text><sodipodi:x {SODIPODI}/>a<?x y?>b</text></svg>', False), f'<svg {S}><text>ab</text></svg>')

    def test_facts(self):
        root, pre = parse_svg(f'<svg {S}><rect onclick="f()"/><animate/></svg>'.encode())
        self.assertEqual(document_facts(root, pre), Facts(stylesheet=False, script=True, animation=True))
        root, pre = parse_svg(f'<svg {S}><foreignObject><style xmlns="http://www.w3.org/1999/xhtml"/></foreignObject></svg>'.encode())
        self.assertEqual(document_facts(root, pre), Facts(stylesheet=True, script=False, animation=False))
        root, pre = parse_svg(f'<?xml-stylesheet href="a.css"?><svg {S}/>'.encode())
        self.assertEqual(document_facts(root, pre), Facts(stylesheet=True, script=False, animation=False))


def roundtrip(text):
    root, preceding = parse_svg(text.encode())
    return serialize(root, preceding).decode()


class SerializeTests(unittest.TestCase):
    def test_text_and_quotes(self):
        doc = f'<svg {S}><style>a > b {{ fill: red }}</style><text font-family=\'"A B"\'> x &amp; &lt; é </text></svg>'
        self.assertEqual(roundtrip(doc), doc)
        self.assertEqual(opt(doc), doc)

    def test_prefixed_namespace(self):
        doc = ('<svg:svg xmlns:svg="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
               '<svg:use xlink:href="#a"/></svg:svg>')
        self.assertEqual(roundtrip(doc), doc)
        self.assertEqual(opt(doc), doc)

    def test_whitespace_inheritance(self):
        self.assertEqual(roundtrip(f'<svg {S}><text><a><tspan>a</tspan> <tspan>b</tspan></a></text></svg>'),
                         f'<svg {S}><text><a><tspan>a</tspan> <tspan>b</tspan></a></text></svg>')
        self.assertEqual(roundtrip(f'<svg {S} xml:space="preserve"><g> <g xml:space="default"> <text>t</text> </g> <text>t</text> </g></svg>'),
                         f'<svg {S} xml:space="preserve"><g> <g xml:space="default"><text>t</text></g> <text>t</text> </g></svg>')

    def test_escapes(self):
        self.assertEqual(roundtrip(f'<svg {S}><text>a&#13;b ]]&gt; c</text></svg>'), f'<svg {S}><text>a&#13;b ]]&gt; c</text></svg>')
        root, preceding = parse_svg(f'<svg {S}><g id="a&#10;b&#9;c&#13;d\'&quot;&quot;"/></svg>'.encode())
        self.assertEqual(serialize(root, preceding).decode(), f'<svg {S}><g id=\'a&#10;b&#9;c&#13;d&apos;""\'/></svg>')

    def test_namespace_declarations(self):
        doc = f'<svg {S}><g xmlns:a="urn:a"><a:x a:y="1"/></g><foreignObject><p xmlns="urn:p"/></foreignObject></svg>'
        self.assertEqual(roundtrip(doc), doc)

    def test_default_namespace_undeclared(self):
        doc = f'<svg {S}><g xmlns=""><a/><svg {S}/></g></svg>'
        self.assertEqual(roundtrip(doc), doc)
        root = etree.fromstring(roundtrip(doc).encode())
        self.assertEqual([e.tag for e in root.iter()], [f"{{{SVG_NS}}}svg", "g", "a", f"{{{SVG_NS}}}svg"])

    def test_element_cost(self):
        root, _ = parse_svg(f'<svg {S}><path d="M0 0h1" fill="red"/></svg>'.encode())
        self.assertEqual(element_cost(root[0]), len('<path d="M0 0h1" fill="red">'))
        self.assertEqual(element_cost(root), len(f'<svg {S}>'))


class LookupTests(unittest.TestCase):
    def test_declared(self):
        root, _ = parse_svg(f'<svg {S}><path style="fill:red;fill: blue" stroke=" red "/><path style="/*x*/"/><path fill=" red"/></svg>'.encode())
        first, second, third = list(root)
        self.assertEqual(declared(first, "fill"), "blue")
        self.assertEqual(declared(first, "stroke"), " red ")
        self.assertIsNone(declared(second, "fill"))
        self.assertEqual(declared(third, "fill"), " red")
        self.assertEqual(declared(third, "stroke"), "")

    def test_defined_ids_first_wins(self):
        root, _ = parse_svg(f'<svg {S}><path id="a"/><g><path id="a"/><path id="b"/></g></svg>'.encode())
        defined = defined_ids(root)
        self.assertEqual(list(defined), ["a", "b"])
        self.assertIs(defined["a"], root[0])
