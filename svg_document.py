"""Parsing, cleanup and compact serialization of SVG documents."""
import re
from dataclasses import dataclass

from lxml import etree

from .svg_values import format_style, parse_style

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
XML_NS = "http://www.w3.org/XML/1998/namespace"
HREFS = ("href", f"{{{XLINK_NS}}}href")

METADATA_NS = {
    "http://purl.org/dc/elements/1.1/",
    "http://creativecommons.org/ns#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
}
EDITOR_NS = METADATA_NS | {
    "http://www.inkscape.org/namespaces/inkscape",
    "http://inkscape.sourceforge.net/DTD/sodipodi-0.dtd",
    "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "http://ns.adobe.com/AdobeIllustrator/10.0/",
    "http://ns.adobe.com/AdobeSVGViewerExtensions/3.0/",
    "http://ns.adobe.com/Extensibility/1.0/",
    "http://ns.adobe.com/Flows/1.0/",
    "http://ns.adobe.com/ImageReplacement/1.0/",
    "http://ns.adobe.com/GenericCustomNamespace/1.0/",
    "http://ns.adobe.com/XPath/1.0/",
    "http://schemas.microsoft.com/visio/2003/SVGExtensions/",
    "http://taptrix.com/vectorillustrator/svg_extensions",
    "http://www.figma.com/figma/ns",
    "http://www.serif.com/",
    "http://www.vector.evaxdesign.sk",
    "http://www.bohemiancoding.com/sketch/ns",
}
TEXT_ELEMENTS = {"text", "tspan", "textPath"}
PRESERVING = TEXT_ELEMENTS | {"style", "script", "title", "desc", "metadata", "foreignObject"}
ANIMATION = {"animate", "animateColor", "animateMotion", "animateTransform", "set", "discard"}
WHITESPACE = " \t\r\n"
NAME = re.compile(r"[^ \t\r\n\f]+")


@dataclass(frozen=True)
class Facts:
    stylesheet: bool
    script: bool
    animation: bool


def local_name(element):
    return etree.QName(element).localname


def namespace(element):
    return etree.QName(element).namespace


def parse_svg(data):
    parser = etree.XMLParser(resolve_entities="internal", no_network=True, remove_comments=True)
    try:
        root = etree.fromstring(data, parser)
    except etree.XMLSyntaxError as error:
        raise ValueError(f"not well-formed XML: {error}") from error
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError("the root element is not svg")
    dtd = root.getroottree().docinfo.internalDTD
    if dtd is not None and any("<" in (entity.content or "") for entity in dtd.entities()):
        raise ValueError("an internal entity contains markup")
    preceding = [node for node in root.itersiblings(preceding=True) if node.tag is etree.ProcessingInstruction and node.target == "xml-stylesheet"]
    return root, preceding[::-1]


def elements(root):
    """Every element in document order, without the content of foreignObject."""
    stack = [root]
    while stack:
        element = stack.pop()
        yield element
        if local_name(element) != "foreignObject":
            stack.extend(child for child in reversed(element) if isinstance(child.tag, str))


def style_sets(element, names):
    """True when the style declares one of the names or cannot be parsed."""
    declarations = parse_style(element.get("style", ""))
    return declarations is None or any(name in names for name, _ in declarations)


def has_foreign_object(root):
    return any(local_name(element) == "foreignObject" for element in elements(root))


def is_switch_child(element):
    parent = element.getparent()
    return parent is not None and namespace(parent) == SVG_NS and local_name(parent) == "switch"


def defined_ids(root):
    """The first element for each id."""
    defined = {}
    for element in elements(root):
        if "id" in element.attrib:
            defined.setdefault(element.get("id"), element)
    return defined


def declared(element, name):
    """The value of the last `name` declaration in the style, else of the attribute; None when the style cannot be parsed."""
    return declared_in(element, name, parse_style(element.get("style", "")))


def declared_in(element, name, declarations):
    """`declared` for the already parsed style of the element."""
    if declarations is None:
        return None
    values = [value for key, value in declarations if key == name]
    return values[-1] if values else element.get(name, "")


def use_targets(root):
    """The ids that a `use` element refers to."""
    return {href[1:] for element in svg_elements(root) if local_name(element) == "use"
            for href in map(element.get, HREFS) if href and href.startswith("#")}


def svg_elements(root):
    return (element for element in elements(root) if namespace(element) == SVG_NS)


def document_facts(root, preceding):
    stylesheet = bool(preceding)
    script = animation = False
    for element in root.iter(etree.Element):
        name = local_name(element)
        stylesheet |= name == "style"
        script |= name == "script" or any(key.startswith("on") for key in element.attrib)
        animation |= name in ANIMATION and namespace(element) == SVG_NS
    return Facts(stylesheet, script, animation)


def remove_node(node):
    """Remove a node; its tail text stays in the document."""
    tail = node.tail
    if tail:
        previous = node.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + tail
        else:
            parent = node.getparent()
            parent.text = (parent.text or "") + tail
    node.getparent().remove(node)


def in_metadata(element):
    return any(local_name(ancestor) == "metadata" and namespace(ancestor) == SVG_NS for ancestor in element.iterancestors())


def clean_document(root, keep_metadata):
    for element in list(elements(root)):
        for child in list(element):
            if child.tag is etree.ProcessingInstruction:
                remove_node(child)
        is_svg = namespace(element) == SVG_NS
        name = local_name(element)
        if is_svg and not keep_metadata and name in ("metadata", "title", "desc") and not is_switch_child(element):
            remove_node(element)
            continue
        inside_metadata = None
        if namespace(element) in EDITOR_NS:
            inside_metadata = in_metadata(element)
            if not (inside_metadata and namespace(element) in METADATA_NS):
                remove_node(element)
                continue
        for key in list(element.attrib):
            if key.startswith("{"):
                attribute_ns = key[1:].partition("}")[0]
                if attribute_ns in EDITOR_NS:
                    if inside_metadata is None:
                        inside_metadata = in_metadata(element)
                    if not (inside_metadata and attribute_ns in METADATA_NS):
                        del element.attrib[key]
        for key in ("data-name", "enable-background"):
            element.attrib.pop(key, None)
        if element is root:
            for key in ("version", "baseProfile"):
                element.attrib.pop(key, None)
        style = element.get("style")
        if style is not None and "enable-background" in style:
            declarations = parse_style(style)
            if declarations is not None:
                declarations = [declaration for declaration in declarations if declaration[0] != "enable-background"]
                if declarations:
                    element.set("style", format_style(declarations))
                else:
                    del element.attrib["style"]
        if f"{{{XML_NS}}}space" in element.attrib and not any(local_name(node) in TEXT_ELEMENTS for node in element.iter(etree.Element)):
            del element.attrib[f"{{{XML_NS}}}space"]


def escape_text(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace("]]>", "]]&gt;").replace("\r", "&#13;")


def quoted(value):
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "&#10;").replace("\r", "&#13;").replace("\t", "&#9;")
    if value.count('"') > value.count("'"):
        return "'" + value.replace("'", "&apos;") + "'"
    return '"' + value.replace('"', "&quot;") + '"'


def default_namespace(element):
    """The default namespace in scope below `element` in the output: that of its nearest unprefixed ancestor-or-self."""
    while element is not None:
        if element.prefix is None:
            return namespace(element) or ""
        element = element.getparent()
    return ""


def qualified_name(element):
    return local_name(element) if element.prefix is None else f"{element.prefix}:{local_name(element)}"


def start_tag(element):
    """The start tag, written as `<name attributes>`, with the namespace declarations the element adds."""
    nsmap = element.nsmap
    parent = element.getparent()
    parent_nsmap = parent.nsmap if parent is not None else {}
    parts = [qualified_name(element)]
    if element.prefix is None:
        default = namespace(element) or ""
        if default != default_namespace(parent):
            parts.append(f'xmlns="{default}"')
    for prefix, uri in nsmap.items():
        if prefix is not None and parent_nsmap.get(prefix) != uri:
            parts.append(f'xmlns:{prefix}="{uri}"')
    for key, value in element.attrib.items():
        if key.startswith("{"):
            uri, _, local = key[1:].partition("}")
            prefix = "xml" if uri == XML_NS else next(prefix for prefix, mapped in nsmap.items() if mapped == uri and prefix)
            key = f"{prefix}:{local}"
        parts.append(f"{key}={quoted(value)}")
    return "<" + " ".join(parts) + ">"


def element_cost(element):
    return len(start_tag(element).encode())


def write_element(element, out, kept, space):
    name = local_name(element)
    kept = kept or name in PRESERVING
    space = {"preserve": True, "default": False}.get(element.get(f"{{{XML_NS}}}space"), space)
    preserve = kept or space
    text = element.text
    if text and not preserve and not text.strip(WHITESPACE):
        text = None
    tag = start_tag(element)
    if text is None and len(element) == 0:
        out.append(tag[:-1] + "/>")
        return
    out.append(tag)
    if text:
        out.append(escape_text(text))
    for child in element:
        if child.tag is etree.ProcessingInstruction:
            out.append(f"<?{child.target} {child.text}?>" if child.text else f"<?{child.target}?>")
        else:
            write_element(child, out, kept, space)
        tail = child.tail
        if tail and (preserve or tail.strip(WHITESPACE)):
            out.append(escape_text(tail))
    out.append(f"</{qualified_name(element)}>")


def serialize(root, preceding):
    out = [f"<?{node.target} {node.text}?>" if node.text else f"<?{node.target}?>" for node in preceding]
    write_element(root, out, False, False)
    return "".join(out).encode()
