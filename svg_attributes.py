"""Attribute rewrites: stylesheet and style to presentation attributes, shortest values, compact geometry, default removal."""
import copy

import cssselect2
import tinycss2
from lxml import etree

from .svg_document import (HREFS, NAME, SVG_NS, document_facts, element_cost, has_foreign_object, local_name, namespace, remove_node,
                           style_sets, svg_elements, use_targets)
from .svg_path import rewrite_path
from .svg_values import (GEOMETRY_DEFAULTS, INHERITED_DEFAULTS, LENGTH_ATTRIBUTES, OTHER_DEFAULTS,
                         PRESENTATION_ATTRIBUTES, convert_colors, format_number, format_style, format_transform, is_default, number_list,
                         parse_number, parse_style, parse_transform, shortest_color, shortest_length)

CONDITIONAL = {"requiredFeatures", "requiredExtensions", "systemLanguage"}
TRANSFORMS = ("transform", "gradientTransform", "patternTransform")
NUMBER_LISTS = ("points", "viewBox", "stroke-dasharray")
CSS_ONLY = ("initial", "unset", "revert", "revert-layer")


def styles_to_attributes(root):
    for element in svg_elements(root):
        declarations = parse_style(element.get("style", ""))
        if not declarations or any(name not in PRESENTATION_ATTRIBUTES or value.lower() in CSS_ONLY or "var(" in value.lower() for name, value in declarations):
            continue
        saved = list(element.attrib.items())
        cost = element_cost(element)
        for name, value in dict(declarations).items():
            element.set(name, value)
        del element.attrib["style"]
        if element_cost(element) > cost:
            element.attrib.clear()
            element.attrib.update(saved)


def clean_values(root, round_alpha=False):
    for element in svg_elements(root):
        for name, value in list(element.attrib.items()):
            if name in LENGTH_ATTRIBUTES:
                value = shortest_length(value)
            elif name == "class":
                value = " ".join(NAME.findall(value))
            else:
                value = convert_colors(name, value, lambda color: shortest_color(color, round_alpha))
            if value == "" and name not in CONDITIONAL and not name.startswith("{") and not (name in TRANSFORMS[1:] and any(href in element.attrib for href in HREFS)):
                del element.attrib[name]
            else:
                element.set(name, value)


def format_geometry(root, written):
    for element in svg_elements(root):
        if local_name(element) == "path" and "d" in element.attrib:
            d = element.get("d")
            text = rewrite_path(d, written)
            if text is not None and len(text) <= len(d):
                element.set("d", text)
        for name in TRANSFORMS:
            items = parse_transform(element.get(name)) if name in element.attrib else None
            if items is not None:
                text = format_transform(items)
                if not text and name != "transform" and any(href in element.attrib for href in HREFS):
                    text = "scale(1)"
                if text:
                    element.set(name, text)
                else:
                    del element.attrib[name]
        for name in NUMBER_LISTS:
            values = number_list(element.get(name, ""))
            if values is not None:
                element.set(name, " ".join(format_number(value) for value in values))


def set_names(element, cache):
    """The inherited property names that the element or one of its ancestors sets; a style that cannot be parsed sets all of them."""
    if element not in cache:
        declarations = parse_style(element.get("style", ""))
        own = INHERITED_DEFAULTS.keys() if declarations is None else {name for name, _ in declarations}.union(element.attrib)
        parent = element.getparent()
        cache[element] = own | (set_names(parent, cache) if parent is not None else set())
    return cache[element]


def remove_defaults(root, stylesheet):
    used = use_targets(root)
    cache = {}
    for element in svg_elements(root):
        inherited = {}
        ancestors = list(element.iterancestors())
        if not stylesheet and not any(node.get("id") in used for node in [element, *ancestors]):
            above = set_names(ancestors[0], cache) if ancestors else ()
            inherited = {name: value for name, value in INHERITED_DEFAULTS.items() if name not in above}
        defaults = OTHER_DEFAULTS | inherited
        for name, default in (defaults | GEOMETRY_DEFAULTS.get(local_name(element), {})).items():
            if name in element.attrib and is_default(name, element.get(name), default):
                del element.attrib[name]
        declarations = parse_style(element.get("style", ""))
        if local_name(element) == "rect" and not stylesheet and not style_sets(element, ("rx", "ry")):
            radii = [parse_number(element.get(name, "0")) for name in ("rx", "ry")]
            if radii[0] == 0 and radii[1] == 0:
                element.attrib.pop("rx", None)
                element.attrib.pop("ry", None)
        if declarations and not stylesheet:
            final = dict(declarations)
            kept = [(name, value) for name, value in declarations
                    if name in element.attrib or name not in defaults or not is_default(name, final[name], defaults[name])]
            if len(kept) < len(declarations):
                if kept:
                    element.set("style", format_style(kept))
                else:
                    del element.attrib["style"]


def read_rules(styles):
    """The (selector, declarations) pairs of the style elements in source order, or None when a stylesheet holds more than plain rules."""
    rules = []
    for style in styles:
        if style.get("type", "text/css") != "text/css" or "media" in style.attrib:
            return None
        for rule in tinycss2.parse_stylesheet(style.text or "", skip_comments=True, skip_whitespace=True):
            if rule.type != "qualified-rule" or any(token.type == "literal" and token.value == ":" for token in rule.prelude):
                return None
            declarations = tinycss2.parse_declaration_list(rule.content, skip_comments=True, skip_whitespace=True)
            if any(item.type != "declaration" or item.important or not item.value or "var(" in tinycss2.serialize(item.value).lower() for item in declarations):
                return None
            try:
                selectors = cssselect2.compile_selector_list(rule.prelude)
            except cssselect2.SelectorError:
                return None
            rules.extend((selector, declarations) for selector in selectors)
    return rules


def inline_stylesheet(root, preceding):
    """A copy of the tree without its `<style>` elements and `class` attributes, whose rules are declared in the `style` of each element they match,
    or None when the document is not one whose stylesheet is plain rules in `<style>` elements with no script or animation."""
    facts = document_facts(root, preceding)
    styles = [element for element in root.iter(etree.Element) if local_name(element) == "style"]
    if not styles or preceding or facts.script or facts.animation or has_foreign_object(root) or not all(namespace(style) == SVG_NS for style in styles):
        return None
    rules = read_rules(styles)
    if rules is None:
        return None
    matcher = cssselect2.Matcher()
    for selector, declarations in rules:
        matcher.add_selector(selector, declarations)
    root = copy.deepcopy(root)
    for wrapper in cssselect2.ElementWrapper.from_xml_root(root).iter_subtree():
        element = wrapper.etree_element
        # A later rule of equal specificity, and a higher one, wins.
        values = {item.lower_name: tinycss2.serialize(item.value).strip() for _, _, _, declarations in matcher.match(wrapper) for item in declarations}
        own = element.get("style", "")
        declared = parse_style(own) or []
        names = {name for name, _ in declared}
        rules_style = format_style([(name, value) for name, value in values.items() if name not in names])
        if rules_style:
            element.set("style", f"{rules_style};{own}" if own else rules_style)
    for element in list(root.iter(etree.Element)):
        element.attrib.pop("class", None)
        if local_name(element) == "style":
            remove_node(element)
    return root
