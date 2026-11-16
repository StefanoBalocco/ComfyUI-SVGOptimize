"""Inlining of a plain stylesheet: its rules move into the style attributes of the elements they match."""
import copy

import cssselect2
import tinycss2
from lxml import etree

from .svg_document import SVG_NS, document_facts, has_foreign_object, local_name, namespace, remove_node
from .svg_values import format_style, parse_style


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
