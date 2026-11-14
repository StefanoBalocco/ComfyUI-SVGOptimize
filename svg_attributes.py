"""Attribute rewrites: style to presentation attributes, shortest values, compact geometry, default removal."""
from decimal import Decimal

from .svg_document import NAME, element_cost, local_name, style_sets, svg_elements, use_targets
from .svg_numbers import NUMBER, SEPARATOR, SPACE, format_number, parse_number
from .svg_path import format_path, parse_path
from .svg_transform import format_transform, parse_transform
from .svg_values import (COLOR_PROPERTIES, GEOMETRY_DEFAULTS, INHERITED_DEFAULTS, LENGTH_ATTRIBUTES, OTHER_DEFAULTS,
                         PRESENTATION_ATTRIBUTES, format_style, is_default, parse_style, shortest_color,
                         shortest_length)

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


def clean_values(root):
    for element in svg_elements(root):
        for name, value in list(element.attrib.items()):
            if name in COLOR_PROPERTIES:
                value = shortest_color(value)
            elif name in LENGTH_ATTRIBUTES:
                value = shortest_length(value)
            elif name == "class":
                value = " ".join(NAME.findall(value))
            elif name == "style":
                declarations = parse_style(value)
                if declarations is not None:
                    value = format_style([(key, shortest_color(item) if key in COLOR_PROPERTIES else item) for key, item in declarations])
            if value == "" and name not in CONDITIONAL and not name.startswith("{"):
                del element.attrib[name]
            else:
                element.set(name, value)


def number_list(text):
    """The numbers of a whitespace or comma separated list, or None."""
    values = []
    position = SPACE.match(text).end()
    while position < len(text):
        match = NUMBER.match(text, position)
        if match is None:
            return None
        values.append(Decimal(match.group()))
        separator = SEPARATOR.match(text, match.end())
        if "," in separator.group() and separator.end() == len(text):
            return None
        position = separator.end()
    return values or None


def format_geometry(root):
    for element in svg_elements(root):
        if local_name(element) == "path" and "d" in element.attrib:
            d = element.get("d")
            segments = parse_path(d)
            text = None if segments is None else format_path(segments)
            if text is not None and len(text) <= len(d):
                element.set("d", text)
        for name in TRANSFORMS:
            items = parse_transform(element.get(name)) if name in element.attrib else None
            if items is not None:
                text = format_transform(items)
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
