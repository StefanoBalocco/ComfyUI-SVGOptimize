"""Removal of unused definitions, ids and elements that draw nothing, shortening of the referenced ids, and painting of one-stop gradients as their color."""
import math
import re
from collections import Counter
from decimal import Inexact, InvalidOperation
from itertools import count, product
from string import ascii_letters

from lxml import etree

from .svg_document import HREFS, NAME, SVG_NS, TEXT_ELEMENTS, WHITESPACE, declared, defined_ids, elements, has_foreign_object, is_switch_child, local_name, namespace, remove_node, style_sets, svg_elements, use_targets
from .svg_shapes import has_markers
from .svg_values import EXACT, INHERITED_PROPERTIES, format_style, is_zero_length, parse_color, parse_number, parse_style, parse_transform, shortest_color

DEFINITIONS = {"linearGradient", "radialGradient", "pattern", "clipPath", "mask", "marker", "symbol", "filter"}
KEPT = {"title", "desc", "metadata", "font", "font-face", "color-profile"}
ARIA_LISTS = ("aria-labelledby", "aria-describedby")
URL_REFERENCE = re.compile(r"""url\([ \t\r\n\f]*(?:"#([^"]*)"|'#([^']*)'|#([^\s)"']*))[ \t\r\n\f]*\)""", re.IGNORECASE)
NON_GRAPHICS = {"title", "desc", "metadata"}
MERGEABLE_CHILDREN = {"g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "image", "use", "a", "switch"}

EFFECTS = {"filter", "mask", "clip-path"}
ZERO_SIZE_SOURCES = {"d", "points", "width", "height", "r", "rx", "ry"}
ZERO_SIZE = {"rect": ("width", "height"), "circle": ("r",), "ellipse": ("rx", "ry")}
EMPTY_DATA = {"path": "d", "polyline": "points"}
PAINTS = ("fill", "stroke")
GRADIENTS = {"linearGradient", "radialGradient"}
DISPLAYED = {"svg", "g", "switch", "a", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text"}
MARKED = {"path", "line", "polyline", "polygon"}


def references(element):
    """Every id name the attributes of `element` refer to, once per reference."""
    for key, value in element.attrib.items():
        if key == "id":
            continue
        for match in URL_REFERENCE.finditer(value):
            yield next(group for group in match.groups() if group is not None)
        if key in HREFS and value.startswith("#"):
            yield value[1:]
        elif key in ARIA_LISTS:
            yield from NAME.findall(value)


def short_names(skipped):
    for length in count(1):
        for letters in product(ascii_letters, repeat=length):
            name = "".join(letters)
            if name not in skipped:
                yield name


def is_empty(element):
    return len(element) == 0 and not (element.text or "").strip(WHITESPACE)


def is_removable(element, needed):
    if namespace(element) != SVG_NS or element in needed:
        return False
    name = local_name(element)
    if name in KEPT:
        return False
    if is_switch_child(element):
        return False
    parent = element.getparent()
    if name in DEFINITIONS or (parent is not None and local_name(parent) == "defs" and namespace(parent) == SVG_NS):
        return True
    return name in ("g", "defs") and is_empty(element)


def clean_ids(root):
    if has_foreign_object(root):
        return
    while True:
        counts = Counter(name for element in elements(root) for name in references(element))
        defined = defined_ids(root)
        needed = set()
        for name in counts:
            if name in defined:
                needed.add(defined[name])
                needed.update(defined[name].iterancestors())
        removable = [element for element in elements(root) if element is not root and is_removable(element, needed)]
        if not removable:
            break
        for element in removable:
            remove_node(element)
    order = {element: index for index, element in enumerate(elements(root))}
    used = sorted((name for name in counts if name in defined), key=lambda name: (-counts[name], order[defined[name]]))
    new_names = dict(zip(used, short_names(set(counts) - set(defined))))
    for element in elements(root):
        if "id" in element.attrib:
            name = element.get("id")
            if name in new_names and defined[name] is element:
                element.set("id", new_names[name])
            else:
                del element.attrib["id"]
        for key, value in element.attrib.items():
            if key == "id":
                continue
            rewritten = URL_REFERENCE.sub(lambda match: f"url(#{new_names[name]})" if (name := next(group for group in match.groups() if group is not None)) in new_names else match.group(), value)
            if key in HREFS and value.startswith("#") and value[1:] in new_names:
                rewritten = "#" + new_names[value[1:]]
            elif key in ARIA_LISTS:
                rewritten = NAME.sub(lambda match: new_names.get(match.group(), match.group()), value)
            if rewritten != value:
                element.set(key, rewritten)


def is_hidden(element):
    return declared(element, "display") == "none"


def is_zero_size(element):
    if style_sets(element, ZERO_SIZE_SOURCES):
        return False
    name = local_name(element)
    if name in EMPTY_DATA:
        return not element.get(EMPTY_DATA[name], "").strip(WHITESPACE)
    return any(is_zero_length(element.get(attribute, "")) for attribute in ZERO_SIZE.get(name, ()))


def remove_invisible(root):
    if has_foreign_object(root):
        return
    markers = has_markers(root)
    targets = use_targets(root)
    referrers = {}
    for element in elements(root):
        for name in references(element):
            referrers.setdefault(name, []).append(element)
    affected = set()
    candidates = []
    for element in elements(root):
        parent = element.getparent()
        if parent in affected or element.get("id") in targets or any(name in element.attrib for name in EFFECTS) or style_sets(element, EFFECTS):
            affected.add(element)
        if element is root or namespace(element) != SVG_NS or element in affected:
            continue
        if is_switch_child(element):
            continue
        name = local_name(element)
        if name in DISPLAYED and not (markers and name in MARKED) and is_hidden(element):
            if any(local_name(ancestor) in TEXT_ELEMENTS for ancestor in element.iterancestors()):
                continue
        elif not is_zero_size(element):
            continue
        subtree = set(element.iter(etree.Element))
        if any(local_name(inner) in NON_GRAPHICS for inner in subtree):
            continue
        if any(outside not in subtree for inner in subtree if "id" in inner.attrib for outside in referrers.get(inner.get("id"), ())):
            continue
        candidates.append(element)
    removed = set()
    for element in candidates:
        if not removed.intersection(element.iterancestors()):
            removed.add(element)
            remove_node(element)


def element_children(group):
    return [child for child in group if isinstance(child.tag, str)]


def hoist_common_attributes(group):
    children = [child for child in element_children(group) if local_name(child) not in NON_GRAPHICS]
    if len(children) < 2 or "style" in group.attrib or any("id" in child.attrib for child in children):
        return
    styled = set()
    for child in children:
        declarations = parse_style(child.get("style", ""))
        if declarations is None:
            return
        styled.update(name for name, _ in declarations)
    for name in sorted(INHERITED_PROPERTIES - styled - set(group.attrib)):
        value = children[0].get(name)
        if value is not None and all(child.get(name) == value for child in children):
            group.set(name, value)
            for child in children:
                del child.attrib[name]


def unwrap(group):
    parent = group.getparent()
    children = list(group)
    text = group.text or ""
    if children:
        children[-1].tail = (children[-1].tail or "") + (group.tail or "")
    else:
        text += group.tail or ""
    if text:
        previous = group.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + text
        else:
            parent.text = (parent.text or "") + text
    for child in children:
        group.addprevious(child)
    parent.remove(group)


def can_merge_into(group, child):
    if namespace(child) != SVG_NS or local_name(child) not in MERGEABLE_CHILDREN or "id" in child.attrib:
        return False
    if style_sets(child, group.attrib) or any(name in child.attrib for name in group.attrib if name != "transform"):
        return False
    return "transform" not in group.attrib or "transform" not in child.attrib \
        or (parse_transform(group.get("transform")) is not None and parse_transform(child.get("transform")) is not None)


def collapse_groups(root):
    groups = [element for element in svg_elements(root) if element is not root and local_name(element) == "g"]
    for group in reversed(groups):
        parent = group.getparent()
        if namespace(parent) == SVG_NS and local_name(parent) in ("switch", "clipPath"):
            continue
        hoist_common_attributes(group)
        children = element_children(group)
        if not group.attrib:
            if not any(local_name(child) in NON_GRAPHICS for child in children):
                unwrap(group)
            continue
        if len(children) != 1 or any(name != "transform" and name not in INHERITED_PROPERTIES for name in group.attrib) or not can_merge_into(group, children[0]):
            continue
        child = children[0]
        for name, value in group.attrib.items():
            if name == "transform" and "transform" in child.attrib:
                child.set(name, f"{value} {child.get(name)}")
            else:
                child.set(name, value)
        unwrap(group)


def is_invertible(items):
    determinant = 1.0
    try:
        for name, values in items:
            if name == "matrix":
                a, b, c, d = map(float, values[:4])
                determinant *= a * d - b * c
            elif name == "scale":
                determinant *= float(values[0]) * float(values[-1])
            elif name in ("skewX", "skewY") and abs(EXACT.remainder(values[0], 180)) == 90:
                return False
    except (Inexact, InvalidOperation):
        return False
    return 0 < abs(determinant) < math.inf and math.isfinite(1 / determinant)


def gradient_color(gradient, defined):
    if local_name(gradient) != "linearGradient":
        return None
    stops = None
    units = None
    transform = None
    seen = set()
    while True:
        if gradient in seen or namespace(gradient) != SVG_NS or local_name(gradient) not in GRADIENTS or style_sets(gradient, ("transform",)):
            return None
        if any(namespace(child) != SVG_NS or local_name(child) != "stop" for child in gradient.iterchildren(etree.Element)):
            return None
        if len({gradient.get(name) for name in HREFS} - {None}) > 1:
            return None
        seen.add(gradient)
        if stops is None:
            children = [child for child in gradient if namespace(child) == SVG_NS and local_name(child) == "stop"]
            stops = children or None
        units = gradient.get("gradientUnits") if units is None else units
        transform = gradient.get("gradientTransform") if transform is None else transform
        href = gradient.get(HREFS[0], gradient.get(HREFS[1]))
        if href is None:
            break
        if not href.startswith("#") or href[1:] not in defined:
            return None
        gradient = defined[href[1:]]
    if stops is None or len(stops) != 1 or units != "userSpaceOnUse":
        return None
    if transform is not None:
        items = parse_transform(transform)
        if items is None or not is_invertible(items):
            return None
    stop = stops[0]
    color = declared(stop, "stop-color")
    opacity = declared(stop, "stop-opacity")
    if color is None or opacity is None:
        return None
    color = color.strip() or "#000"
    if parse_color(color) is None or (opacity.strip() and parse_number(opacity.strip()) != 1):
        return None
    return shortest_color(color)


def painted(value, colors):
    """The color a gradient reference stands for, or None."""
    match = URL_REFERENCE.match(value)
    if match is None:
        return None
    fallback = value[match.end():]
    separated = not fallback or fallback[0].isspace()
    if not separated or (fallback.strip().lower() not in ("", "none", "currentcolor") and parse_color(fallback.strip()) is None):
        return None
    return colors.get(next(group for group in match.groups() if group is not None))


def paint_one_stop_gradients(root):
    if has_foreign_object(root):
        return
    defined = defined_ids(root)
    colors = {}
    for name, element in defined.items():
        color = gradient_color(element, defined)
        if color is not None:
            colors[name] = color
    if not colors:
        return
    for element in svg_elements(root):
        for name in PAINTS:
            color = painted(element.get(name, "").strip(), colors)
            if color is not None:
                element.set(name, color)
        declarations = parse_style(element.get("style", ""))
        if declarations is not None:
            rewritten = [(name, (painted(value, colors) or value) if name in PAINTS else value) for name, value in declarations]
            if rewritten != declarations:
                element.set("style", format_style(rewritten))
