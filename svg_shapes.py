"""Basic shapes written as paths or circles, and plain translates baked into coordinates."""
from decimal import Decimal, Inexact

from .svg_attributes import number_list
from .svg_document import SVG_NS, element_cost, local_name, style_sets, svg_elements, use_targets
from .svg_numbers import EXACT, format_number, parse_number
from .svg_path import format_path, parse_path, translate_path
from .svg_transform import as_translate, parse_transform

MARKERS = ("marker-start", "marker-mid", "marker-end", "marker")
COORDINATES = {"line": ("x1", "y1", "x2", "y2"), "polyline": ("points",), "polygon": ("points",),
               "rect": ("x", "y", "width", "height", "rx", "ry"), "ellipse": ("rx", "ry")}
WRITTEN = {"path": ("d",), "polyline": ("points",), "polygon": ("points",), "line": COORDINATES["line"],
           "rect": COORDINATES["rect"], "circle": ("cx", "cy", "r"), "ellipse": ("cx", "cy", "rx", "ry")}
STYLE_OVERRIDES = ("d", "r", "transform")
TRANSLATED = {"rect": (("x", 0), ("y", 1)), "circle": (("cx", 0), ("cy", 1)), "ellipse": (("cx", 0), ("cy", 1)),
              "line": (("x1", 0), ("y1", 1), ("x2", 0), ("y2", 1))}


def coordinate(element, name):
    """The plain number in an attribute (0 when absent), or None."""
    return parse_number(element.get(name)) if name in element.attrib else Decimal(0)


def has_markers(root):
    return any(name in MARKERS or (name == "style" and "marker" in value)
               for element in root.iter() for name, value in element.attrib.items())


def points_segments(element, close):
    values = number_list(element.get("points", ""))
    if values is None or len(values) < 4 or len(values) % 2:
        return None
    segments = [("M" if index == 0 else "L", (values[index], values[index + 1])) for index in range(0, len(values), 2)]
    return segments + [("Z", ())] if close else segments


def rect_segments(element):
    x, y, width, height = (coordinate(element, name) for name in ("x", "y", "width", "height"))
    if None in (x, y, width, height) or width <= 0 or height <= 0:
        return None
    right, bottom = EXACT.add(x, width), EXACT.add(y, height)
    return [("M", (x, y)), ("L", (right, y)), ("L", (right, bottom)), ("L", (x, bottom)), ("Z", ())]


def line_segments(element):
    x1, y1, x2, y2 = (coordinate(element, name) for name in COORDINATES["line"])
    if None in (x1, y1, x2, y2):
        return None
    return [("M", (x1, y1)), ("L", (x2, y2))]


def convert_shape(element, name, rects):
    if name == "ellipse":
        rx, ry = (parse_number(element.get(key, "")) for key in ("rx", "ry"))
        if rx is None or rx != ry:
            return False
        element.tag = f"{{{SVG_NS}}}circle"
        element.set("r", format_number(rx))
    else:
        if name == "line":
            segments = line_segments(element)
        elif name == "rect":
            segments = rect_segments(element) if rects and "rx" not in element.attrib and "ry" not in element.attrib else None
        else:
            segments = points_segments(element, name == "polygon")
        text = None if segments is None else format_path(segments)
        if text is None:
            return False
        element.tag = f"{{{SVG_NS}}}path"
        element.set("d", text)
    for key in COORDINATES[name]:
        element.attrib.pop(key, None)
    return True


def convert_shapes(root):
    rects = not has_markers(root)
    for element in [element for element in svg_elements(root) if local_name(element) in COORDINATES]:
        name = local_name(element)
        if style_sets(element, WRITTEN[name] + STYLE_OVERRIDES):
            continue
        tag, saved, cost = element.tag, list(element.attrib.items()), element_cost(element)
        try:
            converted = convert_shape(element, name, rects)
        except Inexact:
            converted = False
        if not converted or element_cost(element) >= cost:
            element.tag = tag
            element.attrib.clear()
            element.attrib.update(saved)


def translate_element(element, name, dx, dy):
    if name == "path":
        segments = parse_path(element.get("d", ""))
        shifted = None if segments is None else translate_path(segments, dx, dy)
        text = None if shifted is None else format_path(shifted)
        if text is None:
            return False
        element.set("d", text)
    elif name in ("polyline", "polygon"):
        values = number_list(element.get("points", ""))
        if values is None or len(values) % 2:
            return False
        element.set("points", " ".join(format_number(EXACT.add(value, (dx, dy)[index % 2])) for index, value in enumerate(values)))
    else:
        values = [(key, coordinate(element, key), (dx, dy)[axis]) for key, axis in TRANSLATED[name]]
        if any(value is None for _, value, _ in values):
            return False
        for key, value, offset in values:
            shifted = EXACT.add(value, offset)
            if key in element.attrib or shifted != 0:
                element.set(key, format_number(shifted))
    del element.attrib["transform"]
    return True


def apply_translates(root):
    used = use_targets(root)
    for element in [element for element in svg_elements(root) if local_name(element) in WRITTEN]:
        items = parse_transform(element.get("transform", ""))
        offset = None if items is None else as_translate(items)
        if offset is None or style_sets(element, WRITTEN[local_name(element)] + STYLE_OVERRIDES):
            continue
        ancestry = (element, *element.iterancestors())
        if any(node.get("id") in used or any("url(" in value.lower() for value in node.attrib.values()) for node in ancestry):
            continue
        saved, cost = list(element.attrib.items()), element_cost(element)
        try:
            baked = translate_element(element, local_name(element), *offset)
        except Inexact:
            baked = False
        if not baked or element_cost(element) >= cost:
            element.attrib.clear()
            element.attrib.update(saved)
