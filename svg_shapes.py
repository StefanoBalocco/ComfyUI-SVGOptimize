"""Basic shapes written as paths or circles, plain translates baked into coordinates, and removal of the stroke properties that draw nothing."""
from decimal import Decimal, Inexact

from .svg_attributes import number_list, set_names
from .svg_document import SVG_NS, declared_in, element_cost, local_name, style_sets, svg_elements, use_targets
from .svg_numbers import EXACT, format_number, parse_number
from .svg_path import format_path, parse_path, translate_path
from .svg_transform import as_translate, parse_transform
from .svg_values import format_style, is_zero_length, parse_color, parse_style, set_property, shortest_color

MARKERS = ("marker-start", "marker-mid", "marker-end", "marker")
COORDINATES = {"line": ("x1", "y1", "x2", "y2"), "polyline": ("points",), "polygon": ("points",),
               "rect": ("x", "y", "width", "height", "rx", "ry"), "ellipse": ("rx", "ry")}
WRITTEN = {"path": ("d",), "polyline": ("points",), "polygon": ("points",), "line": COORDINATES["line"],
           "rect": COORDINATES["rect"], "circle": ("cx", "cy", "r"), "ellipse": ("cx", "cy", "rx", "ry")}
STROKE_SHAPES = ("path", "rect", "circle", "ellipse", "line", "polyline", "polygon")
STROKE_PROPERTIES = ("stroke-opacity", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit",
                     "stroke-dasharray", "stroke-dashoffset")
STYLE_OVERRIDES = ("d", "r", "transform")
OPACITIES = (("fill", "fill-opacity"), ("stroke", "stroke-opacity"))
TRANSLATED = {"rect": (("x", 0), ("y", 1)), "circle": (("cx", 0), ("cy", 1)), "ellipse": (("cx", 0), ("cy", 1)),
              "line": (("x1", 0), ("y1", 1), ("x2", 0), ("y2", 1))}


def coordinate(element, name):
    """The plain number in an attribute (0 when absent), or None."""
    return parse_number(element.get(name)) if name in element.attrib else Decimal(0)


def has_markers(root):
    return any(name in MARKERS or (name == "style" and "marker" in value)
               for element in root.iter() for name, value in element.attrib.items())


def drop_properties(element, declarations, names):
    """Removes the properties from the attributes and from the parsed style of the element."""
    for name in names:
        element.attrib.pop(name, None)
    declarations = [(name, value) for name, value in declarations if name not in names]
    if declarations:
        element.set("style", format_style(declarations))
    else:
        element.attrib.pop("style", None)


def remove_unused_strokes(root):
    markers = has_markers(root)
    targets = use_targets(root)
    for element in svg_elements(root):
        if local_name(element) not in STROKE_SHAPES:
            continue
        declarations = parse_style(element.get("style", ""))
        width = declared_in(element, "stroke-width", declarations)
        if width is None:
            continue
        ancestors = list(element.iterancestors())
        can_stroke = any("stroke" in item.attrib or style_sets(item, ("stroke",)) for item in ancestors) or any(
            item.get("id") in targets for item in [element, *ancestors])
        paint = declared_in(element, "stroke", declarations)
        if paint == "none" or (paint == "" and not can_stroke):
            names = STROKE_PROPERTIES if markers else STROKE_PROPERTIES + ("stroke-width",)
        elif is_zero_length(width) and not markers:
            names = ("stroke",) + STROKE_PROPERTIES + (() if can_stroke else ("stroke-width",))
        else:
            continue
        drop_properties(element, declarations, names)


def merge_opacities(root, round_alpha=False):
    """Writes the fill-opacity and stroke-opacity of a shape into the alpha of its fill and stroke color, as hex, when the alpha is exact (or rounded to 1/255 with `round_alpha`) and the text is shorter."""
    used = use_targets(root)
    cache = {}
    for element in svg_elements(root):
        declarations = parse_style(element.get("style", ""))
        ancestors = list(element.iterancestors())
        if local_name(element) not in STROKE_SHAPES or declarations is None or any(node.get("id") in used for node in [element, *ancestors]):
            continue
        above = set_names(ancestors[0], cache) if ancestors else ()
        for paint, opacity in OPACITIES:
            color = parse_color(declared_in(element, paint, declarations).strip())
            alpha = parse_number(declared_in(element, opacity, declarations).strip())
            if opacity in above or color is None or alpha is None or not 0 <= alpha <= 1:
                continue
            try:
                merged = EXACT.multiply(color[3], alpha)
            except Inexact:
                continue
            if merged != merged.to_integral_value() and not round_alpha:
                continue
            saved, cost = list(element.attrib.items()), element_cost(element)
            drop_properties(element, parse_style(element.get("style", "")), (opacity,))
            set_property(element, paint, shortest_color("#%02x%02x%02x%02x" % (*color[:3], int(merged.to_integral_value()))))
            if element_cost(element) >= cost:
                element.attrib.clear()
                element.attrib.update(saved)
            declarations = parse_style(element.get("style", ""))


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


def translate_element(element, name, dx, dy, written):
    if name == "path":
        segments = parse_path(element.get("d", ""))
        shifted = None if segments is None else translate_path(segments, dx, dy)
        text = None if shifted is None else format_path(shifted)
        if text is None:
            return False
        written[text] = text
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


def apply_translates(root, written):
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
            baked = translate_element(element, local_name(element), *offset, written)
        except Inexact:
            baked = False
        if not baked or element_cost(element) >= cost:
            element.attrib.clear()
            element.attrib.update(saved)
