"""Paths that fill the viewport: written as its rectangle, and the elements they hide removed."""
from decimal import Decimal, Inexact
from fractions import Fraction
from bisect import bisect_right
from functools import reduce
from itertools import combinations

from lxml import etree

from .svg_attributes import CONDITIONAL
from .svg_document import SVG_NS, declared, has_foreign_object, inherited, local_name, namespace, remove_node, style_sets, svg_elements
from .svg_path import format_path, is_straight, path_segments
from .svg_shapes import MARKERS, STYLE_OVERRIDES, WRITTEN, box_segments, coordinate, line_segments, points_segments, rect_segments
from .svg_values import EXACT, LENGTH, number_list, parse_color, parse_number

MOVERS = ("transform", "filter", "mask", "clip-path")
UNITS = ("", "px", "pt", "pc", "mm", "cm", "in", "em", "ex")
HIDDEN = ("path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "image", "use", "g", "a", "switch", "svg")
HALF = Decimal("0.5")
SHAPES = ("path", "rect", "circle", "ellipse", "line", "polyline", "polygon")


def size(text, units):
    """The number and unit of a length, or None."""
    match = LENGTH.fullmatch(text)
    if match is None or (match[2] or "") not in units:
        return None
    return Decimal(match[1]), match[2] or ""


def viewport(root):
    """The visible rectangle (left, top, right, bottom) in user units, or None when it is not known exactly."""
    overflow = declared(root, "overflow")
    if overflow is None or overflow.strip() not in ("", "hidden", "scroll"):
        return None
    has_box = "viewBox" in root.attrib
    width, height = (size(root.get(name, ""), UNITS if has_box else ("",)) for name in ("width", "height"))
    if width is None or height is None or width[0] <= 0 or height[0] <= 0 or width[1] != height[1]:
        return None
    if not has_box:
        return Decimal(0), Decimal(0), width[0], height[0]
    box = number_list(root.get("viewBox"))
    if box is None or len(box) != 4 or box[2] <= 0 or box[3] <= 0 or EXACT.multiply(box[2], height[0]) != EXACT.multiply(box[3], width[0]):
        return None
    return box[0], box[1], EXACT.add(box[0], box[2]), EXACT.add(box[1], box[3])


def apart(points, box):
    """True when a line separates the convex hull of the points from the open box."""
    left, top, right, bottom = box
    xs, ys = [x for x, _ in points], [y for _, y in points]
    if max(xs) <= left or min(xs) >= right or max(ys) <= top or min(ys) >= bottom:
        return True
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    for p, q in combinations(points, 2):
        ax, ay = EXACT.subtract(q[1], p[1]), EXACT.subtract(p[0], q[0])
        if ax == 0 and ay == 0:
            continue
        low, high = ([EXACT.add(EXACT.multiply(ax, x), EXACT.multiply(ay, y)) for x, y in group] for group in (points, corners))
        if max(low) <= min(high) or min(low) >= max(high):
            return True
    return False


def halves(points):
    """The two halves of a Bezier curve, split at t = 1/2."""
    first, second = [points[0]], [points[-1]]
    while len(points) > 1:
        points = [tuple(EXACT.multiply(EXACT.add(a, b), HALF) for a, b in zip(p, q)) for p, q in zip(points, points[1:])]
        first.append(points[0])
        second.insert(0, points[-1])
    return first, second


def line_edges(points, box, depth=4):
    """Line edges with the same winding around the box as the curve through the points, or None."""
    if len(points) == 2:
        return [tuple(points)]
    if apart(points, box):
        return list(zip(points, points[1:]))
    if is_straight([value for point in points[1:] for value in point], *points[0]):
        return [(points[0], points[-1])]
    if depth == 0:
        return None
    edges = []
    for half in halves(points):
        part = line_edges(half, box, depth - 1)
        if part is None:
            return None
        edges += part
    return edges


def path_chains(segments):
    """The point lists of the segments and of the closing line of each subpath, or None with an arc."""
    chains = []
    current = start = None
    for command, values in segments:
        if command == "A":
            return None
        if command == "M":
            if start is not None:
                chains.append([current, start])
            current = start = values
        elif command == "Z":
            chains.append([current, start])
            current = start
        else:
            points = [values[index:index + 2] for index in range(0, len(values), 2)]
            chains.append([current, *points])
            current = points[-1]
    if start is not None:
        chains.append([current, start])
    return chains


def crossing(edge, other):
    """The y where two edges meet, or None."""
    (x0, y0), (x1, y1) = edge
    (x2, y2), (x3, y3) = other
    denominator = (x1 - x0) * (y3 - y2) - (y1 - y0) * (x3 - x2)
    if denominator == 0:
        return None
    t = ((x2 - x0) * (y3 - y2) - (y2 - y0) * (x3 - x2)) / denominator
    u = ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)) / denominator
    return y0 + t * (y1 - y0) if 0 <= t <= 1 and 0 <= u <= 1 else None


def filled(edges, box, rules):
    """True when every point of the open box has a winding number under the fill rule of some layer; each edge is ((p0, p1), layer), not horizontal and within the y range of the box, and `rules` holds the `evenodd` flag of each layer."""
    left, top, right, bottom = box
    inner = [edge for edge, _ in edges if max(edge[0][0], edge[1][0]) > left and min(edge[0][0], edge[1][0]) < right]
    levels = {top, bottom}
    for ((x0, y0), (x1, y1)), _ in edges:
        levels.update((y0, y1))
        levels.update(y0 + (x - x0) * (y1 - y0) / (x1 - x0) for x in (left, right) if (x0 - x) * (x1 - x) < 0)
    levels.update(y for edge, other in combinations(inner, 2) if (y := crossing(edge, other)) is not None)
    levels = sorted(y for y in levels if top <= y <= bottom)
    for low, high in zip(levels, levels[1:]):
        middle = (low + high) / 2
        hits = sorted((x0 + (middle - y0) * (x1 - x0) / (y1 - y0), 1 if y1 > y0 else -1, layer)
                      for ((x0, y0), (x1, y1)), layer in edges if min(y0, y1) < middle < max(y0, y1))
        bounds = [float("-inf"), *(hit[0] for hit in hits), float("inf")]
        windings = [0] * len(rules)
        for index, (lower, upper) in enumerate(zip(bounds, bounds[1:])):
            if index:
                windings[hits[index - 1][2]] -= hits[index - 1][1]
            if lower < right and upper > left and lower < upper and not any(w % 2 if evenodd else w for w, evenodd in zip(windings, rules)):
                return False
    return True


def layer_edges(layers, box):
    """The exact line edges of the layers around the box that are not horizontal and reach its y range, tagged with the layer; a layer with a curve that cannot be turned into lines is left out."""
    edges = []
    for number, (_, chains, _, _) in enumerate(layers):
        parts = [line_edges(chain, box) for chain in chains if max(y for _, y in chain) > box[1] and min(y for _, y in chain) < box[3]]
        if None not in parts:
            edges += [(tuple((Fraction(x), Fraction(y)) for x, y in edge), number) for part in parts for edge in part
                      if edge[0][1] != edge[1][1] and max(edge[0][1], edge[1][1]) > box[1] and min(edge[0][1], edge[1][1]) < box[3]]
    return edges


def covers(chains, box, evenodd):
    """True when the path with these chains fills every point of the open box."""
    return filled(layer_edges([(None, chains, evenodd, None)], box), tuple(map(Fraction, box)), [evenodd])


def sampled(layers, box):
    """False when a point of a 3 by 3 grid in the box is outside every layer, judged on the control polygons in floating point; a heuristic that only saves work."""
    left, top, right, bottom = map(float, box)
    for fx in (.27, .52, .79):
        for fy in (.27, .52, .79):
            x, y = left + fx * (right - left), top + fy * (bottom - top)
            if not any(winding(polygon, x, y, evenodd) for _, _, evenodd, polygon in layers if polygon[0][0] < x < polygon[0][2] and polygon[0][1] < y < polygon[0][3]):
                return False
    return True


def winding(polygon, x, y, evenodd):
    count = 0
    for x0, y0, x1, y1 in polygon[1]:
        if (y0 <= y) != (y1 <= y) and x0 + (y - y0) * (x1 - x0) / (y1 - y0) > x:
            count += 1 if y1 > y0 else -1
    return count % 2 if evenodd else count != 0


def float_polygon(bounds, chains):
    """The bounds and the control polygon edges of a path as floats, for `sampled`."""
    edges = [(float(x0), float(y0), float(x1), float(y1)) for chain in chains for (x0, y0), (x1, y1) in zip(chain, chain[1:])]
    return tuple(map(float, bounds)), edges


def union_covers(box, layers, budget=16, limit=200):
    """True when every point of the open box is filled by some layer, each (bounds, chains, evenodd, polygon): a cell is proven by one layer, or by the straight edges of several with at most `limit` chains in all, and is otherwise split in four, up to `budget` splits; after the first split, layers with more than `limit` chains are left out."""
    cells, splits = [box], 0
    while cells:
        left, top, right, bottom = cell = cells.pop()
        if right <= left or bottom <= top:
            continue
        near = [layer for layer in layers if layer[0][0] < right and layer[0][2] > left and layer[0][1] < bottom and layer[0][3] > top]
        if not near or not sampled(near, cell):
            return False
        usable = [layer for layer in near if not splits or len(layer[1]) <= limit]
        if any(contains(layer[0], cell) and sampled([layer], cell) and covers(layer[1], cell, layer[2]) for layer in usable):
            continue
        if len(usable) > 1 and sum(len(layer[1]) for layer in usable) <= limit and filled(
                layer_edges(usable, cell), tuple(map(Fraction, cell)), [layer[2] for layer in usable]):
            continue
        if splits == budget:
            return False
        splits += 1
        x, y = EXACT.multiply(EXACT.add(left, right), HALF), EXACT.multiply(EXACT.add(top, bottom), HALF)
        cells += [(left, top, x, y), (x, top, right, y), (left, y, x, bottom), (x, y, right, bottom)]
    return True


def bounds_of(points):
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def is_opaque(nodes):
    """True when the path at the start of the chain paints a solid, fully opaque fill."""
    color = parse_color(inherited(nodes, "fill", "black") or "")
    fill_opacity = parse_number(inherited(nodes, "fill-opacity", "1") or "")
    if color is None or color[3] != 255 or fill_opacity is None or fill_opacity < 1 or inherited(nodes, "visibility", "visible") != "visible":
        return False
    for node in nodes:
        opacity = declared(node, "opacity")
        if opacity is None or CONDITIONAL & set(node.attrib) or declared(node, "display").strip() == "none" or declared(node, "mix-blend-mode").strip() not in ("", "normal"):
            return False
        if opacity.strip() and (parse_number(opacity.strip()) or 0) < 1:
            return False
    return True


def is_plain(element, ancestors, blocked):
    """True when the element has no id, only `g` elements without id above it, and none of the blocked properties on it or above."""
    return "id" not in element.attrib and not any(local_name(item) != "g" or "id" in item.attrib for item in ancestors[:-1]) and not any(
        any(name in node.attrib for name in blocked) or style_sets(node, blocked) for node in [element, *ancestors])


def shape_chains(element, name, written):
    """The point chains of a path, rect, line, polyline or polygon, or None."""
    if name == "path":
        segments = path_segments(element.get("d", ""), written)
    elif name == "rect":
        segments = rect_segments(element)
    elif name == "line":
        segments = line_segments(element)
    elif name in ("polyline", "polygon"):
        segments = points_segments(element, name == "polygon")
    else:
        return None
    return None if segments is None else path_chains(segments)


def cover_rule(element, ancestors):
    """The `evenodd` flag of a path without stroke that a later path may be tested against, or None."""
    nodes = [element, *ancestors]
    if style_sets(element, ("d",)) or not is_plain(element, ancestors, MOVERS) or any("url(" in value.lower() for node in nodes for value in node.attrib.values()):
        return None
    rule = inherited(nodes, "fill-rule", "nonzero")
    if inherited(nodes, "stroke", "none") != "none" or rule not in ("evenodd", "nonzero"):
        return None
    return rule == "evenodd"


def painted_bounds(element, name, ancestors, bounds):
    """The (left, top, right, bottom) that the paint of a shape with these point bounds stays within, or None when that is not known."""
    nodes = [element, *ancestors]
    if (not is_plain(element, ancestors, ("transform", "filter")) or style_sets(element, WRITTEN[name] + STYLE_OVERRIDES)
            or any(declared(node, marker) not in ("", "none") for node in nodes for marker in MARKERS)):
        return None
    if name in ("circle", "ellipse"):
        cx, cy = coordinate(element, "cx"), coordinate(element, "cy")
        rx = parse_number(element.get("r" if name == "circle" else "rx", ""))
        ry = rx if name == "circle" else parse_number(element.get("ry", ""))
        if None in (cx, cy, rx, ry) or rx < 0 or ry < 0:
            return None
        bounds = EXACT.subtract(cx, rx), EXACT.subtract(cy, ry), EXACT.add(cx, rx), EXACT.add(cy, ry)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    stroke = inherited(nodes, "stroke", "none")
    if stroke is None:
        return None
    if stroke != "none":
        width = parse_number(inherited(nodes, "stroke-width", "1") or "")
        limit = parse_number(inherited(nodes, "stroke-miterlimit", "4") or "")
        if width is None or limit is None or width < 0 or declared(element, "vector-effect") not in ("", "none"):
            return None
        margin = EXACT.multiply(EXACT.multiply(width, HALF), max(limit, Decimal("1.5")))
        left, top, right, bottom = EXACT.subtract(left, margin), EXACT.subtract(top, margin), EXACT.add(right, margin), EXACT.add(bottom, margin)
    return left, top, right, bottom


def remove_outside(shapes, area):
    """Removes the shapes whose paint does not meet the open viewport; returns the others with their boxes clipped to it."""
    kept = []
    for index, element, (left, top, right, bottom) in shapes:
        if right <= area[0] or left >= area[2] or bottom <= area[1] or top >= area[3]:
            remove_node(element)
        else:
            kept.append((index, element, (max(left, area[0]), max(top, area[1]), min(right, area[2]), min(bottom, area[3]))))
    return kept


def merge(first, second):
    return bounds_of([first[:2], first[2:], second[:2], second[2:]])


def covered_after(box, index, extra, covering, indices, reach):
    """True when the extra layers and the opaque paths after `index` fill every point of the box."""
    start = bisect_right(indices, index)
    layers = [*extra, *covering[start:]]
    if not layers:
        return False
    bounds = reduce(merge, [layer[0] for layer in extra] + ([reach[start]] if reach[start] is not None else []))
    try:
        return contains(bounds, box) and union_covers(box, layers)
    except Inexact:
        return False


def remove_covered(shapes, covering, indices, reach):
    """Removes the shapes that the opaque paths after them fill at every point of their box."""
    for index, element, box in shapes:
        if box[2] > box[0] and box[3] > box[1] and covered_after(box, index, [], covering, indices, reach):
            remove_node(element)


def remove_hidden(cover, ancestors):
    for node in [cover, *ancestors[:-1]]:
        for sibling in list(node.itersiblings(preceding=True)):
            if (isinstance(sibling.tag, str) and namespace(sibling) == SVG_NS and local_name(sibling) in HIDDEN
                    and not any("id" in item.attrib for item in sibling.iter(etree.Element))):
                remove_node(sibling)


def crop_viewport(root, written):
    """Writes a path that fills the whole viewport, alone or with the opaque paths after it, as the viewport rectangle when that is shorter, and removes the shapes outside the viewport and the elements that opaque paths hide."""
    area = viewport(root)
    shapes, paths, covering, indices = [], [], [], []
    for index, element in enumerate(svg_elements(root)):
        name = local_name(element)
        if name not in SHAPES:
            continue
        ancestors = list(element.iterancestors())
        try:
            chains = shape_chains(element, name, written)
            bounds = bounds_of([point for chain in chains for point in chain]) if chains else None
            box = painted_bounds(element, name, ancestors, bounds)
            if box is not None:
                shapes.append((index, element, box))
            evenodd = cover_rule(element, ancestors) if name == "path" and chains else None
        except Inexact:
            continue
        if evenodd is not None:
            layer = bounds, chains, evenodd, float_polygon(bounds, chains)
            opaque = is_opaque([element, *ancestors])
            paths.append((index, element, ancestors, layer, opaque))
            if opaque:
                covering.append(layer)
                indices.append(index)
    if has_foreign_object(root):
        return
    reach = [None]
    for layer in reversed(covering):
        reach.append(layer[0] if reach[-1] is None else merge(reach[-1], layer[0]))
    reach.reverse()
    if area is not None:
        shapes = remove_outside(shapes, area)
    remove_covered(shapes, covering, indices, reach)
    last = None
    for index, element, ancestors, layer, opaque in paths:
        if area is None or element.getparent() is None or not covered_after(area, index, [layer], covering, indices, reach):
            continue
        text = format_path(box_segments(*area))
        if text is not None and len(text) < len(element.get("d")):
            element.set("d", text)
        if opaque:
            last = element, ancestors
    if last is not None:
        remove_hidden(*last)
