"""Capture the drawable geometry of an arbitrary SVG as device-space cubic Beziers.

CairoSVG already resolves CSS classes, inheritance, <use>, transforms, viewBox,
units and arcs. A surface subclass records what it asks cairo to fill, stroke and
clip, so the result is exactly what CairoSVG would draw. Each shape knows the
element it comes from and whether that element can be rewritten and fitted, the
drawing is split into units in paint order, and a surface can draw the document
without some of its elements. Path data is written from device-space geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cairocffi
import numpy as np
from cairosvg.parser import Tree
from cairosvg.surface import PNGSurface
from cairosvg.url import parse_url

from .svg_document import elements, parse_svg, serialize
from .svg_values import convert_colors, read_color

# Attribute that numbers the elements of a document, so that what CairoSVG
# draws can be traced back to them.
KEY = "data-k"
# Elements that paint themselves; their drawing is one unit in paint order.
LEAVES = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "image", "use"}
TEXT = {"text", "tspan", "textPath"}
MARKERS = ("marker", "marker-start", "marker-mid", "marker-end")


@dataclass
class Shape:
    subpaths: list[np.ndarray]
    closed: list[bool]
    # Per subpath, per cubic segment: drawn as a straight line, its control
    # points at 1/3 and 2/3 of the way.
    lines: list[np.ndarray]
    fill: tuple[float, float, float, float] | None
    fill_rule: str
    stroke: tuple[float, float, float, float] | None
    stroke_width: float
    clip: list[list[np.ndarray]]
    # "miter" or "round"; bevel joins are kept as round.
    stroke_join: str = "miter"
    miter_limit: float = 4.0
    stroke_cap: str = "butt"
    # Index of the element and its transform to device space (xx, yx, xy, yy, x0, y0).
    element: int = -1
    matrix: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    # The element can be rewritten; the rasterizer reproduces it, so it is also fitted.
    editable: bool = False
    fittable: bool = False
    # The geometry differs from the captured one.
    changed: bool = False
    tune_color: bool = True
    tune_width: bool = True


@dataclass
class Scene:
    width: int
    height: int
    shapes: list[Shape]
    # In paint order, the element of each unit of drawing and the shapes it recorded.
    units: list[tuple[int, list[Shape]]] = field(default_factory=list)


def _device_path(context):
    """Current path of a cairocffi context as (cubic subpaths, closed flags,
    straight-segment flags) in device space."""
    context.save()
    context.identity_matrix()
    path = context.copy_path()
    context.restore()
    subpaths, closed, lines = [], [], []
    points, straight = [], []

    def finish(is_closed):
        if len(points) > 1:
            subpaths.append(np.array(points, np.float32))
            closed.append(is_closed)
            lines.append(np.array(straight, bool))

    for kind, coords in path:
        if kind == cairocffi.PATH_MOVE_TO:
            finish(False)
            points, straight = [coords], []
        elif kind == cairocffi.PATH_LINE_TO:
            a = np.array(points[-1])
            b = np.array(coords)
            points += [tuple(a + (b - a) / 3), tuple(a + (b - a) * 2 / 3), coords]
            straight.append(True)
        elif kind == cairocffi.PATH_CURVE_TO:
            points += [coords[0:2], coords[2:4], coords[4:6]]
            straight.append(False)
        else:
            finish(True)
            points, straight = [], []
    finish(False)
    return subpaths, closed, lines


class _Recorder:
    """Stands in for the cairocffi Context of a CairoSVG surface and records paint operations."""

    def __init__(self, context, surface):
        self.context = context
        self.surface = surface
        self.recording = False
        self.text_depth = 0
        self.clips = []
        self.saved_clips = []
        self.group_starts = []
        self.group_done = None
        self.pending = None

    def __getattr__(self, name):
        return getattr(self.context, name)

    def _paint(self):
        """RGBA of the source, or None when it is not a solid color."""
        source = self.context.get_source()
        return source.get_rgba() if isinstance(source, cairocffi.SolidPattern) else None

    def _opaque_source(self):
        source = self.context.get_source()
        return not isinstance(source, cairocffi.SolidPattern) or source.get_rgba()[3] > 0

    def _shape(self, subpaths, closed, lines):
        surface = self.surface
        return Shape(subpaths, closed, lines, None, "nonzero", None, 0.0, list(self.clips), element=surface.index, matrix=tuple(self.context.get_matrix()),
                     editable=not (surface.uses or surface.marks), fittable=not surface.effects)

    def _recorded_path(self):
        if not self.recording or self.text_depth or not self._opaque_source():
            return None
        path = _device_path(self.context)
        return path if path[0] else None

    def _record_fill(self):
        path = self._recorded_path()
        if path is None:
            return
        shape = self._shape(*path)
        shape.fill = self._paint()
        shape.fill_rule = "evenodd" if self.context.get_fill_rule() == cairocffi.FILL_RULE_EVEN_ODD else "nonzero"
        shape.fittable &= shape.fill is not None
        self.surface.scene.shapes.append(shape)
        self.pending = shape

    def fill(self):
        self._record_fill()
        self.pending = None
        self.context.fill()

    def fill_preserve(self):
        self._record_fill()
        self.context.fill_preserve()

    def _record_stroke(self):
        path = self._recorded_path()
        if path is None:
            return
        subpaths, closed, lines = path
        xx, yx, xy, yy, _, _ = self.context.get_matrix()
        shape = self.pending
        if shape is None or shape.closed != closed or len(shape.subpaths) != len(subpaths) or not all(map(np.array_equal, shape.subpaths, subpaths)):
            shape = self._shape(subpaths, closed, lines)
            self.surface.scene.shapes.append(shape)
        shape.stroke = self._paint()
        shape.fittable &= shape.stroke is not None and not self.context.get_dash()[0]
        shape.stroke_width = self.context.get_line_width() * math.sqrt(abs(xx * yy - xy * yx))
        shape.stroke_join = "miter" if self.context.get_line_join() == cairocffi.LINE_JOIN_MITER else "round"
        shape.miter_limit = self.context.get_miter_limit()
        shape.stroke_cap = {cairocffi.LINE_CAP_ROUND: "round", cairocffi.LINE_CAP_SQUARE: "square"}.get(self.context.get_line_cap(), "butt")
        self.pending = None

    def stroke(self):
        self._record_stroke()
        self.context.stroke()

    def stroke_preserve(self):
        self._record_stroke()
        self.context.stroke_preserve()

    def _clip_region(self):
        if self.recording:
            subpaths, _, _ = _device_path(self.context)
            self.clips.append(subpaths)

    def clip(self):
        self._clip_region()
        self.context.clip()

    def clip_preserve(self):
        self._clip_region()
        self.context.clip_preserve()

    def save(self):
        self.saved_clips.append(list(self.clips))
        self.context.save()

    def restore(self):
        self.clips = self.saved_clips.pop()
        self.context.restore()

    def push_group(self):
        self.group_starts.append(len(self.surface.scene.shapes))
        self.pending = None
        self.context.push_group()

    def pop_group_to_source(self):
        self.group_done = self.group_starts.pop()
        self.pending = None
        self.context.pop_group_to_source()

    def paint_with_alpha(self, alpha):
        if self.group_done is not None:
            for shape in self.surface.scene.shapes[self.group_done:]:
                shape.fill = shape.fill and shape.fill[:3] + (shape.fill[3] * alpha,)
                shape.stroke = shape.stroke and shape.stroke[:3] + (shape.stroke[3] * alpha,)
            self.group_done = None
        self.context.paint_with_alpha(alpha)

    def mask_surface(self, *args):
        self.group_done = None
        self.context.mask_surface(*args)


def _index(node):
    value = node.element.etree_element.get(KEY)
    return None if value is None else int(value)


class _GrayText:
    """Stands in for a cairocffi Context and draws text with grayscale antialiasing,
    whatever the background, so that every render draws text alike."""

    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        return getattr(self.context, name)

    def set_font_options(self, options):
        options.set_antialias(cairocffi.ANTIALIAS_GRAY)
        self.context.set_font_options(options)


class _Surface(PNGSurface):
    """A CairoSVG surface that draws text in grayscale and does not draw the elements numbered in `skipped`, except inside <use>."""

    def __init__(self, *args, skipped=frozenset(), **kwargs):
        self.skipped = skipped
        self.uses = 0
        super().__init__(*args, **kwargs)

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, context):
        self._context = _GrayText(context)

    def draw(self, node):
        if not self.uses and _index(node) in self.skipped:
            return
        is_use = node.tag == "use"
        self.uses += is_use
        super().draw(node)
        self.uses -= is_use


class _CaptureSurface(_Surface):
    def __init__(self, *args, scene, **kwargs):
        self.scene = scene
        # Element being drawn, and the number of nodes being drawn with a filter or mask and with markers.
        self.index = None
        self.effects = self.marks = 0
        # The outermost unit being drawn: its element, the number of shapes recorded
        # before it and the number of elements that paint themselves inside it.
        self.unit = None
        self.leaves = 0
        super().__init__(*args, **kwargs)

    @property
    def context(self):
        return self._recorder

    @context.setter
    def context(self, context):
        self._recorder = _Recorder(_GrayText(context), self)

    def draw(self, node):
        recorder = self._recorder
        recorder.recording = True
        index, self.index = self.index, _index(node)
        effect = parse_url(node.get("filter")).fragment in self.filters or parse_url(node.get("mask")).fragment in self.masks
        marked = any(parse_url(node.get(name)).fragment for name in MARKERS)
        # Clip paths are drawn through draw() without painting.
        leaf = node.tag in LEAVES and self.stroke_and_fill
        started = self.unit is None and self.stroke_and_fill and (leaf or effect or (float(node.get("opacity", 1)) < 1 and bool(node.children)))
        if started:
            self.unit = self.index, len(self.scene.shapes)
            self.leaves = 0
        self.leaves += leaf
        text = node.tag in TEXT
        self.effects += effect
        self.marks += marked
        recorder.text_depth += text
        super().draw(node)
        recorder.text_depth -= text
        self.marks -= marked
        self.effects -= effect
        if started:
            element, first = self.unit
            shapes = self.scene.shapes[first:]
            if self.leaves != 1 or len(shapes) != 1:
                for shape in shapes:
                    shape.fittable = False
            self.scene.units.append((element, shapes))
            self.unit = None
        self.index = index


def rgba_text(rgb, alpha) -> str:
    return "#%02x%02x%02x" % tuple(rgb) if alpha == 255 else "rgba(%d,%d,%d,%s)" % (*rgb, number(float(alpha) / 255, 4))


def cairo_color(text: str) -> str:
    """The color as `#rrggbb` or `rgba()` with a decimal alpha, the forms CairoSVG reads exactly; text that is not a readable color stays."""
    color = read_color(text.strip())
    if color is None:
        return text
    return rgba_text(color[:3], color[3])


def cairo_colors(data: bytes) -> bytes:
    """The document with the colors of its attributes and style declarations written as `cairo_color` does."""
    root, preceding = parse_svg(data)
    for element in elements(root):
        for name, value in list(element.attrib.items()):
            element.set(name, convert_colors(name, value, cairo_color))
    return serialize(root, preceding)


def capture_svg(data: bytes, width: int, height: int) -> Scene:
    """The shapes CairoSVG draws for the document `data`, whose elements are numbered by KEY."""
    scene = Scene(width, height, [])
    _CaptureSurface(Tree(bytestring=cairo_colors(data), unsafe=False), None, 96, None, None, None, 1, width, height, scene=scene)
    # An element that is also drawn through <use> or markers changes with them.
    shared = {shape.element for shape in scene.shapes if not shape.editable}
    for shape in scene.shapes:
        shape.editable = shape.element not in shared
        shape.fittable &= shape.editable
    return scene


def render_layer(data: bytes, width: int, height: int, skipped: set[int] = frozenset(), background: str | None = None):
    """Premultiplied RGB [3,H,W] and alpha [H,W] in 0..1 of the document `data`
    on a transparent or `background` colored surface, without the elements numbered in `skipped`."""
    surface = _Surface(Tree(bytestring=cairo_colors(data), unsafe=False), None, 96, None, None, None, 1, width, height, background, skipped=skipped)
    surface.cairo.flush()
    pixels = np.frombuffer(surface.cairo.get_data(), np.uint8).reshape(height, width, 4).astype(np.float32) / 255
    return np.ascontiguousarray(pixels[:, :, 2::-1].transpose(2, 0, 1)), pixels[:, :, 3]


def number(value: float, decimals: int = 3) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if text.startswith(("0.", "-0.")):
        text = text.replace("0.", ".", 1)
    return "0" if text == "-0" else text


def _command(previous, token, letter, values, decimals):
    """Text of one path command after the command `previous` and the token
    `token`: the letter only when it does not repeat, a space only where a
    number would run into the one before it."""
    text = "" if letter == previous else letter
    token = letter if text else token
    for value in (number(value, decimals) for value in values):
        if not (token.isalpha() or value[0] == "-" or value[0] == "." and "." in token):
            text += " "
        text += value
        token = value
    return text, token


def path_data(subpaths, closed, lines, decimals) -> str:
    """Each command absolute or relative, whichever is shorter. Relative
    coordinates are differences of rounded absolute ones, so rounding errors
    do not add up along a path."""
    parts, previous, token, current = [], "", "", np.zeros(2)
    for points, is_closed, straight in zip(subpaths, closed, lines):
        points = np.round(points.astype(np.float64), decimals)
        commands = [(("M", points[0]), ("m", points[0] - current))]
        start = current = points[0]
        for i, line in zip(range(1, len(points), 3), straight):
            end = points[i + 2]
            if not line:
                commands.append((("C", points[i:i + 3].ravel()), ("c", (points[i:i + 3] - current).ravel())))
            elif end[1] == current[1]:
                commands.append((("H", end[:1]), ("h", end[:1] - current[0])))
            elif end[0] == current[0]:
                commands.append((("V", end[1:]), ("v", end[1:] - current[1])))
            else:
                commands.append((("L", end), ("l", end - current)))
            current = end
        for candidates in commands:
            text, token = min((_command(previous, token, letter, values, decimals) for letter, values in candidates), key=lambda pair: len(pair[0]))
            parts.append(text)
            if text[0].isalpha():
                # Coordinate pairs after a moveto are linetos.
                previous = {"M": "L", "m": "l"}.get(text[0], text[0])
        if is_closed:
            parts.append("z")
            previous, token, current = "z", "z", start
    return "".join(parts)
