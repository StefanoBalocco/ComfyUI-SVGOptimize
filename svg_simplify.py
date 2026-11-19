"""Lossy simplification of the shapes of an SVG where they are, verified against a render of the input."""
import copy
import math
import operator
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from cairosvg.helpers import node_format
from cairosvg.parser import Tree

from .svg_document import SVG_NS, document_facts, elements, inherited, local_name, parse_svg, remove_node, serialize
from .svg_fit import merge_segments, optimize, removal_costs, render
from .svg_scene import KEY, Scene, capture_svg, number, path_data, render_layer, rgba_text
from .svg_structure import references
from .svg_values import read_color, set_property

# Longest side of the raster the shapes are fitted on, in pixels.
SIZE = 512
# A candidate is acceptable when its CairoSVG render is at least this close to
# the render of the input, as PSNR in dB.
MIN_PSNR = 45
ERROR_LIMIT = 10 ** (-MIN_PSNR / 10)
STEPS = 100
# A cubic whose control points lie this close to its chord, in raster pixels, is drawn straight.
STRAIGHT = 0.05
TOLERANCES = (1.0, 0.5, 0.25)
# Most optimization trials per document: dropping shapes, one per tolerance, dropping shapes again.
TRIALS = 2 + len(TOLERANCES)
# CairoSVG resolves units and em lengths against its surface; these are the values of a fresh one.
SURFACE = SimpleNamespace(dpi=96, font_size=16, context_width=0, context_height=0)
GEOMETRY = {"rect": ("x", "y", "width", "height", "rx", "ry"), "circle": ("cx", "cy", "r"), "ellipse": ("cx", "cy", "rx", "ry"),
            "line": ("x1", "y1", "x2", "y2"), "polyline": ("points",), "polygon": ("points",)}


def raster_size(data):
    """Pixel size of the raster: SIZE on the longest side, with the aspect ratio
    CairoSVG renders (square when the document declares no size)."""
    width, height, _ = node_format(SURFACE, Tree(bytestring=data, unsafe=False))
    if width > 0 and height > 0:
        scale = SIZE / max(width, height)
        return max(1, round(width * scale)), max(1, round(height * scale))
    return SIZE, SIZE


def render_svg(data, width, height, device):
    """[3,H,W] floats of the CairoSVG render on white."""
    return torch.from_numpy(render_layer(data, width, height, background="white")[0]).to(device)


def acceptable(document, reference):
    height, width = reference.shape[1:]
    return (render_svg(document, width, height, reference.device) - reference).square().mean().item() <= ERROR_LIMIT


def is_uniform(matrix):
    """True when the transform scales both axes alike, whatever its rotation."""
    xx, yx, xy, yy = matrix[:4]
    scale = xx * xx + yx * yx
    return abs(scale - (xy * xy + yy * yy)) <= 1e-6 * scale and abs(xx * xy + yx * yy) <= 1e-6 * scale


def straighten(shape, tolerance=STRAIGHT):
    """Makes straight the cubic segments whose control points lie within `tolerance` of the chord, in order along it, and puts their control points at 1/3 and 2/3. True when a segment changes."""
    changed = False
    for subpath, lines in zip(shape.subpaths, shape.lines):
        points = subpath[3 * np.arange(len(lines))[:, None] + np.arange(4)].astype(np.float64)
        chord = points[:, 3] - points[:, 0]
        length = (chord ** 2).sum(1)
        offsets = points[:, 1:3] - points[:, :1]
        cross = offsets[..., 0] * chord[:, None, 1] - offsets[..., 1] * chord[:, None, 0]
        along = (offsets * chord[:, None]).sum(2)
        straight = ~lines & (length > 0) & (cross ** 2 <= tolerance ** 2 * length[:, None]).all(1) & (along[:, 0] >= 0) & (along[:, 0] <= along[:, 1]) & (along[:, 1] <= length)
        if straight.any():
            first = 3 * np.flatnonzero(straight)
            subpath[first + 1] = points[straight, 0] + chord[straight] / 3
            subpath[first + 2] = points[straight, 0] + chord[straight] * 2 / 3
            lines[straight] = True
            shape.changed = changed = True
    return changed


def shape_data(shape):
    """The path data of the shape in the user space of its element, and the decimals it is written with."""
    xx, yx, xy, yy, x0, y0 = shape.matrix
    linear = np.array([[xx, xy], [yx, yy]])
    # Rounding the coordinates moves the points by at most about 0.05 px.
    decimals = max(0, math.ceil(math.log10(10 * np.linalg.norm(linear, 2))))
    subpaths = [(points - (x0, y0)) @ np.linalg.inv(linear).T for points in shape.subpaths]
    return path_data(subpaths, shape.closed, shape.lines, decimals), decimals


def painted_color(element, name):
    """The color of the property `name` that the element paints with, declared on it or on its nearest ancestor, as (red, green, blue, alpha times 255), or None."""
    return read_color(inherited([element, *element.iterancestors()], name, "") or "")


def color_text(paint, painted):
    """The fitted color of `paint` with the alpha of the color the element paints with."""
    rgb = tuple(round(channel * 255) for channel in paint[:3])
    alpha = painted[3] if painted else 255
    if alpha != 255 and alpha == alpha.to_integral_value():
        return "#%02x%02x%02x%02x" % (*rgb, int(alpha))
    return rgba_text(rgb, alpha)


def write_shape(element, shape):
    """Writes the geometry of the shape, in the user space of the element, and its fitted colors and stroke width."""
    if not shape.changed:
        return
    xx, yx, xy, yy = shape.matrix[:4]
    data, decimals = shape_data(shape)
    for name in GEOMETRY.get(local_name(element), ()):
        element.attrib.pop(name, None)
    element.tag = f"{{{SVG_NS}}}path"
    element.set("d", data)
    if shape.fittable and shape.tune_color:
        for name, paint in (("fill", shape.fill), ("stroke", shape.stroke)):
            if paint is not None:
                set_property(element, name, color_text(paint, painted_color(element, name)))
    if shape.fittable and shape.tune_width and shape.stroke is not None:
        set_property(element, "stroke-width", number(shape.stroke_width / math.sqrt(abs(xx * yy - xy * yx)), decimals))


def round_paths(root, scene):
    """A copy of the tree where each editable path drawn as one shape has its data written from the captured geometry when that is shorter, or None when no path is."""
    tree = copy.deepcopy(root)
    nodes = list(elements(tree))
    drawn = Counter(shape.element for shape in scene.shapes)
    rounded = False
    for shape in scene.shapes:
        element = nodes[shape.element]
        if shape.editable and drawn[shape.element] == 1 and local_name(element) == "path":
            data, _ = shape_data(shape)
            if len(data) < len(element.get("d")):
                element.set("d", data)
                rounded = True
    return tree if rounded else None


def write_document(root, shapes, dropped):
    """A copy of the tree with the shapes written to their elements and the elements numbered in `dropped` removed."""
    tree = copy.deepcopy(root)
    nodes = list(elements(tree))
    for shape in shapes:
        write_shape(nodes[shape.element], shape)
    for index in dropped:
        remove_node(nodes[index])
    return tree


def untagged(tree, preceding):
    for element in elements(tree):
        del element.attrib[KEY]
    return serialize(tree, preceding)


def split_runs(scene):
    """The elements of the units that are not fitted, grouped by the fitted shapes
    they lie between: run i lies before fitted shape i, the last after all of them."""
    runs = [[]]
    for index, shapes in scene.units:
        if len(shapes) == 1 and shapes[0].fittable:
            runs.append([])
        else:
            runs[-1].append(index)
    return runs


def stack(below, above):
    """The fixed layers (premultiplied color, alpha) `above` over `below`."""
    if below is None or above is None:
        return below or above
    return above[0] + (1 - above[1]) * below[0], above[1] + (1 - above[1]) * below[1]


def without(layers, keep, combine):
    """`layers` once the shapes that are not kept are removed: layers[i] lies before
    shape i, so the two around a removed shape become one."""
    out = layers[:1]
    for kept, layer in zip(keep, layers[1:]):
        if kept:
            out.append(layer)
        else:
            out[-1] = combine(out[-1], layer)
    return out


def draw_runs(scene, runs, fixed, root, preceding, dropped, device):
    """The fixed layers of the runs, drawn again from the scene's geometry where
    they are missing or the run holds editable shapes."""
    editable = {index for index, shapes in scene.units if any(shape.editable for shape in shapes)}
    stale = [bool(members) and (layer is None or not editable.isdisjoint(members)) for members, layer in zip(runs, fixed)]
    if not any(stale):
        return fixed
    data = serialize(write_document(root, [shape for shape in scene.shapes if not shape.fittable], dropped), preceding)
    units = {index for index, _ in scene.units}
    layers = []
    for members, layer, redraw in zip(runs, fixed, stale):
        if redraw:
            color, alpha = render_layer(data, scene.width, scene.height, units.difference(members))
            layer = torch.from_numpy(color).to(device), torch.from_numpy(alpha).to(device)
        layers.append(layer)
    return layers


def fitted(scene):
    return Scene(scene.width, scene.height, [shape for shape in scene.shapes if shape.fittable])


def drop_shapes(root, preceding, scene, fixed, dropped, reference, device, progress):
    """Optimizes a copy of the scene without its cheapest removable fittable shapes that fit the
    error budget left, never all of them. Returns the trial scene, the kept flags, the numbers of the
    removed elements, the trial's fixed layers and its document, or None when no shape is dropped."""
    fit = fitted(scene)
    if not fit.shapes:
        return None
    nodes = list(elements(root))
    referenced = {name for element in nodes for name in references(element)}
    removable = np.array([nodes[shape.element].get("id") not in referenced for shape in fit.shapes], bool)
    costs = np.where(removable, removal_costs(fit, fixed, reference), np.inf)
    order = np.argsort(costs)
    budget = ERROR_LIMIT - (render(fit, fixed, device) - reference).square().mean().item()
    count = min(int((np.cumsum(costs[order]) <= budget).sum()), len(costs) - 1)
    if not count:
        return None
    keep = np.ones(len(costs), bool)
    keep[order[:count]] = False
    gone = {shape.element for shape, kept in zip(fit.shapes, keep) if not kept}
    trial = copy.deepcopy(scene)
    trial.shapes = [shape for shape in trial.shapes if shape.element not in gone]
    trial_fixed = without(fixed, keep, stack)
    optimize(fitted(trial), trial_fixed, reference, STEPS)
    progress()
    return trial, keep, gone, trial_fixed, untagged(write_document(root, trial.shapes, dropped | gone), preceding)


def simplify_svg(data, device, progress=lambda: None):
    """The input with its shapes simplified where they are, drawing within MIN_PSNR
    of its render, or None when nothing is simplified. Runs on `device` when it is
    CUDA, otherwise on the CPU, and calls `progress` after each optimization trial."""
    root, preceding = parse_svg(data)
    facts = document_facts(root, preceding)
    for index, element in enumerate(elements(root)):
        element.set(KEY, str(index))
    width, height = raster_size(data)
    scene = capture_svg(serialize(root, preceding), width, height)
    if not any(shape.editable for shape in scene.shapes):
        return None
    for shape in scene.shapes:
        shape.tune_color = not facts.stylesheet
        shape.tune_width = not facts.stylesheet and is_uniform(shape.matrix)
        if shape.editable:
            straighten(shape)
    device = device if device.type == "cuda" else torch.device("cpu")
    # ComfyUI runs nodes in inference mode, which the optimizer cannot differentiate in.
    with torch.inference_mode(False), torch.enable_grad():
        reference = render_svg(data, width, height, device)
        best, size = None, len(untagged(write_document(root, [], set()), preceding))
        rounded = round_paths(root, scene)
        if rounded is not None:
            document = untagged(copy.deepcopy(rounded), preceding)
            if len(document) < size and acceptable(document, reference):
                best, size = document, len(document)
        runs = split_runs(scene)
        fixed = draw_runs(scene, runs, [None] * len(runs), root, preceding, set(), device)
        dropped = set()
        attempt = drop_shapes(root, preceding, scene, fixed, dropped, reference, device, progress)
        if attempt is not None:
            trial, keep, gone, trial_fixed, document = attempt
            if len(document) < size and acceptable(document, reference):
                best, size = document, len(document)
                scene, runs, fixed, dropped = trial, without(runs, keep, operator.add), trial_fixed, dropped | gone
        for tolerance in TOLERANCES:
            trial = copy.deepcopy(scene)
            straightened = any([straighten(shape, tolerance) for shape in trial.shapes if shape.editable])
            if not (merge_segments(trial, tolerance) or straightened):
                break
            trial_fixed = draw_runs(trial, runs, fixed, root, preceding, dropped, device)
            fit = fitted(trial)
            if fit.shapes:
                optimize(fit, trial_fixed, reference, STEPS)
            progress()
            document = untagged(write_document(root, trial.shapes, dropped), preceding)
            if len(document) < size and acceptable(document, reference):
                attempt = drop_shapes(root, preceding, trial, trial_fixed, dropped, reference, device, progress)
                if attempt is not None:
                    smaller = attempt[4]
                    if len(smaller) < len(document) and acceptable(smaller, reference):
                        return smaller
                return document
    return best
