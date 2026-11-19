"""Simplification: raster size, fidelity to the input and what is rewritten."""
from base64 import b64encode
import copy
from io import BytesIO
import math
from pathlib import Path
import unittest

import cairosvg
import numpy as np
import torch
from lxml import etree
from PIL import Image

import package_alias  # noqa: F401
from helpers import gradient_document, ring
from svgoptimize import svg_fit
from svgoptimize.svg_document import elements, parse_svg, serialize
from svgoptimize.svg_fit import removal_costs, render
from svgoptimize.svg_optimize import optimize_svg
from svgoptimize.svg_scene import KEY, cairo_colors, capture_svg
from svgoptimize.svg_simplify import MIN_PSNR, color_text, draw_runs, drop_shapes, fitted, is_uniform, painted_color, raster_size, render_svg, simplify_svg, split_runs, stack, straighten, untagged, without, write_document

FIXTURES = Path(__file__).parent / "fixtures"
CPU = torch.device("cpu")


def svg(attributes, body=""):
    return f'<svg xmlns="http://www.w3.org/2000/svg" {attributes}>{body}</svg>'.encode()


def cairo_render(data, size):
    png = cairosvg.svg2png(bytestring=data, output_width=size[0], output_height=size[1], background_color="white")
    return np.asarray(Image.open(BytesIO(png)).convert("RGB"), np.float64) / 255


def psnr(a, b):
    return -10 * np.log10(np.mean((a - b) ** 2))


def similarity(data, result, size):
    return psnr(cairo_render(result, (size, size)), cairo_render(data, (size, size)))


def by_id(data):
    return {element.get("id"): element for element in etree.fromstring(data).iter() if element.get("id")}


def scene_of(data):
    """The tree numbered by KEY, its serialization and the scene captured from it."""
    root, preceding = parse_svg(data)
    for index, element in enumerate(elements(root)):
        element.set(KEY, str(index))
    width, height = raster_size(data)
    return root, preceding, capture_svg(serialize(root, preceding), width, height)


class RasterSizeTests(unittest.TestCase):
    def test_longest_side_is_512(self):
        for attributes in ('width="24" height="24"', 'width="2000" height="1000"', 'width="100mm" height="100mm"', 'viewBox="0 0 64 64"', ""):
            with self.subTest(attributes):
                self.assertEqual(raster_size(svg(attributes)), (512, 512) if "2000" not in attributes else (512, 256))

    def test_aspect_ratio_is_kept(self):
        self.assertEqual(raster_size(svg('width="100" height="50"')), (512, 256))
        self.assertEqual(raster_size(svg('width="30px" height="60px"')), (256, 512))
        self.assertEqual(raster_size(svg('width="100mm" height="50mm"')), (512, 256))
        self.assertEqual(raster_size(svg('viewBox="0 0 320 160"')), (512, 256))

    def test_percentages_are_ignored(self):
        self.assertEqual(raster_size(svg('width="100%" height="100%" viewBox="0 0 400 200"')), (512, 256))

    def test_no_size_is_square(self):
        self.assertEqual(raster_size(svg("")), (512, 512))


class SimplifyTests(unittest.TestCase):
    def check_vtracer(self, device):
        data = (FIXTURES / "vtracer.svg").read_bytes()
        result = simplify_svg(data, device)
        self.assertLess(len(optimize_svg(result, False)), len(optimize_svg(data, False)))
        self.assertGreaterEqual(similarity(data, result, 256), MIN_PSNR)
        source, simplified = etree.fromstring(data), etree.fromstring(result)
        self.assertEqual(dict(simplified.attrib), dict(source.attrib))
        paths = simplified.findall("{*}path")
        self.assertTrue(paths)
        self.assertTrue(all(path.get("transform") == "translate(0,0)" or path.get("transform") in {item.get("transform") for item in source} for path in paths))

    def test_vtracer_fixture(self):
        self.check_vtracer(CPU)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_vtracer_fixture_on_cuda(self):
        self.check_vtracer(torch.device("cuda"))

    def test_nothing_smaller_gives_none(self):
        self.assertIsNone(simplify_svg(svg('width="64" height="64"', '<rect width="30" height="20" fill="#f00"/>'), CPU))

    def test_paths_are_rounded_to_the_raster(self):
        data = (FIXTURES / "chart-icon.svg").read_bytes()
        result = simplify_svg(data, CPU)
        self.assertLess(len(result), len(data))
        self.assertIn(b'd="M27.042 10.5h7.5v31h-7.5z"', result)
        self.assertGreaterEqual(similarity(data, result, 512), MIN_PSNR)

    def test_basic_shapes_are_not_rounded(self):
        self.assertIsNone(simplify_svg(svg('width="48" height="48"', '<rect x="27.042373" y="10.5" width="7.5" height="31" fill="#f39c12"/>'), CPU))

    def test_document_without_shapes_gives_none(self):
        self.assertIsNone(simplify_svg(svg('width="64" height="64"', '<text x="5" y="30">Hi</text>'), CPU))

    def test_gradient_document(self):
        data = gradient_document()
        with self.assertNoLogs(level="WARNING"):
            result = simplify_svg(data, CPU)
        self.assertLess(len(result), len(data))
        self.assertGreaterEqual(similarity(data, result, 512), MIN_PSNR)
        source, simplified = by_id(data), by_id(result)
        for name in ("g", "panel", "label"):
            self.assertEqual(etree.tostring(simplified[name], with_tail=False), etree.tostring(source[name], with_tail=False))
        self.assertEqual([item.tag.rpartition("}")[2] for item in (simplified["a"], simplified["b"])], ["path", "path"])
        self.assertEqual(simplified["b"].get("fill-opacity"), ".8")

    def test_what_is_not_a_shape_stays_as_written(self):
        pixels = BytesIO()
        Image.new("RGB", (4, 4), "#3c6").save(pixels, "PNG")
        defs = f'<marker id="m" markerWidth="4" markerHeight="4" refX="2" refY="2"><circle cx="2" cy="2" r="1.5" fill="#f0f"/></marker><polygon id="t" points="{ring(10, 10, 8, 8, 60)}" fill="#08c"/>'
        middle = (f'<use id="u" href="#t" x="70" y="5"/><polyline id="mk" points="{ring(80, 70, 10, 10, 40)}" fill="none" stroke="#000" marker-mid="url(#m)"/>'
                  f'<image id="img" x="2" y="2" width="12" height="12" href="data:image/png;base64,{b64encode(pixels.getvalue()).decode()}"/>')
        data = gradient_document(defs, middle)
        result = simplify_svg(data, CPU)
        source, simplified = by_id(data), by_id(result)
        for name in ("m", "t", "u", "mk", "img", "label"):
            self.assertEqual(etree.tostring(simplified[name], with_tail=False), etree.tostring(source[name], with_tail=False), name)
        self.assertEqual(simplified["a"].tag.rpartition("}")[2], "path")
        self.assertGreaterEqual(similarity(data, result, 512), MIN_PSNR)

    def test_stylesheet_keeps_colors(self):
        data = gradient_document(middle='<style>.c{fill:#2a6}</style>').replace(b'id="a"', b'id="a" class="c"').replace(b'fill="#2a6"', b"")
        result = simplify_svg(data, CPU)
        shape = by_id(result)["a"]
        self.assertEqual(shape.tag.rpartition("}")[2], "path")
        self.assertEqual((shape.get("class"), shape.get("fill"), shape.get("stroke")), ("c", None, None))

    def test_referenced_shapes_stay(self):
        dots = '<circle id="kept" cx="50" cy="20" r="0.2" fill="#000"/><circle id="gone" cx="52" cy="20" r="0.2" fill="#000"/>'
        data = gradient_document(middle=dots).replace(b'id="panel"', b'id="panel" aria-labelledby="kept"').replace(b'<text id="label" x="10" y="95" font-size="10">Hi</text>', b"")
        result = by_id(simplify_svg(data, CPU))
        self.assertIn("kept", result)
        self.assertNotIn("gone", result)

    def test_drop_shapes_on_a_scene_after_a_drop(self):
        dots = '<circle id="dot1" cx="50" cy="20" r="0.2" fill="#000"/><circle id="dot2" cx="52" cy="20" r="0.2" fill="#000"/>'
        data = gradient_document(middle=dots).replace(b'<text id="label" x="10" y="95" font-size="10">Hi</text>', b"")
        root, preceding, scene = scene_of(data)
        nodes = list(elements(root))
        runs = split_runs(scene)
        fixed = draw_runs(scene, runs, [None] * len(runs), root, preceding, set(), CPU)
        reference = render_svg(data, scene.width, scene.height, CPU)
        # A scene from an earlier drop that took dot1 only.
        keep = np.array([nodes[shape.element].get("id") != "dot1" for shape in fitted(scene).shapes])
        dropped = {shape.element for shape in scene.shapes if nodes[shape.element].get("id") == "dot1"}
        trial = copy.deepcopy(scene)
        trial.shapes = [shape for shape in trial.shapes if shape.element not in dropped]
        calls = []
        again, kept, gone, _, document = drop_shapes(root, preceding, trial, without(fixed, keep, stack), dropped, reference, CPU, lambda: calls.append(1))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(kept), len(fitted(trial).shapes))
        self.assertEqual({nodes[index].get("id") for index in gone}, {"dot2"})
        self.assertEqual(len(fitted(again).shapes), kept.sum())
        self.assertEqual(set(by_id(document)) & {"dot1", "dot2"}, set())


def path_data(data):
    return etree.fromstring(data).find("{*}path").get("d")


def near_straight_polygon(count, offset):
    """A polygon around a circle whose edges are cubics with control points `offset` user units off the chord."""
    points = [(50 + 40 * math.cos(2 * math.pi * i / count), 50 + 40 * math.sin(2 * math.pi * i / count)) for i in range(count)]
    d = f"M{points[0][0]:.3f} {points[0][1]:.3f}"
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        nx, ny = -dy / length * offset, dx / length * offset
        d += f"C{x0 + dx / 3 + nx:.3f} {y0 + dy / 3 + ny:.3f} {x0 + dx * 2 / 3 - nx:.3f} {y0 + dy * 2 / 3 - ny:.3f} {x1:.3f} {y1:.3f}"
    return svg('viewBox="0 0 100 100"', f'<path d="{d}z" fill="#2a6"/>')


class ColorTests(unittest.TestCase):
    def pixel(self, color, style=False):
        paint = f'style="fill:{color}"' if style else f'fill="{color}"'
        data = cairo_colors(svg('width="4" height="4"', f'<rect width="4" height="4" {paint}/>'))
        return np.round(cairo_render(data, (4, 4))[0, 0] * 255).astype(int)

    def test_colors_are_written_as_cairosvg_reads_them(self):
        for color, expected in (("#c42c", (214, 105, 78)), ("#cc4422cc", (214, 105, 78)), ("rgba(204,68,34,20%)", (245, 218, 211)), ("rgb(41 163 102)", (41, 163, 102)),
                                ("rgb(41 163 102 / 50%)", (148, 209, 178)), ("rgba(41,163,102,.5)", (148, 209, 178)), ("hsl(150 60% 40%)", (41, 163, 102)),
                                ("hsla(150deg, 60%, 40%, .5)", (148, 209, 178)), ("hsl(150deg 60% 40% / 50%)", (148, 209, 178)), ("hsl(0 100% 50%)", (255, 0, 0)),
                                ("#0000", (255, 255, 255)), ("red", (255, 0, 0))):
            for style in (False, True):
                with self.subTest(color=color, style=style):
                    np.testing.assert_allclose(self.pixel(color, style), expected, atol=1)

    def test_other_values_stay(self):
        data = svg('width="4"', '<g fill="currentColor" style="stroke:url(#g);fill:none"><rect fill="rgb(1.5 2 3)"/></g>')
        self.assertEqual(cairo_colors(data), data)

    def test_declared_colors_are_captured(self):
        for color, expected in (("hsl(150 60% 40%)", (41, 163, 102, 1)), ("rgb(41 163 102)", (41, 163, 102, 1)), ("#c42c", (204, 68, 34, 0.8))):
            with self.subTest(color=color):
                _, _, scene = scene_of(svg('width="4" height="4"', f'<rect width="4" height="4" fill="{color}"/>'))
                np.testing.assert_allclose(scene.shapes[0].fill, [value / 255 for value in expected[:3]] + [expected[3]], atol=1e-3)

    def test_alpha_of_a_hex_color_is_kept(self):
        root, _, scene = scene_of(svg('width="4" height="4"', '<rect width="4" height="4" fill="#c42c"/>'))
        element = list(elements(root))[scene.shapes[0].element]
        self.assertEqual(color_text(scene.shapes[0].fill, painted_color(element, "fill")), "#cc4422cc")


class StraightenTests(unittest.TestCase):
    def check_straightened(self, offset):
        data = near_straight_polygon(40, offset)
        result = simplify_svg(data, CPU)
        self.assertLess(len(result), len(optimize_svg(data, False)))
        self.assertNotRegex(path_data(optimize_svg(result, False)), r"[CcSsQqTt]")
        self.assertGreaterEqual(similarity(data, result, 512), MIN_PSNR)

    def test_nearly_straight_cubics_become_lines(self):
        self.check_straightened(0.004)

    def test_refined_straight_edges_are_recovered(self):
        self.check_straightened(0.06)

    def test_segments_beyond_the_tolerance_or_out_of_order_stay_curves(self):
        root, preceding, scene = scene_of(svg('viewBox="0 0 100 100"', '<path d="M10 10C30 12 50 8 70 10C90 10 40 10 20 10C30 30 40 20 50 50C60 60 55 55 80 80" fill="#000"/>'))
        shape = scene.shapes[0]
        before = shape.subpaths[0].copy()
        straighten(shape)
        self.assertEqual(list(shape.lines[0]), [False, False, False, False])
        self.assertTrue(np.array_equal(shape.subpaths[0], before))
        self.assertFalse(shape.changed)

    def test_control_points_move_to_thirds_of_the_chord(self):
        root, preceding, scene = scene_of(svg('viewBox="0 0 512 512"', '<path d="M10 10C20.01 20 30 30.01 40 40" fill="#000"/>'))
        shape = scene.shapes[0]
        straighten(shape)
        self.assertTrue(shape.lines[0][0] and shape.changed)
        self.assertTrue(np.allclose(shape.subpaths[0], [(10, 10), (20, 20), (30, 30), (40, 40)], atol=1e-4))


class FixedLayerTests(unittest.TestCase):
    MIXED = svg('viewBox="0 0 100 60"', '<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#f00"/><stop offset="1" stop-color="#00f"/></linearGradient></defs>'
                '<rect width="100" height="60" fill="#eee"/><path d="M10 10L40 10L40 30L10 30z" fill="#336" stroke="#f90" stroke-width="2"/>'
                '<rect x="20" y="15" width="40" height="25" fill="url(#g)"/><circle cx="50" cy="30" r="15" fill="#c33" opacity=".7"/>'
                '<text x="10" y="55" font-size="8">Hi</text><path d="M60 40h30v15h-30z" fill="#fa0"/>')

    def layers(self, data):
        root, preceding, scene = scene_of(data)
        runs = split_runs(scene)
        fixed = draw_runs(scene, runs, [None] * len(runs), root, preceding, set(), CPU)
        return scene, fitted(scene), fixed, render_svg(data, scene.width, scene.height, CPU)

    def test_composite_with_fixed_layers_matches_cairosvg(self):
        scene, fit, fixed, reference = self.layers(self.MIXED)
        self.assertEqual(len(fit.shapes), 4)
        self.assertEqual(sum(layer is not None for layer in fixed), 2)
        self.assertGreaterEqual(-10 * torch.log10((render(fit, fixed, CPU) - reference).square().mean()).item(), 50)

    def test_removal_costs_match_removals(self):
        _, fit, fixed, reference = self.layers(self.MIXED)
        base = (render(fit, fixed, CPU) - reference).square().mean().item()
        costs = removal_costs(fit, fixed, reference)
        for index, cost in enumerate(costs):
            keep = np.arange(len(costs)) != index
            rest = type(fit)(fit.width, fit.height, [shape for shape, kept in zip(fit.shapes, keep) if kept])
            measured = (render(rest, without(fixed, keep, stack), CPU) - reference).square().mean().item() - base
            self.assertAlmostEqual(cost, measured, delta=1e-5)

    def test_written_geometry_is_within_a_twentieth_of_a_pixel(self):
        data = svg('viewBox="0 0 24 24"', '<path d="M2 3C8 1 14 5 21 4S22 14 20 20C14 22 8 18 3 21z" fill="#369"/>')
        root, preceding, scene = scene_of(data)
        noise = np.random.default_rng(0).normal(0, 0.7, scene.shapes[0].subpaths[0].shape)
        scene.shapes[0].subpaths[0] = (scene.shapes[0].subpaths[0] + noise).astype(np.float32)
        scene.shapes[0].changed = True
        written = capture_svg(untagged(write_document(root, scene.shapes, set()), preceding), scene.width, scene.height)
        self.assertLess(np.abs(np.concatenate(written.shapes[0].subpaths) - np.concatenate(scene.shapes[0].subpaths)).max(), 0.05)

    def test_colors_and_widths_are_written_where_declared(self):
        data = svg('width="512" height="512"', '<g fill="#111"><path id="a" d="M5 5h40v40z"/><path id="b" d="M5 5h40v40z" style="fill:#222;stroke:#000;stroke-width:3"/>'
                   '<path id="c" d="M5 5h40v40z" fill="#333" stroke="#000" transform="scale(2)"/></g>')
        root, preceding, scene = scene_of(data)
        for shape in scene.shapes:
            shape.fill = (0, 0, 1, 1)
            shape.stroke = shape.stroke and (1, 0, 0, 1)
            shape.stroke_width = shape.stroke_width and shape.stroke_width * 2
            shape.changed = True
        result = by_id(untagged(write_document(root, scene.shapes, set()), preceding))
        self.assertEqual((result["a"].get("fill"), result["a"].get("style")), ("#0000ff", None))
        self.assertEqual(result["b"].get("style"), "fill:#0000ff;stroke:#ff0000;stroke-width:6")
        self.assertEqual((result["c"].get("fill"), result["c"].get("stroke-width")), ("#0000ff", "2"))
        self.assertEqual(result["c"].get("transform"), "scale(2)")

    def test_uniform_transforms(self):
        self.assertTrue(is_uniform((2, 0, 0, 2, 5, 1)))
        self.assertTrue(is_uniform((0, 3, -3, 0, 0, 0)))
        self.assertTrue(is_uniform((2, 0, 0, -2, 0, 0)))
        self.assertFalse(is_uniform((2, 0, 0, 1, 0, 0)))
        self.assertFalse(is_uniform((1, 0, 1, 1, 0, 0)))


class CompositeTests(unittest.TestCase):
    """Source-over onto a background that is not white, and its gradients, against plain PyTorch."""

    def check(self, composite, device):
        generator = torch.Generator().manual_seed(0)
        inputs = [torch.rand(shape, generator=generator).to(device).requires_grad_() for shape in ((5, 8, 9), (5, 3), (3, 8, 9))]
        weights = torch.rand(3, 8, 9, generator=generator).to(device)
        expected = inputs[2]
        for alpha, color in zip(*inputs[:2]):
            expected = expected + alpha * (color[:, None, None] - expected)
        boxes = torch.tensor([[0, 0, 9, 8]] * 5, dtype=torch.int32, device=device)
        starts = torch.arange(6, device=device) * 72
        out = composite(inputs[0].reshape(-1), inputs[1], inputs[2], boxes, starts, 0)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
        for grad, expected_grad in zip(torch.autograd.grad((out * weights).sum(), inputs), torch.autograd.grad((expected * weights).sum(), inputs)):
            torch.testing.assert_close(grad, expected_grad, rtol=1e-4, atol=1e-5)

    def test_torch_composite(self):
        self.check(svg_fit._TorchComposite.apply, CPU)

    @unittest.skipUnless(torch.cuda.is_available() and svg_fit.triton is not None, "needs CUDA and Triton")
    def test_triton_composite(self):
        self.check(svg_fit._Composite.apply, torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
