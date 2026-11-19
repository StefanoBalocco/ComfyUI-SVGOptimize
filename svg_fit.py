"""Differentiable rasterizer and fitting of an SVG scene to a raster image.

Every fittable shape of the SVG (captured through CairoSVG, see svg_scene) is
rasterized by a differentiable rasterizer: per pixel, the distance to each
paint layer's flattened outline and its winding number give an anti-aliased
coverage, and layers are composited source-over in paint order, between fixed
layers that hold what the rasterizer does not draw. Triton kernels
compute it on CUDA; elsewhere PyTorch computes the same values and gradients
from the pixels near each outline. Adam then moves the Bezier control points
and adjusts colors and stroke widths to match the image. Per-shape removal
costs tell which shapes can be dropped, and adjacent segments of the editable
shapes merge while the outline stays within a tolerance.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from .svg_scene import Scene, Shape

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

TILE = 16
# Outside a layer's padded bounding box: distance of a pixel that no outline reaches.
FAR = 1.0e4
# Flattening: samples per cubic segment, about one edge per 2 px of control polygon.
SAMPLE_COUNTS = (1, 2, 4, 8, 16)
FILL_NONZERO, FILL_EVENODD, STROKE = 0, 1, 2
POINT_LR, COLOR_LR, WIDTH_LR = 0.25, 0.01, 0.1
# A stroke corner whose tangent turns by less than about 1 degree is smooth;
# miter joins apply to sharper corners.
SMOOTH_COS = 0.9998
# Segments merge while the outline moves at most the tolerance from its samples
# at u = 0, 1/8, ..., 1 per original segment.
MERGE_SAMPLES = 9
NEWTON_STEPS = 3

if triton is not None:
    @triton.jit
    def _layer_tiles(boxes, layer, TILE: tl.constexpr):
        """The pixel box [x0,x1) x [y0,y1) of the layer, the tiles it spans across and their number."""
        x0 = tl.load(boxes + layer * 4)
        y0 = tl.load(boxes + layer * 4 + 1)
        x1 = tl.load(boxes + layer * 4 + 2)
        y1 = tl.load(boxes + layer * 4 + 3)
        tiles_x = (x1 - x0 + TILE - 1) // TILE
        return x0, y0, x1, y1, tiles_x, tiles_x * ((y1 - y0 + TILE - 1) // TILE)

    @triton.jit
    def _tile_pixels(tile, tiles_x, x0, y0, x1, y1, TILE: tl.constexpr):
        """Origin of the tile, the coordinates of its pixels and which of them lie in the box."""
        i = tl.arange(0, TILE * TILE)
        tx = x0 + (tile % tiles_x) * TILE
        ty = y0 + (tile // tiles_x) * TILE
        x = tx + i % TILE
        y = ty + i // TILE
        return tx, ty, x, y, (x < x1) & (y < y1)

    @triton.jit
    def _edge_along(rx, ry, dx, dy):
        return (rx * dx + ry * dy) / tl.maximum(dx * dx + dy * dy, 1e-12)

    @triton.jit
    def _cut_band(along, t, first, last, e, rx, ry, dx, dy, half):
        """The position on the edge, the distance and the band of a stroke edge,
        where the end edges of a cut line continue straight up to the cuts."""
        t = tl.where(first == e, tl.minimum(along, 1.0), t)
        t = tl.where(last == e, tl.maximum(along, t), t)
        d = tl.sqrt((rx - t * dx) * (rx - t * dx) + (ry - t * dy) * (ry - t * dy))
        return t, d, tl.maximum(tl.minimum(d + half, 0.5) - tl.maximum(d - half, -0.5), 0.0)

    @triton.jit
    def _reach_start(edges, first, px, py, extend, valid):
        """How far past the start of edge `first` the pixel lies."""
        fx = tl.load(edges + first * 4, mask=valid, other=0.0)
        fy = tl.load(edges + first * 4 + 1, mask=valid, other=0.0)
        fdx = tl.load(edges + first * 4 + 2, mask=valid, other=1.0) - fx
        fdy = tl.load(edges + first * 4 + 3, mask=valid, other=0.0) - fy
        return ((px - fx) * fdx + (py - fy) * fdy) / tl.maximum(tl.sqrt(fdx * fdx + fdy * fdy), 1e-6) + extend

    @triton.jit
    def _reach_end(edges, last, px, py, extend, valid):
        """How far before the end of edge `last` the pixel lies."""
        lx = tl.load(edges + last * 4 + 2, mask=valid, other=1.0)
        ly = tl.load(edges + last * 4 + 3, mask=valid, other=0.0)
        ldx = lx - tl.load(edges + last * 4, mask=valid, other=0.0)
        ldy = ly - tl.load(edges + last * 4 + 1, mask=valid, other=0.0)
        return ((lx - px) * ldx + (ly - py) * ldy) / tl.maximum(tl.sqrt(ldx * ldx + ldy * ldy), 1e-6) + extend

    @triton.jit
    def _coverage_forward(edges, cuts, squares, widths, offsets, boxes, starts, modes, cover, dist, index, E, TILE: tl.constexpr):
        """Box-filtered coverage of each pixel by one layer and the edge its
        gradient goes to. A fill (nonzero or even-odd winding) also keeps its
        signed distance to the outline (positive inside). A stroke is the union
        of the bands around its edges; near a butt or square open end, they are
        cut where the first (cuts[0]) or last (cuts[1]) edge of the line starts
        or ends, or half a width beyond for square caps. The pixels of the layer's
        box are stored row by row from its start in the packed arrays."""
        tile = tl.program_id(0)
        layer = tl.program_id(1)
        x0, y0, x1, y1, tiles_x, tiles = _layer_tiles(boxes, layer, TILE)
        if tile < tiles:
            tx, ty, x, y, valid = _tile_pixels(tile, tiles_x, x0, y0, x1, y1, TILE)
            out = tl.load(starts + layer) + (y - y0) * (x1 - x0) + x - x0
            px = x.to(tl.float32) + 0.5
            py = y.to(tl.float32) + 0.5
            mode = tl.load(modes + layer)
            half = tl.load(widths + layer) * 0.5
            best = tl.full((TILE * TILE,), 3.0e38, tl.float32)
            closest = tl.full((TILE * TILE,), -1, tl.int32)
            winding = tl.zeros((TILE * TILE,), tl.int32)
            covered = tl.zeros((TILE * TILE,), tl.float32)
            if mode == 2:
                extend = tl.where(tl.load(squares + layer) != 0, half, 0.0)
                for e in range(tl.load(offsets + layer), tl.load(offsets + layer + 1)):
                    ax = tl.load(edges + e * 4)
                    ay = tl.load(edges + e * 4 + 1)
                    dx = tl.load(edges + e * 4 + 2) - ax
                    dy = tl.load(edges + e * 4 + 3) - ay
                    rx = px - ax
                    ry = py - ay
                    along = _edge_along(rx, ry, dx, dy)
                    t = tl.minimum(tl.maximum(along, 0.0), 1.0)
                    d2 = (rx - t * dx) * (rx - t * dx) + (ry - t * dy) * (ry - t * dy)
                    first = tl.load(cuts + e)
                    last = tl.load(cuts + E + e)
                    _, _, band = _cut_band(along, t, first, last, e, rx, ry, dx, dy, half)
                    cut_first = first >= 0
                    cut_last = last >= 0
                    band *= tl.where(cut_first, tl.minimum(tl.maximum(_reach_start(edges, first, px, py, extend, cut_first) + 0.5, 0.0), 1.0), 1.0)
                    band *= tl.where(cut_last, tl.minimum(tl.maximum(_reach_end(edges, last, px, py, extend, cut_last) + 0.5, 0.0), 1.0), 1.0)
                    better = (band > covered) | ((band == covered) & (d2 < best))
                    covered = tl.where(better, band, covered)
                    best = tl.where(better, d2, best)
                    closest = tl.where(better, e, closest)
                signed = tl.sqrt(best)
            else:
                for e in range(tl.load(offsets + layer), tl.load(offsets + layer + 1)):
                    ax = tl.load(edges + e * 4)
                    ay = tl.load(edges + e * 4 + 1)
                    by = tl.load(edges + e * 4 + 3)
                    dx = tl.load(edges + e * 4 + 2) - ax
                    dy = by - ay
                    rx = px - ax
                    ry = py - ay
                    t = tl.minimum(tl.maximum(_edge_along(rx, ry, dx, dy), 0.0), 1.0)
                    qx = rx - t * dx
                    qy = ry - t * dy
                    d2 = qx * qx + qy * qy
                    closer = d2 < best
                    best = tl.where(closer, d2, best)
                    closest = tl.where(closer, e, closest)
                    cross = dx * ry - rx * dy
                    winding += ((ay <= py) & (by > py) & (cross > 0)).to(tl.int32)
                    winding -= ((ay > py) & (by <= py) & (cross < 0)).to(tl.int32)
                inside = tl.where(mode == 1, (winding & 1) != 0, winding != 0)
                d = tl.sqrt(best)
                signed = tl.where(inside, d, -d)
                covered = tl.minimum(tl.maximum(signed + 0.5, 0.0), 1.0)
            tl.store(cover + out, covered, mask=valid)
            tl.store(dist + out, signed, mask=valid)
            tl.store(index + out, closest, mask=valid)

    @triton.jit
    def _kite_cover(cover, boxes, starts, kite, x, y, valid):
        """Coverage of a miter-join layer at the pixels (x, y) of its owner's box, zero outside the kite layer's own box."""
        kx0 = tl.load(boxes + kite * 4)
        ky0 = tl.load(boxes + kite * 4 + 1)
        kx1 = tl.load(boxes + kite * 4 + 2)
        ky1 = tl.load(boxes + kite * 4 + 3)
        inside = valid & (x >= kx0) & (x < kx1) & (y >= ky0) & (y < ky1)
        out = tl.load(starts + kite) + (y - ky0) * (kx1 - kx0) + x - kx0
        return tl.load(cover + out, mask=inside, other=0.0), out, inside

    @triton.jit
    def _alpha_forward(cover, kites, scale, scale_layer, scale_y, scale_x, boxes, starts, alpha, TILE: tl.constexpr):
        """Alpha of each layer: its coverage, joined with its miter join's by
        the larger value, times its opacity and clip."""
        tile = tl.program_id(0)
        layer = tl.program_id(1)
        x0, y0, x1, y1, tiles_x, tiles = _layer_tiles(boxes, layer, TILE)
        if tile < tiles:
            tx, ty, x, y, valid = _tile_pixels(tile, tiles_x, x0, y0, x1, y1, TILE)
            out = tl.load(starts + layer) + (y - y0) * (x1 - x0) + x - x0
            c = tl.load(cover + out, mask=valid, other=0.0)
            kite = tl.load(kites + layer)
            if kite >= 0:
                kite_c, kite_out, inside = _kite_cover(cover, boxes, starts, kite, x, y, valid)
                c = tl.maximum(c, kite_c)
            tl.store(alpha + out, c * tl.load(scale + layer * scale_layer + y * scale_y + x * scale_x, mask=valid, other=0.0), mask=valid)

    @triton.jit
    def _alpha_backward(grad_alpha, cover, kites, scale, scale_layer, scale_y, scale_x, boxes, starts, grad_cover, TILE: tl.constexpr):
        """Gradient of each layer's and its miter join's coverage; the larger of the two takes it, a tie halves it."""
        tile = tl.program_id(0)
        layer = tl.program_id(1)
        x0, y0, x1, y1, tiles_x, tiles = _layer_tiles(boxes, layer, TILE)
        if tile < tiles:
            tx, ty, x, y, valid = _tile_pixels(tile, tiles_x, x0, y0, x1, y1, TILE)
            out = tl.load(starts + layer) + (y - y0) * (x1 - x0) + x - x0
            g = tl.load(grad_alpha + out, mask=valid, other=0.0) * tl.load(scale + layer * scale_layer + y * scale_y + x * scale_x, mask=valid, other=0.0)
            kite = tl.load(kites + layer)
            if kite >= 0:
                c = tl.load(cover + out, mask=valid, other=0.0)
                kite_c, kite_out, inside = _kite_cover(cover, boxes, starts, kite, x, y, valid)
                owner = tl.where(c > kite_c, 1.0, tl.where(c == kite_c, 0.5, 0.0))
                tl.store(grad_cover + kite_out, g * (1.0 - owner), mask=inside)
                g = g * owner
            tl.store(grad_cover + out, g, mask=valid)

    @triton.jit
    def _cut_gradient(edges, grad_edges, cut, start: tl.constexpr, g, px, py, m):
        """Gradient of a cut's reach (from the start or before the end of edge
        `cut`) into that edge's end points."""
        ax = tl.load(edges + cut * 4, mask=m, other=0.0)
        ay = tl.load(edges + cut * 4 + 1, mask=m, other=0.0)
        dx = tl.load(edges + cut * 4 + 2, mask=m, other=1.0) - ax
        dy = tl.load(edges + cut * 4 + 3, mask=m, other=1.0) - ay
        length = tl.maximum(tl.sqrt(dx * dx + dy * dy), 1e-6)
        nx = dx / length
        ny = dy / length
        # reach = (p - a).n at the start, |b - a| - (p - a).n at the end; w is
        # the part of p - a normal to the edge, over |b - a|.
        along = (px - ax) * nx + (py - ay) * ny
        wx = (px - ax - along * nx) / length
        wy = (py - ay - along * ny) / length
        if start:
            tl.atomic_add(grad_edges + cut * 4, g * (-nx - wx), mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 1, g * (-ny - wy), mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 2, g * wx, mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 3, g * wy, mask=m)
        else:
            tl.atomic_add(grad_edges + cut * 4, g * wx, mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 1, g * wy, mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 2, g * (nx - wx), mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 3, g * (ny - wy), mask=m)

    @triton.jit
    def _coverage_backward(edges, cuts, squares, widths, modes, boxes, starts, dist, index, grad_cover, grad_edges, grad_widths, E, TILE: tl.constexpr):
        tile = tl.program_id(0)
        layer = tl.program_id(1)
        x0, y0, x1, y1, tiles_x, tiles = _layer_tiles(boxes, layer, TILE)
        if tile >= tiles:
            return
        tx, ty, x, y, valid = _tile_pixels(tile, tiles_x, x0, y0, x1, y1, TILE)
        out = tl.load(starts + layer) + (y - y0) * (x1 - x0) + x - x0
        e = tl.load(index + out, mask=valid, other=-1)
        g = tl.load(grad_cover + out, mask=valid, other=0.0)
        m = (e >= 0) & (g != 0)
        ax = tl.load(edges + e * 4, mask=m, other=0.0)
        ay = tl.load(edges + e * 4 + 1, mask=m, other=0.0)
        dx = tl.load(edges + e * 4 + 2, mask=m, other=1.0) - ax
        dy = tl.load(edges + e * 4 + 3, mask=m, other=1.0) - ay
        px = x.to(tl.float32) + 0.5
        py = y.to(tl.float32) + 0.5
        rx = px - ax
        ry = py - ay
        along = _edge_along(rx, ry, dx, dy)
        t = tl.minimum(tl.maximum(along, 0.0), 1.0)
        # The box filter's derivative is 1 within half a pixel of an outline,
        # hence 0 at every pixel center when outlines lie on the pixel grid.
        # Back-propagate the 2 px tent filter's instead: same total along a
        # straight outline, never zero next to it.
        if tl.load(modes + layer) == 2:
            half = tl.load(widths + layer) * 0.5
            extend = tl.where(tl.load(squares + layer) != 0, half, 0.0)
            first = tl.load(cuts + e, mask=m, other=-1)
            last = tl.load(cuts + E + e, mask=m, other=-1)
            t, d, band = _cut_band(along, t, first, last, e, rx, ry, dx, dy, half)
            outer = tl.maximum(1 - tl.abs(half - d), 0.0)
            inner = tl.maximum(1 - tl.abs(half + d), 0.0)
            cut_first = m & (first >= 0)
            reach_start = _reach_start(edges, first, px, py, extend, cut_first)
            cut_last = m & (last >= 0)
            reach_end = _reach_end(edges, last, px, py, extend, cut_last)
            cut_start = tl.where(cut_first, tl.minimum(tl.maximum(reach_start + 0.5, 0.0), 1.0), 1.0)
            cut_end = tl.where(cut_last, tl.minimum(tl.maximum(reach_end + 0.5, 0.0), 1.0), 1.0)
            g_start = tl.where(cut_first, g * band * cut_end * tl.maximum(1 - tl.abs(reach_start), 0.0), 0.0)
            g_end = tl.where(cut_last, g * band * cut_start * tl.maximum(1 - tl.abs(reach_end), 0.0), 0.0)
            g_d = g * cut_start * cut_end * (inner - outer)
            g_half = g * cut_start * cut_end * (outer + inner) + tl.where(tl.load(squares + layer) != 0, g_start + g_end, 0.0)
            tl.atomic_add(grad_widths + layer, tl.sum(tl.where(m, 0.5 * g_half, 0.0), 0))
            _cut_gradient(edges, grad_edges, first, True, g_start, px, py, cut_first & (g_start != 0))
            _cut_gradient(edges, grad_edges, last, False, g_end, px, py, cut_last & (g_end != 0))
        else:
            signed = tl.load(dist + out, mask=m, other=0.0)
            g_d = tl.where(signed >= 0, g, -g) * tl.maximum(1 - tl.abs(signed), 0.0)
        qx = t * dx - rx
        qy = t * dy - ry
        scale = g_d / tl.maximum(tl.sqrt(qx * qx + qy * qy), 1e-6)
        tl.atomic_add(grad_edges + e * 4, scale * qx * (1 - t), mask=m)
        tl.atomic_add(grad_edges + e * 4 + 1, scale * qy * (1 - t), mask=m)
        tl.atomic_add(grad_edges + e * 4 + 2, scale * qx * t, mask=m)
        tl.atomic_add(grad_edges + e * 4 + 3, scale * qy * t, mask=m)

    @triton.jit
    def _kite_corners(corners, c, C, m):
        """Offsets in the points of the head and tail of the arriving and leaving edge of each corner."""
        in_head = tl.load(corners + C + c, mask=m, other=0) * 2
        in_tail = tl.load(corners + 2 * C + c, mask=m, other=0) * 2
        out_head = tl.load(corners + 3 * C + c, mask=m, other=0) * 2
        out_tail = tl.load(corners + 4 * C + c, mask=m, other=0) * 2
        return in_head, in_tail, out_head, out_tail

    @triton.jit
    def _kite_geometry(points, widths, corners, layers, limits, c, C, m):
        """Unit tangents arriving at (a) and leaving (b) each corner and their
        norms, the outer side of the turn, the half width signed toward it, the
        miter scale k of the tip and whether k varies (miter limit holds)."""
        in_head, in_tail, out_head, out_tail = _kite_corners(corners, c, C, m)
        ax = tl.load(points + in_head, mask=m, other=1.0) - tl.load(points + in_tail, mask=m, other=0.0)
        ay = tl.load(points + in_head + 1, mask=m, other=0.0) - tl.load(points + in_tail + 1, mask=m, other=0.0)
        bx = tl.load(points + out_head, mask=m, other=1.0) - tl.load(points + out_tail, mask=m, other=0.0)
        by = tl.load(points + out_head + 1, mask=m, other=0.0) - tl.load(points + out_tail + 1, mask=m, other=0.0)
        la = tl.maximum(tl.sqrt(ax * ax + ay * ay), 1e-3)
        lb = tl.maximum(tl.sqrt(bx * bx + by * by), 1e-3)
        ax, ay, bx, by = ax / la, ay / la, bx / lb, by / lb
        cross = ax * by - ay * bx
        cos = ax * bx + ay * by
        side = tl.where(cross > 0, -1.0, tl.where(cross < 0, 1.0, 0.0))
        half = side * 0.5 * tl.load(widths + tl.load(layers + c, mask=m, other=0), mask=m, other=0.0)
        miter = (1 + cos) * 0.5 >= tl.load(limits + c, mask=m, other=1.0)
        k = tl.where(miter, 1 / tl.maximum(1 + cos, 1e-6), 0.5)
        return ax, ay, bx, by, la, lb, side, half, k, miter & (1 + cos > 1e-6)

    @triton.jit
    def _kite_forward(points, widths, corners, layers, limits, ring, C, BLOCK: tl.constexpr):
        """Corner, outer offsets and miter tip (or bevel midpoint) of each miter join."""
        c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = c < C
        ax, ay, bx, by, la, lb, _, half, k, _ = _kite_geometry(points, widths, corners, layers, limits, c, C, m)
        v = tl.load(corners + c, mask=m, other=0) * 2
        vx = tl.load(points + v, mask=m, other=0.0)
        vy = tl.load(points + v + 1, mask=m, other=0.0)
        # Left normals (-y, x) scaled by the signed half width.
        tl.store(ring + c * 8, vx, mask=m)
        tl.store(ring + c * 8 + 1, vy, mask=m)
        tl.store(ring + c * 8 + 2, vx - half * ay, mask=m)
        tl.store(ring + c * 8 + 3, vy + half * ax, mask=m)
        tl.store(ring + c * 8 + 4, vx - half * k * (ay + by), mask=m)
        tl.store(ring + c * 8 + 5, vy + half * k * (ax + bx), mask=m)
        tl.store(ring + c * 8 + 6, vx - half * by, mask=m)
        tl.store(ring + c * 8 + 7, vy + half * bx, mask=m)

    @triton.jit
    def _kite_backward(points, widths, corners, layers, limits, grad_ring, grad_points, grad_widths, C, BLOCK: tl.constexpr):
        c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = c < C
        ax, ay, bx, by, la, lb, side, half, k, varying = _kite_geometry(points, widths, corners, layers, limits, c, C, m)
        g0x = tl.load(grad_ring + c * 8, mask=m, other=0.0)
        g0y = tl.load(grad_ring + c * 8 + 1, mask=m, other=0.0)
        g1x = tl.load(grad_ring + c * 8 + 2, mask=m, other=0.0)
        g1y = tl.load(grad_ring + c * 8 + 3, mask=m, other=0.0)
        g2x = tl.load(grad_ring + c * 8 + 4, mask=m, other=0.0)
        g2y = tl.load(grad_ring + c * 8 + 5, mask=m, other=0.0)
        g3x = tl.load(grad_ring + c * 8 + 6, mask=m, other=0.0)
        g3y = tl.load(grad_ring + c * 8 + 7, mask=m, other=0.0)
        # Normals na = (-ay, ax), nb = (-by, bx); first = v + half na,
        # second = v + half nb, tip = v + half k (na + nb).
        gnax = half * (g1x + k * g2x)
        gnay = half * (g1y + k * g2y)
        gnbx = half * (g3x + k * g2x)
        gnby = half * (g3y + k * g2y)
        tip = g2x * (-ay - by) + g2y * (ax + bx)
        g_half = g1x * -ay + g1y * ax + g3x * -by + g3y * bx + k * tip
        # k = 1 / (1 + a.b) within the miter limit.
        g_cos = tl.where(varying, -k * k * half * tip, 0.0)
        gax = gnay + g_cos * bx
        gay = -gnax + g_cos * by
        gbx = gnby + g_cos * ax
        gby = -gnbx + g_cos * ay
        # Through the normalization of a and b (linear below the 1e-3 floor).
        da = tl.where(la > 1e-3, ax * gax + ay * gay, 0.0)
        db = tl.where(lb > 1e-3, bx * gbx + by * gby, 0.0)
        gax, gay = (gax - ax * da) / la, (gay - ay * da) / la
        gbx, gby = (gbx - bx * db) / lb, (gby - by * db) / lb
        v = tl.load(corners + c, mask=m, other=0) * 2
        tl.atomic_add(grad_points + v, g0x + g1x + g2x + g3x, mask=m)
        tl.atomic_add(grad_points + v + 1, g0y + g1y + g2y + g3y, mask=m)
        in_head, in_tail, out_head, out_tail = _kite_corners(corners, c, C, m)
        tl.atomic_add(grad_points + in_head, gax, mask=m)
        tl.atomic_add(grad_points + in_head + 1, gay, mask=m)
        tl.atomic_add(grad_points + in_tail, -gax, mask=m)
        tl.atomic_add(grad_points + in_tail + 1, -gay, mask=m)
        tl.atomic_add(grad_points + out_head, gbx, mask=m)
        tl.atomic_add(grad_points + out_head + 1, gby, mask=m)
        tl.atomic_add(grad_points + out_tail, -gbx, mask=m)
        tl.atomic_add(grad_points + out_tail + 1, -gby, mask=m)
        tl.atomic_add(grad_widths + tl.load(layers + c, mask=m, other=0), 0.5 * side * g_half, mask=m)

    @triton.jit
    def _load_rgb(base, HW, p, m, other):
        return (tl.load(base + p, mask=m, other=other), tl.load(base + HW + p, mask=m, other=other),
                tl.load(base + 2 * HW + p, mask=m, other=other))

    @triton.jit
    def _box_pixels(boxes, starts, layer, tx, ty, x, y, valid, TILE: tl.constexpr):
        """Whether the tile meets the layer's box, which of its pixels lie in it
        and their offsets in the packed arrays."""
        x0 = tl.load(boxes + layer * 4)
        y0 = tl.load(boxes + layer * 4 + 1)
        x1 = tl.load(boxes + layer * 4 + 2)
        y1 = tl.load(boxes + layer * 4 + 3)
        meets = (tx < x1) & (tx + TILE > x0) & (ty < y1) & (ty + TILE > y0)
        inside = valid & (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        return meets, inside, tl.load(starts + layer) + (y - y0) * (x1 - x0) + x - x0

    @triton.jit
    def _composite_forward(alpha, colors, background, boxes, starts, first, out, N, H, W, tiles_x, TILE: tl.constexpr):
        tx, ty, x, y, valid = _tile_pixels(tl.program_id(0), tiles_x, 0, 0, W, H, TILE)
        HW = H * W
        p = y * W + x
        r, g, b = _load_rgb(background, HW, p, valid, 1.0)
        for n in range(N):
            meets, inside, packed = _box_pixels(boxes, starts, first + n, tx, ty, x, y, valid, TILE)
            if meets:
                a = tl.load(alpha + packed, mask=inside, other=0.0)
                r += a * (tl.load(colors + n * 3) - r)
                g += a * (tl.load(colors + n * 3 + 1) - g)
                b += a * (tl.load(colors + n * 3 + 2) - b)
        tl.store(out + p, r, mask=valid)
        tl.store(out + HW + p, g, mask=valid)
        tl.store(out + 2 * HW + p, b, mask=valid)

    @triton.jit
    def _composite_backward(alpha, colors, background, boxes, starts, first, grad_out, weight, grad_alpha, grad_colors, grad_background, N, H, W, tiles_x, TILE: tl.constexpr):
        """Gradients of source-over compositing: `weight` holds the transmittance
        of the layers above each layer. The background shows through the
        transmittance of all the layers."""
        tx, ty, x, y, valid = _tile_pixels(tl.program_id(0), tiles_x, 0, 0, W, H, TILE)
        HW = H * W
        p = y * W + x
        gr, gg, gb = _load_rgb(grad_out, HW, p, valid, 0.0)
        above = tl.zeros((TILE * TILE,), tl.float32) + 1.0
        for k in range(N):
            n = N - 1 - k
            meets, inside, packed = _box_pixels(boxes, starts, first + n, tx, ty, x, y, valid, TILE)
            if meets:
                tl.store(weight + packed, above, mask=inside)
                above *= 1 - tl.load(alpha + packed, mask=inside, other=0.0)
        tl.store(grad_background + p, gr * above, mask=valid)
        tl.store(grad_background + HW + p, gg * above, mask=valid)
        tl.store(grad_background + 2 * HW + p, gb * above, mask=valid)
        r, g, b = _load_rgb(background, HW, p, valid, 1.0)
        for n in range(N):
            meets, inside, packed = _box_pixels(boxes, starts, first + n, tx, ty, x, y, valid, TILE)
            if meets:
                a = tl.load(alpha + packed, mask=inside, other=0.0)
                above = tl.load(weight + packed, mask=inside, other=0.0)
                cr = tl.load(colors + n * 3)
                cg = tl.load(colors + n * 3 + 1)
                cb = tl.load(colors + n * 3 + 2)
                tl.store(grad_alpha + packed, above * (gr * (cr - r) + gg * (cg - g) + gb * (cb - b)), mask=inside)
                visible = a * above
                tl.atomic_add(grad_colors + n * 3, tl.sum(visible * gr, 0))
                tl.atomic_add(grad_colors + n * 3 + 1, tl.sum(visible * gg, 0))
                tl.atomic_add(grad_colors + n * 3 + 2, tl.sum(visible * gb, 0))
                r += a * (cr - r)
                g += a * (cg - g)
                b += a * (cb - b)


class _LayerAlpha(torch.autograd.Function):
    """Alpha of every layer over the pixels of its box, packed: the boxes' pixels row by row, one layer after the other."""

    @staticmethod
    def forward(ctx, edges, cuts, squares, widths, offsets, modes, kites, scale, boxes, starts, height, width):
        visible = len(kites)
        total, shown = starts[[-1, visible]].tolist()
        cover = torch.empty(total, device=edges.device)
        dist = torch.empty_like(cover)
        index = torch.empty(total, device=edges.device, dtype=torch.int32)
        alpha = torch.empty(shown, device=edges.device)
        tiles = triton.cdiv(width, TILE) * triton.cdiv(height, TILE)
        _coverage_forward[(tiles, len(modes))](edges, cuts, squares, widths, offsets, boxes, starts, modes, cover, dist, index, len(edges), TILE=TILE)
        ctx.strides = scale.stride()
        _alpha_forward[(tiles, visible)](cover, kites, scale, *ctx.strides, boxes, starts, alpha, TILE=TILE)
        ctx.save_for_backward(edges, cuts, squares, widths, modes, kites, scale, boxes, starts, cover, dist, index)
        ctx.tiles = tiles
        return alpha

    @staticmethod
    def backward(ctx, grad_alpha):
        edges, cuts, squares, widths, modes, kites, scale, boxes, starts, cover, dist, index = ctx.saved_tensors
        grad_cover = torch.empty_like(cover)
        grad_edges = torch.zeros_like(edges)
        grad_widths = torch.zeros_like(widths)
        _alpha_backward[(ctx.tiles, len(kites))](grad_alpha.contiguous(), cover, kites, scale, *ctx.strides, boxes, starts, grad_cover, TILE=TILE)
        _coverage_backward[(ctx.tiles, len(modes))](edges, cuts, squares, widths, modes, boxes, starts, dist, index, grad_cover, grad_edges, grad_widths, len(edges), TILE=TILE)
        return grad_edges, None, None, grad_widths, None, None, None, None, None, None, None, None


class _MiterKites(torch.autograd.Function):
    """[C,4,2] corner, outer offsets and miter tip of each miter join."""

    @staticmethod
    def forward(ctx, points, widths, corners, layers, limits):
        count = corners.shape[1]
        ring = torch.empty((count, 4, 2), device=points.device)
        _kite_forward[(triton.cdiv(count, 128),)](points, widths, corners, layers, limits, ring, count, BLOCK=128)
        ctx.save_for_backward(points, widths, corners, layers, limits)
        return ring

    @staticmethod
    def backward(ctx, grad_ring):
        points, widths, corners, layers, limits = ctx.saved_tensors
        count = corners.shape[1]
        grad_points = torch.zeros_like(points)
        grad_widths = torch.zeros_like(widths)
        _kite_backward[(triton.cdiv(count, 128),)](points, widths, corners, layers, limits, grad_ring.contiguous(), grad_points, grad_widths, count, BLOCK=128)
        return grad_points, grad_widths, None, None, None


class _Flatten(torch.autograd.Function):
    """Product with a constant sparse matrix, whose transpose is prebuilt for the backward."""

    @staticmethod
    def forward(ctx, points, matrix, transposed):
        ctx.transposed = transposed
        return matrix @ points

    @staticmethod
    def backward(ctx, grad):
        return ctx.transposed @ grad, None, None


class _Composite(torch.autograd.Function):
    """Source-over of solid-color layers with packed alpha onto a [3,H,W] background: `colors` are those of the layers from `first` on."""

    @staticmethod
    def forward(ctx, alpha, colors, background, boxes, starts, first):
        _, height, width = background.shape
        background = background.contiguous()
        out = torch.empty_like(background)
        tiles_x = triton.cdiv(width, TILE)
        _composite_forward[(tiles_x * triton.cdiv(height, TILE),)](alpha, colors, background, boxes, starts, first, out, len(colors), height, width, tiles_x, TILE=TILE)
        ctx.save_for_backward(alpha, colors, background, boxes, starts)
        ctx.first = first
        return out

    @staticmethod
    def backward(ctx, grad_out):
        alpha, colors, background, boxes, starts = ctx.saved_tensors
        _, height, width = background.shape
        weight = torch.empty_like(alpha)
        grad_alpha = torch.zeros_like(alpha)
        grad_colors = torch.zeros_like(colors)
        grad_background = torch.empty_like(background)
        tiles_x = triton.cdiv(width, TILE)
        _composite_backward[(tiles_x * triton.cdiv(height, TILE),)](alpha, colors, background, boxes, starts, ctx.first, grad_out.contiguous(), weight, grad_alpha, grad_colors, grad_background, len(colors), height, width, tiles_x, TILE=TILE)
        return grad_alpha, grad_colors, grad_background, None, None, None


def _near_pixels(edges, pad, height, width):
    """(edge, y, x) of every pixel whose center lies in an edge's bounding box grown by `pad`."""
    start = ((torch.minimum(edges[:, :2], edges[:, 2:]) - pad[:, None] - 0.5).ceil().long()).clamp(min=0)
    stop = torch.minimum((torch.maximum(edges[:, :2], edges[:, 2:]) + pad[:, None] - 0.5).floor().long() + 1, torch.tensor([width, height], device=edges.device))
    size = (stop - start).clamp(min=0)
    counts = size[:, 0] * size[:, 1]
    edge = torch.repeat_interleave(torch.arange(len(edges), device=edges.device), counts)
    local = torch.arange(len(edge), device=edges.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    return edge, start[edge, 1] + local // size[edge, 0], start[edge, 0] + local % size[edge, 0]


def _first_by_rank(key, *ranks):
    """Mask of the one pair per key with the lexicographically smallest ranks."""
    group = torch.unique(key, return_inverse=True)[1]
    keep = torch.ones_like(key, dtype=torch.bool)
    for rank in ranks:
        top = torch.finfo(rank.dtype).max if rank.is_floating_point() else torch.iinfo(rank.dtype).max
        rank = rank.masked_fill(~keep, top)
        keep &= rank == torch.full((len(key),), top, dtype=rank.dtype, device=key.device).scatter_reduce(0, group, rank, "amin")[group]
    return keep


def _winding(edges, edge_layer, boxes, offsets, fill, height):
    """Winding number of every pixel center in each fill layer's box, the boxes
    packed back to back: each edge crossing a pixel row counts for the pixels
    left of the crossing, +1 upward, -1 downward."""
    x0, y0, x1, y1 = boxes.unbind(1)
    width, rows = x1 - x0, torch.where(fill, y1 - y0, 0)
    stride = width + 1
    steps_start = (rows * stride).cumsum(0) - rows * stride
    ax, ay, bx, by = edges.unbind(1)
    first = (torch.minimum(ay, by) - 0.5).ceil().long().clamp(0, height)
    counts = ((torch.maximum(ay, by) - 0.5).ceil().long().clamp(0, height) - first).clamp(min=0)
    edge = torch.repeat_interleave(torch.arange(len(edges), device=edges.device), counts)
    row = first[edge] + torch.arange(len(edge), device=edges.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    cross = ax[edge] + (row + 0.5 - ay[edge]) * (bx[edge] - ax[edge]) / (by[edge] - ay[edge])
    sign = torch.where(by[edge] > ay[edge], 1, -1).int()
    layer = edge_layer[edge]
    base = steps_start[layer] + (row - y0[layer]) * stride[layer]
    steps = torch.zeros(int((rows * stride).sum()), dtype=torch.int32, device=edges.device)
    steps.index_add_(0, base, sign)
    steps.index_add_(0, base + torch.minimum(((cross - 0.5).ceil().long() - x0[layer]).clamp(min=0), width[layer]), -sign)
    winding = torch.zeros(int(offsets[-1]), dtype=torch.int32, device=edges.device)
    for start, offset, w, h in zip(steps_start.tolist(), offsets.tolist(), width.tolist(), rows.tolist()):
        winding[offset:offset + h * w].view(h, w).copy_(steps[start:start + h * (w + 1)].view(h, w + 1).cumsum(1, dtype=torch.int32)[:, :w])
    return winding


def _along(edges, px, py):
    """Direction of each edge, the pixel's offset from its start and its position along it, unclamped."""
    ax, ay, bx, by = edges.unbind(1)
    dx, dy, rx, ry = bx - ax, by - ay, px - ax, py - ay
    return dx, dy, rx, ry, (rx * dx + ry * dy) / (dx * dx + dy * dy).clamp(min=1e-12)


def _stroke_terms(edges, cuts, half, extend, e, px, py):
    """Per pixel and stroke edge, as in _coverage_forward: the band before the
    cuts at butt or square open ends, each cut's reach and factor."""
    dx, dy, rx, ry, along = _along(edges[e], px, py)
    t = along.clamp(0, 1)
    d2 = (rx - t * dx).square() + (ry - t * dy).square()
    first, last = cuts[0, e].long(), cuts[1, e].long()
    t = torch.where(first == e, along.clamp(max=1), t)
    t = torch.where(last == e, torch.maximum(along, t), t)
    d = ((rx - t * dx).square() + (ry - t * dy).square()).sqrt()
    band = ((d + half).clamp(max=0.5) - (d - half).clamp(min=-0.5)).clamp(min=0)
    fx, fy, fbx, fby = edges[first.clamp(min=0)].unbind(1)
    reach_start = ((px - fx) * (fbx - fx) + (py - fy) * (fby - fy)) / ((fbx - fx).square() + (fby - fy).square()).sqrt().clamp(min=1e-6) + extend
    lax, lay, lx, ly = edges[last.clamp(min=0)].unbind(1)
    reach_end = ((lx - px) * (lx - lax) + (ly - py) * (ly - lay)) / ((lx - lax).square() + (ly - lay).square()).sqrt().clamp(min=1e-6) + extend
    cut_start = torch.where(first >= 0, (reach_start + 0.5).clamp(0, 1), 1.0)
    cut_end = torch.where(last >= 0, (reach_end + 0.5).clamp(0, 1), 1.0)
    return dx, dy, rx, ry, t, d, d2, band, first, last, reach_start, reach_end, cut_start, cut_end


def _add_edge_gradient(grad_edges, edge, g_d, t, dx, dy, rx, ry):
    """Gradient `g_d` of each pixel's distance to its edge, into the edge's end points."""
    qx, qy = t * dx - rx, t * dy - ry
    scale = g_d / (qx * qx + qy * qy).sqrt().clamp(min=1e-6)
    grad_edges.index_add_(0, edge, torch.stack([scale * qx * (1 - t), scale * qy * (1 - t), scale * qx * t, scale * qy * t], 1))


def _add_cut_gradient(edges, grad_edges, cut, start, g, px, py):
    """As _cut_gradient: gradient of a cut's reach into the end points of edge `cut`."""
    keep = (cut >= 0) & (g != 0)
    cut, g, px, py = cut[keep], g[keep, None], px[keep], py[keep]
    ax, ay, bx, by = edges[cut].unbind(1)
    length = ((bx - ax).square() + (by - ay).square()).sqrt().clamp(min=1e-6)
    nx, ny = (bx - ax) / length, (by - ay) / length
    along = (px - ax) * nx + (py - ay) * ny
    wx, wy = (px - ax - along * nx) / length, (py - ay - along * ny) / length
    grad = torch.stack([-nx - wx, -ny - wy, wx, wy] if start else [wx, wy, nx - wx, ny - wy], 1)
    grad_edges.index_add_(0, cut, g * grad)


class _NearCoverage(torch.autograd.Function):
    """_LayerAlpha's coverage and gradients in PyTorch, for the CPU and other
    devices. Only pixels near an outline need distances: a fill's coverage is
    its inside test beyond half a pixel and its gradient vanishes beyond one;
    a stroke reaches half its width, plus its square cap and that pixel. The
    coverage is packed: the pixels of each layer's box, row by row, one layer
    after the other (`offsets` marks where each begins)."""

    @staticmethod
    def forward(ctx, edges, cuts, squares, widths, modes, edge_layer, boxes, offsets, height, width):
        boxes = boxes.long()
        x0, y0, x1, _ = boxes.unbind(1)
        box_width, start = x1 - x0, offsets[:-1]
        stroke = modes[edge_layer] == STROKE
        fills = (~stroke).nonzero()[:, 0]
        winding = _winding(edges[fills], edge_layer[fills], boxes, offsets, modes != STROKE, height)
        parity = torch.where(modes == FILL_EVENODD, 1, -1).int().repeat_interleave(offsets.diff())
        cover = ((winding & parity) != 0).to(edges.dtype)
        e, y, x = _near_pixels(edges[fills], edges.new_ones(len(fills)), height, width)
        e = fills[e]
        dx, dy, rx, ry, t = _along(edges[e], x + 0.5, y + 0.5)
        t = t.clamp(0, 1)
        d2 = (rx - t * dx).square() + (ry - t * dy).square()
        layer = edge_layer[e]
        key = start[layer] + (y - y0[layer]) * box_width[layer] + x - x0[layer]
        nearest = _first_by_rank(key, d2, e)
        fill_key, fill_edge, fill_x, fill_y = key[nearest], e[nearest], x[nearest], y[nearest]
        signed = torch.where(cover[fill_key] != 0, d2[nearest].sqrt(), -d2[nearest].sqrt())
        cover[fill_key] = (signed + 0.5).clamp(0, 1)
        strokes = stroke.nonzero()[:, 0]
        half = widths[edge_layer[strokes]] * 0.5
        e, y, x = _near_pixels(edges[strokes], half + torch.where(squares[edge_layer[strokes]] != 0, half, 0.0) + 2, height, width)
        e = strokes[e]
        half = widths[edge_layer[e]] * 0.5
        *_, d2, band, _, _, _, _, cut_start, cut_end = _stroke_terms(edges, cuts, half, torch.where(squares[edge_layer[e]] != 0, half, 0.0), e, x + 0.5, y + 0.5)
        band = band * cut_start * cut_end
        layer = edge_layer[e]
        key = start[layer] + (y - y0[layer]) * box_width[layer] + x - x0[layer]
        best = _first_by_rank(key, -band, d2, e)
        stroke_key, stroke_edge, stroke_x, stroke_y = key[best], e[best], x[best], y[best]
        cover[stroke_key] = band[best]
        ctx.save_for_backward(edges, cuts, squares, widths, edge_layer, fill_key, fill_edge, fill_x, fill_y, signed, stroke_key, stroke_edge, stroke_x, stroke_y)
        return cover

    @staticmethod
    def backward(ctx, grad_cover):
        edges, cuts, squares, widths, edge_layer, fill_key, fill_edge, fill_x, fill_y, signed, stroke_key, stroke_edge, stroke_x, stroke_y = ctx.saved_tensors
        grad_edges = torch.zeros_like(edges)
        grad_widths = torch.zeros_like(widths)
        # The 2 px tent filter's derivative, as in _coverage_backward.
        g = grad_cover[fill_key]
        dx, dy, rx, ry, t = _along(edges[fill_edge], fill_x + 0.5, fill_y + 0.5)
        t = t.clamp(0, 1)
        _add_edge_gradient(grad_edges, fill_edge, torch.where(signed >= 0, g, -g) * (1 - signed.abs()).clamp(min=0), t, dx, dy, rx, ry)
        g = grad_cover[stroke_key]
        px, py = stroke_x + 0.5, stroke_y + 0.5
        layer = edge_layer[stroke_edge]
        half = widths[layer] * 0.5
        square = squares[layer] != 0
        dx, dy, rx, ry, t, d, _, band, first, last, reach_start, reach_end, cut_start, cut_end = _stroke_terms(edges, cuts, half, torch.where(square, half, 0.0), stroke_edge, px, py)
        outer = (1 - (half - d).abs()).clamp(min=0)
        inner = (1 - (half + d).abs()).clamp(min=0)
        g_start = torch.where(first >= 0, g * band * cut_end * (1 - reach_start.abs()).clamp(min=0), 0.0)
        g_end = torch.where(last >= 0, g * band * cut_start * (1 - reach_end.abs()).clamp(min=0), 0.0)
        g = g * cut_start * cut_end
        grad_widths.index_add_(0, layer, 0.5 * (g * (outer + inner) + torch.where(square, g_start + g_end, 0.0)))
        _add_cut_gradient(edges, grad_edges, first, True, g_start, px, py)
        _add_cut_gradient(edges, grad_edges, last, False, g_end, px, py)
        _add_edge_gradient(grad_edges, stroke_edge, g * (inner - outer), t, dx, dy, rx, ry)
        return grad_edges, None, None, grad_widths, None, None, None, None, None, None


def _kites(points, widths, corners, layers, limits):
    """_MiterKites in PyTorch: [C,4,2] corner, outer offsets and miter tip (or bevel midpoint)."""
    vertex, in_head, in_tail, out_head, out_tail = points.index_select(0, corners.reshape(-1)).view(5, -1, 2)
    tangents = F.normalize(torch.stack([in_head, out_head]) - torch.stack([in_tail, out_tail]), dim=2, eps=1e-3)
    normals = tangents.flip(2) * tangents.new_tensor([-1.0, 1.0])
    cos = (tangents[0] * tangents[1]).sum(1, keepdim=True)
    offsets = normals * (-torch.sign((normals[0] * tangents[1]).sum(1, keepdim=True)) * widths.index_select(0, layers)[:, None] / 2)
    first, second = vertex + offsets
    tip = torch.where((1 + cos) / 2 >= limits[:, None], vertex + offsets.sum(0) / (1 + cos).clamp(min=1e-6), (first + second) / 2)
    return torch.stack([vertex, first, tip, second], 1)


class _TorchComposite(torch.autograd.Function):
    """_Composite in PyTorch: each layer's alpha covers its box [x0,y0,x1,y1)
    only and is zero elsewhere. One layer at a time like the kernels: no
    buffer of all layers in the forward pass, the transmittances in the
    backward pass take the size of the alpha."""

    @staticmethod
    def forward(ctx, alpha, colors, background, boxes, starts, first):
        out = background.clone()
        last = first + len(colors)
        for color, (x0, y0, x1, y1), start, stop in zip(colors, boxes[first:last].tolist(), starts[first:last].tolist(), starts[first + 1:last + 1].tolist()):
            out[:, y0:y1, x0:x1].lerp_(color[:, None, None], alpha[start:stop].view(y1 - y0, x1 - x0))
        ctx.save_for_backward(alpha, colors, background, boxes, starts)
        ctx.first = first
        return out

    @staticmethod
    def backward(ctx, grad_out):
        alpha, colors, background, boxes, starts = ctx.saved_tensors
        last = ctx.first + len(colors)
        grad_out = grad_out.contiguous()
        layers = list(zip(boxes[ctx.first:last].tolist(), starts[ctx.first:last].tolist(), starts[ctx.first + 1:last + 1].tolist()))
        # Transmittance of the layers above each layer, built from the top; what
        # is left at the end is the background's.
        above = torch.empty_like(alpha)
        through = torch.ones_like(grad_out[0])
        for (x0, y0, x1, y1), start, stop in reversed(layers):
            box = through[y0:y1, x0:x1]
            above[start:stop].view_as(box).copy_(box)
            box.mul_(1 - alpha[start:stop].view_as(box))
        # grad_out . composite below each layer, built bottom-up from the background.
        below = (grad_out * background).sum(0)
        grad_alpha = torch.zeros_like(alpha)
        grad_colors = torch.empty_like(colors)
        for color, grad_color, ((x0, y0, x1, y1), start, stop) in zip(colors, grad_colors, layers):
            grad = grad_out[:, y0:y1, x0:x1].reshape(3, -1)
            box = below[y0:y1, x0:x1]
            toward = (color @ grad).view_as(box)
            torch.mul(above[start:stop].view_as(box), toward - box, out=grad_alpha[start:stop].view_as(box))
            grad_color.copy_(grad @ (alpha[start:stop] * above[start:stop]))
            box.lerp_(toward, alpha[start:stop].view_as(box))
        return grad_alpha, grad_colors, grad_out * through, None, None, None


class _Plan:
    """Flattening and paint-layer plan of a scene: which control points make which
    polyline vertices, and which vertex pairs are the edges of each fill or stroke.
    `fixed` holds, for each place before a shape and after the last one, the
    premultiplied color [3,H,W] and alpha [H,W] of what is drawn there, or None."""

    def __init__(self, scene, fixed, device):
        self.triton = triton is not None and torch.device(device).type == "cuda"
        # Each polyline vertex is a fixed Bernstein combination of 4 control points
        # (a subpath's endpoint is its last control point).
        controls, line_starts, columns, weights, rings = [], [], [], [], []
        base = vertex = 0
        for shape in scene.shapes:
            shape_rings = []
            for subpath, closed, straight in zip(shape.subpaths, shape.closed, shape.lines):
                controls.append(subpath)
                line_starts.append(base + 3 * np.flatnonzero(straight))
                lengths = np.linalg.norm(np.diff(subpath, axis=0), axis=1).reshape(-1, 3).sum(1)
                counts = np.asarray(SAMPLE_COUNTS)[np.minimum(np.searchsorted(2 * np.asarray(SAMPLE_COUNTS), lengths), len(SAMPLE_COUNTS) - 1)]
                segment = np.repeat(np.arange(len(counts)), counts)
                t = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) / counts[segment]
                columns += [base + 3 * segment[:, None] + np.arange(4), np.full((1, 4), base + len(subpath) - 1)]
                weights += [np.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], 1), np.array([[1.0, 0, 0, 0]])]
                shape_rings.append((np.arange(vertex, vertex + counts.sum() + 1), closed, subpath, base))
                vertex += counts.sum() + 1
                base += len(subpath)
            rings.append(shape_rings)
        points = np.concatenate(controls)
        columns, weights = np.concatenate(columns), np.concatenate(weights)
        # The control points of a straight segment stay at 1/3 and 2/3 of its
        # endpoints: only the other points are parameters, so it stays straight.
        starts = np.concatenate(line_starts)
        free = np.ones(len(points), bool)
        free[starts + 1] = free[starts + 2] = False
        index = np.cumsum(free) - 1
        dependent = np.concatenate([starts + 1, starts + 1, starts + 2, starts + 2])
        rows = np.concatenate([np.flatnonzero(free), dependent])
        cols = np.concatenate([index[free], index[starts], index[starts + 3], index[starts], index[starts + 3]])
        values = np.concatenate([np.ones(free.sum()), np.full(len(starts), 2 / 3), np.full(len(starts), 1 / 3), np.full(len(starts), 1 / 3), np.full(len(starts), 2 / 3)])
        self.expand = sparse.csr_matrix((values, (rows, cols)), shape=(len(points), free.sum()))
        self.points = torch.from_numpy(points[free]).to(device)

        edge_starts, edge_ends, cuts, offsets, modes, colors, opacity, widths, clips = [], [], [], [0], [], [], [], [], []
        starts, tune_colors, tune_widths = [], [], []
        squares = []
        # Initial polyline vertices, to find the edges near open stroke ends.
        vertices = (weights[:, :, None] * points[columns]).sum(1)
        corners, corner_limits, miter_owners = [], [], []
        count = 0
        for shape, shape_rings in zip(scene.shapes, rings):
            starts.append(len(modes))
            for mode, paint in ((FILL_EVENODD if shape.fill_rule == "evenodd" else FILL_NONZERO, shape.fill), (STROKE, shape.stroke)):
                if paint is None:
                    continue
                for ids, closed, _, _ in shape_rings:
                    if mode != STROKE or closed:
                        edge_starts.append(ids)
                        edge_ends.append(np.roll(ids, -1))
                        cuts.append(np.full((2, len(ids)), -1, np.int32))
                    else:
                        edge_starts.append(ids[:-1])
                        edge_ends.append(ids[1:])
                        cuts.append(np.full((2, len(ids) - 1), -1, np.int32))
                        if shape.stroke_cap != "round":
                            # Round caps are the bands' own ends. Butt and square ends
                            # cut the edges within about a width of them.
                            lengths = np.linalg.norm(np.diff(vertices[ids], axis=0), axis=1)
                            reach = shape.stroke_width + 2
                            cuts[-1][0, np.cumsum(lengths) - lengths < reach] = count
                            cuts[-1][1, lengths[::-1].cumsum()[::-1] - lengths < reach] = count + len(lengths) - 1
                    count += len(edge_starts[-1])
                if mode == STROKE and shape.stroke_join == "miter":
                    layer_corners = np.concatenate([base + _corners(subpath, closed) for _, closed, subpath, base in shape_rings])
                    if len(layer_corners):
                        corners.append(layer_corners)
                        corner_limits.append(np.full(len(layer_corners), shape.miter_limit))
                        miter_owners.append(len(modes))
                offsets.append(count)
                squares.append(mode == STROKE and shape.stroke_cap == "square")
                modes.append(mode)
                colors.append(paint[:3])
                opacity.append(paint[3])
                widths.append(shape.stroke_width if mode == STROKE else 0.0)
                clips.append(shape.clip)
                tune_colors.append(shape.tune_color)
                tune_widths.append(shape.tune_width)
        starts.append(len(modes))
        self.runs = [(start, *layer) for start, layer in zip(starts, fixed) if layer is not None]
        # Edge endpoints as one constant sparse matrix over the free control points.
        ids = np.concatenate(edge_starts + edge_ends)
        flatten = (sparse.csr_matrix((weights[ids].reshape(-1), (np.repeat(np.arange(len(ids)), 4), columns[ids].reshape(-1))), shape=(len(ids), len(points))) @ self.expand).tocoo()
        values = torch.from_numpy(flatten.data).float()
        self.flatten = torch.sparse_coo_tensor(np.stack([flatten.row, flatten.col]), values, flatten.shape, check_invariants=True).coalesce().to(device)
        self.flatten_t = torch.sparse_coo_tensor(np.stack([flatten.col, flatten.row]), values, flatten.shape[::-1], check_invariants=True).coalesce().to(device)
        # Miter joins: one more fill layer per miter-joined stroke, made of a
        # 4-edge quadrilateral per corner (see _miter_kites).
        self.layers = len(modes)
        self.miter_owners = torch.tensor(miter_owners, device=device, dtype=torch.int64)
        if corners:
            counts = [len(layer_corners) for layer_corners in corners]
            self.corners = torch.from_numpy(np.concatenate(corners).T.copy()).to(device)
            self.corner_layer = torch.from_numpy(np.repeat(miter_owners, counts)).to(device)
            self.corner_limit = torch.from_numpy(1 / np.concatenate(corner_limits) ** 2).float().to(device)
            full = self.expand.tocoo()
            values = torch.from_numpy(full.data).float()
            self.expand_points = torch.sparse_coo_tensor(np.stack([full.row, full.col]), values, full.shape, check_invariants=True).coalesce().to(device)
            self.expand_points_t = torch.sparse_coo_tensor(np.stack([full.col, full.row]), values, full.shape[::-1], check_invariants=True).coalesce().to(device)
            offsets += (count + 4 * np.cumsum(counts)).tolist()
            modes += [FILL_NONZERO] * len(counts)
            cuts.append(np.full((2, 4 * sum(counts)), -1, np.int32))
            squares += [False] * len(counts)
        self.cuts = torch.from_numpy(np.concatenate(cuts, 1)).to(device)
        self.squares = torch.tensor(squares, device=device, dtype=torch.int32)
        self.kites = torch.full((self.layers,), -1, device=device, dtype=torch.int32)
        self.kites[self.miter_owners] = torch.arange(len(self.miter_owners), device=device, dtype=torch.int32) + self.layers
        self.edge_layer = torch.repeat_interleave(torch.arange(len(modes), device=device), torch.from_numpy(np.diff(offsets)).to(device))
        self.offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
        self.modes = torch.tensor(modes, device=device, dtype=torch.int32)
        stroke = self.modes == STROKE
        self.reach = torch.where(stroke, 0.5 * (1 + (self.squares != 0)), 0.0)
        self.margin = torch.where(stroke, 2.0, 1.0)
        self.limit = torch.tensor([scene.width, scene.height], device=device)
        self.colors = torch.tensor(colors, device=device, dtype=torch.float32)
        self.widths = torch.tensor(widths, device=device, dtype=torch.float32)
        self.tune_colors = torch.tensor(tune_colors, device=device, dtype=torch.float32)[:, None]
        self.tune_widths = torch.tensor(tune_widths, device=device, dtype=torch.float32)
        self.scale = torch.tensor(opacity, device=device, dtype=torch.float32)[:, None, None]
        if any(clips):
            self.scale = self.scale * _clip_coverage(clips, scene.width, scene.height, device)
        self.size = (scene.height, scene.width)

    def _miter_kites(self, points, widths):
        """Miter join of each stroke corner as a quadrilateral: the corner, the
        two outer offsets of the stroke and the miter tip. Beyond the miter
        limit it collapses to the bevel triangle, inside the round join."""
        full = _Flatten.apply(points, self.expand_points, self.expand_points_t)
        ring = (_MiterKites.apply if self.triton else _kites)(full, widths, self.corners, self.corner_layer, self.corner_limit)
        return torch.cat([ring, ring.roll(-1, 1)], 2).reshape(-1, 4)

    def _boxes(self, edges, widths):
        """Pixel box (x0, y0, x1, y1) of each layer, the bounding box of its edges
        grown by the reach of _near_pixels, and the offsets of the layers' boxes
        in a packed array of their pixels. A miter-joined stroke's box holds its
        quadrilaterals' too."""
        ends = edges.view(-1, 2, 2)
        layer = self.edge_layer[:, None].expand(-1, 4)
        # The least corner of each layer's edges and the negated greatest.
        extent = torch.full((len(self.modes), 4), FAR, device=edges.device).scatter_reduce(0, layer, torch.cat([ends.amin(1), -ends.amax(1)], 1), "amin")
        extent -= (widths * self.reach + self.margin)[:, None]
        start = (extent[:, :2] - 0.5).ceil().long()
        stop = (-extent[:, 2:] - 0.5).floor().long() + 1
        owners = self.miter_owners
        start[owners] = torch.minimum(start[owners], start[self.layers:])
        stop[owners] = torch.maximum(stop[owners], stop[self.layers:])
        start = torch.minimum(start.clamp(min=0), self.limit)
        stop = torch.maximum(torch.minimum(stop, self.limit), start)
        areas = (stop - start).prod(1)
        return torch.cat([start, stop], 1).int(), F.pad(areas.cumsum(0), (1, 0))

    def _merge_kites(self, cover, boxes, offsets):
        """A miter join adds its quadrilaterals to the round-joined stroke band."""
        parts = list(cover.split(offsets.diff().tolist()))
        boxes = boxes.tolist()
        for owner, kite in zip(self.miter_owners.tolist(), range(self.layers, len(boxes))):
            x0, y0, x1, y1 = boxes[owner]
            kx0, ky0, kx1, ky1 = boxes[kite]
            if kx1 > kx0 and ky1 > ky0:
                ring = F.pad(parts[kite].view(ky1 - ky0, kx1 - kx0), (kx0 - x0, x1 - kx1, ky0 - y0, y1 - ky1)).view(-1)
            else:
                ring = torch.zeros_like(parts[owner])
            parts[owner] = torch.maximum(parts[owner], ring)
        return torch.cat(parts[:self.layers])

    def alpha(self, points, widths):
        """The layers' alpha packed over their boxes, with the boxes and the packed offsets."""
        edges = _Flatten.apply(points, self.flatten, self.flatten_t).view(2, -1, 2).transpose(0, 1).reshape(-1, 4)
        if len(self.miter_owners):
            edges = torch.cat([edges, self._miter_kites(points, widths)])
            widths = torch.cat([widths, widths.new_zeros(len(self.miter_owners))])
        boxes, offsets = self._boxes(edges.detach(), widths.detach())
        scale = self.scale.expand(-1, *self.size)
        if self.triton:
            alpha = _LayerAlpha.apply(edges, self.cuts, self.squares, widths, self.offsets, self.modes, self.kites, scale, boxes, offsets, *self.size)
        else:
            cover = _NearCoverage.apply(edges, self.cuts, self.squares, widths, self.modes, self.edge_layer, boxes, offsets, *self.size)
            if len(self.miter_owners):
                cover = self._merge_kites(cover, boxes, offsets)
            alpha = cover * torch.cat([scale[layer, y0:y1, x0:x1].reshape(-1) for layer, (x0, y0, x1, y1) in enumerate(boxes[:self.layers].tolist())])
        return alpha, boxes[:self.layers], offsets[:self.layers + 1]

    def dense(self, alpha, boxes, offsets):
        """Alpha as [L,H,W]."""
        dense = alpha.new_zeros(self.layers, *self.size)
        for layer, (x0, y0, x1, y1), start, stop in zip(range(self.layers), boxes.tolist(), offsets.tolist(), offsets[1:].tolist()):
            dense[layer, y0:y1, x0:x1] = alpha[start:stop].view(y1 - y0, x1 - x0)
        return dense

    def composite(self, alpha, colors, boxes, offsets):
        """The layers over white, with the fixed layers between them."""
        over = (_Composite if self.triton else _TorchComposite).apply
        out, start = alpha.new_ones(3, *self.size), 0
        for stop, color, cover in self.runs + [(self.layers, None, None)]:
            if stop > start:
                out, start = over(alpha, colors[start:stop], out, boxes, offsets, start), stop
            if color is not None:
                out = color + (1 - cover) * out
        return out

    def render(self, points, colors, widths):
        alpha, boxes, offsets = self.alpha(points, widths)
        return self.composite(alpha, colors, boxes, offsets)

    def write_back(self, scene, points, colors, widths):
        points = (self.expand @ points.detach().cpu().numpy()).astype(np.float32)
        colors, widths = colors.detach().cpu().tolist(), widths.detach().cpu().tolist()
        base = layer = 0
        for shape in scene.shapes:
            for i, subpath in enumerate(shape.subpaths):
                shape.subpaths[i] = points[base:base + len(subpath)].copy()
                base += len(subpath)
            shape.changed = True
            if shape.fill is not None:
                shape.fill = (*colors[layer], shape.fill[3])
                layer += 1
            if shape.stroke is not None:
                shape.stroke = (*colors[layer], shape.stroke[3])
                shape.stroke_width = widths[layer]
                layer += 1


def _corners(subpath, closed):
    """Corners of a stroked subpath where its tangent turns, as rows of control
    point indices: the vertex, then head and tail of the tangent arriving at it
    and of the tangent leaving it."""
    last = len(subpath) - 1

    def distinct(fixed, candidates):
        return next((i for i in candidates if np.linalg.norm(subpath[i] - subpath[fixed]) > 1e-3), candidates[-1])

    def arriving(end):
        return end, distinct(end, (end - 1, end - 2, end - 3))

    def leaving(start):
        return distinct(start, (start + 1, start + 2, start + 3)), start

    corners = [(end, *arriving(end), *leaving(end)) for end in range(3, last, 3)]
    if closed and np.linalg.norm(subpath[last] - subpath[0]) > 1e-3:
        # Around the implicit closing line from the last point to the start.
        corners += [(last, *arriving(last), 0, last), (0, 0, last, *leaving(0))]
    elif closed:
        corners.append((0, *arriving(last), *leaving(0)))
    corners = np.array(corners, dtype=np.int64).reshape(-1, 5)
    arrive = subpath[corners[:, 1]] - subpath[corners[:, 2]]
    leave = subpath[corners[:, 3]] - subpath[corners[:, 4]]
    cos = (arrive * leave).sum(1) / np.maximum(np.linalg.norm(arrive, axis=1) * np.linalg.norm(leave, axis=1), 1e-12)
    return corners[cos < SMOOTH_COS]


def _clip_coverage(clips, width, height, device):
    """Fixed alpha factor of each layer's clip regions (intersection of nonzero fills)."""
    regions, layer_regions = {}, []
    for layer_clips in clips:
        keys = [b"".join(subpath.tobytes() for subpath in region) for region in layer_clips]
        for key, region in zip(keys, layer_clips):
            regions.setdefault(key, (len(regions), region))
        layer_regions.append([regions[key][0] for key in keys])
    shapes = [Shape(list(region), [True] * len(region), [np.zeros(len(subpath) // 3, bool) for subpath in region], (0.0, 0.0, 0.0, 1.0), "nonzero", None, 0.0, []) for _, region in regions.values()]
    plan = _Plan(Scene(width, height, shapes), [None] * (len(shapes) + 1), device)
    region_alpha = plan.dense(*plan.alpha(plan.points, plan.widths))
    factors = torch.ones((len(clips), height, width), device=device)
    for factor, indices in zip(factors, layer_regions):
        for index in indices:
            factor *= region_alpha[index]
    return factors


def optimize(scene, fixed, target, steps):
    """Adam on all control points, colors and stroke widths; keeps the best step."""
    plan = _Plan(scene, fixed, target.device)
    points = plan.points.clone().requires_grad_()
    colors = plan.colors.clone().requires_grad_()
    widths = plan.widths.clone().requires_grad_()
    optimizer = torch.optim.Adam([
        {"params": [points], "lr": POINT_LR},
        {"params": [colors], "lr": COLOR_LR},
        {"params": [widths], "lr": WIDTH_LR},
    ], fused=True)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - 0.9 * step / steps)
    best = [points.detach().clone(), colors.detach().clone(), widths.detach().clone()]
    best_loss = torch.full((), math.inf, device=target.device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        image = plan.render(points, colors, widths)
        loss = (image - target).square().mean()
        loss.backward()
        colors.grad *= plan.tune_colors
        widths.grad *= plan.tune_widths
        # Compare the losses on the device: the CPU queues the next steps while
        # the GPU computes, where reading each loss into Python would wait for it.
        improved = loss.detach() < best_loss
        best_loss = torch.minimum(best_loss, loss.detach())
        for kept, current in zip(best, (points, colors, widths)):
            kept.copy_(torch.where(improved, current.detach(), kept))
        optimizer.step()
        schedule.step()
        colors.detach().clamp_(0, 1)
        widths.detach().clamp_(min=0)
    plan.write_back(scene, *best)
    return plan.render(*best).detach()


def removal_costs(scene, fixed, target):
    """Per shape, the increase of the mean squared error if only that shape
    (its fill and stroke layers together) were removed."""
    plan = _Plan(scene, fixed, target.device)
    alpha, boxes, offsets = plan.alpha(plan.points, plan.widths)
    residual = plan.composite(alpha, plan.colors, boxes, offsets) - target
    alpha = plan.dense(alpha, boxes, offsets)
    # Transmittance of what is above each layer, fixed layers included, built in place from the top.
    above = torch.empty_like(alpha)
    through = torch.ones_like(target[0])
    runs = list(plan.runs)
    for layer in range(len(alpha) - 1, -1, -1):
        while runs and runs[-1][0] > layer:
            through = through * (1 - runs.pop()[2])
        above[layer] = through
        through = through * (1 - alpha[layer])
    runs = iter(plan.runs)
    run = next(runs, None)
    below = torch.ones_like(target)
    costs = []
    layer = 0
    for shape in scene.shapes:
        while run is not None and run[0] <= layer:
            below = run[1] + (1 - run[2]) * below
            run = next(runs, None)
        start = below
        for _ in range((shape.fill is not None) + (shape.stroke is not None)):
            color = plan.colors[layer][:, None, None]
            below = below + alpha[layer] * (color - below)
            layer += 1
        delta = above[layer - 1] * (start - below)
        costs.append((delta * (2 * residual + delta)).mean())
    return torch.stack(costs).cpu().numpy()


def _bernstein(u):
    u = u[:, None]
    return np.concatenate([(1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u ** 2, u ** 3], 1)


def _span_samples(ref, lo, hi):
    """Reference samples of the original segments lo[k]..hi[k] of each span, the
    span of each sample and the first sample of each span."""
    count = hi - lo
    ids = np.repeat(lo - np.cumsum(count) + count, count) + np.arange(count.sum())
    return ref[ids].reshape(-1, 2), np.repeat(np.arange(len(lo)), count * MERGE_SAMPLES), np.cumsum(count * MERGE_SAMPLES) - count * MERGE_SAMPLES


def _fit_lines(samples, group, first, a, b, tolerance):
    """Straight segments from a to b, and which keep every sample within the tolerance."""
    ab = b - a
    p = samples - a[group]
    t = ((p * ab[group]).sum(1) / np.maximum((ab * ab).sum(1), 1e-12)[group]).clip(0, 1)
    error = np.linalg.norm(p - t[:, None] * ab[group], axis=1)
    return np.stack([a, a + ab / 3, a + ab * 2 / 3, b], 1), np.maximum.reduceat(error, first) <= tolerance


def _direction(vectors):
    """Unit vector of the first nonzero one of [n,3,2], and whether there is one."""
    norms = np.linalg.norm(vectors, axis=2)
    nonzero = norms > 1e-9
    k = nonzero.argmax(1)
    rows = np.arange(len(vectors))
    return vectors[rows, k] / np.maximum(norms[rows, k], 1e-9)[:, None], nonzero.any(1)


def _fit_curves(samples, group, first, head, tail, tolerance):
    """Cubics over the spans of the cubics head and tail, keeping the end points
    and end tangent directions: least squares for the tangent lengths and Newton
    steps on the sample parameters. Also which have positive lengths and keep
    every sample within the tolerance."""
    a, b = head[:, 0], tail[:, 3]
    t0, found0 = _direction(head[:, 1:] - head[:, :1])
    t1, found1 = _direction(tail[:, [2, 1, 0]] - tail[:, 3:])
    last = np.append(first[1:], len(samples)) - 1
    steps = np.linalg.norm(np.diff(samples, axis=0, prepend=samples[:1]), axis=1)
    steps[first] = 0
    length = np.cumsum(steps)
    length -= length[first][group]
    u = length / np.maximum(length[last], 1e-12)[group]
    ok = found0 & found1 & (length[last] > 1e-9)
    for step in range(NEWTON_STEPS + 1):
        basis = _bernstein(u)
        a1, a2 = basis[:, 1:2] * t0[group], basis[:, 2:3] * t1[group]
        rhs = samples - (basis[:, :1] + basis[:, 1:2]) * a[group] - (basis[:, 2:3] + basis[:, 3:]) * b[group]
        c11, c12, c22, x1, x2 = np.add.reduceat(np.stack([(a1 * a1).sum(1), (a1 * a2).sum(1), (a2 * a2).sum(1), (a1 * rhs).sum(1), (a2 * rhs).sum(1)], 1), first).T
        det = c11 * c22 - c12 * c12
        ok &= np.abs(det) > 1e-12
        det = np.where(ok, det, 1)
        alpha, beta = (x1 * c22 - x2 * c12) / det, (c11 * x2 - c12 * x1) / det
        ok &= (alpha > 0) & (beta > 0)
        curve = np.stack([a, a + alpha[:, None] * t0, b + beta[:, None] * t1, b], 1)
        error = np.einsum("nk,nkd->nd", basis, curve[group]) - samples
        if step == NEWTON_STEPS:
            return curve, ok & (np.maximum.reduceat(np.linalg.norm(error, axis=1), first) <= tolerance)
        # First and second derivative of the cubic, from the differences of its control points.
        v = u[:, None]
        diff = np.diff(curve, axis=1)[group]
        d1 = 3 * np.einsum("nk,nkd->nd", np.concatenate([(1 - v) ** 2, 2 * (1 - v) * v, v ** 2], 1), diff)
        d2 = 6 * ((1 - v) * (diff[:, 1] - diff[:, 0]) + v * (diff[:, 2] - diff[:, 1]))
        den = (d1 * d1).sum(1) + (error * d2).sum(1)
        valid = np.abs(den) > 1e-12
        u = (u - np.where(valid, (error * d1).sum(1) / np.where(valid, den, 1), 0)).clip(0, 1)
        u[first], u[last] = 0, 1


def merge_segments(scene, tolerance):
    """Merges adjacent segments of each subpath, both straight or both curved,
    while the merged outline stays within `tolerance` pixels of the samples of the
    original segments it replaces. Disjoint pairs of all subpaths merge in each
    round, alternating the pairing offset. False when nothing merges."""
    shapes = [shape for shape in scene.shapes if shape.editable]
    if not shapes:
        return False
    owners = [(shape, i) for shape in shapes for i in range(len(shape.subpaths))]
    sizes = [len(shape.lines[i]) for shape, i in owners]
    ctrl = np.concatenate([subpath[3 * np.arange(len(straight))[:, None] + np.arange(4)] for shape in shapes for subpath, straight in zip(shape.subpaths, shape.lines)])
    straight = np.concatenate([straight for shape in shapes for straight in shape.lines])
    sub = np.repeat(np.arange(len(owners)), sizes)
    ref = np.einsum("uk,skd->sud", _bernstein(np.linspace(0, 1, MERGE_SAMPLES, dtype=np.float32)), ctrl)
    lo = np.arange(len(sub))
    hi = lo + 1
    failed = np.zeros(len(sub), bool)
    idle = offset = 0
    while idle < 2:
        # Pairs start at even or odd positions within each subpath, so other subpaths do not change which pairs a subpath tries.
        local = np.arange(len(sub)) - np.searchsorted(sub, sub)
        pairs = np.flatnonzero((sub[:-1] == sub[1:]) & (straight[:-1] == straight[1:]) & (local[:-1] % 2 == offset) & ~failed[:-1])
        offset = 1 - offset
        absorbed = np.zeros(len(sub), bool)
        for kind in (True, False):
            i = pairs[straight[pairs] == kind]
            if len(i):
                samples, group, first = _span_samples(ref, lo[i], hi[i + 1])
                merged, ok = _fit_lines(samples, group, first, ctrl[i, 0], ctrl[i + 1, 3], tolerance) if kind else _fit_curves(samples, group, first, ctrl[i], ctrl[i + 1], tolerance)
                failed[i[~ok]] = True
                i = i[ok]
                ctrl[i], hi[i], absorbed[i + 1] = merged[ok], hi[i + 1], True
                failed[i] = failed[i - 1] = False
        idle = 0 if absorbed.any() else idle + 1
        ctrl, straight, sub, lo, hi, failed = (v[~absorbed] for v in (ctrl, straight, sub, lo, hi, failed))
    if len(sub) == sum(sizes):
        return False
    bounds = np.cumsum(np.bincount(sub, minlength=len(owners)))
    for (shape, i), end, size in zip(owners, bounds, np.bincount(sub, minlength=len(owners))):
        shape.changed |= bool(size != len(shape.lines[i]))
        shape.subpaths[i] = np.concatenate([ctrl[end - size, :1], ctrl[end - size:end, 1:].reshape(-1, 2)]).astype(np.float32)
        shape.lines[i] = straight[end - size:end]
    return True


def render(scene, fixed, device):
    plan = _Plan(scene, fixed, device)
    return plan.render(plan.points, plan.colors, plan.widths)
