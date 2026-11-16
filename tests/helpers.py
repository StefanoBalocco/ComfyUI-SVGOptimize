import math

from lxml import etree

import package_alias  # noqa: F401
from svgoptimize.svg_document import parse_svg, serialize
from svgoptimize.svg_optimize import optimize_svg

S = 'xmlns="http://www.w3.org/2000/svg"'
X = 'xmlns:xlink="http://www.w3.org/1999/xlink"'


def opt(text, keep=False, round_alpha=False):
    return optimize_svg(text.encode(), keep, round_alpha).decode()


def run(rewrite, text, *args):
    """One pass alone: parse, rewrite, drop unused namespaces, serialize."""
    root, preceding = parse_svg(text.encode())
    rewrite(root, *args)
    etree.cleanup_namespaces(root)
    return serialize(root, preceding).decode()


def ring(cx, cy, rx, ry, count):
    """Points of a polygon around an ellipse."""
    return " ".join(f"{cx + rx * math.cos(2 * math.pi * i / count):.3f},{cy + ry * math.sin(2 * math.pi * i / count):.3f}" for i in range(count))


GRADIENT = '<defs><linearGradient id="g"><stop offset="0" stop-color="#f00"/><stop offset="1" stop-color="#00f"/></linearGradient>{}</defs>'


def gradient_document(defs="", middle=""):
    """A background, two 120-point polygons around a rect filled with a gradient,
    and a text; `defs` and `middle` add to the definitions and the shapes."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100" height="100">{GRADIENT.format(defs)}'
            f'<rect width="100" height="100" fill="#fff8e8"/><polygon id="a" points="{ring(40, 40, 30, 30, 120)}" fill="#2a6"/>'
            f'<rect id="panel" x="20" y="55" width="60" height="30" fill="url(#g)"/>'
            f'<polygon id="b" points="{ring(60, 50, 25, 15, 120)}" fill="#c42" fill-opacity=".8"/>{middle}<text id="label" x="10" y="95" font-size="10">Hi</text></svg>').encode()
