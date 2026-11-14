from lxml import etree

import package_alias  # noqa: F401
from svgoptimize.svg_document import parse_svg, serialize
from svgoptimize.svg_optimize import optimize_svg

S = 'xmlns="http://www.w3.org/2000/svg"'
X = 'xmlns:xlink="http://www.w3.org/1999/xlink"'


def opt(text, keep=False):
    return optimize_svg(text.encode(), keep).decode()


def run(rewrite, text, *args):
    """One pass alone: parse, rewrite, drop unused namespaces, serialize."""
    root, preceding = parse_svg(text.encode())
    rewrite(root, *args)
    etree.cleanup_namespaces(root)
    return serialize(root, preceding).decode()
