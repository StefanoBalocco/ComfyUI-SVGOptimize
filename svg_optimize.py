"""The optimizer entry point."""
from lxml import etree

from .svg_attributes import clean_values, format_geometry, remove_defaults, styles_to_attributes
from .svg_document import clean_document, document_facts, parse_svg, serialize
from .svg_shapes import apply_translates, convert_shapes, remove_zero_width_strokes
from .svg_structure import clean_ids, collapse_groups, paint_one_stop_gradients, remove_invisible


def optimize_svg(data, keep_metadata):
    root, preceding = parse_svg(data)
    clean_document(root, keep_metadata)
    etree.cleanup_namespaces(root)
    facts = document_facts(root, preceding)
    previous = None
    written = {}
    # Each round is lossless, so stopping at the bound still gives a correct document.
    for _ in range(10):
        if not facts.stylesheet:
            styles_to_attributes(root)
        if not (facts.stylesheet or facts.script or facts.animation):
            remove_invisible(root)
            remove_zero_width_strokes(root)
            paint_one_stop_gradients(root)
        clean_values(root)
        if not (facts.stylesheet or facts.script or facts.animation):
            clean_ids(root)
        if not (facts.stylesheet or facts.animation):
            collapse_groups(root)
            convert_shapes(root)
            apply_translates(root, written)
        if not facts.animation:
            format_geometry(root, written)
        remove_defaults(root, facts.stylesheet or facts.animation)
        etree.cleanup_namespaces(root)
        output = serialize(root, preceding)
        if output == previous:
            break
        previous = output
    return output
