"""The optimizer entry point."""
from lxml import etree

from .svg_attributes import clean_values, format_geometry, remove_defaults, styles_to_attributes
from .svg_document import clean_document, document_facts, parse_svg, serialize
from .svg_shapes import apply_translates, convert_shapes
from .svg_structure import clean_ids, collapse_groups


def optimize_svg(data, keep_metadata):
    root, preceding = parse_svg(data)
    clean_document(root, keep_metadata)
    etree.cleanup_namespaces(root)
    facts = document_facts(root, preceding)
    previous = None
    # Each round is lossless, so stopping at the bound still gives a correct document.
    for _ in range(10):
        if not facts.stylesheet:
            styles_to_attributes(root)
        clean_values(root)
        if not (facts.stylesheet or facts.script or facts.animation):
            clean_ids(root)
        if not (facts.stylesheet or facts.animation):
            collapse_groups(root)
            convert_shapes(root)
            apply_translates(root)
        if not facts.animation:
            format_geometry(root)
        remove_defaults(root, facts.stylesheet or facts.animation)
        etree.cleanup_namespaces(root)
        output = serialize(root, preceding)
        if output == previous:
            break
        previous = output
    return output
