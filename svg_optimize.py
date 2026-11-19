"""The optimizer entry point."""
from lxml import etree

from .svg_attributes import clean_values, format_geometry, inline_stylesheet, remove_defaults, styles_to_attributes
from .svg_document import clean_document, document_facts, parse_svg, serialize
from .svg_shapes import apply_translates, convert_shapes, merge_opacities, remove_unused_strokes
from .svg_structure import clean_ids, collapse_groups, paint_one_stop_gradients, remove_invisible
from .svg_viewport import crop_viewport


def optimize_svg(data, keep_metadata, round_alpha=False):
    root, preceding = parse_svg(data)
    clean_document(root, keep_metadata)
    etree.cleanup_namespaces(root)
    inlined = inline_stylesheet(root, preceding)
    outputs = [rewrite(root, preceding, round_alpha)]
    if inlined is not None:
        outputs.append(rewrite(inlined, preceding, round_alpha))
    return min(outputs, key=len)


def rewrite(root, preceding, round_alpha=False):
    facts = document_facts(root, preceding)
    previous = None
    written = {}
    # The structural pass has run on `previous`.
    settled = False
    # Each round is lossless, so stopping at the bound still gives a correct document.
    for _ in range(10):
        if not facts.stylesheet:
            styles_to_attributes(root)
        if not (facts.stylesheet or facts.script or facts.animation):
            remove_invisible(root)
            remove_unused_strokes(root)
            paint_one_stop_gradients(root)
        clean_values(root, round_alpha)
        if not (facts.stylesheet or facts.script or facts.animation):
            clean_ids(root)
            merge_opacities(root, round_alpha)
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
            if settled or facts.stylesheet or facts.script or facts.animation:
                break
            crop_viewport(root, written)
            settled = True
            output = serialize(root, preceding)
            if output == previous:
                break
        else:
            settled = False
        previous = output
    return output
