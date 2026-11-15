# ComfyUI-SVGOptimize

An SVG optimizer node for ComfyUI. It rewrites each SVG of a batch into fewer bytes. Coordinates are computed with exact decimal arithmetic, and every rewrite is one the SVG specification defines as equivalent, so the document draws the same image. It works with the native `SVG` type from any source, such as StarVector or VTracer output.

## Install

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/StefanoBalocco/ComfyUI-SVGOptimize.git
# Install with ComfyUI's own Python interpreter:
python -m pip install -r ComfyUI-SVGOptimize/requirements.txt
```

The only dependency is `lxml`.

## Node

`Optimize SVG` appears under `image`.

- Input `svg`: the SVG documents to rewrite.
- Input `keep_metadata` (default off): keeps `<metadata>`, `<title>` and `<desc>`. When off, all three are removed.
- Output `svg`: one UTF-8 document per input document, starting at the `<svg>` element.

A document that is not well-formed XML, whose root is not `svg`, or that declares an entity containing markup passes through with the same bytes, and the node logs one warning.

## Rewrites

Document cleanup:

- The XML declaration, comments, DOCTYPE and processing instructions other than `xml-stylesheet` are removed.
- Editor data is removed: elements and attributes in editor namespaces (Inkscape, Sodipodi, Illustrator, Sketch, Figma, Serif and others), `data-name`, `enable-background`, and `xml:space` where the subtree has no text content elements.
- Namespace declarations that nothing uses, and `version` and `baseProfile` on the root, are removed.

Ids and definitions:

- Ids that nothing references are removed, and referenced ids get the shortest free names, most referenced first.
- Definitions that nothing references (gradients, patterns, clip paths, masks, markers, symbols, filters, anything inside `<defs>`) are removed, as are empty `<defs>` and empty groups.
- Direct children of `<switch>` stay.

Elements that draw nothing:

- Shapes, groups, links, switches, text and nested `svg` elements with `display:none` are removed when nothing outside them references them.
- Zero-size shapes (a `rect` with zero width or height, a `circle` with zero `r`, an `ellipse` with zero `rx` or `ry`, a `path` without `d`, a `polyline` without `points`) are removed when nothing outside them references them.

Groups:

- A `<g>` without attributes is replaced by its children.
- A `<g>` with one child and only inheritable presentation attributes and `transform` passes them to the child when the child sets none of them.
- Inheritable attributes shared by all children of a group move to the group when that is shorter.

Shapes:

- `line`, `polyline` and `polygon` become `path` when shorter.
- A plain `rect` becomes `path` when shorter and the document has no marker property.
- An `ellipse` with equal radii becomes `circle`.

Path data:

- Each segment is written in the shortest of absolute and relative form, using `H`, `V`, `S` and `T` where the values allow.
- Repeated command letters are omitted, numbers use their shortest text, and arc flags keep a separator.

Transforms:

- `translate(x 0)`, `scale(s s)`, `rotate(a 0 0)`, `matrix(1 0 0 1 x y)` and `matrix(a 0 0 d 0 0)` take their shorter forms, and identity transforms are removed, except `gradientTransform` and `patternTransform` on an element with `href`, which become `scale(1)`.
- A single `translate` on a shape is applied to its coordinates when that is shorter and no `url()` reference applies to the element or its ancestors.

Attributes and values:

- Colors take the shortest of `#rrggbb`, `#rgb` and keyword forms.
- Numeric values use their shortest text, and `px` is dropped from length attributes.
- Whitespace is collapsed in `viewBox`, `points`, `class`, `transform` and `d`.
- Empty attributes are removed, except conditional processing attributes.
- Default values of presentation and geometry attributes are removed, in attributes and in `style`.
- `style` declarations become presentation attributes when shorter.
- A `fill` or `stroke` that references a `linearGradient` with one opaque stop and `userSpaceOnUse` units, and an invertible or no `gradientTransform`, takes the stop color. The stop, units and transform may come through `href`.
- On a shape with `stroke-width` 0, the other stroke properties are removed, and `stroke-width` too when no ancestor or `<use>` can give the shape a stroke.

Serialization:

- Whitespace between elements is removed, except in text content and under `xml:space="preserve"`.
- Each attribute uses the quote character that needs fewer escapes, and empty elements self-close.
- The text of `text`, `tspan`, `textPath`, `style`, `script`, `title` and `desc`, and the content of `foreignObject`, is kept as written.

Documents with a stylesheet (`<style>` or `xml-stylesheet`), a script, SMIL animation elements or `<use>` keep the parts those can reach:

- With a stylesheet, script or SMIL animation, ids and definitions are kept, and with a stylesheet or SMIL animation groups are kept.
- With a stylesheet, shapes keep their element names, `style` declarations stay in `style`, inherited defaults stay, and translates stay in `transform`.
- With a stylesheet, script or SMIL animation, and in documents with `foreignObject`, gradient references stay.
- With a stylesheet, script or SMIL animation, and in documents with `foreignObject`, hidden and zero-size elements stay. Children of `switch`, elements under `filter`, `mask` or `clip-path`, elements inside a `<use>` target, hidden elements inside text content, and elements containing kept metadata stay too.
- With a marker property, hidden paths, lines, polylines and polygons stay.
- With a marker property, stroke properties of zero-width strokes stay.
- Inherited defaults inside a subtree that a `<use>` references stay.
- Translates stay in `transform` with SMIL animation elements and on elements a `<use>` references, directly or through an ancestor.

## Relative coordinates

Path segments use relative coordinates where they are shorter. Renderers built on Cairo keep the current point at 1/256 px, so on long relative paths edges can land up to a fraction of a pixel from their exact position.

## Metadata

ComfyUI's Save SVG node writes the workflow into the saved file as `<metadata>`, unless ComfyUI runs with `--disable-metadata`.

## Tests

```sh
cd ComfyUI/custom_nodes/ComfyUI-SVGOptimize
python -m unittest discover -s tests -v
```

## License

BSD-3-Clause. See `LICENSE`.
