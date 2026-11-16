# ComfyUI-SVGOptimize

An SVG optimizer node for ComfyUI. It rewrites each SVG of a batch into fewer bytes. Coordinates are computed with exact decimal arithmetic, and every rewrite is one the SVG specification defines as equivalent, so the document draws the same image. An optional `simplify` step first rounds path data, merges path segments, re-fits and drops shapes in place, keeping the drawing within 45 dB PSNR of a render of the input. It works with the native `SVG` type from any source, such as StarVector or VTracer output.

## Install

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/StefanoBalocco/ComfyUI-SVGOptimize.git
# Install with ComfyUI's own Python interpreter:
python -m pip install -r ComfyUI-SVGOptimize/requirements.txt
```

The lossless rewrites need `lxml`, `tinycss2` and `cssselect2`. Simplify also uses `cairosvg`, `numpy` and `scipy`, and PyTorch from ComfyUI. It runs on the GPU ComfyUI uses when that is CUDA, with Triton kernels when Triton is installed, and on the CPU otherwise.

## Node

`Optimize SVG` appears under `image`.

- Input `svg`: the SVG documents to rewrite.
- Input `keep_metadata` (default off): keeps `<metadata>`, `<title>` and `<desc>`. When off, all three are removed.
- Input `simplify` (default off): lossy simplification before the lossless rewrites. See Simplify.
- Output `svg`: one UTF-8 document per input document, starting at the `<svg>` element.

The progress bar advances with each document and, with `simplify` on, with each optimization trial of 100 steps.

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
- A cubic whose two control points lie exactly on the line through its end points, at positions 0 <= t1 <= t2 <= 1 along the chord, becomes a line, as does a quadratic whose control point lies on the chord at 0 <= t <= 1. Such a segment traverses the chord once, so fill, stroke, dashes and marker directions are those of the line. Segments with a zero-length chord stay, and a following `S` or `T` keeps its geometry.

Transforms:

- `translate(x 0)`, `scale(s s)`, `rotate(a 0 0)`, `matrix(1 0 0 1 x y)` and `matrix(a 0 0 d 0 0)` take their shorter forms, and identity transforms are removed, except `gradientTransform` and `patternTransform` on an element with `href`, which become `scale(1)`.
- A single `translate` on a shape is applied to its coordinates when that is shorter and no `url()` reference applies to the element or its ancestors.

Attributes and values:

- Colors take the shortest of keyword, `#rgb`, `#rrggbb`, `#rgba`, `#rrggbbaa`, `rgb()` and `hsl()` forms (the last two with spaces and `/ alpha`). Hex alpha forms are used when the alpha times 255 is an integer, and `rgb()` and `hsl()` when their numbers are exact decimals; `transparent` becomes `#0000`. The input may use `rgba()`, `rgb()` with commas or with spaces and `/`, percentages, and `#rgba` or `#rrggbbaa`.
- A shape (`path`, `rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon`) that declares both a solid `fill` and `fill-opacity`, or `stroke` and `stroke-opacity`, takes the opacity as the alpha of the color when the resulting alpha times 255 is an integer and the text is shorter. No ancestor may declare that opacity property, the shape and its ancestors must not be referenced by a `<use>`, and the document has no stylesheet, script or SMIL animation. `opacity` is not merged.
- Numeric values use their shortest text, and `px` is dropped from length attributes.
- Whitespace is collapsed in `viewBox`, `points`, `class`, `transform` and `d`.
- Empty attributes are removed, except conditional processing attributes.
- Default values of presentation and geometry attributes are removed, in attributes and in `style`.
- `style` declarations become presentation attributes when shorter.
- A stylesheet of plain rules is inlined, see Stylesheets.
- A `fill` or `stroke` that references a `linearGradient` with one opaque stop and `userSpaceOnUse` units, and an invertible or no `gradientTransform`, takes the stop color. The stop, units and transform may come through `href`.
- On a shape with `stroke-width` 0, the other stroke properties are removed, and `stroke-width` too when no ancestor or `<use>` can give the shape a stroke.
- On a shape whose stroke is `none` (set to `none`, or set by neither the shape nor an ancestor), `stroke-width`, `stroke-linecap`, `stroke-linejoin`, `stroke-miterlimit`, `stroke-dasharray`, `stroke-dashoffset` and `stroke-opacity` are removed. A shape that does not set `stroke` keeps them when an ancestor sets `stroke` or an ancestor or the shape itself is referenced by a `<use>`.

Serialization:

- Whitespace between elements is removed, except in text content and under `xml:space="preserve"`.
- Each attribute uses the quote character that needs fewer escapes, and empty elements self-close.
- The text of `text`, `tspan`, `textPath`, `style`, `script`, `title` and `desc`, and the content of `foreignObject`, is kept as written.

Stylesheets:

- A stylesheet is inlined when every stylesheet of the document is a `<style>` element without `media`, with no `type` or `type="text/css"`, holding only plain qualified rules: no at-rules, `!important`, pseudo-classes, pseudo-elements or `var()`. The document has no `xml-stylesheet` instruction, `foreignObject`, script or SMIL animation.
- Each element gets the declarations of the rules that match it (selectors are matched by `cssselect2`, rules parsed by `tinycss2`) at the start of its `style`, in cascade order: higher specificity wins, then the later rule. Declarations that its own `style` sets are not added. Then every `<style>` element and `class` attribute is removed.
- It draws the same image because a rule overrides presentation attributes and yields to the `style` of the element, and a declaration in `style` has the same priority relations. Inheritance does not change, as each declaration stays on the element the rule matched.
- Once inlined, the rewrites that a stylesheet blocks apply to the document. The node writes the rewrites of the document as it is and, when it can be inlined, of the inlined document, and outputs the shorter, since a class shared by many elements can be shorter as a rule.

Documents whose stylesheet is not inlined (`<style>` or `xml-stylesheet`), and documents with a script, SMIL animation elements or `<use>` keep the parts those can reach:

- With a stylesheet, script or SMIL animation, ids and definitions are kept, and with a stylesheet or SMIL animation groups are kept.
- With a stylesheet, shapes keep their element names, `style` declarations stay in `style`, inherited defaults stay, and translates stay in `transform`.
- With a stylesheet, script or SMIL animation, and in documents with `foreignObject`, gradient references stay.
- With a stylesheet, script or SMIL animation, and in documents with `foreignObject`, hidden and zero-size elements stay. Children of `switch`, elements under `filter`, `mask` or `clip-path`, elements inside a `<use>` target, hidden elements inside text content, and elements containing kept metadata stay too.
- With a marker property, hidden paths, lines, polylines and polygons stay.
- With a marker property, stroke properties of zero-width strokes stay, and `stroke-width` stays on shapes whose stroke is `none`.
- Inherited defaults inside a subtree that a `<use>` references stay.
- Translates stay in `transform` with SMIL animation elements and on elements a `<use>` references, directly or through an ancestor.

## Simplify

With `simplify` on, each document first has its shapes simplified where they are, staying within 45 dB PSNR of the CairoSVG render of the input. The lossless rewrites then run on the simplified document, and the node outputs the shorter of the lossless output of the input and the lossless output of the simplified document. A document where no simplified candidate is acceptable and smaller is left to the lossless rewrites.

Both lossless runs also round the alpha of colors to the nearest 1/255, so `rgba(255,0,0,.5)` becomes `#ff000080` and `fill="#f00" fill-opacity=".5"` becomes `fill="#ff000080"`, where the merge conditions of Rewrites hold and the text is shorter. This is not checked against the 45 dB limit: CairoSVG ignores the alpha of `#rrggbbaa`, and the change is at most 1/510 of the alpha, below the 8-bit resolution renderers composite with.

What changes and what stays:

- Each shape keeps its place, id, classes, other attributes and transform; only its geometry changes. A `rect`, `circle`, `ellipse`, `line`, `polyline` or `polygon` whose geometry changes becomes a `path` without its geometry attributes. Gradients, text, filters, images, definitions, styles and groups stay as written.
- Segment merging applies to every editable shape: a `path` or basic shape drawn directly, once, in the normal drawing order. Shapes drawn through `<use>` (and the elements `<use>` references), shapes inside `defs`, `clipPath`, `mask`, `marker`, `pattern` or `symbol`, shapes with a marker property, and text are not edited.
- Re-fitting of points, colors and stroke widths, and removal of shapes, apply to fittable shapes: editable shapes with a solid fill and stroke, no dashes, no filter or mask on the shape or an ancestor, and no group opacity on an ancestor that holds more than the shape. A shape is removed only when nothing references its id (`url(#id)`, `href`, `xlink:href`).
- Everything else that is drawn enters the fit as fixed layers: CairoSVG renders what lies between two consecutive fittable shapes, in paint order, so fittable shapes adapt to what is above and below them. Fixed layers are rendered again when merging changes the editable shapes in them.
- Colors and stroke widths are fitted and written where the element declares them (attribute or `style`), or as an attribute on the element when it inherits them. In a document with a stylesheet (`<style>` or `xml-stylesheet`) they stay as written. A stroke width also stays as written when the transform of the element to the raster is not a uniform scale with rotation and translation.
- Points are written in the user space of the element, with the inverse of its transform and as many decimals as keep each point within about 0.05 px of the raster.

Procedure:

- Straightening comes first: each cubic segment of an editable shape whose control points lie within 0.05 px of its chord, in order along it, becomes a straight segment with its control points at 1/3 and 2/3 of the chord. The rounding candidate writes it as a line, and the fits keep it straight.
- Rounding comes first: each editable `path` that the capture recorded as one shape has its `d` rewritten from the captured geometry with the decimals that keep each point within about 0.05 px of the raster, wherever that is shorter than its current `d`. Basic shapes stay as written. When the result is acceptable and smaller, it is the best candidate so far; the steps below start from the input and must beat it.
- The raster is the CairoSVG render of the input on white, 512 px on the longest side whatever the document declares. It keeps the aspect ratio CairoSVG renders from the document's width and height, or from its viewBox when width and height are missing, and is square when the document declares no size. Absolute units are converted to px; percentages and missing values are ignored.
- CairoSVG captures every shape at that size as cubic Beziers with its fill, stroke, clip and the element it comes from.
- A candidate is acceptable when the CairoSVG render of its written SVG is at least 45 dB PSNR from the render of the input and its SVG is smaller than the best candidate so far, the first being the input.
- Drop shapes: the cost of each removable shape is the increase of the mean squared error if it alone were removed. The cheapest shapes are dropped while the current error plus their summed costs stays within the 45 dB limit, never all the fittable ones. The remaining fittable shapes are optimized for 100 Adam steps toward the render of the input, and the result is kept if it is acceptable.
- Merge segments: at each tolerance below, the cubic segments of editable shapes whose control points lie within the tolerance of their chord, in order along it, are first made straight. Then adjacent straight segments, or adjacent curved segments, of editable shapes merge into one while the outline stays within a tolerance of 1.0, then 0.5, then 0.25 px of the original, in units of the 512 px raster. Each tolerance is optimized for 100 steps and written; the first acceptable result is kept, and the search stops when a tolerance straightens and merges nothing.

The optimization uses a differentiable rasterizer: per pixel, the distance to each paint layer's outline and its winding number give an anti-aliased coverage, and layers composite source-over in paint order, between the fixed layers.

Measured on an RTX 3060 (GPU) and with 5 CPU threads (CPU), sizes of the lossless output without and with simplify:

| Documents | Without | With | PSNR | GPU | CPU |
|---|---|---|---|---|---|
| A watercolor refined by ComfyUI-VTracer | 378 KB | 116 KB | 46.1 dB | 7.4 s | 203 s |
| VTracer's trace of the same watercolor | 134 KB | 65 KB | 45.6 dB | 7.0 s | 196 s |
| VTracer's traces of six flat icons | 0.4-7.5 KB | 0.3-1.0 KB | 46.6-48.8 dB | 0.5-2.0 s | 4-15 s |
| StarVector's SVGs of the same icons | 191-442 B | 191-442 B | - | 0.0-2.9 s | 0.1-13 s |
| Two VTracer traces (watercolor, one icon) with a gradient panel and text added | 134 KB, 4.0 KB | 65 KB, 1.2 KB | 45.6 dB, 47.8 dB | 7.7 s, 0.9 s | 220 s, 6.7 s |

Text is rendered with grayscale antialiasing in the reference, the fixed layers and the candidates, so it adds no error to the fit. On StarVector's icons, five come out unchanged, because no simplified candidate is shorter; one gains 16 B from rounding.

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
