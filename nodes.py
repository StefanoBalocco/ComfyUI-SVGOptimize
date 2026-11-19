from io import BytesIO
import logging

import comfy.model_management
from comfy.utils import ProgressBar
from comfy_api.latest import io

from .svg_optimize import optimize_svg
from .svg_simplify import TRIALS, simplify_svg


class OptimizeSVG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SVGOptimize",
            display_name="Optimize SVG",
            category="image",
            description="Rewrites each SVG into fewer bytes that draw the same image, or with simplify on, nearly the same image.",
            inputs=[
                io.SVG.Input("svg"),
                io.Boolean.Input("keep_metadata", default=False, tooltip="Keep <metadata>, <title> and <desc>."),
                io.Boolean.Input("simplify", default=False, tooltip="Lossy: simplifies shapes where they are by merging segments, dropping shapes and re-fitting colors, keeping the drawing within 45 dB PSNR of a render of the input SVG."),
            ],
            outputs=[io.SVG.Output("svg")],
        )

    @classmethod
    def execute(cls, svg, keep_metadata, simplify) -> io.NodeOutput:
        # One unit for the lossless rewrites of each document, and with simplify
        # one per optimization trial; a document that needs fewer trials jumps ahead.
        budget = 1 + TRIALS * simplify
        progress = ProgressBar(len(svg.data) * budget)
        documents = []
        for index, item in enumerate(svg.data):
            data = item.getvalue()
            try:
                optimized = optimize_svg(data, keep_metadata, simplify)
                progress.update(1)
                if simplify:
                    try:
                        simplified = simplify_svg(optimized, comfy.model_management.get_torch_device(), lambda: progress.update(1))
                    except ValueError as exc:
                        logging.warning("Optimize SVG: simplify skipped for document %d: %s", index, exc)
                        simplified = None
                    if simplified is not None:
                        optimized = min(optimized, optimize_svg(simplified, keep_metadata, simplify), key=len)
                data = optimized
            except ValueError as exc:
                logging.warning("Optimize SVG: document %d left unchanged: %s", index, exc)
            progress.update_absolute((index + 1) * budget)
            documents.append(BytesIO(data))
        return io.NodeOutput(io.SVG.Type(documents))
