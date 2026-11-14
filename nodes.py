from io import BytesIO
import logging

from comfy_api.latest import io

from .svg_optimize import optimize_svg


class OptimizeSVG(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SVGOptimize",
            display_name="Optimize SVG",
            category="image",
            description="Rewrites each SVG into fewer bytes that draw the same image.",
            inputs=[
                io.SVG.Input("svg"),
                io.Boolean.Input("keep_metadata", default=False, tooltip="Keep <metadata>, <title> and <desc>."),
            ],
            outputs=[io.SVG.Output("svg")],
        )

    @classmethod
    def execute(cls, svg, keep_metadata) -> io.NodeOutput:
        documents = []
        for index, item in enumerate(svg.data):
            data = item.getvalue()
            try:
                data = optimize_svg(data, keep_metadata)
            except ValueError as exc:
                logging.warning("Optimize SVG: document %d left unchanged: %s", index, exc)
            documents.append(BytesIO(data))
        return io.NodeOutput(io.SVG.Type(documents))
