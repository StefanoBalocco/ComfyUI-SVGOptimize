from comfy_api.latest import ComfyExtension

from .nodes import OptimizeSVG


class SVGOptimizeExtension(ComfyExtension):
    async def get_node_list(self):
        return [OptimizeSVG]


async def comfy_entrypoint() -> SVGOptimizeExtension:
    return SVGOptimizeExtension()
