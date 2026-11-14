"""Node tests with a stand-in for comfy_api."""
from io import BytesIO
import importlib
import sys
import types
import unittest

import package_alias  # noqa: F401


class Recorder:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


class SVGType:
    def __init__(self, data):
        self.data = data


class Namespace:
    Input = Output = Recorder


def stub_comfy_api():
    package = types.ModuleType("comfy_api")
    package.__path__ = []
    latest = types.ModuleType("comfy_api.latest")
    latest.io = types.SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}),
        Schema=Recorder,
        NodeOutput=Recorder,
        SVG=types.SimpleNamespace(Type=SVGType, Input=Recorder, Output=Recorder),
        Boolean=Namespace,
    )
    latest.ComfyExtension = type("ComfyExtension", (), {})
    sys.modules["comfy_api"] = package
    sys.modules["comfy_api.latest"] = latest
    package.latest = latest


class OptimizeSVGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub_comfy_api()
        cls.addClassCleanup(sys.modules.pop, "comfy_api")
        cls.addClassCleanup(sys.modules.pop, "comfy_api.latest")
        sys.modules.pop("svgoptimize.nodes", None)
        cls.node = importlib.import_module("svgoptimize.nodes").OptimizeSVG

    def run_node(self, documents, keep_metadata=False):
        svg = SVGType([BytesIO(document) for document in documents])
        return self.node.execute(svg, keep_metadata).args[0].data

    def test_batch_with_invalid_document(self):
        valid = b'<svg xmlns="http://www.w3.org/2000/svg"><g><rect width="1" height="1"/></g></svg>'
        document = BytesIO(valid)
        document.read()
        with self.assertLogs(level="WARNING"):
            out = self.node.execute(SVGType([document, BytesIO(b"not xml")]), False).args[0].data
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].getvalue(), b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h1v1H0z"/></svg>')
        self.assertEqual(out[1].getvalue(), b"not xml")

    def test_keep_metadata_both_ways(self):
        document = b'<svg xmlns="http://www.w3.org/2000/svg"><title>Cat</title><path d="M0 0L5 5"/></svg>'
        self.assertNotIn(b"<title>", self.run_node([document])[0].getvalue())
        self.assertIn(b"<title>Cat</title>", self.run_node([document], True)[0].getvalue())

    def test_schema(self):
        schema = self.node.define_schema()
        self.assertEqual(schema.kwargs["node_id"], "SVGOptimize")
        self.assertEqual([item.args[0] for item in schema.kwargs["inputs"]], ["svg", "keep_metadata"])


if __name__ == "__main__":
    unittest.main()
