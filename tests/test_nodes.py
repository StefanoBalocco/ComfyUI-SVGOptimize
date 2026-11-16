"""Node tests with a stand-in for comfy_api."""
from io import BytesIO
import importlib
from pathlib import Path
import sys
import types
import unittest

import torch

import package_alias  # noqa: F401
from helpers import gradient_document
from svgoptimize.svg_simplify import TRIALS

FIXTURES = Path(__file__).parent / "fixtures"


class Recorder:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


class SVGType:
    def __init__(self, data):
        self.data = data


class Namespace:
    Input = Output = Recorder


class Progress:
    """Stands in for comfy.utils.ProgressBar and records its total and calls."""
    bars = []

    def __init__(self, total):
        self.total, self.events = total, []
        Progress.bars.append(self)

    def update(self, value):
        self.events.append(("update", value))

    def update_absolute(self, value):
        self.events.append(("absolute", value))


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
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = types.SimpleNamespace(get_torch_device=lambda: torch.device("cpu"))
    comfy.utils = types.SimpleNamespace(ProgressBar=Progress)
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = comfy.model_management
    sys.modules["comfy.utils"] = comfy.utils


class OptimizeSVGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub_comfy_api()
        cls.addClassCleanup(sys.modules.pop, "comfy_api")
        cls.addClassCleanup(sys.modules.pop, "comfy_api.latest")
        cls.addClassCleanup(sys.modules.pop, "comfy")
        cls.addClassCleanup(sys.modules.pop, "comfy.model_management")
        cls.addClassCleanup(sys.modules.pop, "comfy.utils")
        sys.modules.pop("svgoptimize.nodes", None)
        cls.node = importlib.import_module("svgoptimize.nodes").OptimizeSVG

    def run_node(self, documents, keep_metadata=False, simplify=False):
        svg = SVGType([BytesIO(document) for document in documents])
        return self.node.execute(svg, keep_metadata, simplify).args[0].data

    def test_batch_with_invalid_document(self):
        valid = b'<svg xmlns="http://www.w3.org/2000/svg"><g><rect width="1" height="1"/></g></svg>'
        document = BytesIO(valid)
        document.read()
        with self.assertLogs(level="WARNING"):
            out = self.node.execute(SVGType([document, BytesIO(b"not xml")]), False, False).args[0].data
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].getvalue(), b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h1v1H0z"/></svg>')
        self.assertEqual(out[1].getvalue(), b"not xml")

    def test_keep_metadata_both_ways(self):
        document = b'<svg xmlns="http://www.w3.org/2000/svg"><title>Cat</title><path d="M0 0L5 5"/></svg>'
        self.assertNotIn(b"<title>", self.run_node([document])[0].getvalue())
        self.assertIn(b"<title>Cat</title>", self.run_node([document], True)[0].getvalue())

    def test_simplify_shortens_a_trace(self):
        document = (FIXTURES / "vtracer.svg").read_bytes()
        plain = self.run_node([document])[0].getvalue()
        simplified = self.run_node([document], simplify=True)[0].getvalue()
        self.assertLess(len(simplified), len(plain))

    def test_simplify_keeps_title(self):
        document = b'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><title>Cat</title><rect width="30" height="20" fill="#f00"/></svg>'
        self.assertIn(b"<title>Cat</title>", self.run_node([document], True, True)[0].getvalue())

    def test_simplify_keeps_gradients_and_simplifies_the_rest(self):
        document = gradient_document()
        with self.assertNoLogs(level="WARNING"):
            simplified = self.run_node([document], simplify=True)[0].getvalue()
        self.assertLess(len(simplified), len(self.run_node([document])[0].getvalue()))
        self.assertIn(b"linearGradient", simplified)

    def test_simplify_rounds_alpha_to_a_255th(self):
        document = b'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><path d="M0 0h30v20z" fill="#f00" fill-opacity=".5"/></svg>'
        self.assertIn(b'fill="red" fill-opacity=".5"', self.run_node([document])[0].getvalue())
        self.assertIn(b'fill="#ff000080"', self.run_node([document], simplify=True)[0].getvalue())

    def test_progress(self):
        documents = [(FIXTURES / "vtracer.svg").read_bytes(), b"not xml"]
        for simplify, budget in ((False, 1), (True, 1 + TRIALS)):
            with self.subTest(simplify=simplify), self.assertLogs(level="WARNING"):
                self.run_node(documents, simplify=simplify)
                bar = Progress.bars[-1]
                self.assertEqual(bar.total, 2 * budget)
                self.assertEqual([event for event in bar.events if event[0] == "absolute"], [("absolute", budget), ("absolute", 2 * budget)])
                steps = sum(value for kind, value in bar.events[:bar.events.index(("absolute", budget))] if kind == "update")
                self.assertGreater(steps, 1 if simplify else 0)
                self.assertLessEqual(steps, budget)

    def test_schema(self):
        schema = self.node.define_schema()
        self.assertEqual(schema.kwargs["node_id"], "SVGOptimize")
        self.assertEqual([item.args[0] for item in schema.kwargs["inputs"]], ["svg", "keep_metadata", "simplify"])
        self.assertIs(schema.kwargs["inputs"][2].kwargs["default"], False)


if __name__ == "__main__":
    unittest.main()
