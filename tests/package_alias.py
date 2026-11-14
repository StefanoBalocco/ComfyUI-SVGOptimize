"""Register the custom node folder as package `svgoptimize` for the tests."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "svgoptimize" not in sys.modules:
    package = types.ModuleType("svgoptimize")
    package.__path__ = [str(ROOT)]
    sys.modules["svgoptimize"] = package
