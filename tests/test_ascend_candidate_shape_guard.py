"""The experimental candidate must never silently omit rows."""
import ast
from pathlib import Path

import pytest


def _shape_selector():
    path = (Path(__file__).resolve().parents[1] / "docs/ascend950pr/candidates"
            / "rmsnorm_cw4_guarded/kernel.py")
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_rows_per_core")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"N_CORES": 64}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_rows_per_core"]


@pytest.mark.parametrize("batch,rows", [(64, 1), (128, 2), (192, 3), (256, 4), (8192, 4)])
def test_divisible_batches_cover_all_rows(batch, rows):
    assert _shape_selector()(batch) == rows
    assert batch % (64 * rows) == 0


@pytest.mark.parametrize("batch", [0, 1, 63, 65, 127, 320, 8193])
def test_unsupported_batches_fail_instead_of_dropping_rows(batch):
    with pytest.raises(ValueError, match="remainder handling"):
        _shape_selector()(batch)
