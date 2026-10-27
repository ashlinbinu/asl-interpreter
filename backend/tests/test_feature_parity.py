"""Python vs JavaScript feature pipelines must agree (docs/FEATURE_SPEC.md)."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from app import features as fx
from app.config import REPO_ROOT
from tests import synth

NODE = shutil.which("node")
CLI = REPO_ROOT / "frontend" / "scripts" / "features_cli.mjs"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_python_and_js_features_match(tmp_path):
    cases = synth.parity_cases()
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    res = subprocess.run([NODE, str(CLI), str(path)], capture_output=True, text=True, check=True,
                         cwd=REPO_ROOT / "frontend")
    js = json.loads(res.stdout)
    assert len(js) == len(cases)
    for case, out in zip(cases, js):
        ex = fx.extract(case["frames"], case["left_handed"])
        np.testing.assert_allclose(np.array(out["feats"]), ex.feats, atol=1e-6)
        assert np.array(out["hand_mask"]).tolist() == ex.hand_mask.tolist()
        if ex.body is None:
            assert out["body"] is None
        else:
            assert out["body"]["scale"] == pytest.approx(ex.body.scale, abs=1e-9)
            assert out["body"]["roll"] == pytest.approx(ex.body.roll, abs=1e-9)
        seq = fx.resample(ex.feats, ex.t, ex.hand_mask)
        if seq is None:
            assert out["seq"] is None
        else:
            np.testing.assert_allclose(np.array(out["seq"]), seq, atol=1e-6)
        for row, js_static in zip(ex.feats, out["static"]):
            py_static = fx.static_vector(row)
            if py_static is None:
                assert js_static is None
            else:
                np.testing.assert_allclose(np.array(js_static), py_static, atol=1e-6)
