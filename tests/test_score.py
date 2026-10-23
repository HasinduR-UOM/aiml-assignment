"""Azure ML scoring script tests. Skipped when no trained model is present."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "models" / "final_model.joblib").exists(),
                                reason="no trained model")


@pytest.fixture(scope="module")
def score():
    import os
    os.environ["AZUREML_MODEL_DIR"] = str(ROOT / "artifacts")
    from aml import score as s
    s.init()
    return s


def test_single_request(score):
    out = score.run((ROOT / "aml" / "sample-request.json").read_text())
    assert 500 < out["yield_kg_ha"] < 9000 and "shortfall" in out


def test_batch_with_bad_district(score):
    out = score.run(json.dumps([{"district": "Polonnaruwa", "season": "Maha"}, {"district": "Atlantis", "season": "Maha"}]))
    assert len(out) == 2 and "error" in out[1]
