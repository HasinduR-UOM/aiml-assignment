import json

from src.gate import main


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_first_model_passes(tmp_path):
    ch = write(tmp_path, "c.json", {"best_model": "Ridge", "test_rmse_normal": 470.0})
    assert main(["--challenger", ch, "--champion", write(tmp_path, "n.json", None),
                 "--report", str(tmp_path / "r.md")]) == 0


def test_worse_model_fails(tmp_path):
    ch = write(tmp_path, "c.json", {"best_model": "Ridge", "test_rmse_normal": 470.0})
    champ = write(tmp_path, "champ.json", {"test_rmse_normal": "400.0", "version": "2"})
    assert main(["--challenger", ch, "--champion", champ, "--report", str(tmp_path / "r.md")]) == 1
