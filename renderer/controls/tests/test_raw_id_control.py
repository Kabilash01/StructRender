"""Model N is a control: it must carry the raw ID, and nothing else may."""

import ast
import json
from pathlib import Path

import torch

from renderer.training.train_baseline import ROOT


def test_raw_id_channel_is_the_scaled_entity_id():
    from data_pipeline.teacher.structural import StructuralSession
    from renderer.controls.raw_id import ID_SCALE, RawIDFrameDataset
    from renderer.training.dataset import StyledFrameDataset, target_dir_for
    sid = json.load(open(ROOT / "configs/data/session_split_v2.json"))["val_sessions"][0]
    s = StructuralSession(ROOT / "data_pipeline/recorder" / sid)
    ds = RawIDFrameDataset(StyledFrameDataset(s, target_dir_for(s.session_id, "oil_impasto"), s.frame_ids()[:3]))
    x, _, fid = ds[1]
    assert x.shape[0] == 8
    ids = torch.from_numpy(s.load_structure(int(fid)).entity_id.astype("float32")) / ID_SCALE
    assert torch.equal(x[7], ids)


def test_no_production_module_imports_the_control():
    """Decision 9: only the control may feed IDs as values."""
    offenders = []
    for p in (Path(ROOT) / "renderer").rglob("*.py"):
        if "controls" in p.parts or "tests" in p.parts or p.name == "train_baseline.py":
            continue
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and "controls" in node.module:
                offenders.append(str(p))
    assert not offenders, offenders


def test_only_the_n_config_turns_the_control_on():
    on = [p.name for p in (Path(ROOT) / "configs").rglob("*.json")
          if json.load(open(p)).get("data", {}).get("raw_id_control")]
    assert on == ["model_n.json"], on
