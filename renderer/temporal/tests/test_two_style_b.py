"""Two-style Model B and per-frame exclusion — KB Decision 94."""

import json

import pytest
import torch

from renderer.temporal.style_unet import StyleTemporalUNet
from renderer.temporal.temporal_unet import TemporalUNet
from renderer.training import multi_session as MS
from renderer.training import train_temporal as T
from renderer.training.train_baseline import ROOT

STYLES = ["oil_impasto", "watercolor_bold_s055"]


def _pair(seed=0):
    torch.manual_seed(seed)
    s = StyleTemporalUNet(STYLES, image_size=(64, 64), base_channels=8)
    b = TemporalUNet(image_size=(64, 64), base_channels=8)
    b.load_state_dict({k: v for k, v in s.state_dict().items() if k in b.state_dict()})
    return s, b


def _history(m, x):
    feats, _ = m.encode(x)
    return feats.detach(), torch.randn(1, 2, 64, 64), torch.rand(1, 1, 64, 64)


@torch.no_grad()
def test_untrained_two_style_b_is_exactly_model_b_in_both_styles():
    s, b = _pair()
    x = torch.randn(1, 7, 64, 64)
    h = _history(b, x)
    for i in range(len(STYLES)):
        idx = torch.tensor([i])
        assert torch.equal(s(x, style_idx=idx), b(x))
        assert torch.equal(s(x, *h, style_idx=idx), b(x, *h))


def test_film_gets_gradient_at_init_so_styles_can_separate():
    s, _ = _pair()
    x = torch.randn(2, 7, 64, 64)
    s(x, style_idx=torch.tensor([0, 1])).mean().backward()
    for film in s.film:
        assert float(film.weight.grad.abs().sum()) > 0
    assert float(s.style_table.grad.abs().sum()) == 0   # zero FiLM -> no signal yet, by design


@torch.no_grad()
def test_styles_differ_once_film_is_nonzero_and_encoder_ignores_style():
    s, _ = _pair()
    for f in s.film:
        f.weight.normal_(0, 0.1)
    x = torch.randn(1, 7, 64, 64)
    a, c = s(x, style_idx=torch.tensor([0])), s(x, style_idx=torch.tensor([1]))
    assert not torch.allclose(a, c)
    # The encoder has no style input: the history features are style-independent,
    # so switching style mid-game cannot disturb the temporal memory.
    assert "style_idx" not in s.encode.__code__.co_varnames


def test_forward_without_style_is_refused():
    s, _ = _pair()
    with pytest.raises(ValueError):
        s(torch.randn(1, 7, 64, 64))


def test_parameter_overhead_is_under_one_percent_of_b():
    cfg = json.load(open(ROOT / "configs/game/model_b_two_style.json"))
    base = json.load(open(ROOT / "configs/game/model_b_oil.json"))
    n = sum(p.numel() for p in T.build_model(cfg).parameters())
    nb = sum(p.numel() for p in T.build_model(base).parameters())
    assert nb < n < 1.01 * nb


def test_config_style_order_must_match_data():
    cfg = json.load(open(ROOT / "configs/game/model_b_two_style.json"))
    cfg["model"]["style_ids"] = cfg["model"]["style_ids"][::-1]
    with pytest.raises(ValueError):
        T.build_model(cfg)


def test_training_and_eval_forward_route_the_style():
    torch.manual_seed(0)
    s = StyleTemporalUNet(STYLES, image_size=(64, 64), base_channels=8)
    for f in s.film:
        torch.nn.init.normal_(f.weight, 0, 0.1)
    z = torch.randn(1, 7, 64, 64)
    b = {"structure_prev2": z, "structure_prev": z, "structure_curr": z,
         "motion_prev": torch.zeros(1, 2, 64, 64), "mask_prev": torch.ones(1, 1, 64, 64),
         "motion_curr": torch.zeros(1, 2, 64, 64), "mask_curr": torch.ones(1, 1, 64, 64)}
    with torch.no_grad():
        p0 = T.eval_forward(s, {**b, "style_idx": torch.tensor([0])})
        p1 = T.eval_forward(s, {**b, "style_idx": torch.tensor([1])})
        t0, _ = T.training_forward(s, {**b, "style_idx": torch.tensor([0])})
    assert not torch.allclose(p0, p1)
    assert torch.allclose(p0, t0, atol=1e-6)


# --------------------------------------------------------------------------- #
# per-frame exclusion
# --------------------------------------------------------------------------- #
class _FakeSession:
    validated = True

    def __init__(self, path):
        self.session_id = path.name

    def frame_ids(self):
        return list(range(10))


class _Recorder:
    calls = []

    def __init__(self, session, tdir, frame_ids, **kw):
        self.ids = list(frame_ids)
        _Recorder.calls.append((session.session_id, tdir, self.ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"frame_id": self.ids[i]}


def _patched(monkeypatch):
    from renderer.temporal import pair_dataset
    _Recorder.calls = []
    monkeypatch.setattr(MS, "StructuralSession", _FakeSession)
    monkeypatch.setattr(MS, "target_dir_for", lambda sid, style, root=None: f"{style}/{sid}")
    monkeypatch.setattr(pair_dataset, "TemporalTripleDataset", _Recorder)
    monkeypatch.setattr(pair_dataset, "TemporalFrameDataset", _Recorder)


SPLIT = {"train_sessions": ["s1"], "val_sessions": ["s2"], "test_sessions": []}


def _ex(**over):
    ex = {"styles": STYLES, "admitted": {st: ["s1", "s2"] for st in STYLES},
          "refused": {st: {} for st in STYLES},
          # frame 3 fails oil only, frame 7 fails watercolour only
          "excluded": {"s1": {"3": {"oil_impasto": ["align.x"]},
                              "7": {"watercolor_bold_s055": ["align.y"]}}}}
    ex.update(over)
    return ex


def test_excluded_frames_are_the_union_over_styles_and_removed_from_both(monkeypatch):
    _patched(monkeypatch)
    train, val = MS.build_two_style_temporal_splits(STYLES, SPLIT, _ex())
    s1 = [c for c in _Recorder.calls if c[0] == "s1"]
    assert [c[1] for c in s1] == [f"{st}/s1" for st in STYLES]
    for _, _, ids in s1:
        assert 3 not in ids and 7 not in ids and len(ids) == 8
    assert [c[2] for c in _Recorder.calls if c[0] == "s2"] == [list(range(10))] * 2
    # every item carries its style index, oil first
    assert train[0]["style_idx"] == 0 and train[len(train) - 1]["style_idx"] == 1
    assert len(train) == 16 and len(val) == 20


def test_unadmitted_session_is_refused(monkeypatch):
    _patched(monkeypatch)
    ex = _ex(admitted={"oil_impasto": ["s1", "s2"], "watercolor_bold_s055": ["s2"]},
             refused={"oil_impasto": {}, "watercolor_bold_s055": {"s1": ["layout.bad"]}})
    with pytest.raises(RuntimeError, match="not admitted"):
        MS.build_two_style_temporal_splits(STYLES, SPLIT, ex)


def test_exclusion_list_for_other_styles_is_refused(monkeypatch):
    _patched(monkeypatch)
    with pytest.raises(RuntimeError, match="exclusion list covers"):
        MS.build_two_style_temporal_splits(STYLES, SPLIT, _ex(styles=STYLES[::-1]))


def test_train_baseline_refuses_two_style_b(monkeypatch):
    from renderer.training import train_baseline as B
    monkeypatch.setattr("sys.argv", ["x", "--config", "configs/game/model_b_two_style.json", "--mode", "train"])
    with pytest.raises(SystemExit) as e:
        B.main()
    assert "train_temporal" in str(e.value)


def test_two_style_checkpoint_round_trips(tmp_path):
    from renderer.training.train_baseline import load_checkpoint
    cfg = {"model": {"arch": "style_temporal_unet", "image_size": [64, 64], "base_channels": 8,
                     "style_ids": STYLES}, "data": {"styles": STYLES}}
    m = T.build_model(cfg)
    for f in m.film:
        torch.nn.init.normal_(f.weight, 0, 0.1)
    torch.save({"model": m.state_dict(), "arch": T.arch_of(cfg), "model_kwargs": T.model_kwargs(cfg)},
               tmp_path / "best.pt")
    r = load_checkpoint(tmp_path / "best.pt", torch.device("cpu"))
    x = torch.randn(1, 7, 64, 64)
    with torch.no_grad():
        assert r.style_ids == STYLES
        assert torch.equal(r(x, style_idx=torch.tensor([1])), m.eval()(x, style_idx=torch.tensor([1])))
