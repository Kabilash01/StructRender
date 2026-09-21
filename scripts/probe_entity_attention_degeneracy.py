"""Why Model E uses a lookup, not attention — the measurement behind Decision 75.

Regenerates the two numbers that decision rests on: how many bottleneck cells
have more than one legal entity, and how many parameters can therefore never
receive gradient. Decisions 44/51 require a committed artifact to have both a
generator and a test; this is the generator.
"""
import json, torch
from renderer.entity.dataset import build_entity_splits
from renderer.entity.model import downsample_masks_disjoint
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT, build_model, load_config

def main() -> int:
    sp = load_session_split(ROOT / "configs/data/session_split_v2.json")
    tr, _, _ = build_entity_splits("oil_impasto", sp, k_max=8)
    tot = multi = 0
    for i in range(0, len(tr), 913):
        d = downsample_masks_disjoint(tr[i]["entity_masks"][None], (36, 64))
        per = d.sum(1)
        tot += int((per > 0).sum()); multi += int((per > 1).sum())
    m = build_model(load_config(ROOT / "configs/ladder_v3/model_c.json"))
    pc = sum(p.numel() for p in m.parameters())
    rep = {"cells_with_an_entity": tot, "cells_with_more_than_one": multi,
           "softmax_is_constant": multi == 0,
           "model_c_params": pc,
           "note": ("With every cell owned by at most one entity the masked softmax "
                    "returns 1.0 by construction, so Q and K receive exactly zero "
                    "gradient. See KB Decision 75.")}
    out = ROOT / "eval" / "reports" / "entity_attention_degeneracy.json"
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
