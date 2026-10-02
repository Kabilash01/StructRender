"""Score two-style B with NO frame history (every frame a cold start) — the
single-input mode a style-switching live export would run if the temporal
inputs are not wired. Same scorer, same frames, same exclusions as Decision 95;
only the history is withheld, so the gap is exactly what history buys."""
from __future__ import annotations
import sys
import torch
import scripts.eval_game as E

class NoHistory(torch.nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner = inner; self.style_ids = inner.style_ids
    def encode(self, x):
        return self.inner.encode(x)
    def forward(self, x, prev=None, motion=None, mask=None, style_idx=None):
        return self.inner(x, style_idx=style_idx)

if __name__ == "__main__":
    style, out = sys.argv[1], sys.argv[2]
    load = E.load_checkpoint
    E.load_checkpoint = lambda p, d: NoHistory(load(p, d))
    raise SystemExit(E.main("checkpoints/game_b_two_style/best.pt", "configs/data/game_split_v1.json",
                            style, out, "configs/data/game_frame_exclusions_v1.json"))
