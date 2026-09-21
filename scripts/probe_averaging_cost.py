"""Probe: how much sharpness does AVERAGING cost?

The renderer is trained under L1+LPIPS, so its optimum is ~the conditional mean
of the teacher given structure. The teacher flickers 32x the engine, so that mean
is an average over many different-looking samples of the SAME entity. If averaging
is what blurs, then EPAM's "commit to one appearance" has a sharpness story —
and if it is not, EPAM has no sharpness story and that is worth knowing BEFORE
spending 1.5 weeks on M8.

Measured with no training: take an entity's frames within a reappearance window
(same entity, near-identical pose), and compare the sharpness of INDIVIDUAL
teacher targets against the sharpness of their MEAN.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0,'/home/kabilash01/StructRender')
from PIL import Image
from data_pipeline.teacher.structural import StructuralSession
from eval.metrics.rcs import entity_mask_at, reappearance_sequences
from renderer.training.dataset import target_dir_for
from renderer.training.multi_session import load_session_split
from renderer.training.train_baseline import ROOT

def grad_energy(img, mask=None):
    g = torch.from_numpy(img).permute(2,0,1)[None].float().mean(1, keepdim=True)
    kx = torch.tensor([[[[-1.,0,1],[-2,0,2],[-1,0,1]]]])
    gx = torch.nn.functional.conv2d(g, kx, padding=1)
    gy = torch.nn.functional.conv2d(g, kx.transpose(2,3), padding=1)
    e = (gx**2+gy**2).sqrt()[0,0].numpy()
    if mask is not None:
        m = mask.astype(bool)
        return float(e[m].mean()) if m.sum() > 50 else None
    return float(e.mean())

split = load_session_split(ROOT/"configs/data/session_split_v2.json")
rows_ind, rows_avg = [], []
for sid in split["test_sessions"]:
    s = StructuralSession(ROOT/"data_pipeline"/"recorder"/sid)
    td = target_dir_for(s.session_id, "oil_impasto")
    idx = {int(json.loads(l)["frame_id"]): json.loads(l)["target_rgb_path"]
           for l in (td/"manifest.jsonl").read_text().splitlines() if l.strip()}
    def tgt(f): return np.asarray(Image.open(td/idx[f]).convert("RGB"),dtype=np.float32)/255.
    for q in reappearance_sequences(s):
        iid = int(q["target_instance_id"]); t0 = int(q["last_seen_frame"])
        # consecutive frames where the entity is visible: near-identical pose
        run = [f for f in range(t0-3, t0+1) if f in set(s.frame_ids())
               and entity_mask_at(s, f, iid).sum() > 200]
        if len(run) < 3: continue
        imgs = [tgt(f) for f in run]
        m = entity_mask_at(s, run[-1], iid)
        ind = [grad_energy(i, m) for i in imgs]
        ind = [v for v in ind if v]
        avg = grad_energy(np.mean(imgs, axis=0), m)
        if ind and avg:
            rows_ind.append(float(np.mean(ind))); rows_avg.append(avg)

i, a = np.array(rows_ind), np.array(rows_avg)
print(f"n = {len(i)} entity windows (same entity, {3}-4 consecutive frames)\n")
print(f"  individual teacher frames : {i.mean():.5f}  (sd {i.std(ddof=1):.5f})")
print(f"  MEAN of those frames      : {a.mean():.5f}  (sd {a.std(ddof=1):.5f})")
loss = 100*(a.mean()-i.mean())/i.mean()
print(f"\n  sharpness change from averaging: {loss:+.1f}%")
se = (i-a).std(ddof=1)/np.sqrt(len(i))
print(f"  paired diff {np.mean(i-a):+.5f} +/- {se:.5f} (SE)  -> {abs(np.mean(i-a))/se:.1f} sigma")
print(f"\n  => averaging {'COSTS real sharpness -> EPAM has a sharpness story' if loss < -5 else 'costs little -> EPAM has NO sharpness story here'}")
