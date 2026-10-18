"""IK round-trip test + check that MuJoCo's foot positions match our IK.  python tools/check_ik.py"""
import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import mujoco, numpy as np
from quadsim.kinematics import leg_ik, leg_fk, leg_to_body
from quadsim.robot import load_walker
from quadsim.scene import build_model

w = load_walker("stick_hexapod")
random.seed(0)
for _ in range(2000):
    t = np.array([random.uniform(0.08, 0.2), random.uniform(-0.1, 0.1), random.uniform(-0.15, 0.05)])
    if not abs(w.L2 - w.L3) < np.linalg.norm([np.hypot(t[0], t[1]) - w.L1, t[2]]) < w.L2 + w.L3: continue
    err = np.linalg.norm(leg_fk(*leg_ik(*t, w.L1, w.L2, w.L3), w.L1, w.L2, w.L3) - t)
    assert err < 1e-9, (t, err)
print("IK round-trip OK")

m = build_model(w, "flat"); d = mujoco.MjData(m)
d.qpos[:7] = [0, 0, 1, 1, 0, 0, 0]
q = np.array([0.3, 0.4, -1.2] * 6) + np.random.default_rng(1).normal(0, 0.2, 18)
d.qpos[7:] = q
mujoco.mj_forward(m, d)
for i, mt in enumerate(w.mounts):
    fk = leg_to_body(leg_fk(*q[3*i:3*i+3], w.L1, w.L2, w.L3), mt.x, mt.y, mt.yaw) + [0, 0, 1]
    sim = d.site(f"{mt.name}_foot").xpos
    assert np.allclose(fk, sim, atol=1e-6), (mt.name, fk, sim)
print("MuJoCo frames match kinematics OK")
