"""Phase-based gaits: tripod (fast), ripple, wave (stable). Defined in the walker JSON.

Each leg has a phase offset in the cycle and spends `duty` of the cycle on the
ground. The gait works in the BODY frame: a foot on the ground slides backwards
relative to the body (so the body moves forwards); a swinging foot is lifted and
carried forwards. Turning uses the same maths: each foot's slide velocity is
-(v + w x p), so a yaw command makes the feet sweep around the body centre.

Top speed is not a separate setting: it follows from stride and timing,
v_max = max_stride / (duty * cycle). Longer stride = faster with a natural look;
shorter cycle (faster stepping) = faster but looks "sped up".

Cadence follows the stick: at low speed the cycle is stretched by `cadence_slowdown`
(long, unhurried steps) and only reaches `cycle_s` at full throttle, like an animal
lengthening its stride first and quickening its steps last.
"""
from __future__ import annotations

import math

import numpy as np

from .kinematics import body_to_leg, leg_ik, leg_to_body
from .robot import Walker


class PhaseGait:
    def __init__(self, walker: Walker, gait_type: str | None = None, speed_mult: float = 1.0,
                 stride_mult: float = 1.0):
        self.w = walker
        self.speed_mult = speed_mult    # faster motors -> shorter step cycle
        self.stride_mult = stride_mult  # longer strides -> faster without faster stepping
        g = walker.gait
        self.types = list(g["types"])
        self.base_stride = g["max_stride_m"]
        self.slowdown = g.get("cadence_slowdown", 1.8)
        self.stride_limit = g.get("stride_limit_m", self.base_stride * 1.6)  # hard reach limit (boost/stride)
        self.boost = 1.0
        self.response = g.get("response_s", 0.06)

        self.base_height = walker.stance["body_height_m"]
        reach = walker.L1 + walker.stance["foot_reach_m"]
        self.neutral = {m.name: leg_to_body((reach, 0.0, 0.0), m.x, m.y, m.yaw) for m in walker.mounts}
        self.r_max = max(math.hypot(p[0], p[1]) for p in self.neutral.values())

        self.phase = 0.0
        self.cmd = np.zeros(3)   # filtered (vx, vy, wz)
        self.lift = 0.0          # swing-height scale (0 when idle so feet stay planted)
        self.height = self.base_height
        self.set_type(gait_type or g.get("default", self.types[0]))

    # ------------------------------------------------------------------ config
    def set_type(self, name: str):
        t = self.w.gait["types"][name]
        self.name = name
        self.label = t.get("label", name)
        self.max_stride = self.base_stride * self.stride_mult
        self.cycle = t["cycle_s"] / self.speed_mult   # cycle at full speed
        self.cycle_now = self.cycle * self.slowdown   # current (stretched at low speed)
        self.duty = t["duty"]
        self.swing_frac = 1.0 - self.duty
        self.swing_h = t["swing_height_m"]
        self.offset = {m.name: t["offsets"].get(m.name, 0.0) for m in self.w.mounts}
        t_stance = self.duty * self.cycle
        self.max_v = self.max_stride / t_stance
        self.max_wz = self.max_stride / (t_stance * self.r_max)

    def next_type(self):
        i = self.types.index(self.name)
        self.set_type(self.types[(i + 1) % len(self.types)])

    # ------------------------------------------------------------------ update
    def update(self, dt: float, cmd_norm, height_delta: float = 0.0, boost: float = 1.0) -> np.ndarray:
        """cmd_norm = (forward, left, turn_left), 1.0 = normal top speed (may exceed 1 while boosting).
        boost > 1 lengthens strides and quickens the cycle by sqrt(boost) each. Returns 18 joint targets."""
        self.boost = boost
        bs = math.sqrt(max(1.0, boost))
        target = np.array([cmd_norm[0] * self.max_v, cmd_norm[1] * self.max_v, cmd_norm[2] * self.max_wz])
        a = min(1.0, dt / max(self.response, 1e-3))
        self.cmd += a * (target - self.cmd)
        moving = float(np.linalg.norm(cmd_norm) > 0.05 or np.linalg.norm(self.cmd[:2]) > 0.01)
        self.lift += min(1.0, dt / max(self.response, 1e-3)) * (moving - self.lift)
        self.height = float(np.clip(self.height + height_delta, self.base_height - 0.035, self.base_height + 0.035))

        # cadence: slow, long steps at low speed; full step rate only at full throttle
        u = min(1.0, max(abs(self.cmd[0]) / self.max_v, abs(self.cmd[1]) / self.max_v, abs(self.cmd[2]) / self.max_wz))
        self.cycle_now = self.cycle * (self.slowdown - (self.slowdown - 1.0) * u) / bs
        self.phase = (self.phase + dt / self.cycle_now) % 1.0
        t_stance = self.duty * self.cycle_now

        vx, vy, wz = self.cmd
        strides = {n: np.array([vx - wz * p[1], vy + wz * p[0]]) * t_stance for n, p in self.neutral.items()}
        cap = min(self.max_stride * bs, self.stride_limit)
        worst = max(np.linalg.norm(s) for s in strides.values())
        if worst > cap:
            k = cap / worst
            strides = {n: s * k for n, s in strides.items()}

        foot_z = -(self.height - self.w.foot_r)
        ctrl = []
        for m in self.w.mounts:
            ph = (self.phase - self.offset[m.name]) % 1.0
            d = strides[m.name]
            if ph < self.swing_frac:  # swing: -d/2 -> +d/2, lifted
                s = ph / self.swing_frac
                xy = -d / 2 + d * (0.5 - 0.5 * math.cos(math.pi * s))
                z = foot_z + self.lift * self.swing_h * math.sin(math.pi * s)
            else:  # stance: +d/2 -> -d/2 on the ground
                s = (ph - self.swing_frac) / self.duty
                xy = d / 2 - d * s
                z = foot_z
            n = self.neutral[m.name]
            ctrl.extend(leg_ik(*body_to_leg((n[0] + xy[0], n[1] + xy[1], z), m.x, m.y, m.yaw),
                               self.w.L1, self.w.L2, self.w.L3))
        return np.array(ctrl)

    def standing_pose(self) -> np.ndarray:
        foot_z = -(self.base_height - self.w.foot_r)
        out = []
        for m in self.w.mounts:
            n = self.neutral[m.name]
            out.extend(leg_ik(*body_to_leg((n[0], n[1], foot_z), m.x, m.y, m.yaw), self.w.L1, self.w.L2, self.w.L3))
        return np.array(out)


def make_gait(walker: Walker, gait_type: str | None = None, speed_mult: float = 1.0,
              stride_mult: float = 1.0) -> PhaseGait:
    return PhaseGait(walker, gait_type, speed_mult, stride_mult)
