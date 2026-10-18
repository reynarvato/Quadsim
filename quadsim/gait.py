"""Gait generators. For now: a wave gait (one leg in the air at a time).

The gait works in the BODY frame: each foot has a neutral position; while a
foot is on the ground it slides backwards relative to the body (so the body
moves forwards), and while it swings it is lifted and carried forwards.
Turning is handled by the same maths: the foot's slide velocity is
-(v + w x p), so a yaw command makes each foot sweep around the body centre.
"""
from __future__ import annotations

import math

import numpy as np

from .kinematics import body_to_leg, leg_ik, leg_to_body
from .robot import Walker


class WaveGait:
    def __init__(self, walker: Walker):
        self.w = walker
        g = walker.gait
        self.cycle = g["cycle_s"]
        self.swing_h = g["swing_height_m"]
        self.max_stride = g["max_stride_m"]
        self.max_v = g["max_speed_mps"]
        self.max_wz = g["max_yaw_rate_rps"]
        n = len(walker.mounts)
        order = g.get("order", [m.name for m in walker.mounts])
        self.offset = {name: k / n for k, name in enumerate(order)}
        self.swing_frac = 1.0 / n  # wave gait: exactly one leg swings at a time
        self.duty = 1.0 - self.swing_frac

        self.base_height = walker.stance["body_height_m"]
        reach = walker.L1 + walker.stance["foot_reach_m"]
        self.neutral = {m.name: leg_to_body((reach, 0.0, 0.0), m.x, m.y, m.yaw) for m in walker.mounts}

        self.phase = 0.0
        self.cmd = np.zeros(3)   # filtered (vx, vy, wz) in m/s, m/s, rad/s
        self.lift = 0.0          # filtered swing-height scale (0 when idle)
        self.height = self.base_height

    # --------------------------------------------------------------------
    def update(self, dt: float, cmd_norm, height_delta: float = 0.0) -> np.ndarray:
        """cmd_norm = (forward, left, turn_left) each in [-1, 1]. Returns ctrl targets."""
        target = np.array([cmd_norm[0] * self.max_v, cmd_norm[1] * self.max_v, cmd_norm[2] * self.max_wz])
        a = min(1.0, dt / 0.3)  # ~0.3 s smoothing so feet don't jump when the stick moves
        self.cmd += a * (target - self.cmd)
        moving = float(np.linalg.norm(cmd_norm) > 0.05)
        self.lift += min(1.0, dt / 0.25) * (moving - self.lift)
        self.height = float(np.clip(self.height + height_delta, self.base_height - 0.035, self.base_height + 0.035))

        self.phase = (self.phase + dt / self.cycle) % 1.0
        t_stance = self.duty * self.cycle

        # stride vector for each foot; scale the whole command down if any foot would over-stride
        vx, vy, wz = self.cmd
        strides = {}
        for name, p in self.neutral.items():
            strides[name] = np.array([vx - wz * p[1], vy + wz * p[0]]) * t_stance
        worst = max(np.linalg.norm(s) for s in strides.values())
        if worst > self.max_stride:
            k = self.max_stride / worst
            strides = {n: s * k for n, s in strides.items()}

        foot_z = -(self.height - self.w.foot_r)
        ctrl = []
        for m in self.w.mounts:
            ph = (self.phase - self.offset[m.name]) % 1.0
            d = strides[m.name]
            if ph < self.swing_frac:  # swing: -d/2 -> +d/2, lifted
                s = ph / self.swing_frac
                s_smooth = 0.5 - 0.5 * math.cos(math.pi * s)
                xy = -d / 2 + d * s_smooth
                z = foot_z + self.lift * self.swing_h * math.sin(math.pi * s)
            else:  # stance: +d/2 -> -d/2 on the ground
                s = (ph - self.swing_frac) / self.duty
                xy = d / 2 - d * s
                z = foot_z
            n = self.neutral[m.name]
            p_body = (n[0] + xy[0], n[1] + xy[1], z)
            ctrl.extend(leg_ik(*body_to_leg(p_body, m.x, m.y, m.yaw), self.w.L1, self.w.L2, self.w.L3))
        return np.array(ctrl)

    def standing_pose(self) -> np.ndarray:
        foot_z = -(self.base_height - self.w.foot_r)
        out = []
        for m in self.w.mounts:
            n = self.neutral[m.name]
            out.extend(leg_ik(*body_to_leg((n[0], n[1], foot_z), m.x, m.y, m.yaw), self.w.L1, self.w.L2, self.w.L3))
        return np.array(out)


GAITS = {"wave": WaveGait}


def make_gait(walker: Walker):
    return GAITS[walker.gait.get("type", "wave")](walker)
