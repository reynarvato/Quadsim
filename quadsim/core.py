"""Simulation core: no rendering, no input. Used by the game loop and the headless tests."""
from __future__ import annotations

import math
import mujoco
import numpy as np

from .gait import make_gait
from .robot import Walker
from .scene import build_model


class SimSession:
    CONTROL_DT = 0.01  # gait / controller runs at 100 Hz

    def __init__(self, walker: Walker, level_key: str):
        self.walker = walker
        self.level_key = level_key
        self.model = build_model(walker, level_key)
        self.data = mujoco.MjData(self.model)
        self.gait = make_gait(walker)
        self.torso_id = self.model.body("torso").id
        self.finish_id = self._site_id("finish")
        self.spawn_id = self._site_id("spawn")
        self.n_sub = max(1, round(self.CONTROL_DT / self.model.opt.timestep))
        self.reset()

    def _site_id(self, name):
        try:
            return self.model.site(name).id
        except KeyError:
            return None

    # ------------------------------------------------------------------
    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)  # so site positions are valid
        spawn = d.site_xpos[self.spawn_id].copy() if self.spawn_id is not None else np.zeros(3)
        d.qpos[0:3] = spawn + [0, 0, self.walker.stance["body_height_m"] + 0.01]
        d.qpos[3:7] = [1, 0, 0, 0]
        pose = self.gait.standing_pose()
        d.qpos[7:7 + len(pose)] = pose
        d.ctrl[:] = pose
        self.gait = make_gait(self.walker)
        mujoco.mj_forward(m, d)
        self.sim_time = 0.0
        self.run_start = None   # racecourse timer
        self.run_time = None
        self._acc = 0.0

    def step(self, dt: float, cmd, height_rate: float = 0.0):
        """Advance by dt seconds. cmd = (forward, left, turn_left) in [-1, 1]; height_rate in m/s."""
        self._acc += dt
        while self._acc >= self.CONTROL_DT:
            self._acc -= self.CONTROL_DT
            self.data.ctrl[:] = self.gait.update(self.CONTROL_DT, cmd, height_rate * self.CONTROL_DT)
            for _ in range(self.n_sub):
                mujoco.mj_step(self.model, self.data)
            self.sim_time += self.CONTROL_DT
        self._update_race(cmd)

    def _update_race(self, cmd):
        if self.finish_id is None or self.run_time is not None:
            return
        if self.run_start is None and np.linalg.norm(cmd) > 0.05:
            self.run_start = self.sim_time
        if self.run_start is not None and self.in_finish():
            self.run_time = self.sim_time - self.run_start

    # ------------------------------------------------------------------ state
    def in_finish(self) -> bool:
        if self.finish_id is None:
            return False
        c = self.data.site_xpos[self.finish_id]
        half = self.model.site_size[self.finish_id]
        p = self.torso_pos
        return bool(np.all(np.abs(p - c) <= half))

    @property
    def torso_pos(self):
        return self.data.xpos[self.torso_id]

    @property
    def yaw(self):
        w, x, y, z = self.data.xquat[self.torso_id]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    @property
    def tilt_deg(self):
        zaxis = self.data.xmat[self.torso_id].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(max(-1.0, min(1.0, zaxis[2]))))

    @property
    def fallen(self) -> bool:
        return self.tilt_deg > 60

    @property
    def speed(self):
        return float(np.linalg.norm(self.data.qvel[0:2]))

    @property
    def total_mass(self):
        return float(self.model.body_subtreemass[self.torso_id])

    def max_servo_load(self) -> float:
        """Peak |actuator torque| / limit across all servos (1.0 = saturated)."""
        lim = self.model.actuator_forcerange[:, 1]
        return float(np.max(np.abs(self.data.actuator_force) / lim))
