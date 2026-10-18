"""Simulation core: no rendering, no input. Used by the game loop, the debug viewer and the tests."""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .gait import make_gait
from .robot import Walker
from .scene import build_model

RHO_AIR = 1.2
# Ray-casts (BBs, laser) only test geom groups 0 (world + simple robot) and 3 (URDF collision shapes),
# skipping the hundreds of purely visual URDF geoms in group 1.
RAY_GROUPS = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
G = np.array([0.0, 0.0, -9.81])


@dataclass
class Controls:
    """One frame of player input, all normalised."""
    forward: float = 0.0     # +1 = forward
    strafe: float = 0.0      # +1 = left
    turn: float = 0.0        # +1 = turn left
    height: float = 0.0      # +1 = raise body, -1 = lower
    pan: float = 0.0         # +1 = turret left
    tilt: float = 0.0        # +1 = turret up
    fire: bool = False       # held

    @property
    def move(self):
        return (self.forward, self.strafe, self.turn)


FEEL_DEFAULTS = {
    # Arcade movement layer between the stick and the gait (not physics - flavour).
    "accel_mps2": 2.5,        # how quickly it gets up to speed
    "friction_mps2": 1.4,     # coasting deceleration when you let go (lower = more slide)
    "brake_mps2": 4.0,        # deceleration when you push the opposite way
    "yaw_accel_rps2": 10.0,   # how quickly turning starts/stops
    "grip_g": 0.16,           # sideways grip: faster = wider turns (lateral accel limit, in g)
    "boost": {
        "threshold": 0.95,    # stick must be at least this far forward...
        "hold_s": 1.2,        # ...for this long to kick in
        "factor": 1.5,        # speed multiplier at full boost
        "ramp_s": 0.35,       # time to spool up / down
        "duration_s": 3.0,    # full meter lasts this long
        "recharge_s": 5.0,    # empty -> full while not boosting
        "min_restart": 0.25,  # meter needed to boost again
    },
}


class MovementFeel:
    """Momentum, coasting friction, a grip limit on turning, and a boost meter."""

    def __init__(self, cfg: dict | None = None):
        c = {**FEEL_DEFAULTS, **(cfg or {})}
        c["boost"] = {**FEEL_DEFAULTS["boost"], **(cfg or {}).get("boost", {})}
        self.c = c
        self.v = np.zeros(2)      # body-frame velocity command, m/s
        self.wz = 0.0             # yaw rate command, rad/s
        self.hold = 0.0           # seconds at full throttle
        self.level = 0.0          # 0..1 boost spool
        self.energy = 1.0         # 0..1 meter
        self.boosting = False
        self.skid = 0.0           # 0..1, how hard we're braking / sliding (for dust)
        self.grip_limited = False

    def update(self, dt, c: "Controls", max_v, max_wz):
        b = self.c["boost"]
        full = c.forward >= b["threshold"]
        self.hold = self.hold + dt if full else 0.0
        if self.boosting:
            self.energy -= dt / b["duration_s"]
            if not full or self.energy <= 0:
                self.boosting = False
                self.energy = max(self.energy, 0.0)
        else:
            if full and self.hold >= b["hold_s"] and self.energy >= b["min_restart"]:
                self.boosting = True
            elif not full:
                self.energy = min(1.0, self.energy + dt / b["recharge_s"])
        self.level += np.clip((1.0 if self.boosting else 0.0) - self.level, -dt / b["ramp_s"], dt / b["ramp_s"])
        boost = 1.0 + (b["factor"] - 1.0) * self.level

        # linear: accelerate toward the stick, coast with friction when you let go, brake when reversing
        target = np.array([c.forward, c.strafe]) * max_v * np.array([boost, 1.0])
        dv = target - self.v
        speed, tspeed = np.linalg.norm(self.v), np.linalg.norm(target)
        reversing = speed > 0.05 and float(target @ self.v) < 0
        if reversing:
            rate = self.c["brake_mps2"]
        elif tspeed < speed:
            rate = self.c["friction_mps2"]
        else:
            rate = self.c["accel_mps2"] * (1.6 if self.boosting else 1.0)
        n = np.linalg.norm(dv)
        if n > rate * dt:
            dv *= rate * dt / n
        self.v += dv

        # yaw with inertia, limited by grip at speed: |v| * |wz| <= grip * g
        wt = c.turn * max_wz
        speed = float(np.linalg.norm(self.v))
        lim = self.c["grip_g"] * 9.81 / max(speed, 1e-3)
        self.grip_limited = abs(wt) > lim and speed > 0.3
        wt = float(np.clip(wt, -lim, lim))
        ya = self.c["yaw_accel_rps2"] * dt
        self.wz += float(np.clip(wt - self.wz, -ya, ya))

        # skid amount for dust: braking/coasting hard or turning at the grip limit
        braking = (reversing or tspeed < speed - 0.15) and speed > 0.25
        want = 1.0 if (braking or self.grip_limited) else 0.0
        self.skid += (want - self.skid) * min(1.0, dt * 8)

        return (self.v[0] / max_v, self.v[1] / max_v, self.wz / max_wz), boost


class SimSession:
    CONTROL_DT = 0.01  # gait / controllers run at 100 Hz

    def __init__(self, walker: Walker, level_key: str, speed_mult: float = 1.0,
                 turret_sens=(1.0, 1.0), stride_mult: float = 1.0):
        """speed_mult scales motor speed (step rate + servo top speed); stride_mult scales stride length;
        turret_sens = (pan, tilt) rate scales."""
        self.walker = walker
        self.speed_mult = speed_mult
        self.stride_mult = stride_mult
        self.turret_sens = turret_sens
        self.level_key = level_key
        self.model = build_model(walker, level_key)
        self.data = mujoco.MjData(self.model)
        self.torso_id = self.model.body("torso").id
        self.finish_id = self._site_id("finish")
        self.spawn_id = self._site_id("spawn")
        self.n_sub = max(1, round(self.CONTROL_DT / self.model.opt.timestep))
        self.n_leg = 3 * len(walker.mounts)
        self.servo_rate = math.radians(walker.cfg["servo"].get("max_speed_dps", 1e6)) * speed_mult

        self.has_turret = walker.has_turret
        if self.has_turret:
            t = walker.turret
            self.pan_range = np.radians(t["pan_range_deg"])
            self.tilt_range = np.radians(t["tilt_range_deg"])
            self.pan_rate = math.radians(t["pan_rate_dps"]) * turret_sens[0]
            self.tilt_rate = math.radians(t["tilt_rate_dps"]) * turret_sens[1]
            bb = t["bb"]
            self.bb_r = bb["diameter_mm"] / 2000.0
            self.bb_m = bb["mass_g"] / 1000.0
            self.bb_v0 = bb["muzzle_speed_mps"]
            self.bb_period = 1.0 / bb["fire_rate_hz"]
            # >1 exaggerates how hard BBs shove things (1.0 = real momentum, 0.2 g at 45 m/s)
            self.bb_impulse_scale = bb.get("impulse_scale", 3.0)
            # quadratic drag constant: a = -k |v| v
            self.bb_k = 0.5 * RHO_AIR * bb["drag_cd"] * math.pi * self.bb_r ** 2 / self.bb_m
            self.muzzle_id = self.model.site("muzzle").id
            self.tilt_body = self.model.body("turret_tilt").id
        self.gait_type = None
        self.feel_enabled = True
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
        self.spawn_z = float(spawn[2])
        d.qpos[3:7] = [1, 0, 0, 0]
        if self.spawn_id is not None:   # face the way the spawn site points (yaw only)
            R = d.site_xmat[self.spawn_id].reshape(3, 3)
            yaw = math.atan2(R[1, 0], R[0, 0])
            d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.gait = make_gait(self.walker, self.gait_type, self.speed_mult, self.stride_mult)
        pose = self.gait.standing_pose()
        d.qpos[7:7 + self.n_leg] = pose
        d.ctrl[:self.n_leg] = pose
        self.pan = self.tilt = 0.0
        mujoco.mj_forward(m, d)
        self.sim_time = 0.0
        self.run_start = None   # racecourse timer
        self.run_time = None
        self._acc = 0.0
        # BBs: live projectiles + recent impact points
        self.bb_pos = np.zeros((0, 3))
        self.bb_vel = np.zeros((0, 3))
        self.impacts: list[tuple[np.ndarray, float]] = []
        self.shots = 0
        self._cooldown = 0.0
        self._pending_hits = []
        self._clear_qfrc = False
        self.feel = MovementFeel(self.walker.cfg.get("feel"))
        self.boost = 1.0
        self.dust: list[list] = []   # [pos(3), vel(3), t_spawn]
        self._dust_acc = 0.0

    def set_speed_mult(self, k: float):
        """Live change of motor speed (step rate + servo top speed)."""
        self.speed_mult = k
        self.servo_rate = math.radians(self.walker.cfg["servo"].get("max_speed_dps", 1e6)) * k
        self.gait.speed_mult = k
        self.gait.set_type(self.gait.name)

    def set_stride_mult(self, k: float):
        """Live change of stride length (faster travel at the same step rate)."""
        self.stride_mult = k
        self.gait.stride_mult = k
        self.gait.set_type(self.gait.name)

    def set_turret_sens(self, pan: float, tilt: float):
        self.turret_sens = (pan, tilt)
        if self.has_turret:
            t = self.walker.turret
            self.pan_rate = math.radians(t["pan_rate_dps"]) * pan
            self.tilt_rate = math.radians(t["tilt_rate_dps"]) * tilt

    def cycle_gait(self):
        self.gait.next_type()
        self.gait_type = self.gait.name

    def step(self, dt: float, c: Controls):
        """Advance by dt seconds of sim time."""
        self._acc += dt
        while self._acc >= self.CONTROL_DT:
            self._acc -= self.CONTROL_DT
            h = self.CONTROL_DT
            if self.feel_enabled:
                move, self.boost = self.feel.update(h, c, self.gait.max_v, self.gait.max_wz)
            else:
                move, self.boost = c.move, 1.0
            target = self.gait.update(h, move, 0.04 * c.height * h, boost=self.boost)
            # hobby servos have a top speed: move each target at most rate*h per tick (boost overclocks them)
            rate = self.servo_rate * self.boost * h
            cur = self.data.ctrl[:self.n_leg]
            self.data.ctrl[:self.n_leg] = cur + np.clip(target - cur, -rate, rate)
            self._update_dust(h)
            if self.has_turret:
                self.pan = float(np.clip(self.pan + c.pan * self.pan_rate * h, *self.pan_range))
                self.tilt = float(np.clip(self.tilt + c.tilt * self.tilt_rate * h, *self.tilt_range))
                self.data.ctrl[self.n_leg:self.n_leg + 2] = (self.pan, self.tilt)
            if self._pending_hits:   # BB impulses, spread over this control tick
                for bid, p, J in self._pending_hits:
                    mujoco.mj_applyFT(self.model, self.data, J / h, np.zeros(3), p, bid, self.data.qfrc_applied)
                self._pending_hits = []
                self._clear_qfrc = True
            for _ in range(self.n_sub):
                mujoco.mj_step(self.model, self.data)
            if self._clear_qfrc:
                self.data.qfrc_applied[:] = 0
                self._clear_qfrc = False
            self.sim_time += h
            if self.has_turret:
                self._update_gun(h, c.fire)
        self._update_race(c)

    # ------------------------------------------------------------------ dust (visual only)
    def _update_dust(self, h: float):
        skid = self.feel.skid if self.feel_enabled else 0.0
        self._dust_acc += h * 60.0 * skid      # up to ~60 puffs/s while skidding hard
        rng = np.random.default_rng()
        while self._dust_acc >= 1.0:
            self._dust_acc -= 1.0
            m = self.walker.mounts[rng.integers(len(self.walker.mounts))]
            p = self.data.site_xpos[self.model.site(f"{m.name}_foot").id].copy()
            if p[2] - self._ground_z(p) > 0.03:   # only feet near the ground kick up dust
                continue
            v = np.array([rng.normal(0, 0.15), rng.normal(0, 0.15), rng.uniform(0.1, 0.35)])
            v[:2] -= self.data.qvel[0:2] * 0.3
            self.dust.append([p, v, self.sim_time])
        for d in self.dust:
            d[1][2] -= 0.4 * h
            d[0] = d[0] + d[1] * h
        self.dust = [d for d in self.dust if self.sim_time - d[2] < 0.7][-120:]

    def _ground_z(self, p):
        gid = np.zeros(1, dtype=np.int32)
        d = mujoco.mj_ray(self.model, self.data, p + [0, 0, 0.05], np.array([0, 0, -1.0]), RAY_GROUPS, 1,
                          self.torso_id, gid)
        return p[2] + 0.05 - d if d >= 0 else 0.0

    # ------------------------------------------------------------------ BBs
    def muzzle(self):
        """(position, unit direction) of the barrel."""
        p = self.data.site_xpos[self.muzzle_id].copy()
        direction = self.data.site_xmat[self.muzzle_id].reshape(3, 3)[:, 0].copy()
        return p, direction

    def _update_gun(self, h: float, fire: bool):
        self._cooldown -= h
        if fire and self._cooldown <= 0:
            self._cooldown = self.bb_period
            p, u = self.muzzle()
            body_v = self.data.qvel[0:3]
            self.bb_pos = np.vstack([self.bb_pos, p])
            self.bb_vel = np.vstack([self.bb_vel, u * self.bb_v0 + body_v])
            self.shots += 1
        if len(self.bb_pos) == 0:
            return
        # integrate (gravity + quadratic drag) and ray-cast each BB's path this tick for hits
        speed = np.linalg.norm(self.bb_vel, axis=1, keepdims=True)
        acc = G - self.bb_k * speed * self.bb_vel
        new_vel = self.bb_vel + acc * h
        seg = (self.bb_vel + new_vel) * 0.5 * h
        keep = np.ones(len(self.bb_pos), dtype=bool)
        gid = np.zeros(1, dtype=np.int32)
        for i in range(len(self.bb_pos)):
            dist = mujoco.mj_ray(self.model, self.data, self.bb_pos[i], seg[i], RAY_GROUPS, 1, self.tilt_body, gid)
            if 0 <= dist <= 1.0:
                hit = self.bb_pos[i] + seg[i] * dist
                self.impacts.append((hit, self.sim_time))
                keep[i] = False
                # knock loose objects (anything with its own free joint that isn't the robot)
                bid = int(self.model.geom_bodyid[gid[0]])
                if bid > 0 and self.model.body_rootid[bid] != self.torso_id and self.model.body_dofnum[bid] > 0:
                    impulse = self.bb_m * self.bb_vel[i] * self.bb_impulse_scale
                    self._pending_hits.append((bid, hit, impulse))
        self.bb_pos = (self.bb_pos + seg)[keep]
        self.bb_vel = new_vel[keep]
        alive = (self.bb_pos[:, 2] > -1) & (np.linalg.norm(self.bb_pos - self.torso_pos, axis=1) < 60)
        self.bb_pos, self.bb_vel = self.bb_pos[alive], self.bb_vel[alive]
        self.impacts = self.impacts[-60:]

    def aim_point(self, max_dist=30.0):
        """Where a straight line down the barrel hits the world (laser dot), or None."""
        if not self.has_turret:
            return None
        p, u = self.muzzle()
        gid = np.zeros(1, dtype=np.int32)
        d = mujoco.mj_ray(self.model, self.data, p, u, RAY_GROUPS, 1, self.tilt_body, gid)
        return p + u * d if 0 <= d <= max_dist else None

    # ------------------------------------------------------------------ race
    def _update_race(self, c: Controls):
        if self.finish_id is None or self.run_time is not None:
            return
        if self.run_start is None and np.linalg.norm(c.move) > 0.05:
            self.run_start = self.sim_time
        if self.run_start is not None and self.in_finish():
            self.run_time = self.sim_time - self.run_start

    def in_finish(self) -> bool:
        if self.finish_id is None:
            return False
        c = self.data.site_xpos[self.finish_id]
        half = self.model.site_size[self.finish_id]
        return bool(np.all(np.abs(self.torso_pos - c) <= half))

    # ------------------------------------------------------------------ state
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
        """Tipped over, or dropped off the edge of a floating level."""
        return self.tilt_deg > 60 or self.torso_pos[2] < self.spawn_z - 0.5

    @property
    def speed(self):
        return float(np.linalg.norm(self.data.qvel[0:2]))

    @property
    def total_mass(self):
        return float(self.model.body_subtreemass[self.torso_id])

    def max_servo_load(self) -> float:
        """Peak |torque| / limit across the leg servos (1.0 = saturated)."""
        lim = self.model.actuator_forcerange[:self.n_leg, 1]
        return float(np.max(np.abs(self.data.actuator_force[:self.n_leg]) / lim))


# ---------------------------------------------------------------------- drawing helpers
def add_overlay(scene: mujoco.MjvScene, sess: SimSession, show_beam=True):
    """Draw BBs, impact marks and the aim laser into a MuJoCo scene (after update_scene)."""
    def new_geom():
        if scene.ngeom >= scene.maxgeom:
            return None
        g = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return g

    eye = np.eye(3).flatten()
    for p, _, t in sess.dust:      # skid dust: grows and fades
        age = (sess.sim_time - t) / 0.7
        g = new_geom()
        if g is None:
            break
        r = 0.01 + 0.03 * age
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]), p, eye,
                            np.array([0.62, 0.58, 0.5, 0.45 * (1 - age)], dtype=np.float32))
    if not sess.has_turret:
        return
    r = max(sess.bb_r * 1.5, 0.004)  # drawn a bit bigger than life so you can see them
    for p, v in zip(sess.bb_pos, sess.bb_vel):
        g = new_geom()
        if g is None:
            return
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]), p, eye,
                            np.array([1.0, 0.85, 0.2, 1.0], dtype=np.float32))
        g = new_geom()
        if g is None:
            return
        tail = p - v * 0.012  # short tracer streak
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye,
                            np.array([1.0, 0.75, 0.2, 0.45], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r * 0.6, tail, p)
    for p, t in sess.impacts:
        age = sess.sim_time - t
        g = new_geom()
        if g is None:
            return
        a = max(0.25, 1.0 - age / 4.0)
        s = 0.008 if age < 0.15 else 0.005
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([s, s, s]), p, eye,
                            np.array([1.0, 0.45, 0.1, a], dtype=np.float32))
    hit = sess.aim_point()
    if show_beam:
        p, u = sess.muzzle()
        end = hit if hit is not None else p + u * 8.0
        g = new_geom()
        if g is not None:
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye,
                                np.array([1.0, 0.1, 0.1, 0.25], dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0012, p, end)
    if hit is not None:  # laser dot
        g = new_geom()
        if g is not None:
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.007] * 3), hit, eye,
                                np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32))
