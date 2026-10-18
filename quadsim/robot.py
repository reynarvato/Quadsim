"""Walker definitions: load a walker config (JSON) and generate its MJCF.

Leg convention (used by kinematics.py too):
  - Each leg has a mount on the chassis at (x, y) with a yaw angle. In the leg
    frame, +x points outward from the body, +z is up.
  - coxa joint:  rotation about +z (positive = counter-clockwise from above)
  - femur joint: rotation about -y (positive = femur lifts UP)
  - tibia joint: rotation about -y (positive = tibia lifts UP relative to femur)
  - all-zero joint angles = leg straight out horizontally.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WALKER_DIR = ROOT / "walkers"


@dataclass
class LegMount:
    name: str
    x: float
    y: float
    yaw: float  # radians


class Walker:
    def __init__(self, cfg: dict, key: str):
        self.key = key
        self.cfg = cfg
        self.name = cfg.get("name", key)
        legs = cfg["legs"]
        self.L1 = legs["coxa_m"]
        self.L2 = legs["femur_m"]
        self.L3 = legs["tibia_m"]
        self.foot_r = legs["foot_radius_m"]
        self.mounts = [
            LegMount(m["name"], m["pos_m"][0], m["pos_m"][1], math.radians(m["yaw_deg"]))
            for m in legs["mounts"]
        ]
        self.gait = cfg["gait"]
        self.stance = cfg["stance"]

    # ------------------------------------------------------------------ MJCF
    def to_mjcf_body(self) -> str:
        """Return the <body name="torso"> XML (with freejoint) and actuator XML."""
        c = self.cfg
        hx, hy, hz = c["chassis_half_size_m"]
        eb = c["electronics_box"]
        ex, ey, ez = (v / 2000.0 for v in eb["size_mm"])  # mm -> half-size m
        ox, oy, oz = (v / 1000.0 for v in eb["offset_mm"])
        seg_m = c["legs"]["segment_mass_kg"]
        servo = c["servo"]
        plate = 0.003

        lines = []
        lines.append('<body name="torso" pos="0 0 0.3">')
        lines.append('  <freejoint name="root"/>')
        lines.append('  <camera name="track" mode="trackcom" pos="-0.9 0 0.5" xyaxes="0 -1 0 0.45 0 1"/>')
        # chassis = top + bottom plates (they carry the chassis mass)
        pm = c["chassis_mass_kg"] / 2
        for zs, nm in ((hz - plate, "top"), (-hz + plate, "bottom")):
            lines.append(
                f'  <geom name="plate_{nm}" type="box" size="{hx} {hy} {plate}" pos="0 0 {zs:.4f}" '
                f'mass="{pm}" material="chassis"/>'
            )
        # electronics bay: a payload box you can resize / re-weight in the walker JSON
        lines.append(
            f'  <geom name="electronics" type="box" size="{ex} {ey} {ez}" pos="{ox} {oy} {oz}" '
            f'mass="{eb["mass_kg"]}" material="electronics" contype="0" conaffinity="0"/>'
        )
        lines.append('  <site name="imu" pos="0 0 0" size="0.01" group="3"/>')
        # heading marker so you can see which way is forward
        lines.append(
            f'  <geom type="box" size="0.02 0.01 0.004" pos="{hx - 0.02} 0 {hz + 0.004}" '
            f'material="accent" mass="0.001" contype="0" conaffinity="0"/>'
        )

        for m in self.mounts:
            n = m.name
            qw, qz = math.cos(m.yaw / 2), math.sin(m.yaw / 2)
            lines.append(f'  <body name="{n}_coxa" pos="{m.x} {m.y} 0" quat="{qw:.6f} 0 0 {qz:.6f}">')
            lines.append(f'    <joint name="{n}_coxa" axis="0 0 1" range="-80 80"/>')
            lines.append(f'    <geom type="capsule" fromto="0 0 0 {self.L1} 0 0" size="0.008" mass="{seg_m}" material="leg"/>')
            lines.append(f'    <body name="{n}_femur" pos="{self.L1} 0 0">')
            lines.append(f'      <joint name="{n}_femur" axis="0 -1 0" range="-100 100"/>')
            lines.append(f'      <geom type="capsule" fromto="0 0 0 {self.L2} 0 0" size="0.007" mass="{seg_m}" material="leg"/>')
            lines.append(f'      <body name="{n}_tibia" pos="{self.L2} 0 0">')
            lines.append(f'        <joint name="{n}_tibia" axis="0 -1 0" range="-160 30"/>')
            lines.append(f'        <geom type="capsule" fromto="0 0 0 {self.L3 - self.foot_r} 0 0" size="0.006" mass="{seg_m}" material="leg"/>')
            lines.append(f'        <geom name="{n}_foot" type="sphere" pos="{self.L3} 0 0" size="{self.foot_r}" '
                         f'mass="0.005" material="foot" friction="1.2 0.02 0.001"/>')
            lines.append(f'        <site name="{n}_foot" pos="{self.L3} 0 0" size="0.005" group="3"/>')
            lines.append('      </body>')
            lines.append('    </body>')
            lines.append('  </body>')
        lines.append('</body>')

        acts = []
        for m in self.mounts:
            for j in ("coxa", "femur", "tibia"):
                acts.append(
                    f'<position name="{m.name}_{j}" joint="{m.name}_{j}" kp="{servo["kp"]}" '
                    f'kv="{servo["damping"]}" forcerange="-{servo["max_torque_nm"]} {servo["max_torque_nm"]}"/>'
                )
        return "\n".join(lines), "\n".join(acts)


def list_walkers() -> dict[str, str]:
    """key -> display name for every walker JSON in walkers/."""
    out = {}
    for p in sorted(WALKER_DIR.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())["name"]
        except Exception:  # noqa: BLE001 - a broken file shouldn't kill the menu
            continue
    return out


def load_walker(key: str) -> Walker:
    return Walker(json.loads((WALKER_DIR / f"{key}.json").read_text()), key)
