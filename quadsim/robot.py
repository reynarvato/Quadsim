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
import tempfile
import xml.etree.ElementTree as ET
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
        self.turret = cfg.get("turret", {"enabled": False})
        self.has_turret = bool(self.turret.get("enabled"))

    # ------------------------------------------------------------------ MJCF
    def to_mjcf_body(self):
        """Return (<body name="torso"> XML with freejoint, actuator XML, [asset elements])."""
        if self.cfg.get("urdf"):
            body, assets = self._urdf_body()
            return body, self._actuators_xml(), assets
        return self._generated_body(), self._actuators_xml(), []

    def _payload_xml(self, hz: float) -> list[str]:
        """Electronics bay box + IMU site + tracking camera (shared by both model sources)."""
        eb = self.cfg["electronics_box"]
        ex, ey, ez = (v / 2000.0 for v in eb["size_mm"])  # mm -> half-size m
        ox, oy, oz = (v / 1000.0 for v in eb["offset_mm"])
        return [
            # own (welded) body, so its mass counts even when the parent has explicit URDF inertia
            f'<body name="electronics_bay" pos="{ox} {oy} {oz}"><geom name="electronics" type="box" '
            f'size="{ex} {ey} {ez}" mass="{eb["mass_kg"]}" material="electronics" contype="0" conaffinity="0"/></body>',
            '<site name="imu" pos="0 0 0" size="0.01" group="3"/>',
            '<camera name="track" mode="trackcom" pos="-0.9 0 0.5" xyaxes="0 -1 0 0.45 0 1"/>',
        ]

    def _urdf_body(self):
        """Load a URDF (e.g. from onshape-to-robot), convert it with MuJoCo and adapt it for the sim."""
        import mujoco  # local import: robot.py stays importable without MuJoCo for menus/tests

        path = (WALKER_DIR / self.cfg["urdf"]).resolve()
        model = mujoco.MjModel.from_xml_path(str(path))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "converted.xml"
            mujoco.mj_saveLastXML(str(out), model)
            root = ET.parse(out).getroot()

        comp = root.find("compiler")
        radians = comp is None or comp.get("angle", "degree") == "radian"
        meshdir = Path(comp.get("meshdir", "")) if comp is not None and comp.get("meshdir") else Path()
        assets = []
        asset = root.find("asset")
        if asset is not None:
            for el in asset:
                f = el.get("file")
                if f and not Path(f).is_absolute():  # make mesh/texture paths absolute
                    el.set("file", str((path.parent / meshdir / f).resolve()))
                assets.append(el)

        base = root.find("worldbody").find("body")
        base.set("name", "torso")
        base.set("pos", "0 0 0.3")
        base.attrib.pop("quat", None)
        base.insert(0, ET.Element("freejoint", {"name": "root"}))
        for j in root.iter("joint"):
            if radians and j.get("range"):
                lo, hi = (float(v) for v in j.get("range").split())
                j.set("range", f"{math.degrees(lo):.3f} {math.degrees(hi):.3f}")
            j.attrib.pop("actuatorfrcrange", None)
        for g in root.iter("geom"):
            if g.get("contype") != "0":   # collision geoms: keep physics, hide from view
                g.set("group", "3")
        bodies = {b.get("name"): b for b in root.iter("body")}
        for el in self._payload_xml(0.0):
            base.append(ET.fromstring(el))
        for m in self.mounts:
            tib = bodies.get(f"{m.name}_tibia")
            if tib is None:
                raise ValueError(f"URDF has no link '{m.name}_tibia' (see tools/make_hexapod_a.py for naming)")
            ET.SubElement(tib, "site", {"name": f"{m.name}_foot", "pos": f"{self.L3} 0 0", "size": "0.005", "group": "3"})
        if self.has_turret and "turret_tilt" in bodies:
            bodies["turret_tilt"].append(ET.fromstring(self._fpv_camera_xml()))
        for name, spec in self.cfg.get("sites", {}).items():
            ET.SubElement(bodies[spec["body"]], "site",
                          {"name": name, "pos": " ".join(str(v) for v in spec["pos"]), "size": "0.004", "group": "3"})
        return ET.tostring(base, encoding="unicode"), assets

    def _fpv_camera_xml(self) -> str:
        """FPV camera on the turret tilt link, looking along the barrel (+x). MuJoCo cameras look down -z."""
        cam = self.turret.get("fpv_camera", {})
        # default: just above and ahead of the muzzle, so the view is a clean line of sight down the barrel
        default = [self.turret.get("barrel_mm", 120) / 1000.0 + 0.006, 0.0, 0.02]
        pos = " ".join(f"{v:.4f}" for v in cam.get("pos_m", default))
        return f'<camera name="fpv" pos="{pos}" xyaxes="0 -1 0 0 0 1" fovy="{cam.get("fovy_deg", 80)}"/>'

    def _actuators_xml(self) -> str:
        servo = self.cfg["servo"]
        acts = []
        for m in self.mounts:
            for j in ("coxa", "femur", "tibia"):
                acts.append(
                    f'<position name="{m.name}_{j}" joint="{m.name}_{j}" kp="{servo["kp"]}" '
                    f'kv="{servo["damping"]}" forcerange="-{servo["max_torque_nm"]} {servo["max_torque_nm"]}"/>'
                )
        if self.has_turret:
            t = self.turret
            for j in ("pan", "tilt"):
                acts.append(
                    f'<position name="turret_{j}" joint="turret_{j}" kp="{t["servo_kp"]}" kv="{t["servo_kp"] * 0.04:.3f}" '
                    f'forcerange="-{t["servo_max_torque_nm"]} {t["servo_max_torque_nm"]}"/>'
                )
        return "\n".join(acts)

    def _generated_body(self) -> str:
        """Stick-figure model built from the JSON numbers."""
        c = self.cfg
        hx, hy, hz = c["chassis_half_size_m"]
        seg_m = c["legs"]["segment_mass_kg"]
        plate = 0.003

        lines = []
        lines.append('<body name="torso" pos="0 0 0.3">')
        lines.append('  <freejoint name="root"/>')
        # chassis = top + bottom plates (they carry the chassis mass)
        pm = c["chassis_mass_kg"] / 2
        for zs, nm in ((hz - plate, "top"), (-hz + plate, "bottom")):
            lines.append(
                f'  <geom name="plate_{nm}" type="box" size="{hx} {hy} {plate}" pos="0 0 {zs:.4f}" '
                f'mass="{pm}" material="chassis"/>'
            )
        # electronics bay (payload box you can resize / re-weight in the walker JSON), IMU, camera
        lines.extend("  " + el for el in self._payload_xml(hz))
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
        if self.has_turret:
            lines.extend(self._turret_mjcf(hz))
        lines.append('</body>')

        return "\n".join(lines)

    def _turret_mjcf(self, top_z: float) -> list[str]:
        """2-DOF pan/tilt turret with an arrow-shaped barrel. Muzzle = site 'muzzle'."""
        t = self.turret
        mx, my = (v / 1000.0 for v in t["mount_mm"])
        L = t["barrel_mm"] / 1000.0
        m_pan, m_tilt = t["mass_kg"] * 0.55, t["mass_kg"] * 0.45
        p0, p1 = t["pan_range_deg"]
        t0, t1 = t["tilt_range_deg"]
        hl = 0.018  # arrowhead fin length
        return [
            f'  <body name="turret_pan" pos="{mx} {my} {top_z + 0.004:.4f}">',
            f'    <joint name="turret_pan" axis="0 0 1" range="{p0} {p1}" damping="0.02"/>',
            f'    <geom type="cylinder" size="0.032 0.006" pos="0 0 0.006" mass="{m_pan * 0.6:.4f}" material="turret"/>',
            f'    <geom type="box" size="0.006 0.018 0.016" pos="0 0 0.026" mass="{m_pan * 0.4:.4f}" material="turret"/>',
            '    <body name="turret_tilt" pos="0 0 0.04">',
            f'      <joint name="turret_tilt" axis="0 -1 0" range="{t0} {t1}" damping="0.02"/>',
            f'      <geom type="cylinder" size="0.012 0.022" quat="0.7071 0.7071 0 0" mass="{m_tilt * 0.5:.4f}" material="turret"/>',
            f'      <geom name="barrel" type="capsule" fromto="-0.02 0 0 {L} 0 0" size="0.0055" mass="{m_tilt * 0.5:.4f}" material="barrel"/>',
            # arrowhead: two fins swept back from the tip, so you can read the aim direction
            f'      <geom type="capsule" fromto="{L} 0 0 {L - hl} {hl * 0.6:.4f} 0" size="0.003" mass="0.001" material="accent" contype="0" conaffinity="0"/>',
            f'      <geom type="capsule" fromto="{L} 0 0 {L - hl} {-hl * 0.6:.4f} 0" size="0.003" mass="0.001" material="accent" contype="0" conaffinity="0"/>',
            f'      <geom type="capsule" fromto="{L} 0 0 {L - hl} 0 {hl * 0.6:.4f}" size="0.003" mass="0.001" material="accent" contype="0" conaffinity="0"/>',
            f'      <site name="muzzle" pos="{L + 0.01} 0 0" size="0.004" group="3"/>',
            '      ' + self._fpv_camera_xml(),
            '    </body>',
            '  </body>',
        ]


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
