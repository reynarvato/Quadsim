"""Generate the "A" hexapod: walkers/hexapod_a.urdf + walkers/hexapod_a.json.

Style: round two-plate body with orange standoffs and a controller board on top,
black servo blocks with orange brackets at every joint, twin-plate femurs, and
long curved lattice "scythe" tibias that end in a point. Plus a 2-DOF turret.

All geometry is primitives (box / cylinder / sphere), so no mesh files are needed.
When the real Onshape model exists, export it with onshape-to-robot (URDF) and write
a walker JSON like the one this script writes. The sim needs these names:
  joints  <LEG>_coxa / <LEG>_femur / <LEG>_tibia  (LEG = L1 L2 L3 R1 R2 R3)
          coxa axis +z, femur/tibia axis -y (positive = lift up), all-zero = leg straight out
  links   <LEG>_tibia (foot tip at +L3 on its x axis), turret_tilt (barrel along +x)
          plus turret_pan / turret_tilt joints if there is a turret.

    python tools/make_hexapod_a.py
"""
from __future__ import annotations

import json
import math
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "walkers"

# ---------------------------------------------------------------- dimensions (m)
BODY_R = 0.112          # body plate radius (150x150 mm electronics bay fits inside)
PLATE_Z = 0.034         # plate centre height above/below body centre
MOUNT_R = 0.122         # coxa joint radius
L1, L2, L3 = 0.052, 0.088, 0.165   # coxa, femur, tibia (knee -> tip chord)
TIBIA_BULGE = 0.042     # how far the scythe bows out from the knee-tip chord
TIBIA_W0, TIBIA_W1 = 0.034, 0.004  # lattice depth at knee / at tip
TILT_H = 0.085          # turret tilt axis above the pan base (high enough to fire over the knees)
LEGS = [("R1", -30), ("R2", -90), ("R3", -150), ("L1", 30), ("L2", 90), ("L3", 150)]

BLACK = "0.07 0.07 0.08 1"
SERVO = "0.12 0.12 0.13 1"
ORANGE = "0.95 0.45 0.08 1"
STEEL = "0.75 0.76 0.78 1"
WHITE = "0.92 0.92 0.92 1"
BOARD = "0.10 0.14 0.12 1"
CHIP = "0.85 0.70 0.15 1"
RED = "0.85 0.10 0.10 1"


def f(*v):
    return " ".join(f"{x:.5f}" for x in v)


def box(size, xyz=(0, 0, 0), rpy=(0, 0, 0), rgba=BLACK):
    return (f'    <visual><origin xyz="{f(*xyz)}" rpy="{f(*rpy)}"/><geometry><box size="{f(*size)}"/></geometry>'
            f'<material name=""><color rgba="{rgba}"/></material></visual>')


def cyl(r, length, xyz=(0, 0, 0), rpy=(0, 0, 0), rgba=BLACK):
    return (f'    <visual><origin xyz="{f(*xyz)}" rpy="{f(*rpy)}"/><geometry><cylinder radius="{r:.5f}" length="{length:.5f}"/></geometry>'
            f'<material name=""><color rgba="{rgba}"/></material></visual>')


def sph(r, xyz, rgba=BLACK):
    return (f'    <visual><origin xyz="{f(*xyz)}"/><geometry><sphere radius="{r:.5f}"/></geometry>'
            f'<material name=""><color rgba="{rgba}"/></material></visual>')


def bar_xz(p, q, thick, width, y=0.0, rgba=BLACK):
    """Box from p to q in the link's x-z plane (thick = z-size, width = y-size)."""
    dx, dz = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dz)
    pitch = -math.atan2(dz, dx)
    c = ((p[0] + q[0]) / 2, y, (p[1] + q[1]) / 2)
    return box((length + thick * 0.6, width, thick), c, (0, pitch, 0), rgba)


def inertial(mass, sx, sy, sz, xyz=(0, 0, 0)):
    ixx = mass * (sy * sy + sz * sz) / 12
    iyy = mass * (sx * sx + sz * sz) / 12
    izz = mass * (sx * sx + sy * sy) / 12
    return (f'    <inertial><origin xyz="{f(*xyz)}"/><mass value="{mass}"/>'
            f'<inertia ixx="{ixx:.3e}" iyy="{iyy:.3e}" izz="{izz:.3e}" ixy="0" ixz="0" iyz="0"/></inertial>')


def servo_block(xyz, horn_axis="z"):
    """Black hobby servo with a white horn and a steel screw, horn facing +z or +y."""
    out = [box((0.040, 0.020, 0.038) if horn_axis == "z" else (0.040, 0.038, 0.020), xyz, rgba=SERVO)]
    if horn_axis == "z":
        out.append(cyl(0.010, 0.004, (xyz[0], xyz[1], xyz[2] + 0.021), rgba=WHITE))
        out.append(cyl(0.003, 0.005, (xyz[0], xyz[1], xyz[2] + 0.023), rgba=STEEL))
    else:
        out.append(cyl(0.010, 0.004, (xyz[0], xyz[1] + 0.021, xyz[2]), (math.pi / 2, 0, 0), WHITE))
        out.append(cyl(0.003, 0.005, (xyz[0], xyz[1] + 0.023, xyz[2]), (math.pi / 2, 0, 0), STEEL))
    return out


# ---------------------------------------------------------------- links
def base_link():
    v = []
    for z in (PLATE_Z, -PLATE_Z):
        v.append(cyl(BODY_R, 0.004, (0, 0, z), rgba=BLACK))
    for k in range(6):  # orange standoffs between the plates
        a = math.radians(60 * k)
        v.append(cyl(0.004, 2 * PLATE_Z, (0.085 * math.cos(a), 0.085 * math.sin(a), 0), rgba=ORANGE))
    v.append(box((0.060, 0.050, 0.008), (0, 0, PLATE_Z + 0.008), rgba=BOARD))  # controller board
    v.append(box((0.020, 0.020, 0.004), (0.005, 0, PLATE_Z + 0.014), rgba=CHIP))
    v.append(box((0.004, 0.044, 0.006), (-0.022, 0, PLATE_Z + 0.015), rgba=BLACK))  # header pins
    for k, (_, yaw) in enumerate(LEGS):  # red servo leads from board to each coxa
        a = math.radians(yaw)
        r0, r1 = 0.03, 0.09
        mid = ((r0 + r1) / 2 * math.cos(a), (r0 + r1) / 2 * math.sin(a), PLATE_Z + 0.011)
        v.append(cyl(0.0018, r1 - r0, mid, (0, math.pi / 2, a), RED))
    v.append(box((0.016, 0.012, 0.006), (0.088, 0, PLATE_Z + 0.006), rgba=RED))  # front marker
    col = (f'    <collision><origin xyz="0 0 0"/><geometry><cylinder radius="{BODY_R:.4f}" length="{2 * PLATE_Z + 0.004:.4f}"/>'
           f'</geometry></collision>')
    return "\n".join(x for x in v if x) + "\n" + inertial(0.30, 0.22, 0.22, 0.07) + "\n" + col


def coxa_link():
    v = servo_block((-0.004, 0, PLATE_Z - 0.004))                  # hip servo standing on the body
    v.append(box((L1 + 0.006, 0.026, 0.004), (L1 / 2, 0, PLATE_Z + 0.019), rgba=BLACK))  # bracket top
    v.append(box((0.004, 0.026, PLATE_Z + 0.02), (L1 - 0.012, 0, PLATE_Z / 2 + 0.01), rgba=BLACK))
    v.append(box((0.006, 0.028, 0.010), (L1 - 0.012, 0, PLATE_Z + 0.012), rgba=ORANGE))  # orange clamp
    v.append(box((L1 - 0.01, 0.026, 0.004), ((L1 + 0.01) / 2, 0, -0.014), rgba=BLACK))
    return "\n".join(v) + "\n" + inertial(0.065, L1, 0.03, 0.04, (L1 / 2, 0, 0.01))


def femur_link():
    v = servo_block((0.0, -0.002, 0.0), horn_axis="y")             # femur servo, horn sideways
    for y in (0.021, -0.021):                                        # twin lattice side plates
        v.append(box((L2, 0.003, 0.016), (L2 / 2, y, 0), rgba=BLACK))
        for k in range(3):
            x = L2 * (0.25 + 0.25 * k)
            v.append(cyl(0.0035, 0.004, (x, y + math.copysign(0.002, y), 0), (math.pi / 2, 0, 0), STEEL))
    v.append(box((0.006, 0.040, 0.012), (L2 * 0.5, 0, 0), rgba=ORANGE))   # orange cross brace
    v += servo_block((L2, -0.002, 0.0), horn_axis="y")               # knee servo
    v.append(box((0.006, 0.046, 0.018), (L2 - 0.026, 0, 0), rgba=ORANGE))
    return "\n".join(v) + "\n" + inertial(0.09, L2, 0.045, 0.03, (L2 / 2, 0, 0))


def arc_points(n):
    """Points along the bowed tibia from knee (0,0) to tip (L3,0), bulging toward +z."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = L3 * t
        z = 4 * TIBIA_BULGE * t * (1 - t)            # parabola through knee and tip
        w = TIBIA_W0 + (TIBIA_W1 - TIBIA_W0) * t ** 0.8
        pts.append((x, z, w))
    return pts


def tibia_link():
    v = [box((0.026, 0.030, 0.026), (0.0, 0, 0), rgba=BLACK),    # knee bracket
         box((0.008, 0.034, 0.028), (0.012, 0, 0), rgba=ORANGE),
         cyl(0.004, 0.036, (0, 0, 0), (math.pi / 2, 0, 0), STEEL)]
    pts = arc_points(10)
    outer, inner = [], []
    for i, (x, z, w) in enumerate(pts):
        # normal to the arc in x-z, so the rails sit on either side of the centre line
        if i < len(pts) - 1:
            dx, dz = pts[i + 1][0] - x, pts[i + 1][1] - z
        else:
            dx, dz = x - pts[i - 1][0], z - pts[i - 1][1]
        n = math.hypot(dx, dz)
        nx, nz = -dz / n, dx / n
        outer.append((x + nx * w / 2, z + nz * w / 2))
        inner.append((x - nx * w / 2, z - nz * w / 2))
    for i in range(len(pts) - 1):  # two rails + zig-zag struts = lattice
        v.append(bar_xz(outer[i], outer[i + 1], 0.005, 0.010))
        v.append(bar_xz(inner[i], inner[i + 1], 0.005, 0.010))
        a, b = (outer[i], inner[i + 1]) if i % 2 == 0 else (inner[i], outer[i + 1])
        if i < len(pts) - 2:
            v.append(bar_xz(a, b, 0.003, 0.008))
    v.append(sph(0.005, (L3, 0, 0)))  # tip
    col = f'    <collision><origin xyz="{L3:.4f} 0 0"/><geometry><sphere radius="0.007"/></geometry></collision>'
    return "\n".join(v) + "\n" + inertial(0.045, L3, 0.01, 0.03, (L3 / 2, 0, TIBIA_BULGE / 2)) + "\n" + col


def turret_links():
    barrel = 0.12
    pan = [cyl(0.034, 0.008, (0, 0, 0.004), rgba=BLACK),
           cyl(0.024, 0.003, (0, 0, 0.0095), rgba=ORANGE)]
    pan.append(cyl(0.012, TILT_H - 0.03, (0, 0, (TILT_H - 0.03) / 2 + 0.008), rgba=BLACK))  # mast
    pan += servo_block((0, 0, TILT_H - 0.03), horn_axis="y")
    for y in (0.026, -0.026):  # yoke
        pan.append(box((0.03, 0.004, 0.05), (0, y, TILT_H - 0.012), rgba=ORANGE))
    tilt = [cyl(0.012, 0.046, (0, 0, 0), (math.pi / 2, 0, 0), BLACK),
            box((0.05, 0.03, 0.026), (0.01, 0, 0), rgba=SERVO),
            cyl(0.0065, barrel + 0.02, (barrel / 2 - 0.01, 0, 0), (0, math.pi / 2, 0), BLACK),
            cyl(0.0085, 0.02, (barrel - 0.01, 0, 0), (0, math.pi / 2, 0), ORANGE)]
    hl = 0.02  # arrowhead fins
    ang = math.atan(0.6)
    for dy, dz in ((1, 0), (-1, 0), (0, 1)):  # fins swept back from the tip
        rpy = (0, ang * dz, -ang * dy)
        tilt.append(box((math.hypot(hl, hl * 0.6), 0.004, 0.004),
                        (barrel - hl / 2, dy * hl * 0.3, dz * hl * 0.3), rpy, RED))
    return ("\n".join(pan) + "\n" + inertial(0.08, 0.06, 0.06, 0.05, (0, 0, 0.02)),
            "\n".join(tilt) + "\n" + inertial(0.07, 0.12, 0.03, 0.03, (0.04, 0, 0)), barrel)


# ---------------------------------------------------------------- assemble
def joint(name, parent, child, xyz, rpy, axis, lo, hi, effort=3.0):
    return (f'  <joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{f(*xyz)}" rpy="{f(*rpy)}"/><axis xyz="{axis}"/>'
            f'<limit lower="{math.radians(lo):.4f}" upper="{math.radians(hi):.4f}" effort="{effort}" velocity="12"/></joint>')


def build():
    parts = ['<?xml version="1.0"?>',
             '<!-- Generated by tools/make_hexapod_a.py - edit the script, not this file. -->',
             '<robot name="hexapod_a">',
             '  <mujoco><compiler discardvisual="false" fusestatic="false" strippath="false"/></mujoco>',
             f'  <link name="base_link">\n{base_link()}\n  </link>']
    for name, yaw in LEGS:
        a = math.radians(yaw)
        parts.append(f'  <link name="{name}_coxa">\n{coxa_link()}\n  </link>')
        parts.append(f'  <link name="{name}_femur">\n{femur_link()}\n  </link>')
        parts.append(f'  <link name="{name}_tibia">\n{tibia_link()}\n  </link>')
        parts.append(joint(f"{name}_coxa", "base_link", f"{name}_coxa",
                           (MOUNT_R * math.cos(a), MOUNT_R * math.sin(a), 0), (0, 0, a), "0 0 1", -80, 80))
        parts.append(joint(f"{name}_femur", f"{name}_coxa", f"{name}_femur", (L1, 0, 0), (0, 0, 0), "0 -1 0", -100, 100))
        parts.append(joint(f"{name}_tibia", f"{name}_femur", f"{name}_tibia", (L2, 0, 0), (0, 0, 0), "0 -1 0", -160, 30))
    pan_v, tilt_v, barrel = turret_links()
    parts.append(f'  <link name="turret_pan">\n{pan_v}\n  </link>')
    parts.append(f'  <link name="turret_tilt">\n{tilt_v}\n  </link>')
    parts.append(joint("turret_pan", "base_link", "turret_pan", (-0.045, 0, PLATE_Z + 0.004), (0, 0, 0), "0 0 1", -170, 170, 1.5))
    parts.append(joint("turret_tilt", "turret_pan", "turret_tilt", (0, 0, TILT_H), (0, 0, 0), "0 -1 0", -20, 60, 1.5))
    parts.append('</robot>')
    (OUT / "hexapod_a.urdf").write_text("\n".join(parts) + "\n")

    base = json.loads((OUT / "stick_hexapod.json").read_text())
    cfg = {
        "name": "Hexapod A",
        "description": "Round-body hexapod with lattice scythe legs, loaded from hexapod_a.urdf.",
        "urdf": "hexapod_a.urdf",
        "chassis_half_size_m": [BODY_R, BODY_R, PLATE_Z + 0.002],
        "electronics_box": base["electronics_box"],
        "legs": {
            "coxa_m": L1, "femur_m": L2, "tibia_m": L3, "foot_radius_m": 0.007,
            "mounts": [{"name": n, "pos_m": [round(MOUNT_R * math.cos(math.radians(y)), 5),
                                             round(MOUNT_R * math.sin(math.radians(y)), 5)], "yaw_deg": y}
                       for n, y in LEGS],
        },
        "servo": base["servo"],
        "stance": {"foot_reach_m": 0.105, "body_height_m": 0.12},
        "gait": base["gait"],
        "turret": dict(base["turret"], mount_mm=[-45, 0], barrel_mm=round(barrel * 1000)),
        "sites": {"muzzle": {"body": "turret_tilt", "pos": [barrel + 0.012, 0, 0]}},
    }
    cfg["gait"]["max_stride_m"] = 0.14   # long legs -> longer natural stride than the stick walker
    cfg["gait"]["types"]["tripod"]["offsets"] = {"L1": 0.0, "R2": 0.0, "L3": 0.0, "R1": 0.5, "L2": 0.5, "R3": 0.5}
    (OUT / "hexapod_a.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("wrote", OUT / "hexapod_a.urdf", "and", OUT / "hexapod_a.json")


if __name__ == "__main__":
    build()
