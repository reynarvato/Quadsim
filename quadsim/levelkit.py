"""levelkit: build Quadsim levels in Python instead of hand-writing MuJoCo XML.

Conventions (all metres / degrees):
  - size = FULL dimensions (x, y, z), not MuJoCo's half-sizes
  - pos  = (x, y, z_bottom): where the object's footprint centre sits and how high its BOTTOM is
  - yaw  = rotation about vertical, degrees (0 = long side along x)
  - static things (walls, furniture) never move; `free_*` things are loose props with a mass that
    the robot can push and BBs can knock over

    lv = Level("My room")
    lv.floor("wood")
    lv.box((1.0, 0.5, 0.4), (0, 0, 0), "sofa")          # a block on the floor
    lv.free_cyl(0.033, 0.115, (0.2, 0, 0.42), 0.015, "can")  # a can on top of something 0.42 high
    lv.spawn(0, -1.5, yaw=90)
    lv.finish((1.0, 1.5, 0.0), (0.6, 0.4, 0.3))
    lv.write("levels/my_room.xml")

Furniture helpers (table, sofa, shelf, ...) are just groups of boxes/cylinders - copy and edit them.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path


def _f(*v):
    return " ".join(f"{x:.4f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x) for x in v)


class Level:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.root = ET.Element("mujoco", {"model": name})
        self.asset = ET.SubElement(self.root, "asset")
        self.world = ET.SubElement(self.root, "worldbody")
        self._n = 0
        self._has_spawn = False

    # ------------------------------------------------------------------ materials
    def texture(self, name: str, file: str | None = None, builtin: str | None = None, **kw):
        """Image texture (path relative to levels/) or MuJoCo builtin ('checker', 'gradient', 'flat')."""
        a = {"name": name, "type": kw.pop("type", "2d")}
        if file:
            a["file"] = file
        if builtin:
            a["builtin"] = builtin
        a.update({k: (_f(*v) if isinstance(v, (tuple, list)) else str(v)) for k, v in kw.items()})
        ET.SubElement(self.asset, "texture", a)

    def material(self, name: str, rgba=(0.7, 0.7, 0.7, 1), texture: str | None = None, repeat=None, **kw):
        a = {"name": name, "rgba": _f(*rgba)}
        if texture:
            a["texture"] = texture
        if repeat:
            a["texrepeat"] = _f(*repeat)
            a["texuniform"] = "true"
        a.update({k: str(v) for k, v in kw.items()})
        ET.SubElement(self.asset, "material", a)

    def _mat(self, a: dict, look):
        """look = material name (str) or an (r, g, b[, a]) colour."""
        if isinstance(look, str):
            a["material"] = look
        else:
            rgba = tuple(look) + ((1.0,) if len(look) == 3 else ())
            a["rgba"] = _f(*rgba)

    def _name(self, prefix):
        self._n += 1
        return f"{prefix}{self._n}"

    # ------------------------------------------------------------------ static shapes
    def floor(self, material="grid", friction=None, area=None, thick=0.12, underside=(0.35, 0.33, 0.3)):
        """Infinite ground plane, or - with area=(x0, x1, y0, y1) - a finite slab whose top is at z=0
        (a floating island: walk off the edge and you fall)."""
        if area is None:
            a = {"name": "floor", "type": "plane", "size": "0 0 0.05", "material": material}
        else:
            x0, x1, y0, y1 = area
            # visible top skin + a thick plain underside, so the texture isn't stretched down the sides
            a = {"name": "floor", "type": "box", "size": _f((x1 - x0) / 2, (y1 - y0) / 2, 0.005),
                 "pos": _f((x0 + x1) / 2, (y0 + y1) / 2, -0.005), "material": material}
            self.box((x1 - x0, y1 - y0, thick - 0.01), ((x0 + x1) / 2, (y0 + y1) / 2, -thick), underside,
                     name="floor_slab")
        if friction:
            a["friction"] = _f(friction, 0.02, 0.001)
        ET.SubElement(self.world, "geom", a)

    def floating(self, background_top=(0.08, 0.09, 0.12), background_bottom=(0.01, 0.01, 0.02)):
        """Turn off the endless sky and the sun: the level floats in a dark void and brings its own lights.
        Less to draw, and shadows are sharper because the world is small."""
        custom = ET.SubElement(self.root, "custom")
        ET.SubElement(custom, "numeric", {"name": "quadsim_no_sun", "data": "1"})
        ET.SubElement(custom, "numeric", {"name": "quadsim_no_skybox", "data": "1"})
        self.texture("void_sky", builtin="gradient", type="skybox", rgb1=background_top, rgb2=background_bottom,
                     width=64, height=64)

    def ceiling(self, x0, x1, y0, y1, height, look=(0.9, 0.9, 0.88), thick=0.08):
        self.box((x1 - x0, y1 - y0, thick), ((x0 + x1) / 2, (y0 + y1) / 2, height), look, name="ceiling")

    def ceiling_light(self, x, y, height, shadows=False, brightness=0.55, cutoff=75):
        """Spotlight pointing down from the ceiling, with a small glowing fitting."""
        self.ensure_glow_materials()
        ET.SubElement(self.world, "light", {"pos": _f(x, y, height - 0.12), "dir": "0 0 -1", "cutoff": str(cutoff),
                                            "exponent": "2", "diffuse": _f(brightness, brightness * 0.95, brightness * 0.85),
                                            "specular": "0.1 0.1 0.1", "castshadow": str(shadows).lower()})
        self.cyl(0.18, 0.06, (x, y, height - 0.06), "lamp_glow", collide=False)

    def window(self, x, y, z_bottom, width, height, yaw=0.0):
        """A glowing window pane set into a wall (visual only)."""
        self.ensure_glow_materials()
        self.box((width, 0.02, height), (x, y, z_bottom), "window_glow", yaw, collide=False)

    def box(self, size, pos, look=(0.6, 0.6, 0.6), yaw=0.0, name=None, collide=True, friction=None, parent=None):
        sx, sy, sz = size
        x, y, z = pos
        a = {"name": name or self._name("box"), "type": "box", "size": _f(sx / 2, sy / 2, sz / 2),
             "pos": _f(x, y, z + sz / 2), "euler": _f(0, 0, yaw)}
        self._mat(a, look)
        if not collide:
            a.update(contype="0", conaffinity="0")
        if friction:
            a["friction"] = _f(friction, 0.02, 0.001)
        return ET.SubElement(parent if parent is not None else self.world, "geom", a)

    def cyl(self, r, h, pos, look=(0.6, 0.6, 0.6), name=None, collide=True, parent=None):
        x, y, z = pos
        a = {"name": name or self._name("cyl"), "type": "cylinder", "size": _f(r, h / 2), "pos": _f(x, y, z + h / 2)}
        self._mat(a, look)
        if not collide:
            a.update(contype="0", conaffinity="0")
        return ET.SubElement(parent if parent is not None else self.world, "geom", a)

    def sphere(self, r, pos_center, look=(0.6, 0.6, 0.6), collide=True, parent=None):
        a = {"name": self._name("sph"), "type": "sphere", "size": _f(r), "pos": _f(*pos_center)}
        self._mat(a, look)
        if not collide:
            a.update(contype="0", conaffinity="0")
        return ET.SubElement(parent if parent is not None else self.world, "geom", a)

    # ------------------------------------------------------------------ loose props
    def _free_body(self, prefix, pos_center, yaw=0.0):
        b = ET.SubElement(self.world, "body", {"name": self._name(prefix), "pos": _f(*pos_center), "euler": _f(0, 0, yaw)})
        ET.SubElement(b, "freejoint")
        return b

    def free_box(self, size, pos, mass, look=(0.8, 0.3, 0.3), yaw=0.0, name="prop"):
        sx, sy, sz = size
        b = self._free_body(name, (pos[0], pos[1], pos[2] + sz / 2), yaw)
        a = {"type": "box", "size": _f(sx / 2, sy / 2, sz / 2), "mass": _f(mass)}
        self._mat(a, look)
        ET.SubElement(b, "geom", a)
        return b

    def free_cyl(self, r, h, pos, mass, look=(0.8, 0.8, 0.85), name="prop"):
        b = self._free_body(name, (pos[0], pos[1], pos[2] + h / 2))
        a = {"type": "cylinder", "size": _f(r, h / 2), "mass": _f(mass)}
        self._mat(a, look)
        ET.SubElement(b, "geom", a)
        return b

    def free_sphere(self, r, pos, mass, look=(0.9, 0.2, 0.2), name="ball"):
        b = self._free_body(name, (pos[0], pos[1], pos[2] + r))
        a = {"type": "sphere", "size": _f(r), "mass": _f(mass)}
        self._mat(a, look)
        ET.SubElement(b, "geom", a)
        return b

    # ------------------------------------------------------------------ game sites
    def spawn(self, x, y, yaw=0.0, z=0.0):
        """Where the robot starts, and which way it faces (yaw 0 = +x)."""
        ET.SubElement(self.world, "site", {"name": "spawn", "pos": _f(x, y, z), "euler": _f(0, 0, yaw),
                                           "size": "0.01", "rgba": "0 0 0 0"})
        self._has_spawn = True

    def finish(self, pos, size):
        """Racecourse goal: a box zone (pos = footprint centre + bottom z, size = full dims)."""
        sx, sy, sz = size
        ET.SubElement(self.world, "site", {"name": "finish", "type": "box", "size": _f(sx / 2, sy / 2, sz / 2),
                                           "pos": _f(pos[0], pos[1], pos[2] + sz / 2), "rgba": "0.2 0.9 0.3 0.25"})

    def light(self, pos, diffuse=(0.4, 0.4, 0.4), castshadow=False):
        ET.SubElement(self.world, "light", {"pos": _f(*pos), "dir": "0 0 -1", "diffuse": _f(*diffuse),
                                            "castshadow": str(castshadow).lower(), "attenuation": "1 0.1 0.05"})

    # ------------------------------------------------------------------ furniture (groups of shapes)
    def walls(self, x0, x1, y0, y1, height=1.0, thick=0.1, look=(0.85, 0.83, 0.78), door=None):
        """Four walls around [x0,x1] x [y0,y1]. door = (side, centre, width) e.g. ("south", 0.0, 0.9)."""
        sides = {
            "south": ((x0 + x1) / 2, y0 - thick / 2, x1 - x0 + 2 * thick, thick, "x"),
            "north": ((x0 + x1) / 2, y1 + thick / 2, x1 - x0 + 2 * thick, thick, "x"),
            "west": (x0 - thick / 2, (y0 + y1) / 2, thick, y1 - y0, "y"),
            "east": (x1 + thick / 2, (y0 + y1) / 2, thick, y1 - y0, "y"),
        }
        for side, (cx, cy, sx, sy, axis) in sides.items():
            if door and door[0] == side:
                c, w = door[1], door[2]
                lo, hi = (cx - sx / 2, cx + sx / 2) if axis == "x" else (cy - sy / 2, cy + sy / 2)
                for a, b in ((lo, c - w / 2), (c + w / 2, hi)):
                    if b - a > 1e-3:
                        if axis == "x":
                            self.box((b - a, sy, height), ((a + b) / 2, cy, 0), look, name=self._name("wall"))
                        else:
                            self.box((sx, b - a, height), (cx, (a + b) / 2, 0), look, name=self._name("wall"))
            else:
                self.box((sx, sy, height), (cx, cy, 0), look, name=self._name("wall"))

    def table(self, x, y, length, width, height, top=0.035, leg=0.045, look="wood_dark", yaw=0.0):
        """Table: top + 4 legs. You can walk underneath if it's tall enough."""
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        self.box((length, width, top), (x, y, height - top), look, yaw)
        for dx in (-1, 1):
            for dy in (-1, 1):
                lx, ly = dx * (length / 2 - leg), dy * (width / 2 - leg)
                self.box((leg, leg, height - top), (x + c * lx - s * ly, y + s * lx + c * ly, 0), look, yaw)

    def sofa(self, x, y, length=2.0, depth=0.85, seat_h=0.42, back_h=0.82, arm_h=0.6, look="fabric", yaw=0.0):
        """Sofa whose seat faces +x before rotation (back on the -x side)."""
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

        def at(lx, ly):
            return x + c * lx - s * ly, y + s * lx + c * ly

        self.box((depth, length, seat_h - 0.1), (x, y, 0.1), look, yaw)                      # base
        for fx, fy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):                                   # feet
            px, py = at(fx * (depth / 2 - 0.06), fy * (length / 2 - 0.06))
            self.cyl(0.025, 0.1, (px, py, 0), "wood_dark")
        bx, by = at(-depth / 2 + 0.1, 0)
        self.box((0.2, length, back_h - seat_h), (bx, by, seat_h), look, yaw)              # back
        for side in (-1, 1):                                                                # arms
            ax, ay = at(0, side * (length / 2 - 0.09))
            self.box((depth, 0.18, arm_h - seat_h), (ax, ay, seat_h), look, yaw)

    def shelf(self, x, y, width, depth, height, shelves=4, look="wood", yaw=0.0):
        """Open bookcase with shelves and a few books."""
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

        def at(lx, ly):
            return x + c * lx - s * ly, y + s * lx + c * ly

        t = 0.02
        for side in (-1, 1):
            px, py = at(side * (width / 2 - t / 2), 0)
            self.box((t, depth, height), (px, py, 0), look, yaw)
        self.box((width, t, height), at(0, depth / 2 - t / 2) + (0,), look, yaw)
        colours = [(0.6, 0.15, 0.1), (0.15, 0.3, 0.55), (0.2, 0.45, 0.25), (0.75, 0.6, 0.2), (0.3, 0.2, 0.35)]
        for k in range(shelves + 1):
            z = k * (height - t) / shelves
            self.box((width - 2 * t, depth, t), at(0, 0) + (z,), look, yaw)
            if 0 < k < shelves or k == 0:
                lx = -width / 2 + 0.06
                i = 0
                while lx < width / 2 - 0.08 and k < shelves:
                    bw = 0.03 + 0.015 * ((i * 7 + k) % 3)
                    bh = 0.18 + 0.05 * ((i * 5 + k) % 3)
                    px, py = at(lx + bw / 2, 0)
                    self.box((bw, depth * 0.8, bh), (px, py, z + t), colours[(i + k) % len(colours)], yaw,
                             collide=False)
                    lx += bw + 0.004
                    i += 1

    def plant(self, x, y, pot_r=0.16, pot_h=0.32):
        self.cyl(pot_r, pot_h, (x, y, 0), (0.75, 0.42, 0.28))
        for k in range(7):  # leaves: visual only
            a = k * 2.4
            r = 0.05 + 0.06 * (k % 3)
            self.sphere(0.12 - 0.01 * (k % 3), (x + r * math.cos(a), y + r * math.sin(a), pot_h + 0.12 + 0.07 * (k % 4)),
                        (0.2, 0.5 + 0.05 * (k % 3), 0.2), collide=False)

    def ensure_glow_materials(self):
        names = {m.get("name") for m in self.asset.findall("material")}
        if "lamp_glow" not in names:
            self.material("lamp_glow", rgba=(1.0, 0.95, 0.8, 1), emission=1.0)
        if "window_glow" not in names:
            self.material("window_glow", rgba=(0.65, 0.8, 0.95, 1), emission=0.8)

    def write(self, path):
        if not self._has_spawn:
            self.spawn(0, 0)
        ET.indent(self.root, space="  ")
        header = f"<!-- {self.description} -->\n" if self.description else ""
        Path(path).write_text(header + ET.tostring(self.root, encoding="unicode") + "\n")
        n_geom = sum(1 for _ in self.root.iter("geom"))
        n_free = sum(1 for _ in self.root.iter("freejoint"))
        print(f"wrote {path}: {n_geom} geoms, {n_free} loose props")
