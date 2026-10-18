"""Combine a walker and a level into one MuJoCo model."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from .robot import ROOT, Walker

LEVEL_DIR = ROOT / "levels"

SCENE_TEMPLATE = """
<mujoco model="quadsim">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="2048"/>
  </visual>
  <default>
    <geom condim="3" friction="1.0 0.02 0.001" solref="0.004 1"/>
    <joint damping="0.05" armature="0.002"/>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.45 0.6 0.8" rgb2="0.1 0.12 0.18" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.24 0.26 0.29" rgb2="0.30 0.32 0.35" width="512" height="512" mark="edge" markrgb="0.4 0.42 0.45"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true"/>
    <material name="stairs" rgba="0.55 0.5 0.42 1"/>
    <material name="platform" rgba="0.45 0.42 0.36 1"/>
    <material name="chassis" rgba="0.85 0.86 0.88 0.55"/>
    <material name="electronics" rgba="0.15 0.55 0.95 0.55"/>
    <material name="leg" rgba="0.92 0.55 0.15 1"/>
    <material name="foot" rgba="0.1 0.1 0.1 1"/>
    <material name="accent" rgba="0.95 0.2 0.2 1"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 3" dir="0.3 0.2 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
  </worldbody>
  <actuator/>
  <sensor/>
</mujoco>
"""


def list_levels() -> dict[str, str]:
    out = {}
    for p in sorted(LEVEL_DIR.glob("*.xml")):
        try:
            out[p.stem] = ET.parse(p).getroot().get("model", p.stem)
        except ET.ParseError:
            continue
    return out


def level_has_finish(level_key: str) -> bool:
    root = ET.parse(LEVEL_DIR / f"{level_key}.xml").getroot()
    return any(s.get("name") == "finish" for s in root.iter("site"))


def build_model(walker: Walker, level_key: str) -> mujoco.MjModel:
    scene = ET.fromstring(SCENE_TEMPLATE)
    world = scene.find("worldbody")
    level = ET.parse(LEVEL_DIR / f"{level_key}.xml").getroot()

    for section in ("asset", "default"):
        src = level.find(section)
        if src is not None:
            dst = scene.find(section)
            for child in src:
                dst.append(child)
    lw = level.find("worldbody")
    if lw is not None:
        for child in lw:
            world.append(child)

    body_xml, act_xml = walker.to_mjcf_body()
    world.append(ET.fromstring(body_xml))
    act = scene.find("actuator")
    for el in ET.fromstring(f"<a>{act_xml}</a>"):
        act.append(el)
    sens = scene.find("sensor")
    for tag, attrs in (("framequat", {"name": "body_quat", "objtype": "site", "objname": "imu"}),
                       ("gyro", {"name": "body_gyro", "site": "imu"}),
                       ("accelerometer", {"name": "body_acc", "site": "imu"})):
        ET.SubElement(sens, tag, attrs)

    xml = ET.tostring(scene, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)


def export_xml(walker: Walker, level_key: str, path: Path) -> None:
    """Save the combined scene (handy for opening in MuJoCo's own viewer: python -m mujoco.viewer)."""
    model = build_model(walker, level_key)
    mujoco.mj_saveLastXML(str(path), model)
