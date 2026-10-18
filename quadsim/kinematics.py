"""Leg inverse kinematics for a 3-DOF (coxa-femur-tibia) leg.

Frames match robot.py: leg frame +x outward, +z up, femur/tibia positive = up.
"""
from __future__ import annotations

import math

import numpy as np


def leg_ik(x: float, y: float, z: float, L1: float, L2: float, L3: float):
    """Foot target (leg frame, origin at coxa joint) -> (coxa, femur, tibia) radians.

    Unreachable targets are clamped to the nearest reachable point.
    """
    coxa = math.atan2(y, x)
    r = math.hypot(x, y) - L1
    d2 = r * r + z * z
    d = math.sqrt(d2)
    d = min(max(d, abs(L2 - L3) + 1e-4), L2 + L3 - 1e-4)
    d2 = d * d
    cos_t = (d2 - L2 * L2 - L3 * L3) / (2 * L2 * L3)
    tibia = -math.acos(max(-1.0, min(1.0, cos_t)))  # knee-up configuration
    femur = math.atan2(z, r) - math.atan2(L3 * math.sin(tibia), L2 + L3 * math.cos(tibia))
    return coxa, femur, tibia


def leg_fk(coxa: float, femur: float, tibia: float, L1: float, L2: float, L3: float):
    """Inverse of leg_ik (used by the tests)."""
    r = L1 + L2 * math.cos(femur) + L3 * math.cos(femur + tibia)
    z = L2 * math.sin(femur) + L3 * math.sin(femur + tibia)
    return np.array([r * math.cos(coxa), r * math.sin(coxa), z])


def body_to_leg(p_body, mount_x, mount_y, mount_yaw):
    """Point in the body frame -> that leg's frame."""
    dx, dy = p_body[0] - mount_x, p_body[1] - mount_y
    c, s = math.cos(-mount_yaw), math.sin(-mount_yaw)
    return c * dx - s * dy, s * dx + c * dy, p_body[2]


def leg_to_body(p_leg, mount_x, mount_y, mount_yaw):
    c, s = math.cos(mount_yaw), math.sin(mount_yaw)
    return np.array([mount_x + c * p_leg[0] - s * p_leg[1], mount_y + s * p_leg[0] + c * p_leg[1], p_leg[2]])
