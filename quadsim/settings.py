"""Persistent user settings (config/settings.json). Missing keys fall back to DEFAULTS."""
from __future__ import annotations

import copy
import json

from .robot import ROOT

SETTINGS_PATH = ROOT / "config" / "settings.json"

# Keyboard actions -> pygame key names (pygame.key.name()).
KEY_ACTIONS = [
    ("forward", "Walk forward"),
    ("back", "Walk backward"),
    ("strafe_left", "Strafe left"),
    ("strafe_right", "Strafe right"),
    ("turn_left", "Turn left"),
    ("turn_right", "Turn right"),
    ("turret_left", "Turret pan left"),
    ("turret_right", "Turret pan right"),
    ("turret_up", "Turret tilt up"),
    ("turret_down", "Turret tilt down"),
    ("flip_tilt", "Flip turret tilt"),
    ("fire", "Fire BB"),
    ("height_up", "Body up"),
    ("height_down", "Body down"),
    ("gait", "Next gait"),
    ("camera", "Cycle camera"),
    ("hud", "Show / hide info panel"),
    ("perf", "Show / hide FPS timings"),
    ("reset", "Reset robot"),
    ("menu", "Pause / menu"),
]

# Joystick analogue axes: (action, label, what to do when capturing = the POSITIVE direction).
AXIS_ACTIONS = [
    ("forward", "Walk forward / back", "push FORWARD"),
    ("turn", "Turn left / right", "push LEFT"),
    ("strafe", "Strafe left / right", "push LEFT"),
    ("turret_pan", "Turret pan", "push LEFT"),
    ("turret_tilt", "Turret tilt", "push UP"),
]
BUTTON_ACTIONS = [
    ("fire", "Fire BB"),
    ("gait", "Next gait"),
    ("flip_tilt", "Flip turret tilt"),
    ("height_up", "Body up"),
    ("height_down", "Body down"),
    ("camera", "Cycle camera"),
    ("reset", "Reset robot"),
    ("menu", "Pause / menu"),
]

# render_scale: 3D view resolution vs window. shadows: an extra render pass into a fixed-size shadow
# map (shadow_size^2 pixels, whatever the view size), so they cost the same in the small PiP as in the
# main view - pip_shadows is off below High. pip_every: redraw the picture-in-picture every N frames.
GRAPHICS = {
    "high":   {"label": "High",   "render_scale": 1.0,  "shadows": True,  "shadow_size": 2048, "msaa": 4,
               "pip_shadows": True,  "pip_every": 1, "pip_scale": 1.0},
    "medium": {"label": "Medium", "render_scale": 0.75, "shadows": True,  "shadow_size": 1024, "msaa": 2,
               "pip_shadows": False, "pip_every": 2, "pip_scale": 0.75},
    "low":    {"label": "Low",    "render_scale": 0.5,  "shadows": False, "shadow_size": 512, "msaa": 0,
               "pip_shadows": False, "pip_every": 3, "pip_scale": 0.5},
}
FPS_CAPS = [30, 60, 90, 120, 144, 0]

SETTINGS_VERSION = 2  # bump when defaults change in a way old files should not override

DEFAULTS = {
    "version": SETTINGS_VERSION,
    "walker": "hexapod_a",
    "level": "flat",
    "mode": "freestyle",  # or "racecourse"
    "window": [1280, 720],
    "flip_tilt": False,  # invert turret tilt (like "invert Y" in a shooter)
    # Motor speed multiplier: 1.0 = realistic hobby servos (0.1 s/60 deg). Higher = faster stepping
    # and faster joints (same torque), for faster-paced play. Above ~2 the stairs get hard.
    "speed_mult": 1.0,
    # Stride multiplier: longer steps at the same step rate = faster without looking sped up.
    # x1.0 = the walker's own max_stride_m; much above x1.3 the servos run out of speed/torque.
    "stride_mult": 1.0,
    # Arcade movement: momentum, coasting friction, grip-limited turning and boost (hold full forward)
    "arcade_feel": True,
    # Turret aim sensitivity, multiplies the walker's pan / tilt rates: [x (pan), y (tilt)]
    "turret_sens": [1.0, 1.0],
    # Graphics preset (see GRAPHICS below) and frame-rate cap (0 = uncapped)
    "graphics": "high",
    "fps_cap": 60,
    "keys": {
        "forward": "w", "back": "s", "strafe_left": "a", "strafe_right": "d",
        "turn_left": "q", "turn_right": "e",
        "turret_left": "left", "turret_right": "right", "turret_up": "up", "turret_down": "down",
        "flip_tilt": "v", "fire": "space", "height_up": "r", "height_down": "f", "gait": "g",
        "camera": "c", "hud": "h", "perf": "f3", "reset": "backspace", "menu": "escape",
    },
    "controller": {
        "device": None,  # joystick name; None = first one found
        "deadzone": 0.08,
        # Default layout (radio-style, AETR channel order):
        #   axis 2 = walk forward/back, axis 3 = turn, axis 1 = turret tilt, axis 0 = turret pan, button 1 = fire.
        # Positive output = forward / left / up; "invert" flips the raw axis; "center" = stick rest value.
        # If a direction is backwards, re-capture it on the Controller screen.
        "axes": {
            "forward": {"axis": 2, "invert": False, "center": 0.0},
            "turn": {"axis": 3, "invert": True, "center": 0.0},
            "strafe": {"axis": None, "invert": False, "center": 0.0},
            "turret_pan": {"axis": 0, "invert": True, "center": 0.0},
            "turret_tilt": {"axis": 1, "invert": False, "center": 0.0},
        },
        # Unbound by default: on a radio, buttons are often switches that stay "held".
        "buttons": {
            "fire": 1, "gait": None, "flip_tilt": None, "height_up": None, "height_down": None,
            "camera": None, "reset": None, "menu": None,
        },
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load() -> dict:
    if SETTINGS_PATH.exists():
        try:
            stored = json.loads(SETTINGS_PATH.read_text())
            if stored.get("version") != SETTINGS_VERSION:
                # controller layout changed: keep the user's other choices, take the new defaults
                stored.pop("controller", None)
                stored.pop("keys", None)
                stored["version"] = SETTINGS_VERSION
            return _merge(DEFAULTS, stored)
        except (json.JSONDecodeError, OSError):
            print(f"[settings] could not read {SETTINGS_PATH}, using defaults")
    return copy.deepcopy(DEFAULTS)


def save(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2))
    tmp.replace(SETTINGS_PATH)
