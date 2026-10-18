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
    ("height_up", "Body up"),
    ("height_down", "Body down"),
    ("fire", "Fire (turret, later)"),
    ("camera", "Toggle chase camera"),
    ("reset", "Reset robot"),
    ("menu", "Pause / menu"),
]

# Joystick analogue axes and buttons.
AXIS_ACTIONS = [
    ("forward", "Forward / back"),
    ("strafe", "Strafe left / right"),
    ("turn", "Turn left / right"),
]
BUTTON_ACTIONS = [
    ("height_up", "Body up"),
    ("height_down", "Body down"),
    ("fire", "Fire (turret, later)"),
    ("camera", "Toggle chase camera"),
    ("reset", "Reset robot"),
    ("menu", "Pause / menu"),
]

DEFAULTS = {
    "walker": "stick_hexapod",
    "level": "flat",
    "mode": "freestyle",  # or "racecourse"
    "window": [1280, 720],
    "keys": {
        "forward": "w", "back": "s", "strafe_left": "a", "strafe_right": "d",
        "turn_left": "q", "turn_right": "e", "height_up": "r", "height_down": "f",
        "fire": "space", "camera": "c", "reset": "backspace", "menu": "escape",
    },
    "controller": {
        "device": None,  # joystick name; None = first one found
        "deadzone": 0.08,
        # Defaults suit a PS4 pad (left stick = move, right stick X = turn).
        # Positive output = forward / left / turn-left; "invert" flips the raw axis.
        "axes": {
            "forward": {"axis": 1, "invert": True, "center": 0.0},
            "strafe": {"axis": 0, "invert": True, "center": 0.0},
            "turn": {"axis": 2, "invert": True, "center": 0.0},
        },
        "buttons": {
            "height_up": 3, "height_down": 0, "fire": 10,
            "camera": 1, "reset": 4, "menu": 6,
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
            return _merge(DEFAULTS, json.loads(SETTINGS_PATH.read_text()))
        except (json.JSONDecodeError, OSError):
            print(f"[settings] could not read {SETTINGS_PATH}, using defaults")
    return copy.deepcopy(DEFAULTS)


def save(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2))
    tmp.replace(SETTINGS_PATH)
