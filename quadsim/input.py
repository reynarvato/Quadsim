"""Keyboard + joystick input -> normalised walk command.

Works with anything SDL sees as a joystick: a PS4 pad over Bluetooth/USB, or an
ELRS / EdgeTX radio in USB-joystick mode (its sticks show up as axes; use the
Controller screen to capture which axis is which).
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from .core import Controls  # noqa: E402


def key_code(name: str):
    try:
        return pygame.key.key_code(name)
    except (ValueError, TypeError):
        return None


class InputManager:
    def __init__(self, settings: dict):
        self.settings = settings
        self.joy: pygame.joystick.JoystickType | None = None
        self.refresh_devices()

    # ---------------------------------------------------------------- devices
    def devices(self) -> list[str]:
        return [pygame.joystick.Joystick(i).get_name() for i in range(pygame.joystick.get_count())]

    def refresh_devices(self):
        want = self.settings["controller"].get("device")
        self.joy = None
        names = self.devices()
        if not names:
            return
        idx = names.index(want) if want in names else 0
        self.joy = pygame.joystick.Joystick(idx)
        self.joy.init()

    def handle_event(self, ev) -> None:
        if ev.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
            self.refresh_devices()

    @property
    def joy_name(self) -> str:
        return self.joy.get_name() if self.joy else "none"

    # ---------------------------------------------------------------- reading
    def axis_value(self, action: str) -> float:
        if not self.joy:
            return 0.0
        c = self.settings["controller"]
        a = c["axes"].get(action)
        if not a or a.get("axis") is None or a["axis"] >= self.joy.get_numaxes():
            return 0.0
        raw = self.joy.get_axis(a["axis"]) - a.get("center", 0.0)
        if a.get("invert"):
            raw = -raw
        dz = c.get("deadzone", 0.08)
        if abs(raw) < dz:
            return 0.0
        v = (abs(raw) - dz) / (1 - dz)
        return max(-1.0, min(1.0, v if raw > 0 else -v))

    def button_held(self, action: str) -> bool:
        if not self.joy:
            return False
        b = self.settings["controller"]["buttons"].get(action)
        return b is not None and b < self.joy.get_numbuttons() and bool(self.joy.get_button(b))

    def key_held(self, action: str) -> bool:
        k = key_code(self.settings["keys"].get(action, ""))
        return bool(k is not None and pygame.key.get_pressed()[k])

    def is_key(self, ev, action: str) -> bool:
        return ev.type == pygame.KEYDOWN and ev.key == key_code(self.settings["keys"].get(action, ""))

    def is_button(self, ev, action: str) -> bool:
        return (ev.type == pygame.JOYBUTTONDOWN and self.joy is not None
                and ev.instance_id == self.joy.get_instance_id()
                and ev.button == self.settings["controller"]["buttons"].get(action))

    def pressed(self, ev, action: str) -> bool:
        """True if this event is the key or button bound to a one-shot action."""
        return self.is_key(ev, action) or self.is_button(ev, action)

    def controls(self) -> Controls:
        """Combine joystick axes, buttons and keys into one Controls frame."""
        kh = self.key_held
        clamp = lambda v: max(-1.0, min(1.0, v))  # noqa: E731
        return Controls(
            forward=clamp(self.axis_value("forward") + kh("forward") - kh("back")),
            strafe=clamp(self.axis_value("strafe") + kh("strafe_left") - kh("strafe_right")),
            turn=clamp(self.axis_value("turn") + kh("turn_left") - kh("turn_right")),
            height=float((kh("height_up") or self.button_held("height_up"))
                         - (kh("height_down") or self.button_held("height_down"))),
            pan=clamp(self.axis_value("turret_pan") + kh("turret_left") - kh("turret_right")),
            tilt=clamp(self.axis_value("turret_tilt") + kh("turret_up") - kh("turret_down"))
            * (-1.0 if self.settings.get("flip_tilt") else 1.0),
            fire=kh("fire") or self.button_held("fire"),
        )
