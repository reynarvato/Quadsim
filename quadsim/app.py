"""The game: one pygame window with menus, controller/keybind setup and the 3D sim view.

The 3D view is rendered by MuJoCo off-screen and drawn into the pygame window,
so the keyboard and joystick are handled in one place.
"""
from __future__ import annotations

import copy
import math
import time

from . import input as qinput  # sets SDL env hints before pygame import
import mujoco
import numpy as np
import pygame

from . import settings as qsettings
from .core import SimSession
from .robot import list_walkers, load_walker
from .scene import level_has_finish, list_levels

# palette
BG = (18, 20, 24)
PANEL = (30, 33, 40)
TEXT = (225, 228, 235)
DIM = (130, 136, 150)
ACCENT = (245, 150, 40)
GOOD = (90, 210, 120)
BAD = (235, 85, 85)


class MenuList:
    """Vertical list of selectable rows. Each item: dict(label=callable->str, ok=, left=, right=)."""

    def __init__(self, items, x, y, w, row_h=38):
        self.items, self.x, self.y, self.w, self.row_h = items, x, y, w, row_h
        self.sel = 0

    def rect(self, i):
        return pygame.Rect(self.x, self.y + i * self.row_h, self.w, self.row_h - 4)

    def handle(self, ev, nav):
        """nav: 'up'/'down'/'left'/'right'/'ok' derived from keys/joystick, or None."""
        if ev is not None and ev.type == pygame.MOUSEMOTION:
            for i in range(len(self.items)):
                if self.rect(i).collidepoint(ev.pos):
                    self.sel = i
        if ev is not None and ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3):
            for i, it in enumerate(self.items):
                r = self.rect(i)
                if r.collidepoint(ev.pos):
                    self.sel = i
                    has_lr = it.get("left") or it.get("right")
                    if has_lr and ev.pos[0] < r.x + 50:
                        nav = "left"
                    elif has_lr and (ev.pos[0] > r.right - 50 or ev.button == 3 or not it.get("ok")):
                        nav = "right"
                    else:
                        nav = "ok"
        if nav == "up":
            self.sel = (self.sel - 1) % len(self.items)
        elif nav == "down":
            self.sel = (self.sel + 1) % len(self.items)
        elif nav in ("left", "right", "ok"):
            fn = self.items[self.sel].get(nav) or (self.items[self.sel].get("right") if nav == "ok" else None)
            if fn:
                fn()

    def draw(self, surf, font):
        for i, it in enumerate(self.items):
            r = self.rect(i)
            on = i == self.sel
            pygame.draw.rect(surf, (48, 52, 62) if on else PANEL, r, border_radius=6)
            if on:
                pygame.draw.rect(surf, ACCENT, (r.x, r.y, 4, r.h), border_radius=2)
            label = it["label"]()
            if it.get("left") or it.get("right"):
                head, _, val = label.partition(":")
                surf.blit(font.render(head, True, TEXT), (r.x + 16, r.y + 7))
                v = font.render(f"<  {val.strip()}  >", True, ACCENT if on else DIM)
                surf.blit(v, (r.right - v.get_width() - 14, r.y + 7))
            else:
                surf.blit(font.render(label, True, TEXT if it.get("ok") else DIM), (r.x + 16, r.y + 7))


class App:
    def __init__(self, settings=None, headless=False):
        pygame.init()
        pygame.joystick.init()
        self.s = settings or qsettings.load()
        w, h = self.s["window"]
        flags = 0 if headless else pygame.RESIZABLE
        self.screen = pygame.display.set_mode((w, h), flags)
        pygame.display.set_caption("Quadsim")
        self.font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 18)
        self.small = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 15)
        self.big = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 40, bold=True)
        self.input = qinput.InputManager(self.s)
        self.clock = pygame.time.Clock()
        self.state = "main"
        self.running = True
        self.capture = None      # what we're waiting for on the setup screens
        self.toast = ("", 0.0)   # (message, expiry time)
        self.session = None
        self.renderer = None
        self._joy_nav_latch = {}
        self._build_menus()

    # ================================================================ helpers
    def save(self, msg="Saved"):
        qsettings.save(self.s)
        self.toast = (msg, time.time() + 1.8)

    def _cycle(self, key, options: dict, step):
        keys = list(options)
        cur = keys.index(self.s[key]) if self.s[key] in keys else 0
        self.s[key] = keys[(cur + step) % len(keys)]
        self.save()

    def _race_ok(self):
        return level_has_finish(self.s["level"])

    def _mode_label(self):
        if self.s["mode"] == "racecourse":
            return "Mode: Racecourse" if self._race_ok() else "Mode: Racecourse (level has no finish)"
        return "Mode: Freestyle"

    def _nav_from_event(self, ev):
        """Map arrows/enter/joystick hat/stick/button-0 to menu navigation."""
        if ev.type == pygame.KEYDOWN:
            return {pygame.K_UP: "up", pygame.K_DOWN: "down", pygame.K_LEFT: "left", pygame.K_RIGHT: "right",
                    pygame.K_RETURN: "ok", pygame.K_KP_ENTER: "ok"}.get(ev.key)
        if ev.type == pygame.JOYHATMOTION:
            x, y = ev.value
            return "up" if y > 0 else "down" if y < 0 else "left" if x < 0 else "right" if x > 0 else None
        if ev.type == pygame.JOYBUTTONDOWN and ev.button == 0 and not self.input.is_button(ev, "menu"):
            return "ok"
        if ev.type == pygame.JOYAXISMOTION:
            # stick flicks navigate menus (edge-triggered)
            for action, pos, neg in (("forward", "up", "down"), ("strafe", "left", "right")):
                v = self.input.axis_value(action)
                latched = self._joy_nav_latch.get(action, False)
                if abs(v) > 0.6 and not latched:
                    self._joy_nav_latch[action] = True
                    return pos if v > 0 else neg
                if abs(v) < 0.3:
                    self._joy_nav_latch[action] = False
        return None

    def _is_back(self, ev):
        return (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE) or self.input.is_button(ev, "menu")

    # ================================================================ menus
    def _build_menus(self):
        walkers, levels = list_walkers(), list_levels()
        W = 520
        self.main_menu = MenuList([
            dict(label=lambda: "Play", ok=self.start_game),
            dict(label=lambda: f"Walker: {walkers.get(self.s['walker'], self.s['walker'])}",
                 left=lambda: self._cycle("walker", walkers, -1), right=lambda: self._cycle("walker", walkers, 1)),
            dict(label=lambda: f"Level: {levels.get(self.s['level'], self.s['level'])}",
                 left=lambda: self._cycle("level", levels, -1), right=lambda: self._cycle("level", levels, 1)),
            dict(label=self._mode_label,
                 left=lambda: self._cycle("mode", {"freestyle": 0, "racecourse": 0}, -1),
                 right=lambda: self._cycle("mode", {"freestyle": 0, "racecourse": 0}, 1)),
            dict(label=lambda: "Controller setup", ok=lambda: self.goto("controller")),
            dict(label=lambda: "Keybinds", ok=lambda: self.goto("keys")),
            dict(label=lambda: "Quit", ok=self.quit),
        ], 60, 170, W)

        def dev_cycle(step):
            names = self.input.devices()
            if not names:
                return
            cur = names.index(self.input.joy_name) if self.input.joy_name in names else 0
            self.s["controller"]["device"] = names[(cur + step) % len(names)]
            self.input.refresh_devices()
            self.save()

        def dz(step):
            c = self.s["controller"]
            c["deadzone"] = round(min(0.4, max(0.0, c["deadzone"] + step)), 2)
            self.save()

        def axis_label(action, text):
            a = self.s["controller"]["axes"].get(action, {})
            if self.capture and self.capture[:2] == ("axis", action):
                return f"{text}  >> push it FORWARD / LEFT / LEFT now (Esc cancels)"
            if a.get("axis") is None:
                return f"{text}  [unbound]"
            return f"{text}  [axis {a['axis']}{' inv' if a.get('invert') else ''}]"

        def button_label(action, text):
            b = self.s["controller"]["buttons"].get(action)
            if self.capture and self.capture[:2] == ("button", action):
                return f"{text}  >> press a button now (Esc cancels)"
            return f"{text}  [{'unbound' if b is None else f'button {b}'}]"

        items = [
            dict(label=lambda: f"Device: {self.input.joy_name}", left=lambda: dev_cycle(-1), right=lambda: dev_cycle(1)),
            dict(label=lambda: f"Deadzone: {self.s['controller']['deadzone']:.2f}", left=lambda: dz(-0.02), right=lambda: dz(0.02)),
        ]
        for action, text in qsettings.AXIS_ACTIONS:
            items.append(dict(label=lambda a=action, t=text: axis_label(a, t), ok=lambda a=action: self.begin_capture("axis", a)))
        for action, text in qsettings.BUTTON_ACTIONS:
            items.append(dict(label=lambda a=action, t=text: button_label(a, t), ok=lambda a=action: self.begin_capture("button", a)))

        def ctrl_defaults():
            dev = self.s["controller"].get("device")
            self.s["controller"] = copy.deepcopy(qsettings.DEFAULTS["controller"])
            self.s["controller"]["device"] = dev
            self.save("Controller reset to defaults")

        items += [dict(label=lambda: "Reset controller to defaults", ok=ctrl_defaults),
                  dict(label=lambda: "Back", ok=lambda: self.goto("main"))]
        self.ctrl_menu = MenuList(items, 40, 130, 600, row_h=34)

        def key_label(action, text):
            if self.capture and self.capture[:2] == ("key", action):
                return f"{text:<24} >> press a key (Esc cancels)"
            return f"{text:<24} [{self.s['keys'].get(action, '?').upper()}]"

        def key_defaults():
            self.s["keys"] = dict(qsettings.DEFAULTS["keys"])
            self.save("Keybinds reset to defaults")

        kitems = [dict(label=lambda a=a, t=t: key_label(a, t), ok=lambda a=a: self.begin_capture("key", a))
                  for a, t in qsettings.KEY_ACTIONS]
        kitems += [dict(label=lambda: "Reset keybinds to defaults", ok=key_defaults),
                   dict(label=lambda: "Back", ok=lambda: self.goto("main"))]
        self.key_menu = MenuList(kitems, 60, 130, 620, row_h=34)

        self.pause_menu = MenuList([
            dict(label=lambda: "Resume", ok=lambda: self.goto("game")),
            dict(label=lambda: "Reset robot", ok=self.reset_robot),
            dict(label=lambda: "Main menu", ok=self.end_game),
        ], 0, 0, 360)

    def goto(self, state):
        self.capture = None
        self.state = state

    def quit(self):
        self.running = False

    # ================================================================ capture
    def begin_capture(self, kind, action):
        if kind != "key" and not self.input.joy:
            self.toast = ("No controller detected - plug it in / pair it first", time.time() + 2.5)
            return
        baseline = None
        if kind == "axis":
            j = self.input.joy
            baseline = [j.get_axis(i) for i in range(j.get_numaxes())]
        self.capture = (kind, action, baseline, time.time())

    def handle_capture(self, ev) -> bool:
        """Returns True if the event was consumed by an active capture."""
        if not self.capture:
            return False
        kind, action, baseline, t0 = self.capture
        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
            self.capture = None
            return True
        if time.time() - t0 < 0.25:  # ignore the press that started the capture
            return ev.type in (pygame.KEYDOWN, pygame.JOYBUTTONDOWN)
        if kind == "key" and ev.type == pygame.KEYDOWN:
            self.s["keys"][action] = pygame.key.name(ev.key)
            self.capture = None
            self.save(f"{action} -> {pygame.key.name(ev.key).upper()}")
            return True
        if kind == "button" and ev.type == pygame.JOYBUTTONDOWN:
            self.s["controller"]["buttons"][action] = ev.button
            self.capture = None
            self.save(f"{action} -> button {ev.button}")
            return True
        if kind == "axis" and ev.type == pygame.JOYAXISMOTION:
            j = self.input.joy
            vals = [j.get_axis(i) for i in range(j.get_numaxes())]
            dev = [v - b for v, b in zip(vals, baseline)]
            i = int(np.argmax(np.abs(dev)))
            if abs(dev[i]) > 0.5:
                # user pushed in the positive direction (forward / left / turn-left)
                self.s["controller"]["axes"][action] = {"axis": i, "invert": dev[i] < 0, "center": round(baseline[i], 3)}
                self.capture = None
                self.save(f"{action} -> axis {i}{' (inverted)' if dev[i] < 0 else ''}")
            return True
        return ev.type in (pygame.KEYDOWN, pygame.JOYBUTTONDOWN, pygame.JOYAXISMOTION)

    # ================================================================ game
    def start_game(self):
        walker = load_walker(self.s["walker"])
        self.session = SimSession(walker, self.s["level"])
        self.race = self.s["mode"] == "racecourse" and self.session.finish_id is not None
        self._make_renderer()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation = 1.1, -22
        self.cam_offset = 0.0     # azimuth offset from robot heading (mouse drag)
        self.chase = True
        self.cam.lookat[:] = self.session.torso_pos
        self.cam.azimuth = math.degrees(self.session.yaw)
        self.scene_opt = mujoco.MjvOption()
        self.fire_flash = 0.0
        self.goto("game")

    def _make_renderer(self):
        w, h = self.screen.get_size()
        m = self.session.model
        w, h = min(w, m.vis.global_.offwidth), min(h, m.vis.global_.offheight)
        if self.renderer:
            self.renderer.close()
        self.renderer = mujoco.Renderer(m, h, w)

    def reset_robot(self):
        self.session.reset()
        self.goto("game")

    def end_game(self):
        if self.renderer:
            self.renderer.close()
        self.renderer, self.session = None, None
        self.goto("main")

    def game_event(self, ev):
        if self.input.pressed(ev, "menu"):
            self.pause_menu.sel = 0
            self.goto("pause")
        elif self.input.pressed(ev, "reset"):
            self.session.reset()
        elif self.input.pressed(ev, "camera"):
            self.chase = not self.chase
            self.toast = (f"Chase camera {'on' if self.chase else 'off'}", time.time() + 1.2)
        elif self.input.pressed(ev, "fire"):
            self.fire_flash = time.time() + 0.15
        elif ev.type == pygame.MOUSEMOTION and ev.buttons[0]:
            self.cam_offset -= ev.rel[0] * 0.3
            if not self.chase:
                self.cam.azimuth -= ev.rel[0] * 0.3
            self.cam.elevation = float(np.clip(self.cam.elevation - ev.rel[1] * 0.3, -85, -2))
        elif ev.type == pygame.MOUSEWHEEL:
            self.cam.distance = float(np.clip(self.cam.distance * (0.9 ** ev.y), 0.3, 6))
        elif ev.type == pygame.VIDEORESIZE:
            self._make_renderer()

    def game_update(self, dt):
        cmd, h = self.input.command()
        self.last_cmd = cmd
        self.session.step(min(dt, 1 / 30), cmd, height_rate=0.04 * h)
        # follow camera
        target = self.session.torso_pos
        self.cam.lookat[:] += (target - self.cam.lookat) * min(1.0, dt * 6)
        if self.chase:
            want = math.degrees(self.session.yaw) + self.cam_offset
            diff = (want - self.cam.azimuth + 180) % 360 - 180
            self.cam.azimuth += diff * min(1.0, dt * 3)

    def draw_game(self, paused=False):
        self.renderer.update_scene(self.session.data, camera=self.cam, scene_option=self.scene_opt)
        img = self.renderer.render()
        h, w, _ = img.shape
        surf = pygame.image.frombuffer(img.tobytes(), (w, h), "RGB")
        if (w, h) != self.screen.get_size():
            surf = pygame.transform.smoothscale(surf, self.screen.get_size())
        self.screen.blit(surf, (0, 0))
        self.draw_hud()
        if paused:
            shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 140))
            self.screen.blit(shade, (0, 0))
            sw, sh = self.screen.get_size()
            self.pause_menu.x, self.pause_menu.y = sw // 2 - 180, sh // 2 - 60
            self.screen.blit(self.big.render("PAUSED", True, TEXT), (sw // 2 - 80, sh // 2 - 130))
            self.pause_menu.draw(self.screen, self.font)

    def draw_hud(self):
        s = self.session
        cmd = getattr(self, "last_cmd", (0, 0, 0))
        load = s.max_servo_load()
        eb = s.walker.cfg["electronics_box"]
        lines = [
            (f"{s.walker.name}  |  {list_levels().get(s.level_key, s.level_key)}  |  {'Racecourse' if self.race else 'Freestyle'}", TEXT),
            (f"speed {s.speed * 100:5.1f} cm/s   body h {s.gait.height * 1000:4.0f} mm   tilt {s.tilt_deg:4.1f} deg", TEXT),
            (f"cmd fwd {cmd[0]:+.2f} left {cmd[1]:+.2f} turn {cmd[2]:+.2f}   gait: {s.walker.gait.get('type', 'wave')}", DIM),
            (f"mass {s.total_mass:.2f} kg  (electronics {eb['mass_kg']:.2f} kg, {eb['size_mm'][0]}x{eb['size_mm'][1]}x{eb['size_mm'][2]} mm)", DIM),
            (f"peak servo load {load * 100:3.0f}%", BAD if load > 0.9 else DIM),
            (f"controller: {self.input.joy_name}   fps {self.clock.get_fps():4.0f}", DIM),
        ]
        pad = pygame.Surface((560, 22 * len(lines) + 12), pygame.SRCALPHA)
        pad.fill((10, 12, 16, 170))
        self.screen.blit(pad, (10, 10))
        for i, (t, c) in enumerate(lines):
            self.screen.blit(self.small.render(t, True, c), (20, 16 + 22 * i))

        sw, sh = self.screen.get_size()
        k = self.s["keys"]
        hint = (f"{k['forward']}{k['strafe_left']}{k['back']}{k['strafe_right']} move  {k['turn_left']}/{k['turn_right']} turn  "
                f"{k['height_up']}/{k['height_down']} height  {k['camera']} camera  {k['reset']} reset  {k['menu']} menu  "
                f"| drag = orbit, wheel = zoom").upper()
        self.screen.blit(self.small.render(hint, True, DIM), (14, sh - 26))

        if self.race:
            if s.run_time is not None:
                t, col, msg = s.run_time, GOOD, "FINISHED!"
            elif s.run_start is not None:
                t, col, msg = s.sim_time - s.run_start, TEXT, "reach the green box"
            else:
                t, col, msg = 0.0, DIM, "move to start the timer"
            ts = self.big.render(f"{t:6.2f}s", True, col)
            self.screen.blit(ts, (sw - ts.get_width() - 20, 14))
            m = self.small.render(msg, True, col)
            self.screen.blit(m, (sw - m.get_width() - 20, 62))
        if s.fallen:
            m = self.big.render("FALLEN - press reset", True, BAD)
            self.screen.blit(m, (sw // 2 - m.get_width() // 2, sh // 2 - 20))
        if time.time() < self.fire_flash:
            pygame.draw.circle(self.screen, ACCENT, (sw // 2, sh // 2), 10, 3)

    # ================================================================ screens
    def draw_title(self, title, sub=None):
        self.screen.fill(BG)
        self.screen.blit(self.big.render(title, True, TEXT), (60, 40))
        if sub:
            self.screen.blit(self.small.render(sub, True, DIM), (62, 92))

    def draw_main(self):
        self.draw_title("QUADSIM", "hexapod walking sim  |  arrows/mouse/d-pad to navigate, Enter/click to select")
        self.main_menu.draw(self.screen, self.font)
        sw, sh = self.screen.get_size()
        x = 620
        info = [
            f"Controller: {self.input.joy_name}",
            f"Settings file: config/settings.json",
            "",
            "Freestyle: walk around.",
            "Racecourse: reach the green finish box",
            "as fast as you can (needs a level with a finish).",
        ]
        for i, t in enumerate(info):
            self.screen.blit(self.small.render(t, True, DIM), (x, 180 + 24 * i))

    def draw_controller(self):
        self.draw_title("CONTROLLER", "Select a binding and move the stick / press the button. Saved automatically.")
        self.ctrl_menu.draw(self.screen, self.small)
        # live view of the raw device + the resulting command
        x, y = 680, 136
        j = self.input.joy
        if not j:
            self.screen.blit(self.font.render("No controller detected.", True, BAD), (x, y))
            tips = ["PS4: pair over Bluetooth (or USB cable).",
                    "ELRS radio: plug in USB, choose 'USB Joystick'",
                    "(HID) on the radio when asked.",
                    "It appears here automatically when connected."]
            for i, t in enumerate(tips):
                self.screen.blit(self.small.render(t, True, DIM), (x, y + 36 + 22 * i))
            return
        self.screen.blit(self.small.render(f"{j.get_name()}  ({j.get_numaxes()} axes, {j.get_numbuttons()} buttons)", True, TEXT), (x, y))
        y += 30
        for i in range(min(j.get_numaxes(), 12)):
            v = j.get_axis(i)
            self.screen.blit(self.small.render(f"axis {i:2d}", True, DIM), (x, y))
            bx = x + 80
            pygame.draw.rect(self.screen, PANEL, (bx, y + 3, 200, 12), border_radius=3)
            pygame.draw.rect(self.screen, ACCENT, (bx + 100 + min(0, v) * 100, y + 3, abs(v) * 100, 12), border_radius=3)
            pygame.draw.line(self.screen, DIM, (bx + 100, y), (bx + 100, y + 17))
            self.screen.blit(self.small.render(f"{v:+.2f}", True, DIM), (bx + 210, y))
            y += 22
        y += 6
        for i in range(min(j.get_numbuttons(), 24)):
            on = j.get_button(i)
            cx, cy = x + 14 + (i % 12) * 28, y + 12 + (i // 12) * 30
            pygame.draw.circle(self.screen, GOOD if on else PANEL, (cx, cy), 11)
            t = self.small.render(str(i), True, BG if on else DIM)
            self.screen.blit(t, (cx - t.get_width() // 2, cy - 8))
        y += 76
        (f, l, t), h = self.input.command()
        self.screen.blit(self.font.render(f"walk cmd: fwd {f:+.2f}  left {l:+.2f}  turn {t:+.2f}", True, TEXT), (x, y))

    def draw_keys(self):
        self.draw_title("KEYBINDS", "Select an action and press the new key. Saved automatically.")
        self.key_menu.draw(self.screen, self.font)

    # ================================================================ loop
    def frame(self, events, dt):
        for ev in events:
            if ev.type == pygame.QUIT:
                self.running = False
                continue
            self.input.handle_event(ev)
            if self.handle_capture(ev):
                continue
            if self.state == "game":
                self.game_event(ev)
                continue
            nav = self._nav_from_event(ev)
            if self.state == "main":
                if self._is_back(ev):
                    continue
                self.main_menu.handle(ev, nav)
            elif self.state == "controller":
                if self._is_back(ev):
                    self.goto("main")
                else:
                    self.ctrl_menu.handle(ev, nav)
            elif self.state == "keys":
                if self._is_back(ev):
                    self.goto("main")
                else:
                    self.key_menu.handle(ev, nav)
            elif self.state == "pause":
                if self._is_back(ev):
                    self.goto("game")
                else:
                    self.pause_menu.handle(ev, nav)

        if self.state == "game":
            self.game_update(dt)
            self.draw_game()
        elif self.state == "pause":
            self.draw_game(paused=True)
        elif self.state == "main":
            self.draw_main()
        elif self.state == "controller":
            self.draw_controller()
        elif self.state == "keys":
            self.draw_keys()

        msg, until = self.toast
        if msg and time.time() < until:
            t = self.font.render(msg, True, BG)
            sw, sh = self.screen.get_size()
            r = pygame.Rect(0, 0, t.get_width() + 30, 36)
            r.midbottom = (sw // 2, sh - 40)
            pygame.draw.rect(self.screen, GOOD, r, border_radius=8)
            self.screen.blit(t, (r.x + 15, r.y + 7))

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.frame(pygame.event.get(), dt)
            pygame.display.flip()
        if self.renderer:
            self.renderer.close()
        pygame.quit()
