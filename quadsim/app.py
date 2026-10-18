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
from .core import Controls, SimSession, add_overlay
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
            for action, pos, neg in (("forward", "up", "down"), ("turn", "left", "right")):
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
        self.level_names = levels
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
            dict(label=lambda: f"Movement: {'Arcade (momentum + boost)' if self.s['arcade_feel'] else 'Direct'}",
                 left=self.toggle_feel, right=self.toggle_feel),
            dict(label=lambda: f"Stride length: x{self.s['stride_mult']:.1f}",
                 left=lambda: self.adj_stride(-0.1), right=lambda: self.adj_stride(0.1)),
            dict(label=lambda: f"Motor speed: x{self.s['speed_mult']:.2f}",
                 left=lambda: self.adj_speed(-0.25), right=lambda: self.adj_speed(0.25)),
            dict(label=lambda: f"Graphics: {qsettings.GRAPHICS[self.s['graphics']]['label']}",
                 left=lambda: self.adj_graphics(-1), right=lambda: self.adj_graphics(1)),
            dict(label=lambda: f"FPS cap: {self.s['fps_cap'] or 'uncapped'}",
                 left=lambda: self.adj_fps(-1), right=lambda: self.adj_fps(1)),
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

        def axis_label(action, text, hint):
            a = self.s["controller"]["axes"].get(action, {})
            if self.capture and self.capture[:2] == ("axis", action):
                return f"{text}  >> {hint} now (Esc cancels)"
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
            dict(label=lambda: f"Turret tilt flip: {'ON (inverted)' if self.s.get('flip_tilt') else 'off'}",
                 left=self.toggle_flip, right=self.toggle_flip),
            dict(label=lambda: f"Turret sensitivity X (pan): x{self.s['turret_sens'][0]:.1f}",
                 left=lambda: self.adj_sens(0, -0.1), right=lambda: self.adj_sens(0, 0.1)),
            dict(label=lambda: f"Turret sensitivity Y (tilt): x{self.s['turret_sens'][1]:.1f}",
                 left=lambda: self.adj_sens(1, -0.1), right=lambda: self.adj_sens(1, 0.1)),
        ]
        for action, text, hint in qsettings.AXIS_ACTIONS:
            items.append(dict(label=lambda a=action, t=text, h=hint: axis_label(a, t, h),
                              ok=lambda a=action: self.begin_capture("axis", a)))
        for action, text in qsettings.BUTTON_ACTIONS:
            items.append(dict(label=lambda a=action, t=text: button_label(a, t), ok=lambda a=action: self.begin_capture("button", a)))

        def ctrl_defaults():
            dev = self.s["controller"].get("device")
            self.s["controller"] = copy.deepcopy(qsettings.DEFAULTS["controller"])
            self.s["controller"]["device"] = dev
            self.save("Controller reset to defaults")

        items += [dict(label=lambda: "Reset controller to defaults", ok=ctrl_defaults),
                  dict(label=lambda: "Back", ok=lambda: self.goto("main"))]
        self.ctrl_menu = MenuList(items, 40, 130, 600, row_h=31)

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
        self.key_menu = MenuList(kitems, 60, 130, 620, row_h=29)

        self.pause_menu = MenuList([
            dict(label=lambda: "Resume", ok=lambda: self.goto("game")),
            dict(label=lambda: f"Movement: {'Arcade' if self.s['arcade_feel'] else 'Direct'}",
                 left=self.toggle_feel, right=self.toggle_feel),
            dict(label=lambda: f"Stride length: x{self.s['stride_mult']:.1f}",
                 left=lambda: self.adj_stride(-0.1), right=lambda: self.adj_stride(0.1)),
            dict(label=lambda: f"Motor speed: x{self.s['speed_mult']:.2f}",
                 left=lambda: self.adj_speed(-0.25), right=lambda: self.adj_speed(0.25)),
            dict(label=lambda: f"Aim sens X: x{self.s['turret_sens'][0]:.1f}",
                 left=lambda: self.adj_sens(0, -0.1), right=lambda: self.adj_sens(0, 0.1)),
            dict(label=lambda: f"Aim sens Y: x{self.s['turret_sens'][1]:.1f}",
                 left=lambda: self.adj_sens(1, -0.1), right=lambda: self.adj_sens(1, 0.1)),
            dict(label=lambda: f"Graphics: {qsettings.GRAPHICS[self.s['graphics']]['label']}",
                 left=lambda: self.adj_graphics(-1), right=lambda: self.adj_graphics(1)),
            dict(label=lambda: "Reset robot", ok=self.reset_robot),
            dict(label=lambda: "Main menu", ok=self.end_game),
        ], 0, 0, 420)

    def adj_speed(self, step):
        self.s["speed_mult"] = round(min(3.0, max(0.5, self.s["speed_mult"] + step)), 2)
        if self.session:
            self.session.set_speed_mult(self.s["speed_mult"])
        self.save(f"Motor speed x{self.s['speed_mult']:.2f}")

    def toggle_feel(self):
        self.s["arcade_feel"] = not self.s["arcade_feel"]
        if self.session:
            self.session.feel_enabled = self.s["arcade_feel"]
        self.save(f"Movement: {'Arcade' if self.s['arcade_feel'] else 'Direct'}")

    def adj_graphics(self, step):
        keys = list(qsettings.GRAPHICS)
        cur = keys.index(self.s["graphics"]) if self.s["graphics"] in keys else 0
        self.s["graphics"] = keys[(cur + step) % len(keys)]
        if self.session:
            self._make_renderer()
        self.save(f"Graphics: {qsettings.GRAPHICS[self.s['graphics']]['label']}")

    def adj_fps(self, step):
        caps = qsettings.FPS_CAPS
        cur = caps.index(self.s["fps_cap"]) if self.s["fps_cap"] in caps else 1
        self.s["fps_cap"] = caps[(cur + step) % len(caps)]
        self.save(f"FPS cap: {self.s['fps_cap'] or 'uncapped'}")

    def adj_stride(self, step):
        self.s["stride_mult"] = round(min(1.6, max(0.5, self.s["stride_mult"] + step)), 1)
        if self.session:
            self.session.set_stride_mult(self.s["stride_mult"])
        self.save(f"Stride length x{self.s['stride_mult']:.1f}")

    def adj_sens(self, axis, step):
        sens = list(self.s["turret_sens"])
        sens[axis] = round(min(3.0, max(0.1, sens[axis] + step)), 1)
        self.s["turret_sens"] = sens
        if self.session:
            self.session.set_turret_sens(*sens)
        self.save(f"Turret sensitivity {'X' if axis == 0 else 'Y'} x{sens[axis]:.1f}")

    def toggle_flip(self):
        self.s["flip_tilt"] = not self.s.get("flip_tilt", False)
        self.save(f"Turret tilt flip {'ON' if self.s['flip_tilt'] else 'off'}")

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
        self.session = SimSession(walker, self.s["level"], speed_mult=self.s["speed_mult"],
                                  turret_sens=tuple(self.s["turret_sens"]), stride_mult=self.s["stride_mult"])
        self.race = self.s["mode"] == "racecourse" and self.session.finish_id is not None
        self.session.feel_enabled = self.s["arcade_feel"]
        m = self.session.model
        self.fpv_cam_id = m.camera("fpv").id if self.session.has_turret else None
        self.fpv_base_fovy = float(m.cam_fovy[self.fpv_cam_id]) if self.fpv_cam_id is not None else 0.0
        self.rng = np.random.default_rng()
        self._make_renderer()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation = 1.1, -22
        self.cam_offset = 0.0     # azimuth offset from robot heading (mouse drag)
        # fpv = turret camera full screen + third-person picture-in-picture
        self.cam_mode = "fpv" if self.session.has_turret else "chase"
        self.cam.lookat[:] = self.session.torso_pos
        self.cam.azimuth = math.degrees(self.session.yaw)
        self.pip_cam = mujoco.MjvCamera()   # third-person chase view used for the PiP in FPV mode
        self.pip_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.pip_cam.distance, self.pip_cam.elevation = 1.0, -24
        self.pip_cam.lookat[:] = self.session.torso_pos
        self.pip_cam.azimuth = math.degrees(self.session.yaw)
        self.scene_opt = mujoco.MjvOption()
        self.last_ctrl = Controls()
        self.show_hud = getattr(self, "show_hud", True)
        self.show_perf = getattr(self, "show_perf", False)
        self.timings, self.timings_avg = {}, {}
        self.goto("game")

    def _make_renderer(self):
        g = qsettings.GRAPHICS.get(self.s.get("graphics"), qsettings.GRAPHICS["high"])
        self.gfx = g
        sw, sh = self.screen.get_size()
        m = self.session.model
        w = min(int(sw * g["render_scale"]), m.vis.global_.offwidth)
        h = min(int(sh * g["render_scale"]), m.vis.global_.offheight)
        if self.renderer:
            self.renderer.close()
        if getattr(self, "pip_renderer", None):
            self.pip_renderer.close()
        m.vis.quality.shadowsize = g["shadow_size"]  # read when the GL context is created
        m.vis.quality.offsamples = g.get("msaa", 4)    # anti-aliasing samples
        self.renderer = mujoco.Renderer(m, h, w)
        if not getattr(self, "gpu_name", None):
            self.gpu_name = self._gpu_name()
            print(f"[quadsim] OpenGL renderer: {self.gpu_name}")
        self.pip_size = (int(sw * 0.3), int(sh * 0.3))  # picture-in-picture on screen
        self.pip_renderer = mujoco.Renderer(m, max(32, int(self.pip_size[1] * g["pip_scale"])),
                                            max(32, int(self.pip_size[0] * g["pip_scale"])))
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(g["shadows"])
        self.pip_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(g["pip_shadows"])
        self.pip_surf, self.frame_no = None, 0

    def reset_robot(self):
        self.session.reset()
        self.goto("game")

    def end_game(self):
        if self.renderer:
            self.renderer.close()
        if getattr(self, "pip_renderer", None):
            self.pip_renderer.close()
        self.renderer, self.pip_renderer, self.session = None, None, None
        self.goto("main")

    def game_event(self, ev):
        if self.input.pressed(ev, "menu"):
            self.pause_menu.sel = 0
            self.goto("pause")
        elif self.input.pressed(ev, "reset"):
            self.session.reset()
        elif self.input.pressed(ev, "camera"):
            modes = (["fpv", "turret"] if self.session.has_turret else []) + ["chase", "free"]
            self.cam_mode = modes[(modes.index(self.cam_mode) + 1) % len(modes)] if self.cam_mode in modes else modes[0]
            if self.cam_mode == "chase":
                self.cam.distance, self.cam.elevation, self.cam_offset = 1.1, -22, 0.0
            self.toast = (f"Camera: {self.cam_mode}", time.time() + 1.2)
        elif self.input.pressed(ev, "perf"):
            self.show_perf = not self.show_perf
        elif self.input.pressed(ev, "hud"):
            self.show_hud = not self.show_hud
        elif self.input.pressed(ev, "flip_tilt"):
            self.toggle_flip()
        elif self.input.pressed(ev, "gait"):
            self.session.cycle_gait()
            self.toast = (f"Gait: {self.session.gait.label}", time.time() + 1.2)
        elif ev.type == pygame.MOUSEMOTION and ev.buttons[0] and self.cam_mode not in ("turret", "fpv"):
            self.cam_offset -= ev.rel[0] * 0.3
            if self.cam_mode == "free":
                self.cam.azimuth -= ev.rel[0] * 0.3
            self.cam.elevation = float(np.clip(self.cam.elevation - ev.rel[1] * 0.3, -85, -2))
        elif ev.type == pygame.MOUSEWHEEL and self.cam_mode not in ("turret", "fpv"):
            self.cam.distance = float(np.clip(self.cam.distance * (0.9 ** ev.y), 0.3, 6))
        elif ev.type == pygame.VIDEORESIZE:
            self._make_renderer()

    def game_update(self, dt):
        c = self.input.controls()
        self.last_ctrl = c
        s = self.session
        t0 = time.perf_counter()
        s.step(min(dt, 1 / 30), c)
        self.timings["physics"] = (time.perf_counter() - t0) * 1000
        lvl = s.feel.level if s.feel_enabled else 0.0
        self.pip_cam.distance = 1.0 + 0.3 * lvl
        if self.fpv_cam_id is not None:      # FOV kick while boosting
            s.model.cam_fovy[self.fpv_cam_id] = self.fpv_base_fovy + 16.0 * lvl
        self._follow(self.pip_cam, dt, 0.0)
        if self.cam_mode == "fpv":
            return
        if self.cam_mode == "turret":
            # behind the turret, turning with it (Doom-style); tilt only nudges the view
            p, u = s.muzzle()
            az = math.degrees(math.atan2(u[1], u[0]))
            el = math.degrees(math.asin(max(-1.0, min(1.0, u[2]))))
            self.cam.azimuth = az
            self.cam.elevation = float(np.clip(0.5 * el - 10, -45, 12))
            self.cam.distance = 1.0
            fwd = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
            look = s.torso_pos + fwd * 0.4
            look[2] = p[2] + 0.07   # eye line just above the barrel
            self.cam.lookat[:] = look
            return
        if self.cam_mode == "chase":
            self.cam.distance += ((1.1 + 0.35 * lvl) - self.cam.distance) * min(1.0, dt * 4)
        self._follow(self.cam, dt, self.cam_offset if self.cam_mode == "chase" else None)

    def _follow(self, cam, dt, az_offset):
        """Smoothly follow the robot; az_offset=None keeps the azimuth free (mouse orbit)."""
        s = self.session
        cam.lookat[:] += (s.torso_pos - cam.lookat) * min(1.0, dt * 8)
        if az_offset is not None:
            want = math.degrees(s.yaw) + az_offset
            diff = (want - cam.azimuth + 180) % 360 - 180
            cam.azimuth += diff * min(1.0, dt * 5)

    def _render(self, renderer, camera, beam):
        """Render the scene (plus BBs/laser) from a camera; returns (pygame surface, fovy deg)."""
        renderer.update_scene(self.session.data, camera=camera, scene_option=self.scene_opt)
        add_overlay(renderer.scene, self.session, show_beam=beam)
        img = renderer.render()
        h, w, _ = img.shape
        if isinstance(camera, str):
            fovy = float(self.session.model.cam_fovy[self.session.model.camera(camera).id])
        else:
            fovy = float(self.session.model.vis.global_.fovy)
        return pygame.image.frombuffer(img, (w, h), "RGB"), fovy  # no copy

    def draw_game(self, paused=False):
        s = self.session
        fpv = self.cam_mode == "fpv"
        main_cam = "fpv" if fpv else self.cam
        t0 = time.perf_counter()
        surf, self.main_fovy = self._render(self.renderer, main_cam, beam=self.cam_mode in ("chase", "free"))
        self.main_scene = self.renderer.scene
        if surf.get_size() != self.screen.get_size():
            surf = pygame.transform.scale(surf, self.screen.get_size())  # plain scale: ~7x cheaper than smoothscale
        lvl = s.feel.level if s.feel_enabled else 0.0
        shake = (self.rng.normal(0, 2.2 * lvl, 2) if lvl > 0.05 and not paused else (0, 0))
        if lvl > 0.05:
            self.screen.fill((0, 0, 0))
        self.screen.blit(surf, (int(shake[0]), int(shake[1])))
        if lvl > 0.05 and not paused:
            self.draw_speed_lines(lvl)
        t1 = time.perf_counter()
        sw, sh = self.screen.get_size()
        if s.has_turret:  # picture-in-picture: the other point of view (redrawn every pip_every frames)
            self.frame_no += 1
            if self.pip_surf is None or paused or self.frame_no % self.gfx["pip_every"] == 0:
                pip, _ = self._render(self.pip_renderer, self.pip_cam if fpv else "fpv", beam=fpv)
                if pip.get_size() != self.pip_size:
                    pip = pygame.transform.scale(pip, self.pip_size)
                else:
                    pip = pip.copy()  # the renderer reuses its buffer
                self.pip_surf = pip
            pw, ph = self.pip_size
            px, py = sw - pw - 14, sh - ph - 40
            pygame.draw.rect(self.screen, (10, 12, 16), (px - 3, py - 3, pw + 6, ph + 6), border_radius=6)
            self.screen.blit(self.pip_surf, (px, py))
            pygame.draw.rect(self.screen, ACCENT if fpv else TEXT, (px - 3, py - 3, pw + 6, ph + 6), 2, border_radius=6)
            tag = self.small.render("3RD PERSON" if fpv else "FPV (turret cam)", True, TEXT)
            self.screen.blit(tag, (px + 8, py + 6))
        t2 = time.perf_counter()
        if s.feel_enabled:
            self.draw_boost_meter()
        self.draw_orientation(14, sh - 40 - 150)
        self.tape_shown = s.has_turret and self.cam_mode in ("fpv", "turret")
        if self.tape_shown:
            self.draw_heading_tape()
        self.draw_hud()
        t3 = time.perf_counter()
        self.timings["render main"] = (t1 - t0) * 1000
        self.timings["render PiP"] = (t2 - t1) * 1000
        self.timings["HUD"] = (t3 - t2) * 1000
        if self.show_perf:
            self.draw_perf()
        if paused:
            shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 140))
            self.screen.blit(shade, (0, 0))
            sw, sh = self.screen.get_size()
            self.pause_menu.x, self.pause_menu.y = sw // 2 - 210, sh // 2 - 110
            self.screen.blit(self.big.render("PAUSED", True, TEXT), (sw // 2 - 80, sh // 2 - 180))
            self.pause_menu.draw(self.screen, self.font)

    def draw_hud(self):
        s = self.session
        c = self.last_ctrl
        load = s.max_servo_load()
        eb = s.walker.cfg["electronics_box"]
        lines = [
            (f"{s.walker.name}  |  {self.level_names.get(s.level_key, s.level_key)}  |  {'Racecourse' if self.race else 'Freestyle'}", TEXT),
            (f"speed {s.speed * 100:5.1f} cm/s   body h {s.gait.height * 1000:4.0f} mm   tilt {s.tilt_deg:4.1f} deg", TEXT),
            (f"gait: {s.gait.label}  stride {s.gait.max_stride * 1000:.0f} mm  motors x{s.speed_mult:.2f}  "
             f"max {s.gait.max_v:.2f} m/s  {1 / s.gait.cycle_now:.1f} cycles/s", TEXT),
            (f"cmd fwd {c.forward:+.2f} left {c.strafe:+.2f} turn {c.turn:+.2f}", DIM),
            (f"mass {s.total_mass:.2f} kg  (electronics {eb['mass_kg']:.2f} kg, {eb['size_mm'][0]}x{eb['size_mm'][1]}x{eb['size_mm'][2]} mm)", DIM),
            (f"peak servo load {load * 100:3.0f}%", BAD if load > 0.9 else DIM),
            (f"controller: {self.input.joy_name}   camera: {self.cam_mode}   fps {self.clock.get_fps():4.0f}", DIM),
        ]
        if s.has_turret:
            flip = " (flipped)" if self.s.get("flip_tilt") else ""
            lines.insert(3, (f"turret pan {math.degrees(s.pan):+4.0f}  tilt {math.degrees(s.tilt):+3.0f} deg{flip}   "
                             f"BBs fired {s.shots}  in air {len(s.bb_pos)}", TEXT))
        if not self.show_hud:  # compact: one line
            lines = [(f"{s.speed * 100:4.0f} cm/s   {s.gait.label}   BBs {s.shots}   [{self.s['keys']['hud'].upper()}] info", TEXT)]
        pad = pygame.Surface((700 if self.show_hud else 420, 22 * len(lines) + 12), pygame.SRCALPHA)
        pad.fill((10, 12, 16, 170))
        y0 = 98 if getattr(self, "tape_shown", False) else 10   # sit below the heading tape
        self.screen.blit(pad, (10, y0))
        for i, (t, col) in enumerate(lines):
            self.screen.blit(self.small.render(t, True, col), (20, y0 + 6 + 22 * i))

        sw, sh = self.screen.get_size()
        k = self.s["keys"]
        hint = (f"{k['forward']}{k['strafe_left']}{k['back']}{k['strafe_right']} move  {k['turn_left']}/{k['turn_right']} turn  "
                f"arrows turret  {k['flip_tilt']} flip tilt  {k['fire']} fire  {k['gait']} gait  {k['height_up']}/{k['height_down']} height  "
                f"{k['camera']} camera  {k['hud']} info  {k['reset']} reset  {k['menu']} menu").upper()
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
        if self.cam_mode in ("turret", "fpv"):
            # crosshair: project the laser dot into screen space
            hit = s.aim_point()
            if hit is None:
                p, u = s.muzzle()
                hit = p + u * 6.0
            if hit is not None:
                xy = self._project(hit)
                if xy:
                    col = ACCENT if c.fire else TEXT
                    x, y = xy
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        pygame.draw.line(self.screen, col, (x + 6 * dx, y + 6 * dy), (x + 14 * dx, y + 14 * dy), 2)

    def _gpu_name(self) -> str:
        """Which GPU/driver MuJoCo is actually rendering on (shown in F3)."""
        try:
            self.renderer._gl_context.make_current()
            from OpenGL import GL
            return GL.glGetString(GL.GL_RENDERER).decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            return f"unknown ({type(e).__name__})"

    def draw_speed_lines(self, lvl):
        """Streaks rushing in from the screen edges while boosting."""
        sw, sh = self.screen.get_size()
        cx, cy = sw / 2, sh / 2
        layer = pygame.Surface((sw, sh), pygame.SRCALPHA)
        for _ in range(int(10 + 30 * lvl)):
            a = self.rng.uniform(0, 2 * math.pi)
            r0 = self.rng.uniform(0.55, 0.95) * math.hypot(cx, cy)
            r1 = r0 - self.rng.uniform(40, 140) * lvl
            p0 = (cx + math.cos(a) * r0, cy + math.sin(a) * r0 * 0.75)
            p1 = (cx + math.cos(a) * r1, cy + math.sin(a) * r1 * 0.75)
            pygame.draw.line(layer, (255, 255, 255, int(40 + 90 * lvl)), p0, p1, int(self.rng.integers(1, 3)))
        self.screen.blit(layer, (0, 0))

    def draw_boost_meter(self):
        """Boost meter (bottom centre): energy, the hold-to-boost charge ring, and state."""
        f = self.session.feel
        b = f.c["boost"]
        sw, sh = self.screen.get_size()
        W, H = 280, 12
        x, y = sw // 2 - W // 2, sh - 64
        pygame.draw.rect(self.screen, (10, 12, 16), (x - 3, y - 3, W + 6, H + 6), border_radius=6)
        col = ACCENT if f.boosting else (90, 200, 230) if f.energy >= b["min_restart"] else (110, 110, 120)
        pygame.draw.rect(self.screen, col, (x, y, int(W * f.energy), H), border_radius=4)
        if f.boosting:
            label, lc = "BOOST", ACCENT
        elif f.energy < b["min_restart"]:
            label, lc = "RECHARGING", DIM
        elif f.hold > 0:
            label, lc = "HOLD FULL THROTTLE", TEXT
        else:
            label, lc = "BOOST READY" if f.energy >= 0.999 else "BOOST", DIM
        t = (self.font if f.boosting else self.small).render(label, True, lc)
        self.screen.blit(t, (sw // 2 - t.get_width() // 2, y - t.get_height() - 4))
        if not f.boosting and f.hold > 0 and f.energy >= b["min_restart"]:  # charge-up ring
            frac = min(1.0, f.hold / b["hold_s"])
            r = pygame.Rect(0, 0, 28, 28)
            r.center = (x + W + 24, y + H // 2)
            pygame.draw.arc(self.screen, ACCENT, r, math.pi / 2, math.pi / 2 + 2 * math.pi * frac, 4)
        if f.skid > 0.5:
            t = self.small.render("SKID", True, (230, 200, 120))
            self.screen.blit(t, (x - t.get_width() - 14, y - 3))

    def draw_heading_tape(self):
        """FPV heading tape, relative to the BODY: 0 = the way the legs walk forward,
        +right / -left (60, 120 on the right; -60, -120 on the left; 180 = rear).
        The centre caret is where the turret/camera is looking. Green = direction of travel."""
        s = self.session
        sw, _ = self.screen.get_size()
        W, H, top = int(sw * 0.5), 46, 10
        x0, cx = (sw - W) // 2, sw // 2
        span = 100.0                       # degrees shown either side of centre
        k = (W / 2) / span                 # px per degree
        look = -math.degrees(s.pan)        # camera bearing relative to body, right = +
        wrap = lambda a: (a + 180.0) % 360.0 - 180.0  # noqa: E731

        bg = pygame.Surface((W, H), pygame.SRCALPHA)
        bg.fill((10, 12, 16, 150))
        self.screen.blit(bg, (x0, top))
        base = top + H - 6
        clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(x0, top, W, H))
        for a in range(-180, 180, 10):     # ticks every 10, labels every 60 (+ 30 mid ticks)
            x = cx + wrap(a - look) * k
            if not (x0 - 20 <= x <= x0 + W + 20):
                continue
            major, mid = a % 60 == 0, a % 30 == 0
            tl = 14 if major else 9 if mid else 5
            col = ACCENT if a == 0 else TEXT if major else DIM
            pygame.draw.line(self.screen, col, (x, base), (x, base - tl), 3 if a == 0 else 2 if major else 1)
            if major:
                label = "FWD" if a == 0 else ("180" if a == -180 else f"{a:+d}")
                t = self.small.render(label, True, col)
                self.screen.blit(t, (x - t.get_width() / 2, top + 3))
        # direction of travel (body velocity, relative to body)
        R = s.data.xmat[s.torso_id].reshape(3, 3)
        vb = R.T @ s.data.qvel[0:3]
        if math.hypot(vb[0], vb[1]) > 0.05:
            mv = -math.degrees(math.atan2(vb[1], vb[0]))
            x = cx + wrap(mv - look) * k
            pygame.draw.polygon(self.screen, GOOD, [(x, base - 16), (x - 6, base - 26), (x + 6, base - 26)])
        self.screen.set_clip(clip)

        # centre caret + readout of where you're looking
        pygame.draw.polygon(self.screen, TEXT, [(cx, top + H + 1), (cx - 7, top + H + 11), (cx + 7, top + H + 11)])
        box = self.small.render(f"{wrap(look):+04.0f}", True, BG)
        r = pygame.Rect(0, 0, box.get_width() + 12, 20)
        r.midtop = (cx, top + H + 12)
        pygame.draw.rect(self.screen, TEXT, r, border_radius=4)
        self.screen.blit(box, (r.x + 6, r.y + 1))
        # forward is off the tape: arrow at the edge pointing back to it
        off = wrap(0 - look)
        if abs(off) > span:
            side = 1 if off > 0 else -1
            ex = x0 + W + 6 if side > 0 else x0 - 6
            pts = [(ex + 14 * side, top + H / 2), (ex, top + 8), (ex, top + H - 8)]
            pygame.draw.polygon(self.screen, ACCENT, pts)
            t = self.small.render(f"FWD {abs(off):.0f}", True, ACCENT)
            self.screen.blit(t, (ex + (18 if side > 0 else -18 - t.get_width()), top + H / 2 - 9))

    def draw_perf(self):
        """F3 overlay: where each frame's time goes (smoothed)."""
        for k, v in self.timings.items():
            self.timings_avg[k] = self.timings_avg.get(k, v) * 0.9 + v * 0.1
        total = sum(self.timings_avg.values())
        rows = [f"FPS {self.clock.get_fps():5.1f}   frame work {total:5.1f} ms   cap {self.s['fps_cap'] or 'none'}",
                f"graphics {self.gfx['label']}  ({self.renderer.width}x{self.renderer.height}, "
                f"shadows {'on' if self.gfx['shadows'] else 'off'}, PiP every {self.gfx['pip_every']})"]
        rows += [f"  {k:<12} {v:5.1f} ms" for k, v in self.timings_avg.items()]
        gpu = self.gpu_name
        rows.append(f"GPU: {gpu[:52]}")
        if any(x in gpu.lower() for x in ("intel", "gdi", "llvmpipe", "software", "basic render")):
            rows.append("! integrated/software GPU - see README 'Performance'")
        sw, _ = self.screen.get_size()
        w = 540
        pad = pygame.Surface((w, 20 * len(rows) + 12), pygame.SRCALPHA)
        pad.fill((10, 12, 16, 200))
        x0 = sw - w - 14
        self.screen.blit(pad, (x0, 90))
        for i, t in enumerate(rows):
            col = GOOD if i == 0 else BAD if t.startswith("!") else TEXT
            self.screen.blit(self.small.render(t, True, col), (x0 + 10, 96 + 20 * i))

    def draw_orientation(self, x, y):
        """Artificial horizon (pitch/roll) + heading-up mini compass with the turret direction."""
        s = self.session
        R = 62
        panel = pygame.Surface((372, 150), pygame.SRCALPHA)
        panel.fill((10, 12, 16, 175))
        self.screen.blit(panel, (x, y))
        xm = s.data.xmat[s.torso_id].reshape(3, 3)
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, xm[2, 0]))))   # nose up +
        roll = math.degrees(math.atan2(xm[2, 1], xm[2, 2]))              # right side down +
        heading = (math.degrees(s.yaw)) % 360

        # --- attitude ball
        ball = pygame.Surface((2 * R, 2 * R), pygame.SRCALPHA)
        ball.fill((120, 82, 48, 255))
        th = math.radians(roll)
        d = np.array([math.cos(th), -math.sin(th)])       # horizon direction (screen, y down)
        n = np.array([math.sin(th), math.cos(th)])        # points to the ground side
        c = np.array([R, R]) + n * pitch * 2.0            # 2 px per degree
        sky = [c - d * 3 * R, c + d * 3 * R, c + d * 3 * R - n * 3 * R, c - d * 3 * R - n * 3 * R]
        pygame.draw.polygon(ball, (70, 130, 200, 255), [tuple(p) for p in sky])
        pygame.draw.line(ball, (240, 240, 240), tuple(c - d * 3 * R), tuple(c + d * 3 * R), 2)
        for deg in (-20, -10, 10, 20):                    # pitch ladder
            q = c - n * deg * 2.0
            w = 14 if abs(deg) == 10 else 22
            pygame.draw.line(ball, (235, 235, 235), tuple(q - d * w), tuple(q + d * w), 1)
        mask = pygame.Surface((2 * R, 2 * R), pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255, 255), (R, R), R)
        ball.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        cx, cy = x + 12 + R, y + 12 + R
        self.screen.blit(ball, (cx - R, cy - R))
        pygame.draw.circle(self.screen, (200, 200, 205), (cx, cy), R, 2)
        pygame.draw.line(self.screen, ACCENT, (cx - 26, cy), (cx - 8, cy), 3)   # fixed "aircraft" marker
        pygame.draw.line(self.screen, ACCENT, (cx + 8, cy), (cx + 26, cy), 3)
        pygame.draw.circle(self.screen, ACCENT, (cx, cy), 3)

        # --- heading-up compass: body points up, turret arrow, north marker
        mx, my = x + 12 + 2 * R + 30 + 44, cy
        r2 = 44
        pygame.draw.circle(self.screen, (40, 44, 52), (mx, my), r2)
        pygame.draw.circle(self.screen, (150, 155, 165), (mx, my), r2, 1)
        nx, ny = -math.cos(s.yaw), -math.sin(s.yaw)       # world +y ("N") in heading-up screen coords
        nt = self.small.render("N", True, BAD)
        self.screen.blit(nt, (mx + nx * (r2 - 10) - nt.get_width() / 2, my + ny * (r2 - 10) - nt.get_height() / 2))
        hexpts = [(mx + 12 * math.cos(math.radians(a)), my + 12 * math.sin(math.radians(a))) for a in range(0, 360, 60)]
        pygame.draw.polygon(self.screen, (200, 200, 205), hexpts, 2)
        pygame.draw.line(self.screen, (200, 200, 205), (mx, my - 12), (mx, my - 20), 2)   # body forward tick
        if s.has_turret:
            a = s.pan                                       # +left
            tx, ty = mx - math.sin(a) * (r2 - 6), my - math.cos(a) * (r2 - 6)
            pygame.draw.line(self.screen, ACCENT, (mx, my), (tx, ty), 3)
            pygame.draw.circle(self.screen, ACCENT, (int(tx), int(ty)), 4)

        # --- numbers
        tx0 = x + 12 + 2 * R + 30 + 2 * r2 + 14
        rows = [f"HDG {heading:5.1f}", f"PIT {pitch:+5.1f}", f"ROL {roll:+5.1f}"]
        if s.has_turret:
            rows += [f"PAN {math.degrees(s.pan):+5.0f}", f"TLT {math.degrees(s.tilt):+5.0f}"]
        for i, t in enumerate(rows):
            self.screen.blit(self.small.render(t, True, ACCENT if i >= 3 else TEXT), (tx0 - 6, y + 14 + 22 * i))

    def _project(self, p):
        """World point -> screen pixel for the main view's camera (None if behind)."""
        scene = getattr(self, "main_scene", self.renderer.scene)
        cam = scene.camera[0]
        fwd = np.array(cam.forward); up = np.array(cam.up)
        pos = (np.array(scene.camera[0].pos) + np.array(scene.camera[1].pos)) / 2
        right = np.cross(fwd, up)
        v = np.asarray(p) - pos
        z = v @ fwd
        if z <= 1e-3:
            return None
        sw, sh = self.screen.get_size()
        fovy = math.radians(getattr(self, "main_fovy", self.session.model.vis.global_.fovy))
        f = (sh / 2) / math.tan(fovy / 2)
        return int(sw / 2 + f * (v @ right) / z), int(sh / 2 - f * (v @ up) / z)

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
        c = self.input.controls()
        self.screen.blit(self.font.render(f"walk:   fwd {c.forward:+.2f}  left {c.strafe:+.2f}  turn {c.turn:+.2f}",
                                          True, TEXT), (x, y))
        self.screen.blit(self.font.render(f"turret: pan {c.pan:+.2f}  tilt {c.tilt:+.2f}  fire {'YES' if c.fire else 'no'}",
                                          True, ACCENT if c.fire else TEXT), (x, y + 26))

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
            dt = self.clock.tick(self.s.get("fps_cap", 60) or 0) / 1000.0
            self.frame(pygame.event.get(), dt)
            pygame.display.flip()
        if self.renderer:
            self.renderer.close()
        if getattr(self, "pip_renderer", None):
            self.pip_renderer.close()
        pygame.quit()
