"""Drive the whole app headlessly (no screen, no controller) and save screenshots.

    SDL_VIDEODRIVER=dummy MUJOCO_GL=osmesa python tools/smoke_test.py OUTDIR
"""
import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pygame
from quadsim import settings as qsettings
from quadsim.app import App

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
s = qsettings.load()
s["level"], s["mode"] = "stairs", "racecourse"
s["window"] = [960, 540]
qsettings.SETTINGS_PATH = out / "settings_test.json"  # don't touch the real settings
app = App(s, headless=True)
key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0)

app.frame([], 1 / 60); pygame.image.save(app.screen, out / "1_main.png")
app.goto("keys"); app.frame([], 1 / 60); pygame.image.save(app.screen, out / "2_keys.png")
app.goto("controller"); app.frame([], 1 / 60); pygame.image.save(app.screen, out / "3_controller.png")
app.goto("main"); app.main_menu.sel = 0; app.frame([key(pygame.K_RETURN)], 1 / 60)
assert app.state == "game", app.state
from quadsim.core import Controls
# drive by overriding the input: walk, swing the turret, fire
app.input.controls = lambda: Controls(forward=1.0, turn=0.1, pan=0.12, tilt=-0.1, fire=True)
for i in range(60 * 2):
    app.frame([], 1 / 60)
pygame.image.save(app.screen, out / "4_game_turretcam.png")
app.game_event(key(pygame.K_c))  # -> chase cam
for i in range(30):
    app.frame([], 1 / 60)
pygame.image.save(app.screen, out / "4b_game_chase.png")
app.frame([key(pygame.K_ESCAPE)], 1 / 60)
assert app.state == "pause"
pygame.image.save(app.screen, out / "5_pause.png")
print("pos", app.session.torso_pos.round(3), "state", app.state, "saved settings exist:", (out / "settings_test.json").exists())
