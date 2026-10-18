# Quadsim

Hexapod walking simulator (later: BB-gun turret). Python + MuJoCo physics, pygame for the
window, menus and controller input.

## Setup (Windows, once)

Keep the virtual environment **outside Google Drive** so Drive doesn't sync thousands of package files.

```powershell
cd "C:\Users\egyrn1\My Drive\Obsidian\PROJECTS 2026+\Q9\QuadsimRepo"
py -3.12 -m venv C:\venvs\quadsim
C:\venvs\quadsim\Scripts\activate
pip install -r requirements.txt
python main.py
```

(Any Python 3.10-3.13 works.)

## Using it

- **Main menu**: pick walker, level (Flat / Stairs to platform), mode (Freestyle / Racecourse), then Play.
- **Controller setup**: shows every connected device with live axis bars and buttons.
  Select a binding, then push the stick in the *positive* direction (forward / left / turn left)
  or press a button. Works for a PS4 pad (Bluetooth or USB) and an ELRS/EdgeTX radio in USB-joystick mode.
- **Keybinds**: select an action, press the new key.
- Everything is saved straight away to `config/settings.json` (git-ignored, per machine).
- In game: WASD move, Q/E turn, R/F body height, C chase camera, Backspace reset, Esc pause.
  Mouse drag orbits, wheel zooms.

`python main.py --viewer` opens MuJoCo's own viewer instead (joystick only), which is handy for
debugging contacts, forces and joints. `python main.py --export scene.xml` writes the combined model.

## Layout

```
main.py                 launcher (game / --viewer / --export)
quadsim/app.py          pygame window: menus, controller + keybind setup, 3D view, HUD
quadsim/core.py         SimSession: physics + gait, no graphics (used by tests too)
quadsim/robot.py        walker JSON -> MuJoCo XML (legs, servos, electronics box)
quadsim/gait.py         wave gait (one leg up at a time), body-frame foot trajectories
quadsim/kinematics.py   3-DOF leg inverse kinematics
quadsim/input.py        keyboard + joystick -> walk command
quadsim/settings.py     defaults + load/save of config/settings.json
walkers/*.json          walker definitions (geometry, masses, servo torque, gait params)
levels/*.xml            levels as plain MuJoCo XML (sites: spawn, optional finish)
tools/                  check_ik.py, headless_walk.py, smoke_test.py, make_stairs.py
```

## Tweaking

- **Electronics bay**: `walkers/stick_hexapod.json` -> `electronics_box` (size in mm, offset, mass).
  More mass -> more servo load (shown in the HUD); push it until the servos saturate.
- **Servos**: `servo.max_torque_nm` (2.4 Nm ~ a 25 kg.cm hobby servo), `kp`, `damping`.
- **Gait**: `gait.cycle_s`, `swing_height_m`, `max_stride_m`, leg `order`.
- **New level**: copy `levels/flat.xml`, add geoms. Add a `finish` box site to make it a racecourse.
- **New walker**: add a JSON in `walkers/`; it appears in the menu.

## Checks

```
python tools/check_ik.py        # IK maths + MuJoCo frames agree
python tools/headless_walk.py   # walks forward/back/strafe/turn, prints distances and servo load
python tools/headless_walk.py stairs
```

## Next steps

1. Onshape model -> `onshape-to-robot` (MuJoCo output) -> a walker that loads a mesh model instead of the stick figure.
2. Terrain-aware stepping (foot contact detection) so stairs are climbed deliberately rather than by compliance.
3. More gaits (tripod, ripple) and gait switching on a button.
4. Electronics picker: components with mass/size/power -> payload box + speed/endurance estimates.
5. Turret: yaw/pitch joints on top + BB projectiles.
