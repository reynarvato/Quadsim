# Quadsim

Hexapod walking simulator with a 2-DOF BB turret. Python + MuJoCo physics, pygame for the
window, menus and controller input.

## Setup (Windows, once)

Keep the virtual environment **outside Google Drive** so Drive doesn't sync thousands of package files.

```powershell
conda create -n quadsim python=3.12 -y
conda activate quadsim
cd "C:\Users\egyrn1\PROJECTS 2026+\Q9\QuadsimRepo"
pip install -r requirements.txt
python main.py
```

Needs Python 3.10-3.13 (MuJoCo has no ready-made package for 3.9). Next time: `conda activate quadsim`, then `python main.py`.

## Using it

- **Main menu**: pick walker, level (Flat / Stairs to platform), mode (Freestyle / Racecourse), then Play.
- **Controller setup**: shows every connected device with live axis bars and buttons.
  Select a binding, then push the stick in the *positive* direction (forward / left / turn left)
  or press a button. Works for a PS4 pad (Bluetooth or USB) and an ELRS/EdgeTX radio in USB-joystick mode.
- **Keybinds**: select an action, press the new key.
- Everything is saved straight away to `config/settings.json` (git-ignored, per machine).
- **Views**: the default is **FPV** from a camera on the turret (it looks straight down the barrel; the
  crosshair is where the BB will go), with a **3rd-person** picture-in-picture bottom-right and an
  **orientation panel** bottom-left (artificial horizon for pitch/roll, heading-up compass with the
  turret direction and north, and HDG/PIT/ROL/PAN/TLT readouts). C cycles the main view
  FPV -> behind-turret -> chase -> free; the PiP always shows the other point of view (FPV when the main
  view is 3rd person). H hides the info panel.
  **Heading tape** (top, FPV and behind-turret views): angles relative to the robot's BODY, so FWD/0 is
  always the way the legs walk; right is +, left is - (+60, +120, 180, -120, -60). The caret and number
  under it show where the camera/turret is looking; the green marker is the direction of travel; when
  forward is off the tape an orange arrow at the edge points the shortest way back to it. The FPV camera position can be set per walker in
  `turret.fpv_camera` (`pos_m` on the tilt link, `fovy_deg`).
- In game (keyboard): WASD move/strafe, Q/E turn, arrows aim the turret, V flip turret tilt, Space fire, G next gait,
  R/F body height, C cycle camera (turret / chase / free), Backspace reset, Esc pause.
- Default controller layout: axis 2 walk forward/back, axis 3 turn, axis 1 turret tilt, axis 0 turret pan,
  button 1 fire. If a direction comes out backwards, re-capture that axis on the Controller screen
  (it also records the stick's rest position, so non-centring sticks work).
- **Performance**: F3 shows FPS and where each frame's time goes (physics / main view / PiP / HUD).
  **Graphics** (main menu or pause menu): High / Medium / Low changes the 3D resolution, shadows and how
  often the picture-in-picture redraws. Rendering is almost all of the frame cost (physics is ~1-2 ms),
  so if FPS is low, drop to Medium first. **FPS cap**: 30-144 or uncapped.
  F3 also shows which GPU is drawing. If it says Intel (UHD/Iris) on a laptop that has an NVIDIA/AMD
  card, Windows is running Python on the integrated chip: Settings > System > Display > Graphics >
  add `python.exe` from your conda env (e.g. `C:\Users\<you>\anaconda3\envs\quadsim\python.exe`)
  > Options > High performance, then restart the sim. If it says "GDI Generic" or "Basic Render",
  you're on software rendering (e.g. over Remote Desktop) - run it locally or update the GPU driver.
- **Movement: Arcade / Direct** (main or pause menu). Arcade adds momentum (it builds up speed and
  coasts to a stop with "friction" when you let go), grip-limited turning (the faster you go, the wider
  you turn - kick up dust when you push past the grip) and **boost**: hold full forward for ~1.2 s and it
  spools up to ~1.5x speed (longer, quicker strides) with FOV kick, speed lines and camera shake. The
  meter (bottom centre) drains in ~3 s and recharges when you ease off. Tune it per walker with a
  `"feel"` block in the walker JSON (keys and defaults in `FEEL_DEFAULTS`, quadsim/core.py).
  Direct = the old 1:1 stick response.
- **Stride length** (main menu or pause menu, saved): x0.5-x1.6. The natural way to go faster:
  longer steps at the same step rate. Step rate also follows the stick (slow, long steps at low
  speed, quicker only near full throttle). Hexapod A: x1.0 = 140 mm, 0.83 m/s; x1.3 = 1.0 m/s.
- **Motor speed** (main menu or pause menu, saved): x0.5-x3.0. Speeds up stepping and joint speed
  (servo torque stays the same), for faster-paced play. x1 = realistic hobby servos; above ~x2 the
  servos run out of torque on the stairs.
- **Aim sensitivity X / Y** (Controller screen or pause menu, saved): scales turret pan / tilt speed.
- **Turret tilt flip** (like "invert Y"): toggle on the Controller screen, or V / a bound button in game. Saved.

`python main.py --viewer` opens MuJoCo's own viewer instead (joystick only), which is handy for
debugging contacts, forces and joints. `python main.py --export scene.xml` writes the combined model.

## Layout

```
main.py                 launcher (game / --viewer / --export)
quadsim/app.py          pygame window: menus, controller + keybind setup, 3D view, HUD
quadsim/core.py         SimSession: physics, gait, turret + BB ballistics, no graphics
quadsim/robot.py        walker JSON -> MuJoCo XML (legs, servos, electronics box)
quadsim/gait.py         phase gaits (tripod / ripple / wave), body-frame foot trajectories
quadsim/kinematics.py   3-DOF leg inverse kinematics
quadsim/input.py        keyboard + joystick -> walk command
quadsim/settings.py     defaults + load/save of config/settings.json
walkers/*.json          walker definitions (geometry, masses, servo torque, gait params)
walkers/hexapod_a.urdf  the "A" hexapod model (generated by tools/make_hexapod_a.py)
levels/*.xml            levels as plain MuJoCo XML (sites: spawn, optional finish)
tools/                  check_ik.py, headless_walk.py, smoke_test.py, make_stairs.py
```

## Tweaking

- **Electronics bay**: `walkers/stick_hexapod.json` -> `electronics_box` (size in mm, offset, mass).
  More mass -> more servo load (shown in the HUD); push it until the servos saturate.
- **Servos**: `servo.max_torque_nm` (2.4 Nm ~ a 25 kg.cm hobby servo), `kp`, `damping`.
- **Stride / cadence**: `gait.max_stride_m` (stride at full speed) and `gait.cadence_slowdown` (how much
  slower the step rate is at low speed, 1.8 = 1.8x longer cycle).
- **Gaits**: `gait.types` - tripod (fast, default), ripple, wave (stable). Each has `cycle_s`, `duty`
  (fraction of the cycle a foot is on the ground), `swing_height_m` and per-leg phase `offsets`.
  Top speed = `max_stride_m / (duty * cycle_s)`, so shorter cycles = faster and twitchier.
  `response_s` sets how quickly the robot reacts to the stick. `servo.max_speed_dps` caps joint speed
  (600 deg/s ~ a 0.1 s/60 deg hobby servo).
- **Turret**: `turret` - mount position, pan/tilt ranges and rates, barrel length, and the BB
  (diameter, mass, muzzle speed, fire rate, drag). BBs fly with gravity + air drag and stop where they hit.
- **New level**: copy `levels/flat.xml`, add geoms. Add a `finish` box site to make it a racecourse.
- **New walker**: add a JSON in `walkers/`; it appears in the menu.

## Making levels

Every `levels/*.xml` file appears in the Level menu. Three ways to make one, easiest first:

1. **Python with levelkit** (recommended) - see `tools/make_living_room.py`. You place things by real size and
   floor position (metres), with helpers for walls (with doors), tables, sofas, shelves, plants, rugs,
   lights, and loose props (`free_box/free_cyl/free_sphere` with a mass - push them or shoot them).
   Copy the script, edit, run `python tools/make_my_room.py`, pick it in the menu.
2. **Hand-edit the XML** - levels are plain MuJoCo XML (`<geom type="box" size="half-x half-y half-z" pos=...>`).
   Fine for small tweaks; `levels/flat.xml` is the minimal example.
3. **Meshes from Blender / Onshape / asset sites** - add `<mesh name="sofa" file="meshes/sofa.stl"/>` under
   `<asset>` (paths relative to `levels/`) and `<geom type="mesh" mesh="sofa" contype="0" conaffinity="0"/>` for
   looks, plus simple invisible boxes for collision (MuJoCo collides meshes as their convex hull, so a
   sofa mesh would behave like a solid lump).

**Floating levels** (`lv.floating()` + `lv.floor(..., area=(x0, x1, y0, y1))`): no endless ground, sky or sun -
the level sits on a finite slab in a dark void and brings its own lights (`lv.ceiling(...)`,
`lv.ceiling_light(...)`, `lv.window(...)`). Less to draw, sharper shadows, and walking off the edge counts as
fallen (reset to respawn). Under the hood these are `<custom><numeric name="quadsim_no_sun"/"quadsim_no_skybox">`
flags in the level XML.

Special sites: `spawn` (start position + facing, `lv.spawn(x, y, yaw)`) and optional `finish` (racecourse goal box).
Images for textures go in `levels/textures/` (`lv.texture(...)` + `lv.material(..., texture=...)`).
Scale tip: the hexapod is ~0.5 m across and ~0.25 m tall, so real furniture sizes give a "toy in a house"
feel; a coffee table at 0.42 m is high enough to walk under, and single steps up to ~5 cm are climbable.

**Living room** (`living_room`): a floating house - 5 x 4 m room with a 2.4 m ceiling, ceiling lights,
windows, and a porch outside the door that ends in a drop. Sofa, coffee table with cans, TV unit with paper cups, bookcase,
armchair, plants, a rug, a toy-block tower and a ball. Racecourse: climb the two books onto the fireplace hearth.

## Walkers and URDF

Two walkers ship with it: **Hexapod A** (default; round body, servo blocks, lattice scythe legs,
loaded from `walkers/hexapod_a.urdf`) and **Stick hexapod** (built from numbers only).

A walker JSON with `"urdf": "file.urdf"` loads that URDF through MuJoCo, so an Onshape export
(`onshape-to-robot`, URDF output, meshes included) can drop straight in. The URDF must use these names:

- joints `L1_coxa L1_femur L1_tibia ... R3_tibia` (legs L1-L3 left front-to-back, R1-R3 right);
  coxa axis +z; femur and tibia axis -y so positive lifts the leg; all joints at 0 = leg straight out
- link `<LEG>_tibia` with the foot tip on its +x axis at `tibia_m` (the straight knee-to-tip distance,
  even if the leg is curved)
- optional turret: joints `turret_pan` (+z) and `turret_tilt` (-y), link `turret_tilt` with the barrel along +x
  (put the muzzle position in the JSON under `sites.muzzle`)

The JSON still holds the leg lengths and mount positions the gait uses; they must match the URDF.
`python tools/check_ik.py <walker>` checks that they do. To change A's look or proportions, edit the
numbers at the top of `tools/make_hexapod_a.py` and run it.

## Checks

```
python tools/check_ik.py hexapod_a   # IK maths + MuJoCo frames agree for a walker
python tools/headless_walk.py   # walks forward/back/strafe/turn, prints distances and servo load
python tools/headless_walk.py stairs wave hexapod_a
```

## Next steps

1. Onshape model -> `onshape-to-robot` (URDF) -> a walker JSON pointing at it (see "Walkers and URDF").
2. Terrain-aware stepping (foot contact detection) so stairs are climbed deliberately rather than by compliance.
3. Gait auto-switching (wave on stairs, tripod on flat) and smooth transitions between gaits.
4. Electronics picker: components with mass/size/power -> payload box + speed/endurance estimates.
5. Targets to shoot (knock-over bodies, score), recoil, ammo/reload.
