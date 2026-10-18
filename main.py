"""Quadsim launcher.

    python main.py            # normal: menus + sim in one window
    python main.py --viewer   # debug: MuJoCo's own viewer (joystick only), uses saved walker/level
    python main.py --export scene.xml   # write the combined MuJoCo XML and exit
"""
import argparse
import sys
import time

from quadsim import settings as qsettings


def run_viewer():
    """Fallback / debugging mode using MuJoCo's built-in viewer (contacts, forces, joint sliders...)."""
    import mujoco.viewer
    from quadsim import input as qinput
    import pygame
    from quadsim.core import SimSession, add_overlay
    from quadsim.robot import load_walker

    s = qsettings.load()
    pygame.display.init()
    pygame.joystick.init()
    inp = qinput.InputManager(s)
    sess = SimSession(load_walker(s["walker"]), s["level"], speed_mult=s["speed_mult"],
                      turret_sens=tuple(s["turret_sens"]), stride_mult=s["stride_mult"])
    flags = {"reset": False, "gait": False}

    def on_key(keycode):  # GLFW keycodes; 259 = backspace, 71 = G
        if keycode == 259:
            flags["reset"] = True
        elif keycode == 71:
            flags["gait"] = True

    print(f"Controller: {inp.joy_name}. Backspace resets, G changes gait. Close the window to quit.")
    with mujoco.viewer.launch_passive(sess.model, sess.data, key_callback=on_key) as v:
        last = time.perf_counter()
        while v.is_running():
            now = time.perf_counter()
            dt, last = min(now - last, 1 / 30), now
            for ev in pygame.event.get():
                inp.handle_event(ev)
            if flags["reset"]:
                sess.reset()
                flags["reset"] = False
            if flags["gait"]:
                sess.cycle_gait()
                flags["gait"] = False
                print("gait:", sess.gait.label)
            with v.lock():
                sess.step(dt, inp.controls())
                v.user_scn.ngeom = 0
                add_overlay(v.user_scn, sess)
            v.sync()
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))


def main():
    ap = argparse.ArgumentParser(description="Quadsim hexapod simulator")
    ap.add_argument("--viewer", action="store_true", help="use MuJoCo's built-in viewer (debug mode)")
    ap.add_argument("--export", metavar="PATH", help="export the current walker+level as MuJoCo XML and exit")
    args = ap.parse_args()

    if args.export:
        from pathlib import Path
        from quadsim.robot import load_walker
        from quadsim.scene import export_xml
        s = qsettings.load()
        export_xml(load_walker(s["walker"]), s["level"], Path(args.export))
        print(f"wrote {args.export}")
        return
    if args.viewer:
        run_viewer()
        return

    from quadsim.app import App
    try:
        App().run()
    except Exception as e:  # noqa: BLE001
        if "GL" in type(e).__name__ or "GL" in str(e):
            print("\nOpenGL problem while creating the 3D view. Try: python main.py --viewer\n", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
