"""Run the walker without graphics and report how it moves.  python tools/headless_walk.py [level]"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from quadsim.core import SimSession
from quadsim.robot import load_walker

level = sys.argv[1] if len(sys.argv) > 1 else "flat"
w = load_walker("stick_hexapod")

def run(label, cmd, secs):
    s = SimSession(w, level)
    s.step(1.0, (0, 0, 0))  # settle
    p0, y0 = s.torso_pos.copy(), s.yaw
    peak = 0
    for _ in range(int(secs / 0.02)):
        s.step(0.02, cmd); peak = max(peak, s.max_servo_load())
    dp = s.torso_pos - p0
    print(f"{label:<12} dx={dp[0]:+.3f} dy={dp[1]:+.3f} z={s.torso_pos[2]:.3f} dyaw={math.degrees(s.yaw-y0):+6.1f}deg "
          f"tilt={s.tilt_deg:4.1f} fallen={s.fallen} peak_servo_load={peak:.2f} mass={s.total_mass:.2f}kg")
    return s

run("stand", (0, 0, 0), 3)
run("forward", (1, 0, 0), 10)
run("backward", (-1, 0, 0), 5)
run("strafe-left", (0, 1, 0), 5)
run("turn-left", (0, 0, 1), 5)
