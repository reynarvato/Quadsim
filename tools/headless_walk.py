"""Run the walker without graphics and report how it moves.

    python tools/headless_walk.py [level] [gait] [walker]
"""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from quadsim.core import Controls, SimSession
from quadsim.robot import load_walker

level = sys.argv[1] if len(sys.argv) > 1 else "flat"
gait = sys.argv[2] if len(sys.argv) > 2 else None
w = load_walker(sys.argv[3] if len(sys.argv) > 3 else "stick_hexapod")


def run(label, c, secs):
    s = SimSession(w, level)
    if gait:
        s.gait_type = gait; s.reset()
    s.step(0.5, Controls())  # settle
    p0, y0 = s.torso_pos.copy(), s.yaw
    peak, max_tilt, turned, last = 0, 0, 0.0, y0
    for _ in range(int(secs / 0.02)):
        s.step(0.02, c); peak = max(peak, s.max_servo_load()); max_tilt = max(max_tilt, s.tilt_deg)
        turned += (s.yaw - last + math.pi) % (2 * math.pi) - math.pi; last = s.yaw
    dp = s.torso_pos - p0
    print(f"[{s.gait.name}] {label:<11} dx={dp[0]:+.2f} dy={dp[1]:+.2f} -> {math.hypot(dp[0], dp[1]) / secs:4.2f} m/s  "
          f"turned={math.degrees(turned):+7.1f}deg  max_tilt={max_tilt:4.1f}  fallen={s.fallen}  peak_load={peak:.2f}",
          flush=True)
    return s

run("stand", Controls(), 2)
run("forward", Controls(forward=1), 4)
run("backward", Controls(forward=-1), 3)
run("strafe-left", Controls(strafe=1), 3)
run("turn-left", Controls(turn=1), 2)
run("fwd+turn", Controls(forward=1, turn=0.6), 4)
s = run("fire", Controls(fire=True, tilt=0.3), 2)
print(f"shots={s.shots} live_bbs={len(s.bb_pos)} impacts={len(s.impacts)} pan={math.degrees(s.pan):.0f} tilt={math.degrees(s.tilt):.0f}")
