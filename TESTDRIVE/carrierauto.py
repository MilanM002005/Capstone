#!/usr/bin/env python3
"""
carrier_auto.py  -  Demonstrate-and-replay timed maneuver (robust build).

You show three states; it identifies the throttle axis from the FORWARD step and
the steering axis from the RIGHT step (separately, so they can't collide), then
builds CLEAN decoupled commands:

  forward drive = throttle pushed, steering locked to your STOP centre (straight)
  right drive   = that forward + your demonstrated steering
  stop          = your STOP snapshot

Replays:  FORWARD (T1) -> RIGHT (T_TURN) -> FORWARD (T2) -> STOP.
Independent of RCMAP / SERVO settings (SERVO1=26, SERVO3=70 stays).

SAFETY: wheels OFF the ground for the confirm; TX in hand; ready to cut power.
Ctrl-C / dropped SSH -> safe stop.
"""

import sys, time, signal
from pymavlink import mavutil

CONNECTION  = '/dev/ttyACM0'
BAUD        = 115200
T_FORWARD_1 = 15.0
T_TURN      = 3.0
T_FORWARD_2 = 10.0
RATE_HZ     = 10
CONFIRM_S   = 1.5
MIN_MOVE    = 70          # a demo push must move a channel at least this much
STICK_CHANS = (1, 2, 3, 4)

master = None
ACTIVE = []
STOP_FRAME = None


def read_frame(t=2.0):
    m = master.recv_match(type=['RC_CHANNELS', 'RC_CHANNELS_RAW'], blocking=True, timeout=t)
    return None if m is None else [getattr(m, 'chan%d_raw' % i, 0) for i in range(1, 9)]


def average_frame(sec=0.8):
    end = time.time()+sec; acc=[0]*8; n=0
    while time.time() < end:
        f = read_frame()
        if f:
            acc=[a+c for a,c in zip(acc,f)]; n+=1
    return None if n==0 else [round(a/n) for a in acc]


def biggest_mover(sample, base, exclude=()):
    best, bestdev = None, -1
    for c in STICK_CHANS:
        if c in exclude:
            continue
        dev = abs(sample[c-1]-base[c-1])
        if dev > bestdev:
            bestdev, best = dev, c
    return best, bestdev


def send_frame(frame):
    ch = [65535]*8
    for c in ACTIVE:
        ch[c-1] = int(frame[c-1])
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component, *ch)


def release():
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component, *([0]*8))


def hold(sec, frame):
    end = time.time()+sec
    while time.time() < end:
        send_frame(frame); time.sleep(1.0/RATE_HZ)


def set_mode(m):
    master.mav.set_mode_send(master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, master.mode_mapping()[m])


def arm():
    master.mav.command_long_send(master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    master.motors_armed_wait(); print("ARMED")


def disarm():
    master.mav.command_long_send(master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)


def cleanup(*_):
    print("\nSTOP -> neutral, release, disarm")
    try:
        if STOP_FRAME:
            for _ in range(5):
                send_frame(STOP_FRAME); time.sleep(0.05)
        release(); disarm()
    finally:
        sys.exit(0)


def capture_axis(prompt, stop_f, exclude=()):
    """Prompt, capture, ensure a real movement; return (channel, frame)."""
    while True:
        input(prompt)
        f = average_frame(0.7)
        ch, dev = biggest_mover(f, stop_f, exclude)
        if dev >= MIN_MOVE:
            print("   -> CH%d moved %d" % (ch, dev))
            return ch, f
        print("   (only moved %d - too small. Push the stick FURTHER and hold.)" % dev)


def main():
    global master, ACTIVE, STOP_FRAME
    master = mavutil.mavlink_connection(CONNECTION, baud=BAUD)
    print("Waiting for heartbeat..."); master.wait_heartbeat()
    print("Heartbeat: system %d, component %d" % (master.target_system, master.target_component))
    master.mav.request_data_stream_send(master.target_system, master.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 20, 1)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGHUP, cleanup)

    print("\n=== DEMONSTRATE (TX ON, wheels OFF the ground) ===")
    print("Tip: let the steering axis spring to centre; push clearly, not gently.")
    input("1) STOP: throttle at stop, steering centred. Hold, ENTER...")
    stop_f = average_frame(1.0)
    if stop_f is None:
        print("No RC input."); sys.exit(1)
    print("   STOP CH1-4 =", stop_f[:4])

    thr_ch, fwd_f = capture_axis(
        "2) CLEAR FORWARD (wheels spinning), hold. ENTER...", stop_f)
    str_ch, turn_f = capture_axis(
        "3) Strong RIGHT TURN, hold. ENTER...", stop_f, exclude=(thr_ch,))

    # clean decoupled commands
    fwd_drive = list(stop_f);  fwd_drive[thr_ch-1] = fwd_f[thr_ch-1]
    turn_drive = list(fwd_drive); turn_drive[str_ch-1] = turn_f[str_ch-1]
    ACTIVE = sorted({thr_ch, str_ch})
    STOP_FRAME = stop_f

    print("\nThrottle = CH%d, Steering = CH%d" % (thr_ch, str_ch))
    print("  FORWARD ->", {c: fwd_drive[c-1] for c in ACTIVE})
    print("  RIGHT   ->", {c: turn_drive[c-1] for c in ACTIVE})

    if input("\n%.1fs FORWARD confirm drive? [y/N] " % CONFIRM_S).strip().lower() != 'y':
        print("Aborted."); sys.exit(0)
    set_mode('MANUAL'); time.sleep(1); arm()
    print("CONFIRM forward %.1fs" % CONFIRM_S)
    hold(CONFIRM_S, fwd_drive); hold(0.4, stop_f)
    if input("Straight FORWARD? [y/N] ").strip().lower() != 'y':
        print("Still curving -> tell me the printed values and I'll add a steering bias.")
        cleanup()

    print("\nForward %.0f s" % T_FORWARD_1); hold(T_FORWARD_1, fwd_drive)
    print("Turn right %.1f s" % T_TURN);    hold(T_TURN, turn_drive)
    print("Forward %.0f s" % T_FORWARD_2);  hold(T_FORWARD_2, fwd_drive)
    cleanup()


if __name__ == '__main__':
    main()
