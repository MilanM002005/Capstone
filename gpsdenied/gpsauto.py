#!/usr/bin/env python3
"""
GPS-denied, time-based straight-line move for a skid-steer ArduRover.
Companion computer: Jetson (JetPack 4.6.1, Python 3.6).
Method: open-loop dead reckoning. No position estimate is used.
        MANUAL mode + RC_CHANNELS_OVERRIDE, throttle forward for N seconds.

Run:
    python3 gpsauto.py

Ctrl+C at any time = immediate stop + release + disarm (handled in finally).
"""

import time
from pymavlink import mavutil

# ============ CONFIG ============
CONN  = '/dev/ttyACM0'   # USB to Cube. UART: '/dev/ttyTHS1'
BAUD  = 115200           # USB: 115200. UART telem: 57600 or 921600

THROTTLE_CH = 2          # RC channel read as throttle == RCMAP_THROTTLE
STEERING_CH = 1          # RC channel read as steering == RCMAP_ROLL

NEUTRAL       = 1500     # PWM for stop
THROTTLE_FWD  = 1650     # PWM forward. 1500=stop. Higher = faster.

# --- STRAIGHTNESS TUNING KNOB ---
# It circled LEFT at 1500, so 1500 is NOT your true steering center.
# Bias RIGHT (raise this) until it tracks straight. Sweep: 1500->1550->1575->1600...
# If it then over-corrects RIGHT, back off. Set to your RC1_TRIM if you know it.
STEERING_STR  = 1600     # <-- change THIS to make it go straight

RUN_SECONDS   = 100       # hard-coded autonomous move duration
REFRESH_HZ    = 10       # override must be refreshed or it times out (~1-3 s)

# Tuning helper: set True for short 5 s bursts while finding STEERING_STR,
# then set back to False for the real 20 s run.
TUNE_MODE     = False
# ================================

IGNORE = 65535           # per-channel "don't touch" value in RC override


def set_rc(master, overrides):
    rc = [IGNORE] * 8
    for ch, val in overrides.items():
        rc[ch - 1] = val
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component, *rc)


def release_rc(master):
    master.mav.rc_channels_override_send(
        master.target_system, master.target_component, 0, 0, 0, 0, 0, 0, 0, 0)


def set_mode(master, name):
    mode_id = master.mode_mapping()[name]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id)


def arm(master, do_arm):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1 if do_arm else 0, 0, 0, 0, 0, 0, 0)


def main():
    run_secs = 5 if TUNE_MODE else RUN_SECONDS
    print("Connecting to %s @ %d ..." % (CONN, BAUD))
    master = mavutil.mavlink_connection(CONN, baud=BAUD)
    master.wait_heartbeat()
    print("Heartbeat OK: sys=%d comp=%d" %
          (master.target_system, master.target_component))
    if TUNE_MODE:
        print("*** TUNE_MODE: %d s burst, STEERING_STR=%d ***" %
              (run_secs, STEERING_STR))

    try:
        print("Setting MANUAL mode ...")
        set_mode(master, 'MANUAL')
        time.sleep(1)

        # Prime override at neutral BEFORE arming so no jump on arm.
        set_rc(master, {STEERING_CH: STEERING_STR, THROTTLE_CH: NEUTRAL})

        print("Arming ...")
        arm(master, True)
        master.motors_armed_wait()
        print("Armed.")

        # ---- autonomous move: t = 0 -> run_secs ----
        start_wall = time.strftime('%Y-%m-%d %H:%M:%S')
        t0 = time.monotonic()
        print("[t=0.00s] AUTONOMOUS MOVE START @ %s" % start_wall)

        period = 1.0 / REFRESH_HZ
        next_log = 1.0
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= run_secs:
                break
            set_rc(master, {STEERING_CH: STEERING_STR,
                            THROTTLE_CH: THROTTLE_FWD})
            if elapsed >= next_log:
                print("[t=%.2fs] driving  thr=%d str=%d" %
                      (elapsed, THROTTLE_FWD, STEERING_STR))
                next_log += 1.0
            time.sleep(period)

        end_wall = time.strftime('%Y-%m-%d %H:%M:%S')
        print("[t=%.2fs] AUTONOMOUS MOVE END @ %s" %
              (time.monotonic() - t0, end_wall))

    except KeyboardInterrupt:
        print("\nAbort requested.")
    finally:
        try:
            print("Neutral -> release -> disarm ...")
            set_rc(master, {STEERING_CH: STEERING_STR, THROTTLE_CH: NEUTRAL})
            time.sleep(0.5)
            release_rc(master)
            time.sleep(0.2)
            arm(master, False)
            print("Disarmed. Done.")
        except Exception as e:
            print("Cleanup error: %s" % e)


if __name__ == '__main__':
    main()
