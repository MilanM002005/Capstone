#!/usr/bin/env python3
"""
GPS-denied, time-based straight-line move for a skid-steer ArduRover.
Companion computer: Jetson (JetPack 4.6.1, Python 3.6).
Method: open-loop dead reckoning. No position estimate is used.
        MANUAL mode + RC_CHANNELS_OVERRIDE, throttle forward for N seconds.
        Periodic left-correction pulse to counter mechanical right-drift.

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
STEERING_STR  = 1600     # tuned straight-line steering center

# --- DRIFT CORRECTION (weld makes it drift RIGHT, so pulse LEFT) ---
CORRECT_EVERY = 7      # seconds of straight driving between corrections
CORRECT_SECS  = 0.3        # seconds to hold the left pulse
STEERING_LEFT = 1475    # left-pulse PWM. Lower = sharper left, ->1600 = gentler
# ------------------------------------------------------------------

RUN_SECONDS   = 100      # hard-coded autonomous move duration
REFRESH_HZ    = 10       # override must be refreshed or it times out (~1-3 s)

# Tuning helper: True = one 5 s straight burst (no corrections) to re-check trim.
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


def steering_for(elapsed):
    """Straight normally; left pulse for CORRECT_SECS after every CORRECT_EVERY s."""
    if TUNE_MODE:
        return STEERING_STR, False
    cycle = CORRECT_EVERY + CORRECT_SECS          # e.g. 16 s
    in_pulse = (elapsed % cycle) >= CORRECT_EVERY  # last CORRECT_SECS of each cycle
    return (STEERING_LEFT, True) if in_pulse else (STEERING_STR, False)


def main():
    run_secs = 5 if TUNE_MODE else RUN_SECONDS
    print("Connecting to %s @ %d ..." % (CONN, BAUD))
    master = mavutil.mavlink_connection(CONN, baud=BAUD)
    master.wait_heartbeat()
    print("Heartbeat OK: sys=%d comp=%d" %
          (master.target_system, master.target_component))
    if TUNE_MODE:
        print("*** TUNE_MODE: %d s straight burst, no corrections ***" % run_secs)

    try:
        print("Setting MANUAL mode ...")
        set_mode(master, 'MANUAL')
        time.sleep(1)

        set_rc(master, {STEERING_CH: STEERING_STR, THROTTLE_CH: NEUTRAL})

        print("Arming ...")
        arm(master, True)
        master.motors_armed_wait()
        print("Armed.")

        start_wall = time.strftime('%Y-%m-%d %H:%M:%S')
        t0 = time.monotonic()
        print("[t=0.00s] AUTONOMOUS MOVE START @ %s" % start_wall)

        period = 1.0 / REFRESH_HZ
        next_log = 1.0
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= run_secs:
                break
            steer, correcting = steering_for(elapsed)
            set_rc(master, {STEERING_CH: steer, THROTTLE_CH: THROTTLE_FWD})
            if elapsed >= next_log:
                tag = "LEFT-CORRECT" if correcting else "driving"
                print("[t=%.2fs] %-12s thr=%d str=%d" %
                      (elapsed, tag, THROTTLE_FWD, steer))
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
