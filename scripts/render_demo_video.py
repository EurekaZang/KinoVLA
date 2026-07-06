"""Render the recorded walking-skeleton demo into an annotated MP4 (scripts/record_demo.py first).

Composites, per control step: (Isaac) the real 3D Go2 render + a telemetry dashboard
— a top-down map (trajectory, O1 ice patch, goal, FSM avoid regions / detour
waypoints, oriented robot) and time-series panels for speed-vs-command, the three
Kino-Monitor channels with their thresholds and fire markers, and the velocity
command. An academic figure: every signal labeled with units, thresholds drawn,
events annotated.

Usage:
    python scripts/render_demo_video.py --rec outputs/recording --out outputs/kino_vla_m2_demo.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

CH = [
    ("slip_ema", "slip_ratio", "slip (O1/O3)", "C0"),
    ("track_ema", "tracking_error", "tracking err [m/s]", "C1"),
    ("effort_ema", "effort_ratio", "effort sat (O5/O10)", "C2"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the recorded demo to MP4")
    parser.add_argument("--rec", default="outputs/recording")
    parser.add_argument("--out", default="outputs/kino_vla_m2_demo.mp4")
    parser.add_argument("--stride", type=int, default=2, help="telemetry steps per output frame")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    rec = Path(args.rec)
    meta = json.loads((rec / "meta.json").read_text())
    tel = np.load(rec / "telemetry.npz", allow_pickle=True)
    n = len(tel["t"])
    t = tel["t"]
    has_video = bool(meta["has_video"]) and (rec / "isaac_raw.mp4").exists()
    reader = imageio.get_reader(str(rec / "isaac_raw.mp4")) if has_video else None

    ice, goal, start = meta["ice"], meta["goal"], meta["start"]
    thresholds = {
        "slip_ema": float(meta["thresholds"]["slip_ratio"]),
        "track_ema": float(meta["thresholds"]["tracking_error"]),
        "effort_ema": float(meta["thresholds"]["effort_ratio"]),
    }
    fire_idx = [i for i in range(n) if tel["fired"][i] > 0.5]

    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(
        3, 12, hspace=0.55, wspace=2.2, left=0.04, right=0.985, top=0.90, bottom=0.07
    )
    ax_go2 = fig.add_subplot(gs[0:2, 0:7]) if has_video else None
    ax_map = fig.add_subplot(gs[2, 0:7]) if has_video else fig.add_subplot(gs[0:3, 0:7])
    ax_spd = fig.add_subplot(gs[0, 7:12])
    ax_mon = fig.add_subplot(gs[1, 7:12])
    ax_cmd = fig.add_subplot(gs[2, 7:12])
    res = meta["result"]
    fig.suptitle(
        "KiNO · M2 Walking Skeleton — trained Go2 policy on the Kino-Fail O1 ice patch "
        f"({meta['backend']})",
        fontsize=15,
        fontweight="bold",
    )
    status = fig.text(0.04, 0.935, "", fontsize=11, family="monospace", va="top")

    out = imageio.get_writer(
        str(Path(args.out)), fps=args.fps, codec="libx264", quality=8, macro_block_size=None
    )
    idxs = list(range(0, n, args.stride))
    vframe = None
    vpos = -1
    for fi, i in enumerate(idxs):
        if reader is not None:
            while vpos < i:  # advance the sequential reader to frame i
                try:
                    vframe = reader.get_next_data()
                    vpos += 1
                except (IndexError, StopIteration):
                    break
        _draw(
            fi,
            i,
            idxs,
            tel,
            meta,
            thresholds,
            fire_idx,
            ax_go2,
            ax_map,
            ax_spd,
            ax_mon,
            ax_cmd,
            status,
            vframe,
            ice,
            goal,
            start,
            t,
        )
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        out.append_data(arr.copy())
    out.close()
    if reader is not None:
        reader.close()
    print(f"wrote {len(idxs)} frames -> {args.out}")
    print(
        f"result: monitor_fired={res['monitor_fired']} fell={res['fell']} "
        f"goal_reached={res['goal_reached']}"
    )
    print("PASS: render_demo_video")
    return 0


def _robot_marker(ax, x, y, heading, color):
    c, s = np.cos(heading), np.sin(heading)
    body = np.array([[0.24, 0], [-0.16, 0.12], [-0.16, -0.12]])
    pts = np.column_stack(
        [x + body[:, 0] * c - body[:, 1] * s, y + body[:, 0] * s + body[:, 1] * c]
    )
    ax.fill(pts[:, 0], pts[:, 1], color=color, zorder=6, ec="k", lw=0.8)


def _draw(
    fi,
    i,
    idxs,
    tel,
    meta,
    thresholds,
    fire_idx,
    ax_go2,
    ax_map,
    ax_spd,
    ax_mon,
    ax_cmd,
    status,
    vframe,
    ice,
    goal,
    start,
    t,
):
    # ----- Go2 3D render -----
    if ax_go2 is not None:
        ax_go2.clear()
        ax_go2.axis("off")
        if vframe is not None:
            ax_go2.imshow(vframe)
        ax_go2.set_title("Isaac Lab — trained RSL-RL Go2 policy (real PhysX)", fontsize=11)
        if tel["fired"][i] > 0.5 or (fire_idx and 0 <= i - _last_fire(fire_idx, i) <= 30):
            ax_go2.text(
                0.5,
                0.95,
                "⚠ KINO-MONITOR FIRED",
                color="red",
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="top",
                transform=ax_go2.transAxes,
            )

    # ----- top-down map -----
    ax_map.clear()
    ext = meta["terrain_extent"]
    ax_map.add_patch(
        Rectangle(
            (ice["cx"] - ice["hx"], ice["cy"] - ice["hy"]),
            2 * ice["hx"],
            2 * ice["hy"],
            color="#7fb3ff",
            alpha=0.6,
            zorder=1,
        )
    )
    ax_map.text(
        ice["cx"],
        ice["cy"],
        f"O1 ice\nμ={ice['mu_d']:.2f}",
        ha="center",
        va="center",
        fontsize=8,
        zorder=2,
    )
    ax_map.plot(start["x"], start["y"], "o", color="green", ms=9, zorder=3, label="start")
    ax_map.add_patch(
        Circle((goal["x"], goal["y"]), goal["tol"], color="gold", alpha=0.35, zorder=2)
    )
    ax_map.plot(goal["x"], goal["y"], "*", color="goldenrod", ms=18, zorder=4, label="goal")
    ax_map.plot(
        tel["x"][: i + 1], tel["y"][: i + 1], "-", color="0.4", lw=1.4, zorder=3, label="path"
    )
    for cx, cy, r in tel["avoid"][i] if len(tel["avoid"][i]) else []:
        ax_map.add_patch(
            Circle((cx, cy), r, fill=False, ec="darkorange", ls="--", lw=1.5, zorder=3)
        )
    wps = tel["waypoints"][i]
    if len(wps) > 1:
        wp = np.asarray(wps)
        ax_map.plot(wp[:, 0], wp[:, 1], ":", color="purple", lw=1.2, marker="x", ms=5, zorder=3)
    for ev in meta["events"]:
        if ev["t"] <= t[i]:
            ax_map.plot(ev["x"], ev["y"], "X", color="red", ms=10, mec="k", mew=0.6, zorder=5)
    _robot_marker(
        ax_map, tel["x"][i], tel["y"][i], tel["heading"][i], "red" if tel["fallen"][i] else "navy"
    )
    ax_map.set_xlim(-1.0, ext[0] / 2 + 1.5)
    ax_map.set_ylim(-ext[1] / 2 * 0.6, ext[1] / 2 * 0.6)
    ax_map.set_aspect("equal")
    ax_map.set_title("top-down map (odometry frame)", fontsize=10)
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.legend(loc="upper left", fontsize=7, ncol=4)
    ax_map.grid(alpha=0.25)

    tcur = t[i]
    # ----- speed vs command -----
    ax_spd.clear()
    ax_spd.plot(t[: i + 1], tel["cmd_speed"][: i + 1], color="C3", lw=1.4, label="|v| commanded")
    ax_spd.plot(t[: i + 1], tel["speed"][: i + 1], color="C0", lw=1.6, label="|v| measured")
    _decorate_ts(ax_spd, t, tcur, fire_idx, "speed [m/s]")
    ax_spd.legend(loc="upper right", fontsize=7)

    # ----- monitor channels -----
    ax_mon.clear()
    for key, _, label, color in CH:
        ax_mon.plot(t[: i + 1], tel[key][: i + 1], color=color, lw=1.5, label=label)
        ax_mon.axhline(thresholds[key], color=color, ls=":", lw=1.0, alpha=0.7)
    _decorate_ts(ax_mon, t, tcur, fire_idx, "Kino-Monitor (EMA)")
    ax_mon.set_ylim(-0.05, 1.05)
    ax_mon.legend(loc="upper right", fontsize=6.5, ncol=1)

    # ----- command -----
    ax_cmd.clear()
    ax_cmd.plot(t[: i + 1], tel["cmd_vx"][: i + 1], color="C4", lw=1.4, label="v_x cmd [m/s]")
    ax_cmd.plot(t[: i + 1], tel["cmd_wz"][: i + 1], color="C5", lw=1.4, label="ω_z cmd [rad/s]")
    _decorate_ts(ax_cmd, t, tcur, fire_idx, "Sport-Client command")
    ax_cmd.set_xlabel("sim time [s]")
    ax_cmd.legend(loc="upper right", fontsize=7)

    # ----- status HUD -----
    dist = float(np.hypot(tel["x"][i] - goal["x"], tel["y"][i] - goal["y"]))
    mon_state = "FIRED" if tel["fired"][: i + 1].sum() > 0 else "armed/clear"
    status.set_text(
        f"t={tcur:5.2f}s   phase={str(tel['phase'][i]):8s}   dist→goal={dist:4.2f}m\n"
        f"monitor={mon_state:11s}  speed={tel['speed'][i]:.2f} m/s   "
        f"fallen={'YES' if tel['fallen'][i] else 'no'}"
    )


def _last_fire(fire_idx, i):
    prev = [f for f in fire_idx if f <= i]
    return prev[-1] if prev else -999


def _decorate_ts(ax, t, tcur, fire_idx, ylabel):
    for f in fire_idx:
        ax.axvline(t[f], color="red", ls="-", lw=0.8, alpha=0.5)
    ax.axvline(tcur, color="k", lw=1.2, alpha=0.8)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25)


if __name__ == "__main__":
    raise SystemExit(main())
