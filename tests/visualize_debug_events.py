#!/usr/bin/env python
"""Visualize TeraMoE in-kernel debug events as a per-rank role timeline.

Reads the ``teramoe_debug_events_rank{R}.npy`` files produced by the DEBUG branch
(see ``teramoe/buffer.py::_dump_teramoe_debug_events``) and draws a Gantt-style
timeline: x = time, y = (rank, role). Each rank shows 3 role bands
(dispatch / compute / combine); within a band every unit (dispatch SM, compute
task, combine SM) is drawn as a thin horizontal bar so overlap and spread are
visible.

Record dtype (mirrors MkDbgEvent in teramoe_orchestrator.cu):
    ts(u8, ns)  type(i4)  sm_id(i4)  aux0(i4)  aux1(i4)
Event types:
    1 dispatch_start   2 dispatch_end      -> unit key = sm_id
    3 compute_task_start 4 compute_task_end -> unit key = (aux0=group_id, aux1=task_idx)
    5 combine_start    6 combine_end        -> unit key = sm_id

IMPORTANT: %globaltimer is per-GPU, so timestamps are NOT comparable across
ranks. Each rank is normalized to its own first event (t=0), which is the
meaningful within-rank timeline. Different ranks are stacked only for visual
comparison of shape/duration, not absolute alignment.

Usage:
    python tests/visualize_debug_events.py                       # all ranks found
    python tests/visualize_debug_events.py --ranks 0,1,2
    python tests/visualize_debug_events.py --ranks 0-3 --out tl.png
    python tests/visualize_debug_events.py --input-dir output --unit us
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

EVENT_DTYPE = np.dtype([
    ("ts", "<u8"), ("type", "<i4"), ("sm_id", "<i4"), ("aux0", "<i4"), ("aux1", "<i4"),
])

# Event type ids.
DISPATCH_START, DISPATCH_END = 1, 2
COMPUTE_START, COMPUTE_END = 3, 4
COMBINE_START, COMBINE_END = 5, 6

ROLES = ["dispatch", "compute", "combine"]
# compute sub-colors by group_id (0=dedicated, 1=dispatch-reused, 2=combine-precompute)
COMPUTE_GROUP_COLORS = ["#2ca02c", "#17becf", "#8c8c1a"]
DISPATCH_COLOR = "#1f77b4"
COMBINE_COLOR = "#d62728"
COMPUTE_GROUP_NAMES = {0: "compute:dedicated", 1: "compute:dispatch-reused", 2: "compute:combine-precompute"}
# Top-to-bottom row order for the compute group sub-rows (by group_id).
# dispatch-reused (1) above dedicated (0), then combine-precompute (2).
COMPUTE_GROUP_ROW_ORDER = [1, 0, 2]


def parse_ranks(spec, available):
    """Parse a --ranks spec like '0,1,2', '0-3', 'all' into a sorted rank list."""
    if spec is None or spec.strip().lower() == "all":
        return sorted(available)
    wanted = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(part))
    ranks = sorted(r for r in wanted if r in available)
    missing = sorted(wanted - set(available))
    if missing:
        print(f"[warn] requested ranks not found and skipped: {missing}")
    return ranks


def discover_rank_files(input_dir):
    """Return {rank: path} for every teramoe_debug_events_rank*.npy in input_dir."""
    out = {}
    for path in glob.glob(os.path.join(input_dir, "teramoe_debug_events_rank*.npy")):
        m = re.search(r"rank(\d+)\.npy$", os.path.basename(path))
        if m:
            out[int(m.group(1))] = path
    return out


def pair_intervals(events, start_type, end_type, key_fields):
    """Pair start/end events into [(t0, t1, key_tuple), ...] by matching key_fields."""
    starts = {}
    for e in events[events["type"] == start_type]:
        starts[tuple(int(e[f]) for f in key_fields)] = int(e["ts"])
    intervals = []
    for e in events[events["type"] == end_type]:
        key = tuple(int(e[f]) for f in key_fields)
        if key in starts:
            intervals.append((starts[key], int(e["ts"]), key))
    return intervals


def load_rank(path):
    """Load one rank's events; normalize ts so the rank's first event is t=0 (ns)."""
    a = np.load(path)
    if a.dtype != EVENT_DTYPE:
        a = a.view(EVENT_DTYPE)
    if len(a) == 0:
        return a, 0
    t0 = int(a["ts"].min())
    a = a.copy()
    a["ts"] = a["ts"] - t0
    return a, t0


def role_intervals(events):
    """Return dict role -> list of (t0, t1, group_id_or_None, unit_key)."""
    disp = [(t0, t1, None, k) for (t0, t1, k) in
            pair_intervals(events, DISPATCH_START, DISPATCH_END, ["sm_id"])]
    comb = [(t0, t1, None, k) for (t0, t1, k) in
            pair_intervals(events, COMBINE_START, COMBINE_END, ["sm_id"])]
    comp = [(t0, t1, k[0], k) for (t0, t1, k) in
            pair_intervals(events, COMPUTE_START, COMPUTE_END, ["aux0", "aux1"])]
    return {"dispatch": disp, "compute": comp, "combine": comb}


def draw_band(ax, intervals, lane_top, scale, role):
    """Draw one role's intervals inside a unit-height band.

    dispatch/combine: one sub-row per SM (these SMs run in parallel, so the
        vertical spread shows the per-SM start/end skew).
    compute: one sub-row PER GROUP (max 3). Tasks inside a group are serial
        (the group leader finishes one task before popping the next), so they
        are drawn as consecutive segments along x on a single row, not stacked.
    """
    if not intervals:
        return
    inner_lo, inner_hi = lane_top + 0.08, lane_top + 0.92

    if role == "compute":
        present = {gid for _t0, _t1, gid, _k in intervals if gid is not None}
        order_list = [g for g in COMPUTE_GROUP_ROW_ORDER if g in present]
        order_list += sorted(g for g in present if g not in COMPUTE_GROUP_ROW_ORDER)
        row_of = {g: i for i, g in enumerate(order_list)}
        n = max(len(order_list), 1)
        h = (inner_hi - inner_lo) / n
        bar_h = max(h * 0.8, h - 0.01)
        for t0, t1, gid, _key in intervals:
            row = row_of.get(gid, 0)
            y = inner_lo + row * h
            color = COMPUTE_GROUP_COLORS[gid % len(COMPUTE_GROUP_COLORS)] if gid is not None else "#2ca02c"
            width = max((t1 - t0) * scale, 0.0)
            # Thin edge so back-to-back serial tasks stay visually separable.
            ax.broken_barh([(t0 * scale, width)], (y, bar_h),
                           facecolors=color, edgecolors="white", linewidth=0.3)
        return

    # dispatch / combine: parallel units, one sub-row each (ordered by unit id).
    order = sorted(intervals, key=lambda iv: iv[3])
    n = len(order)
    h = (inner_hi - inner_lo) / n
    bar_h = max(h * 0.9, h - 0.002)
    color = DISPATCH_COLOR if role == "dispatch" else COMBINE_COLOR
    for i, (t0, t1, _gid, _key) in enumerate(order):
        y = inner_lo + i * h
        width = max((t1 - t0) * scale, 0.0)
        ax.broken_barh([(t0 * scale, width)], (y, bar_h), facecolors=color, edgecolors="none")


def plot(ranks, files, unit, out_path, xlim=None, xtick=None):
    scale = {"ns": 1.0, "us": 1e-3, "ms": 1e-6}[unit]
    n_ranks = len(ranks)
    fig_h = max(2.0, 0.9 * n_ranks * 3 * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    yticks, ylabels = [], []
    lane = 0  # 0 at top after we invert the y-axis
    for r in ranks:
        events, t0 = load_rank(files[r])
        ivs = role_intervals(events)
        for role in ROLES:
            draw_band(ax, ivs[role], lane, scale, role)
            yticks.append(lane + 0.5)
            ylabels.append(f"r{r}:{role}")
            lane += 1
        # separator line between ranks
        ax.axhline(lane, color="0.85", lw=0.6)

    ax.set_ylim(0, lane)
    ax.invert_yaxis()  # rank0 / first role on top
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel(f"time since each rank's first event ({unit})")
    ax.set_title("TeraMoE forward role timeline (per-rank normalized; %globaltimer is per-GPU)")

    # Manual x range / tick interval so different runs can be aligned on the same scale.
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if xtick is not None and xtick > 0:
        lo, hi = ax.get_xlim()
        ax.set_xticks(np.arange(lo, hi + xtick * 1e-6, xtick))
    ax.grid(axis="x", color="0.9", lw=0.5)

    legend = [
        Patch(color=DISPATCH_COLOR, label="dispatch"),
        Patch(color=COMBINE_COLOR, label="combine"),
        Patch(color=COMPUTE_GROUP_COLORS[0], label=COMPUTE_GROUP_NAMES[0]),
        Patch(color=COMPUTE_GROUP_COLORS[1], label=COMPUTE_GROUP_NAMES[1]),
        Patch(color=COMPUTE_GROUP_COLORS[2], label=COMPUTE_GROUP_NAMES[2]),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[ok] wrote {out_path}")


def print_summary(ranks, files, unit, compute_group_size):
    """Per-rank/role timing.

    ``sm_time`` is the busy SM-time = sum over units of (sms_per_unit * duration).
    This is the right occupancy metric because roles do NOT hold all their SMs
    for the whole wall-clock span (compute groups start/finish at different
    times, tasks are serial within a group). sms_per_unit is 1 for dispatch and
    combine (one unit == one SM), and ``compute_group_size`` for compute (each
    task runs on a whole group of SMs).
    """
    scale = {"ns": 1.0, "us": 1e-3, "ms": 1e-6}[unit]
    for r in ranks:
        events, t0 = load_rank(files[r])
        ivs = role_intervals(events)
        print(f"rank {r}:")
        total_sm_time = 0.0
        for role in ROLES:
            iv = ivs[role]
            if not iv:
                print(f"  {role:9s}: (no events)")
                continue
            starts = [a for a, _b, _g, _k in iv]
            ends = [b for _a, b, _g, _k in iv]
            busy = sum((b - a) for a, b, _g, _k in iv)  # ns, summed over units
            sms_per_unit = compute_group_size if role == "compute" else 1
            sm_time = busy * sms_per_unit * scale
            total_sm_time += sm_time
            span = (max(ends) - min(starts)) * scale
            print(f"  {role:9s}: units={len(iv):3d}  "
                  f"first_start={min(starts)*scale:8.2f}  last_end={max(ends)*scale:8.2f}  "
                  f"span={span:8.2f}  sm_time={sm_time:11.2f} SM*{unit}  (sms/unit={sms_per_unit})")
        print(f"  {'TOTAL':9s}: sm_time={total_sm_time:11.2f} SM*{unit}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="output",
                    help="dir with teramoe_debug_events_rank*.npy (default: output)")
    ap.add_argument("--ranks", default="all",
                    help="ranks to plot: 'all', '0,1,2', or '0-3' (default: all)")
    ap.add_argument("--unit", default="ms", choices=["ns", "us", "ms"],
                    help="x-axis time unit (default: ms)")
    ap.add_argument("--out", default=None, help="output png path")
    ap.add_argument("--xlim", default=None,
                    help="fixed x range as 'min,max' in --unit (align plots across runs)")
    ap.add_argument("--xtick", type=float, default=None,
                    help="x tick interval in --unit (align plots across runs)")
    ap.add_argument("--summary", action="store_true", help="also print per-rank/role timing summary")
    ap.add_argument("--compute-group-size", type=int, default=48,
                    help="SMs per compute group, for the sm_time metric (default: 48 = COMPUTE_GROUP_SIZE)")
    args = ap.parse_args()

    files = discover_rank_files(args.input_dir)
    if not files:
        raise SystemExit(f"no teramoe_debug_events_rank*.npy found in {args.input_dir}")
    ranks = parse_ranks(args.ranks, set(files.keys()))
    if not ranks:
        raise SystemExit("no ranks selected")

    out_path = args.out or os.path.join(args.input_dir, f"teramoe_timeline_{'_'.join(map(str, ranks))}.png")
    if len(ranks) > 6:
        out_path = args.out or os.path.join(args.input_dir, "teramoe_timeline.png")

    xlim = None
    if args.xlim is not None:
        lo, hi = (float(v) for v in args.xlim.split(","))
        xlim = (lo, hi)

    if args.summary:
        print_summary(ranks, files, args.unit, args.compute_group_size)
    plot(ranks, files, args.unit, out_path, xlim=xlim, xtick=args.xtick)


if __name__ == "__main__":
    main()


