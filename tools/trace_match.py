"""trace_match.py  --  DEEP DIVE: where is the human hand vs the robot hand, every frame.

For each frame it measures, in the wrist frame (so scale/position are comparable):
  - palm orientation: angle between the human palm normal and the robot palm normal
  - each FINGERTIP: human position vs robot position, and the gap (mm)
  - each finger's MCP/PIP/DIP if available
Then summarizes per finger over the whole clip (mean / max gap, worst frame), so you can
see exactly which finger drifts, how far, and when.

  python tools/trace_match.py ~/dexcanvas/mocap_ver0.1.parquet           # leap
  python tools/trace_match.py ~/dexcanvas/mocap_ver0.1.parquet leap 5    # every 5th frame, leap
"""
import os, sys
if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY") and sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "osmesa"
import numpy as np, mujoco
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
import forge.se3 as se3
from forge import hands
from forge.replay import ReplayScene
from forge.retarget import DexPilotRetargeter

# MediaPipe/MANO 21-joint layout
H_WRIST = 0
H_TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
H_MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def _n(v): return v / (np.linalg.norm(v) + 1e-9)


def palm_normal(wrist, idx_mcp, pinky_mcp):
    return _n(np.cross(_n(idx_mcp - wrist), _n(pinky_mcp - wrist)))


def trace(demo, kp, hand_name="leap", stride=5):
    spec = hands.load_hand(hand_name)
    rt = DexPilotRetargeter(spec, align=True); fq = rt.retarget_sequence(kp)
    eef = demo.eef_poses
    T = min(len(eef), len(fq), len(kp))
    sc = ReplayScene(spec.xml_path, spec.palm, table_z=-1.0)
    m, d = sc.model, sc.data
    fadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    tips = [m.body(t).id for t in spec.tips]
    palm_bid = sc.palm_bid
    thumb_idx = getattr(spec, "thumb_idx", len(tips) - 1)

    # map robot tips -> human finger names. non-thumb robot tips (in order) take
    # index/middle/ring/pinky; the thumb tip takes thumb.
    nonthumb_names = ["index", "middle", "ring", "pinky"]
    robot2human = {}
    j = 0
    for ri in range(len(tips)):
        if ri == thumb_idx: robot2human[ri] = "thumb"
        else: robot2human[ri] = nonthumb_names[j]; j += 1

    def seat(t):
        root = eef[t] @ se3.pose_inv(sc.root_to_palm); p, R = se3.unmake_pose(root)
        d.mocap_pos[sc.mocap_id] = p; d.mocap_quat[sc.mocap_id] = se3.mat2quat(R)
        d.qpos[fadr] = fq[t]; mujoco.mj_forward(m, d)

    def robot_palm_frame():
        pp = d.xpos[palm_bid].copy(); tp = np.array([d.xpos[b] for b in tips])
        nt = np.array([tp[i] for i in range(len(tips)) if i != thumb_idx])
        fwd = _n(nt.mean(0) - pp); side = _n(nt[-1] - nt[0])
        nrm = _n(np.cross(fwd, side)); side = np.cross(nrm, fwd)
        return pp, np.column_stack([fwd, side, nrm])

    def human_palm_frame(kpw):
        w = kpw[H_WRIST]; mcp = kpw[[H_MCP["index"], H_MCP["middle"], H_MCP["ring"], H_MCP["pinky"]]]
        fwd = _n(mcp.mean(0) - w); side = _n(kpw[H_MCP["pinky"]] - kpw[H_MCP["index"]])
        nrm = _n(np.cross(fwd, side)); side = np.cross(nrm, fwd)
        return np.column_stack([fwd, side, nrm])

    frames = list(range(0, T, stride))
    fingers = [robot2human[ri] for ri in range(len(tips))]
    tip_gap = {f: [] for f in fingers}
    tip_gap_aligned = {f: [] for f in fingers}        # after removing the palm-orientation flip
    palm_pos_gap = []; palm_ang = []

    for t in frames:
        seat(t)
        Rw = eef[t, :3, :3]; tw = eef[t, :3, 3]
        kpw = (Rw @ kp[t].T).T + tw
        rp = d.xpos[palm_bid].copy()
        palm_pos_gap.append(np.linalg.norm(kpw[H_WRIST] - rp) * 1000)
        Rh = human_palm_frame(kpw); pp, Rr = robot_palm_frame()
        R_corr = Rh @ Rr.T
        palm_ang.append(float(np.degrees(np.arccos(np.clip((np.trace(R_corr)-1)/2, -1, 1)))))
        for ri in range(len(tips)):
            name = robot2human[ri]
            h_world = kpw[H_TIP[name]]; r_world = d.xpos[tips[ri]].copy()
            tip_gap[name].append(np.linalg.norm(h_world - r_world) * 1000)
            r_aligned = R_corr @ (r_world - pp) + pp        # rotate robot onto human palm frame
            tip_gap_aligned[name].append(np.linalg.norm(h_world - r_aligned) * 1000)

    def stat(a): a = np.array(a); return a.mean(), a.max(), frames[int(a.argmax())]
    print("=" * 72)
    print(f"HUMAN vs ROBOT CORRESPONDENCE  ({hand_name}, {len(frames)} frames, stride {stride})")
    print("=" * 72)
    pm, px, pf = stat(palm_pos_gap); am, ax_, af = stat(palm_ang)
    print(f"  PALM position gap : mean {pm:5.1f}mm  max {px:5.1f}mm @frame {pf}")
    print(f"  PALM orient flip  : mean {am:5.1f} deg max {ax_:5.1f} deg @frame {af}"
          f"   <-- near 180 = robot hand FLIPPED vs human")
    print("-" * 72)
    print(f"  {'finger':<8}{'tip gap RAW (mm)':<26}{'tip gap PALM-ALIGNED (mm)':<26}")
    print(f"  {'':<8}{'mean / max':<26}{'mean / max':<26}")
    for f in fingers:
        wm, wx, _ = stat(tip_gap[f]); lm, lx, _ = stat(tip_gap_aligned[f])
        print(f"  {f:<8}{wm:5.0f} /{wx:5.0f}{'':<15}{lm:5.0f} /{lx:5.0f}")
    print("-" * 72)
    print("  RAW = robot vs human as currently placed (includes the orientation flip).")
    print("  PALM-ALIGNED = after rotating the robot's palm onto the human's palm; this")
    print("  is the TRUE 'is the robot doing the same gesture' error. If PALM-ALIGNED is")
    print("  small but RAW is large, the only problem is a fixed orientation flip we can bake in.")
    # ---- PER-FRAME dump: CSV + time-series plot (every analyzed frame) ----
    import csv
    with open("trace_match.csv", "w", newline="") as fcsv:
        wri = csv.writer(fcsv)
        head = ["frame", "palm_pos_gap_mm", "palm_flip_deg"]
        for f in fingers: head += [f + "_raw_mm", f + "_aligned_mm"]
        wri.writerow(head)
        for k, t in enumerate(frames):
            row = [t, round(palm_pos_gap[k], 1), round(palm_ang[k], 1)]
            for f in fingers: row += [round(tip_gap[f][k], 1), round(tip_gap_aligned[f][k], 1)]
            wri.writerow(row)
    print(f"\n  wrote trace_match.csv  ({len(frames)} rows, one per analyzed frame)")

    # ---- AUTO ISSUES SUMMARY: phases (steady segments), steps, and spikes ----
    fr = np.array(frames); n = len(fr)
    def runmed(a, win=31):
        a = np.array(a, float)
        return np.array([np.median(a[max(0, i-win//2):i+win//2+1]) for i in range(len(a))])
    def phases(a, min_jump, win=20):
        a = np.array(a, float); cps = []; i = win
        while i < n - win:
            if abs(np.median(a[i:i+win]) - np.median(a[i-win:i])) > min_jump:
                cps.append(i); i += win
            else: i += 1
        bounds = [0] + cps + [n-1]
        segs = [[bounds[j], bounds[j+1], float(np.median(a[bounds[j]:bounds[j+1]+1]))]
                for j in range(len(bounds)-1)]
        merged = []                                   # merge adjacent near-equal slivers
        for s in segs:
            if merged and abs(s[2] - merged[-1][2]) < min_jump * 0.6:
                merged[-1][1] = s[1]
                merged[-1][2] = float(np.median(a[merged[-1][0]:s[1]+1]))
            else: merged.append(s)
        return [(int(fr[a0]), int(fr[a1]), v) for a0, a1, v in merged]
    def baseline(a, ph):
        b = np.zeros(n)
        for a0, a1, v in ph:
            i0 = int(np.searchsorted(fr, a0)); i1 = int(np.searchsorted(fr, a1))
            b[i0:i1+1] = v
        return b
    def steps(a, win=20):
        a = np.array(a, float); out = []; i = win
        while i < n - win:
            d = np.median(a[i:i+win]) - np.median(a[i-win:i])
            if abs(d) > 18: out.append((int(fr[i]), float(d))); i += win
            else: i += 1
        return out
    def spikes(a, thr=25):
        a = np.array(a, float); res = a - baseline(a, phases(a, 22))
        hot = np.where(np.abs(res) > thr)[0]; out = []
        for i in hot:
            if not out or i - out[-1][-1] > 8: out.append([i])
            else: out[-1].append(i)
        return [(int(fr[g[0]]), int(fr[g[-1]]),
                 float(a[g][np.abs(res[g]).argmax()])) for g in out if len(g) >= 3]

    print("=" * 72)
    print("  ISSUES SUMMARY  (paste this)")
    print("=" * 72)
    print("  ORIENTATION (palm flip vs human; 0=aligned, 180=backwards):")
    for a0, a1, v in phases(palm_ang, 35):
        tag = "BACKWARDS" if v > 135 else ("SIDEWAYS" if v > 45 else "ALIGNED")
        print(f"    frames {a0:>4}-{a1:<4}  flip ~{v:3.0f} deg  [{tag}]")
    if min(palm_ang) > 30:
        print("    >> NEVER aligned (flip never near 0) -- robot wrist frame from loader is wrong")
    print("  PER-FINGER gesture error (palm-aligned tip gap, mm):")
    worst_f = max(fingers, key=lambda x: np.mean(tip_gap_aligned[x]))
    for f in fingers:
        ph = phases(tip_gap_aligned[f], 22)
        seg = "; ".join(f"{a0}-{a1}:~{v:.0f}" for a0, a1, v in ph[:6])
        if len(ph) > 6: seg += f" ...(+{len(ph)-6} more)"
        sp = sorted(spikes(tip_gap_aligned[f]), key=lambda s: -s[2])[:4]
        sps = ("  SPIKES " + ", ".join(f"{s0}-{s1}:{pk:.0f}" for s0, s1, pk in sp)) if sp else ""
        print(f"    {f:<7} {seg}{sps}{'  <<WORST' if f == worst_f else ''}")
    allj = []
    for nm, a in [("flip", palm_ang)] + [(f, tip_gap_aligned[f]) for f in fingers]:
        for fjmp, delta in steps(a): allj.append((fjmp, nm, abs(delta)))
    allj.sort(key=lambda x: -x[2])
    if allj:
        print("  BIGGEST STEP CHANGES (frame / what / size):")
        seen = set()
        for fjmp, nm, sz in allj:
            if fjmp in seen: continue
            seen.add(fjmp)
            print(f"    frame {fjmp:>4}  {nm:<7} ~{sz:.0f}{'deg' if nm=='flip' else 'mm'}")
            if len(seen) >= 5: break
    print("=" * 72)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        ax[0].plot(frames, palm_ang, color="crimson"); ax[0].axhline(180, ls=":", c="gray")
        ax[0].set_ylabel("palm flip (deg)")
        ax[0].set_title("robot vs human: orientation flip (180 = backwards)")
        for f in fingers: ax[1].plot(frames, tip_gap[f], label=f)
        ax[1].set_ylabel("RAW tip gap (mm)"); ax[1].legend(ncol=4, fontsize=8)
        ax[1].set_title("fingertip gap as-placed (includes the flip)")
        for f in fingers: ax[2].plot(frames, tip_gap_aligned[f], label=f)
        ax[2].set_ylabel("PALM-ALIGNED tip gap (mm)"); ax[2].set_xlabel("frame")
        ax[2].legend(ncol=4, fontsize=8)
        ax[2].set_title("fingertip gap after de-flip (true gesture error)")
        fig.tight_layout(); fig.savefig("trace_match.png", dpi=110); plt.close(fig)
        print("  wrote trace_match.png  (per-frame time series)")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    # sample frames in full: show the actual wrist-frame positions (mm) so the TYPE of
    # mismatch is visible (scale vs rotation vs offset), not just the gap magnitude.
    print("=" * 72)
    print("  sample frames -- fingertip position in WRIST frame (mm), human vs robot:")
    for t in [frames[0], frames[len(frames)//2], frames[-1]]:
        seat(t); Rw = eef[t, :3, :3]; tw = eef[t, :3, 3]
        print(f"  --- frame {t} ---")
        for ri in range(len(tips)):
            name = robot2human[ri]
            h = kp[t, H_TIP[name]] * 1000
            r = (Rw.T @ (d.xpos[tips[ri]] - tw)) * 1000
            print(f"    {name:<7} human[{h[0]:6.0f}{h[1]:6.0f}{h[2]:6.0f}]  "
                  f"robot[{r[0]:6.0f}{r[1]:6.0f}{r[2]:6.0f}]  gap {np.linalg.norm(h-r):.0f}mm")
    print("=" * 72)


def main():
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_factory import load_input
    args = sys.argv[2:]
    nums = [a for a in args if a.isdigit()]
    stride = int(nums[0]) if nums else 5
    rest = [a for a in args if not a.isdigit()]
    hand = rest[0] if rest else "leap"
    hd = load_input(sys.argv[1]); demo, kp = hd.to_demo()
    trace(demo, kp, hand_name=hand, stride=stride)


if __name__ == "__main__":
    main()
