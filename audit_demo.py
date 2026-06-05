"""audit_demo.py  --  quantitative audit of the SAME poses render_demo renders.

Stills lie; motion problems (clipping, jitter, slip, table penetration) only show
over time. This replays the shared clean_trajectory builder and measures the
standard hand-object metrics frame by frame, then writes a time-series plot.

Metrics + the field's validity thresholds:
  PD  penetration depth  max hand-into-object   INVALID if > 5 mm
  TP  table penetration  max hand/obj-into-table INVALID if > 10 mm
  SD  grasp drift        object slip vs the palm after the grasp
  jit object jitter / jerk motion spikes & smoothness

    python audit_demo.py ~/dexcanvas/mocap_ver0.1.parquet [hand]
Output: forge_audit.png + a printed PASS/FAIL report.
"""
import os, sys
if "MUJOCO_GL" not in os.environ and sys.platform != "darwin" and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"; os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from render_demo import clean_trajectory

PD_MAX, TP_MAX = 0.005, 0.010


def _sets(m, sc):
    obj = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_bid)
    table = set(g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
    return obj, table, set(range(m.ngeom)) - obj - table


def _pen(m, d, A, B):
    p = 0.0
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if (g1 in A and g2 in B) or (g1 in B and g2 in A):
            p = max(p, -float(d.contact[c].dist))
    return p


def _table_pen(m, d, HAND, OBJ, table_top):
    """Geometric table penetration: how far the lowest hand/object geom dips
    below the table top (infinite-plane contact distances are unreliable)."""
    lo = min((d.geom_xpos[g][2] - float(np.max(m.geom_size[g])) for g in (HAND | OBJ)),
             default=table_top)
    return max(0.0, table_top - lo)


def audit(demo, kp, hand_name="leap", mode="kinematic", out="forge_audit.png", stride=2,
          grip_mode="human"):
    sc, fadr, tips, mp, mq, ff, op_, oq_, kp_seq, hum_obj, gf, obj_half, table_z, T = \
        clean_trajectory(demo, kp, hand_name, grip_mode=grip_mode)
    m, d = sc.model, sc.data
    OBJ, TABLE, HAND = _sets(m, sc)
    table_top = float(table_z)
    rest_z = float(op_[0, 2])

    def palm():
        return d.xpos[sc.palm_bid].copy(), d.xmat[sc.palm_bid].reshape(3, 3).copy()

    ts, PD, TP, OZ, PZ, REL = [], [], [], [], [], []
    rel0 = None
    for i in range(0, len(mp), stride):
        d.mocap_pos[sc.mocap_id] = mp[i]; d.mocap_quat[sc.mocap_id] = mq[i]
        d.qpos[fadr] = ff[i]
        d.qpos[sc.obj_qadr:sc.obj_qadr+3] = op_[i]
        d.qpos[sc.obj_qadr+3:sc.obj_qadr+7] = oq_[i]
        mujoco.mj_forward(m, d)
        ts.append(i)
        PD.append(_pen(m, d, HAND, OBJ))
        TP.append(_table_pen(m, d, HAND, OBJ, table_top))
        OZ.append(float(d.xpos[sc.obj_bid][2])); PZ.append(float(d.xpos[sc.palm_bid][2]))
        pp, pR = palm()
        rp = pR.T @ (d.xpos[sc.obj_bid] - pp)            # object in palm frame
        if float(op_[i, 2]) > rest_z + 0.003 and rel0 is None: rel0 = rp.copy()
        REL.append(0.0 if rel0 is None else float(np.linalg.norm(rp - rel0)))

    ts = np.array(ts); PD = np.array(PD); TP = np.array(TP)
    OZ = np.array(OZ); PZ = np.array(PZ); REL = np.array(REL)
    speed = np.r_[0, np.abs(np.diff(OZ))] / (stride / 30.0)
    fjerk = np.abs(np.diff(ff, n=3, axis=0)).max() if len(ff) > 3 else 0.0
    wjerk = np.abs(np.diff(mp[:, 2], n=3)).max() if len(mp) > 3 else 0.0

    def v(x, lim): return "PASS" if x <= lim else "FAIL"
    print("="*64)
    print(f"FORGE TRAJECTORY AUDIT  ({hand_name}, {mode}, frames {ts[0]}-{ts[-1]})")
    print("="*64)
    print(f"  PD  hand->object penetration : {PD.max()*1000:6.1f} mm   [{v(PD.max(),PD_MAX)}  limit 5 mm]")
    print(f"  TP  table penetration        : {TP.max()*1000:6.1f} mm   [{v(TP.max(),TP_MAX)}  limit 10 mm]")
    print(f"  SD  object slip vs palm      : {REL.max()*1000:6.1f} mm   [{'PASS' if REL.max()<0.02 else 'FAIL'}  want < 20 mm]")
    print(f"  lift object rose             : {(OZ.max()-OZ.min())*100:6.1f} cm")
    print(f"  jit max object speed         : {speed.max()*100:6.1f} cm/s")
    print(f"  jerk wrist / finger          : {wjerk*1000:5.1f} mm / {fjerk:5.2f} rad")
    flags = []
    if PD.max() > PD_MAX: flags.append(f"clips object {PD.max()*1000:.0f}mm")
    if TP.max() > TP_MAX: flags.append(f"clips table {TP.max()*1000:.0f}mm")
    if REL.max() > 0.02:  flags.append(f"object slips {REL.max()*1000:.0f}mm")
    print("-"*64)
    print("  VERDICT:", "clean" if not flags else "ISSUES -> " + "; ".join(flags))
    print("="*64)

    lift_i = int(np.argmax(OZ > rest_z + 0.003)) if (OZ > rest_z + 0.003).any() else len(OZ)//2
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(ts, OZ*100, label="object z"); ax[0].plot(ts, PZ*100, '--', label="palm z")
    ax[0].axvline(ts[lift_i], color='k', ls=':', lw=1); ax[0].set_ylabel("height (cm)"); ax[0].legend(fontsize=8)
    ax[1].plot(ts, PD*1000, label="hand->object PD"); ax[1].plot(ts, TP*1000, label="table pen")
    ax[1].axhline(5, color='r', ls='--', lw=1); ax[1].axhline(10, color='orange', ls='--', lw=1)
    ax[1].set_ylabel("penetration (mm)"); ax[1].legend(fontsize=8)
    ax[2].plot(ts, REL*1000, label="object slip"); ax[2].plot(ts, speed*100, label="object speed cm/s")
    ax[2].set_ylabel("slip / speed"); ax[2].set_xlabel("output frame (close -> lift)"); ax[2].legend(fontsize=8)
    fig.suptitle(f"forge audit: {hand_name} ({mode})  [VERDICT: {'clean' if not flags else 'issues'}]")
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"wrote {out}")
    return {"PD": PD.max(), "TP": TP.max(), "slip": REL.max()}


def main():
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_factory import load_input
    args = sys.argv[2:]
    grip_mode = "retarget" if "retarget" in args else ("human" if "human" in args else "pinch")
    rest = [a for a in args if a not in ("human", "retarget", "physics", "kinematic")]
    hand = rest[0] if rest else "leap"
    hd = load_input(sys.argv[1]); demo, kp = hd.to_demo()
    audit(demo, kp, hand_name=hand, grip_mode=grip_mode)


if __name__ == "__main__":
    main()
