"""grasp_report.py  --  describe the grasp in NUMBERS, so a screenshot isn't needed.

The penetration/force-closure audit can pass while the grasp looks broken (cube
buried in the palm, fingers splayed, contacts all on one corner). This reports the
coherence numbers that actually catch that:

  fingertip -> cube surface : is each fingertip ON the cube (0) or splayed away (+) or buried (-)
  faces touched             : which cube faces have contact; a real grip touches OPPOSING faces
  cube vs palm/fingers      : is the cube held between the fingers, or swallowed into the palm

Run:
  python grasp_report.py ~/dexcanvas/mocap_ver0.1.parquet            # leap, retarget
  python grasp_report.py ~/dexcanvas/mocap_ver0.1.parquet leap human
"""
import os, sys
if "MUJOCO_GL" not in os.environ and sys.platform != "darwin" and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"; os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np, mujoco
from render_demo import clean_trajectory


def report(demo, kp, hand_name="leap", grip_mode="human", obj_half_mm=None):
    spec_tips_names = None
    sc, fadr, tips, mp, mq, ff, op_, oq_, kp_seq, hum_obj, gf, obj_half, table_z, T = \
        clean_trajectory(demo, kp, hand_name, grip_mode=grip_mode, obj_half_mm=obj_half_mm)
    m, d = sc.model, sc.data
    OBJ = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_bid)
    PALM = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.palm_bid)

    # the formed grasp = last frame the cube is still at rest (fully gripped, pre-lift)
    rest_z = op_[0, 2]
    gi = int(np.max(np.where(op_[:, 2] <= rest_z + 1e-4)[0]))
    d.mocap_pos[sc.mocap_id] = mp[gi]; d.mocap_quat[sc.mocap_id] = mq[gi]
    d.qpos[fadr] = ff[gi]
    d.qpos[sc.obj_qadr:sc.obj_qadr+3] = op_[gi]
    d.qpos[sc.obj_qadr+3:sc.obj_qadr+7] = oq_[gi]
    mujoco.mj_forward(m, d)

    C = np.array(op_[gi]); h = obj_half
    palm_p = d.xpos[sc.palm_bid].copy()
    tip_p = {t: d.xpos[m.body(t).id if isinstance(t, str) else t].copy() for t in tips}
    tip_p = [d.xpos[t].copy() for t in tips]

    def box_sdf(p):
        q = np.abs(p - C) - h
        return np.linalg.norm(np.maximum(q, 0)) + min(max(q[0], q[1], q[2]), 0.0)

    sdf = [box_sdf(p) for p in tip_p]
    engaged = sum(1 for s in sdf if abs(s) < 0.006)

    # which faces have contact
    FACES = {0: ("-x", "+x"), 1: ("-y", "+y"), 2: ("-z", "+z")}
    faces = set(); pd = 0.0
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if (g1 in OBJ) ^ (g2 in OBJ):
            pd = max(pd, -float(d.contact[c].dist))
            rel = d.contact[c].pos - C
            ax = int(np.argmax(np.abs(rel)))
            faces.add(FACES[ax][1 if rel[ax] > 0 else 0])
    opposing = any((a in faces and b in faces) for a, b in [("-x", "+x"), ("-y", "+y"), ("-z", "+z")])
    ncon = sum(1 for c in range(d.ncon) if (d.contact[c].geom1 in OBJ) ^ (d.contact[c].geom2 in OBJ))

    # buried? cube center distance to palm vs to fingertip centroid
    tipcen = np.mean(tip_p, axis=0)
    d_palm = float(np.linalg.norm(C - palm_p))
    d_tips = float(np.linalg.norm(C - tipcen))

    print("=" * 60)
    print(f"GRASP COHERENCE  ({hand_name}, grip={grip_mode}, grasp frame)")
    print("=" * 60)
    print(f"  cube: half={h*1000:.0f}mm  at [{C[0]:.3f} {C[1]:.3f} {C[2]:.3f}]")
    names = ["tip" + str(i) for i in range(len(tips))]
    print("  fingertip -> cube surface (mm, ~0 ON, + splayed, - buried):")
    print("      " + "  ".join(f"{s*1000:+5.0f}" for s in sdf))
    print(f"  fingertips engaged (|dist|<6mm): {engaged}/{len(tips)}     [want >=3]")
    print(f"  faces touched: {' '.join(sorted(faces)) or '(none)'}   "
          f"opposing faces: {'YES' if opposing else 'NO'}   [want YES]")
    print(f"  contacts: {ncon}   max penetration: {pd*1000:.1f}mm   [want pen<5]")
    print(f"  cube-center -> palm: {d_palm*1000:.0f}mm   -> fingertip-centroid: {d_tips*1000:.0f}mm")
    print(f"      (cube buried in palm if palm distance is the SMALL one)")
    # verdict: a coherent grip = contacts on opposing faces, low penetration, the cube
    # held OUT in the fingers (not swallowed by the palm), and fingers not splayed away.
    min_tip = min(abs(s) for s in sdf)
    held_in_fingers = d_tips < d_palm
    not_splayed = min_tip < (h + 0.02)
    good = opposing and (pd < 0.006) and held_in_fingers and not_splayed
    bad = []
    if not opposing: bad.append("no opposing-face contact (not pinched)")
    if pd >= 0.006: bad.append(f"penetration {pd*1000:.0f}mm")
    if not held_in_fingers: bad.append("cube nearer palm than fingers (buried)")
    if not_splayed is False: bad.append(f"fingers splayed (nearest tip {min_tip*1000:.0f}mm off)")
    print("-" * 60)
    print("  COHERENT GRASP" if good else "  INCOHERENT -> " + "; ".join(bad))
    print("=" * 60)


def main():
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_factory import load_input
    args = sys.argv[2:]
    grip_mode = "retarget" if "retarget" in args else ("human" if "human" in args else "pinch")
    nums = [a for a in args if a.replace('.', '', 1).isdigit()]
    obj_half_mm = float(nums[0]) if nums else None
    rest = [a for a in args if a not in ("human","retarget","pinch") and a not in nums]
    hand = rest[0] if rest else "leap"
    hd = load_input(sys.argv[1]); demo, kp = hd.to_demo()
    report(demo, kp, hand_name=hand, grip_mode=grip_mode, obj_half_mm=obj_half_mm)


if __name__ == "__main__":
    main()
