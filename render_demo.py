"""render_demo.py  --  the demo video.

Renders one human hand demonstration retargeted onto real robot hands in MuJoCo.
The cube rests on the table untouched until the hand closes on it, then rides the
grasp through a clean lift, so the motion reads as a real pick, not a glued prop.

Modes:
  (default)  one robot hand performing the pick (clean studio render)
  trio       the SAME demo on LEAP, Allegro and Shadow side by side, one camera
             -- the cross-embodiment shot
  overlay    the human keypoints ghosted onto the robot (correspondence check)

    python render_demo.py mocap.parquet                 # default: leap
    python render_demo.py mocap.parquet trio            # cross-embodiment
    python render_demo.py mocap.parquet shadow 2 az=-110 el=-18 zoom=1.1

Output: forge_demo.mp4 in the current directory.
"""
import os, sys
# pick a headless GL backend only when needed (headless Linux). On macOS the
# default (CGL) renders offscreen fine, so we leave it alone.
if "MUJOCO_GL" not in os.environ and sys.platform != "darwin" and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import mujoco
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import forge.se3 as se3
from forge import hands
from forge.replay import ReplayScene
from forge.retarget import DexPilotRetargeter

# MediaPipe / MANO 21-keypoint skeleton connectivity
BONES = [(0,1),(1,2),(2,3),(3,4),       # thumb
         (0,5),(5,6),(6,7),(7,8),       # index
         (0,9),(9,10),(10,11),(11,12),  # middle
         (0,13),(13,14),(14,15),(15,16),# ring
         (0,17),(17,18),(18,19),(19,20)]# pinky
FINGER_COLOR = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4']  # per finger


def _cube_faces(c, h):
    x, y, z = c
    v = np.array([[x-h, y-h, z-h], [x+h, y-h, z-h], [x+h, y+h, z-h], [x-h, y+h, z-h],
                  [x-h, y-h, z+h], [x+h, y-h, z+h], [x+h, y+h, z+h], [x-h, y+h, z+h]])
    return [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
            [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]


def human_frame(kp, cube_xyz, obj_half, lims, elev, azim, table_z, W=480, H=480):
    """One human-hand frame: skeleton + solid cube + table, fixed frame so the
    lift reads as motion, camera angle matched to the robot panel."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    # table slab
    tb = [[[xlo, ylo, table_z], [xhi, ylo, table_z], [xhi, yhi, table_z], [xlo, yhi, table_z]]]
    ax.add_collection3d(Poly3DCollection(tb, facecolor='#b8b8b8', edgecolor='none', alpha=0.55))
    # solid cube (matches the simulator's orange object)
    ax.add_collection3d(Poly3DCollection(_cube_faces(np.array(cube_xyz), obj_half),
                                         facecolor='#e8632a', edgecolor='#7a2e10', alpha=0.82))
    # skeleton
    for bi, (a, b) in enumerate(BONES):
        ax.plot(*zip(kp[a], kp[b]), c=FINGER_COLOR[bi//4], lw=4)
    ax.scatter(*kp.T, c='k', s=14)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("human hand (input)", fontsize=12)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy()
    plt.close(fig)
    return img


def clean_trajectory(demo, kp, hand_name="leap", grip_mode="human", obj_half_mm=None, approach=90,
                     grip_overclose=0.15, settle=160):
    """Single source of truth for the grasp+lift poses, used by BOTH the renderer
    and the auditor. Settles the grip ONCE in physics (so contact stops the
    fingers at the object/table surface -> low penetration), freezes that resolved
    pose, and carries the object rigidly relative to the palm through the lift.
    Returns (sc, fadr, tips, mocap_pos, mocap_quat, fingers, obj_pos, obj_quat,
    kp_world, gf, obj_half, start)."""
    spec = hands.load_hand(hand_name)
    fq = DexPilotRetargeter(spec).retarget_sequence(kp)
    eef = demo.eef_poses
    obj0 = demo.object_poses[list(demo.object_poses)[0]]
    objw = obj0 if obj0.ndim == 3 else np.tile(obj0, (len(eef), 1, 1))
    T = min(len(eef), len(fq), len(objw))
    gf = int(np.argmin(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1)))
    real_obj = obj0.ndim == 3 and float(np.ptp(objw[:T, :3, 3], 0).max()) > 0.01

    # object sized to fingertip spread
    pr = ReplayScene(spec.xml_path, spec.palm, table_z=-1.0)
    pfa = [pr.model.jnt_qposadr[pr.model.actuator_trnid[i, 0]] for i in range(pr.model.nu)]
    pti = [pr.model.body(t).id for t in spec.tips]
    rt = eef[gf] @ se3.pose_inv(pr.root_to_palm); wp, wR = se3.unmake_pose(rt)
    pr.data.mocap_pos[pr.mocap_id] = wp; pr.data.mocap_quat[pr.mocap_id] = se3.mat2quat(wR)
    pr.data.qpos[pfa] = fq[gf]; mujoco.mj_forward(pr.model, pr.data)
    txyz = np.array([pr.data.xpos[b] for b in pti]); cen0 = txyz.mean(0)
    obj_half = float(np.clip(np.median(np.linalg.norm(txyz - cen0, axis=1)), 0.016, 0.022))
    if obj_half_mm is not None: obj_half = float(obj_half_mm) / 1000.0
    obj_rest_z = float(np.min(objw[:T, 2, 3])) if real_obj else obj_half

    sc = ReplayScene(spec.xml_path, spec.palm, obj_size=(obj_half,)*3,
                     table_z=obj_rest_z - obj_half - 0.002)
    m, d = sc.model, sc.data
    m.vis.headlight.ambient[:] = 0.7; m.vis.headlight.diffuse[:] = 0.9
    fadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    tips = [m.body(t).id for t in spec.tips]
    thumb_idx = getattr(spec, "thumb_idx", len(tips) - 1)
    open_q = fq[0].copy()
    OBJ = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_bid)
    HAND = set(range(m.ngeom)) - OBJ - set(g for g in range(m.ngeom)
                                           if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)

    def palm():
        return se3.make_pose(d.xpos[sc.palm_bid].copy(), d.xmat[sc.palm_bid].reshape(3, 3).copy())
    def seat(wrist_pose, finger_q):
        root = wrist_pose @ se3.pose_inv(sc.root_to_palm); wp, wR = se3.unmake_pose(root)
        d.mocap_pos[sc.mocap_id] = wp; d.mocap_quat[sc.mocap_id] = se3.mat2quat(wR)
        d.qpos[fadr] = finger_q; mujoco.mj_forward(m, d)
    def pen_obj():
        p = 0.0
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in OBJ) ^ (g2 in OBJ): p = max(p, -float(d.contact[c].dist))
        return p

    # WHERE the cube sits at the grasp. Real captured data: the cube is already where
    # the retargeted hand grasps it (diagnostic shows tens of contacts) -> no shift.
    # Synthetic / no-object input: infer a rest spot at the fingertip centroid.
    seat(eef[gf], fq[gf]); grasp_cen = np.mean([d.xpos[b] for b in tips], axis=0)
    if real_obj:
        rest_xyz = objw[gf, :3, 3].copy()
        shift = np.zeros(3)
    else:
        rest_xyz = np.array([grasp_cen[0], grasp_cen[1], sc.table_z + obj_half + 0.001])
        shift = rest_xyz - grasp_cen
    eef_s = eef.copy(); eef_s[:, :3, 3] += shift

    # GRIP = per-finger CLOSE-TO-CONTACT: close each finger until it first touches the
    # cube, then stop. Every finger comes to rest ON the surface -> a natural wrap
    # like the human's, with ~0 penetration, and no stiff servo to bulldoze through.
    PALM = set(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.palm_bid)
    obj_geom = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_bid)
    def pen_split():
        finger_pen = palm_pen = 0.0
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in OBJ) ^ (g2 in OBJ):
                pd = -float(d.contact[c].dist); hg = g2 if g1 in OBJ else g1
                if hg in PALM: palm_pen = max(palm_pen, pd)
                else: finger_pen = max(finger_pen, pd)
        return finger_pen, palm_pen
    def set_obj_rest():
        d.qpos[sc.obj_qadr:sc.obj_qadr+3] = rest_xyz
        d.qpos[sc.obj_qadr+3:sc.obj_qadr+7] = [1, 0, 0, 0]
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    close_dir = fq[gf] - open_q
    m.geom_size[obj_geom] = [obj_half, obj_half, obj_half]

    # OPTIONAL: plant the palm at the HUMAN palm frame (orientation copied directly
    # from the keypoints), instead of the captured robot wrist frame. This is the
    # "make the robot do what the human does" experiment. NOTE: for a hand whose
    # fingers curl toward the palm (e.g. LEAP), a human palm-DOWN frame physically
    # can't wrap a cube resting on a table -- the fingers curl up off it.
    if grip_mode in ("human", "pinch"):
        def _n(v): return v / (np.linalg.norm(v) + 1e-9)
        ti = getattr(spec, "thumb_idx", len(tips) - 1)
        kpw = (eef[gf, :3, :3] @ kp[gf].T).T + eef[gf, :3, 3]
        fwd_h = _n(kpw[[5, 9, 13, 17]].mean(0) - kpw[0]); side_h = _n(kpw[17] - kpw[5])
        nrm_h = _n(np.cross(fwd_h, side_h)); side_h = np.cross(nrm_h, fwd_h)
        Rh = np.column_stack([fwd_h, side_h, nrm_h])
        seat(eef[gf], open_q)
        pp = d.xpos[sc.palm_bid].copy(); tp = np.array([d.xpos[b] for b in tips])
        nt = np.array([tp[i] for i in range(len(tips)) if i != ti])
        fwd_r = _n(nt.mean(0) - pp); side_r = _n(nt[-1] - nt[0])
        nrm_r = _n(np.cross(fwd_r, side_r)); side_r = np.cross(nrm_r, fwd_r)
        Rr = np.column_stack([fwd_r, side_r, nrm_r])
        grasp_R = (Rh @ Rr.T) @ eef[gf, :3, :3]
        # position palm so the open hand just clears the cube from above
        seat(se3.make_pose(rest_xyz + np.array([0, 0, 0.2]), grasp_R), open_q)
        palm_above = d.xpos[sc.palm_bid][2] - min(d.xpos[b][2] for b in tips)
        eef_s = eef.copy()
        for dz in np.linspace(0.10, -0.03, 27):
            cand = se3.make_pose(np.array([rest_xyz[0], rest_xyz[1],
                                           rest_xyz[2] + palm_above + dz]), grasp_R)
            seat(cand, open_q)
            d.qpos[sc.obj_qadr:sc.obj_qadr+3] = rest_xyz
            d.qpos[sc.obj_qadr+3:sc.obj_qadr+7] = [1, 0, 0, 0]; mujoco.mj_forward(m, d)
            eef_s[gf] = cand
            if pen_obj() > 0.001: break

    # map each actuator to the finger (tip body) whose kinematic chain it drives
    chains = {}
    for tb in tips:
        ch = set(); b = tb
        while b != -1 and b != sc.palm_bid:
            ch.add(b); b = m.body_parentid[b]
        chains[tb] = ch
    finger_acts = {tb: [] for tb in tips}
    for i in range(m.nu):
        jb = m.jnt_bodyid[m.actuator_trnid[i, 0]]
        for tb, ch in chains.items():
            if jb in ch: finger_acts[tb].append(i); break
    def finger_pen(tb):
        chgeom = set(g for g in range(m.ngeom) if m.geom_bodyid[g] in chains[tb])
        p = 0.0
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if ((g1 in OBJ) and (g2 in chgeom)) or ((g2 in OBJ) and (g1 in chgeom)):
                p = max(p, -float(d.contact[c].dist))
        return p

    # BACK OFF the palm until the palm itself clears the cube. The captured human
    # wrist can park the (larger) robot palm INSIDE the cube (palm penetration), and
    # close-to-contact only moves fingers -> the palm would stay buried. Slide the
    # palm straight out along the cube->palm axis until palm penetration is gone,
    # THEN reach the fingers back in. This is the fix for "cube buried in the palm".
    C = rest_xyz.copy()
    base = eef_s[gf].copy()
    back = base[:3, 3] - C; back = back / (np.linalg.norm(back) + 1e-9)
    for s in np.linspace(0.0, 0.10, 41):
        cand = base.copy(); cand[:3, 3] = base[:3, 3] + back * s
        seat(cand, open_q); set_obj_rest(); mujoco.mj_forward(m, d)
        eef_s[gf] = cand
        if pen_split()[1] <= 0.0005: break    # palm clear

    # GRIP: (1) close each finger along the retargeted direction to first contact
    # (a wrap baseline), then (2) greedily refine each finger to drive its FINGERTIP
    # onto the nearest cube face (minimize tip->surface), stopping at contact. Step 2
    # only ever improves the tip distance, so it can't make the baseline worse.
    C = rest_xyz.copy(); hbox = np.array([obj_half, obj_half, obj_half])
    def tip_sdf(tb):
        p = d.xpos[tb]; q = np.abs(p - C) - hbox
        return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0)
    def finger_refine():                                       # joints only, all fingers
        for tb, acts in finger_acts.items():
            if not acts: continue
            for _ in range(30):
                seat(eef_s[gf], fq_grip); set_obj_rest(); mujoco.mj_forward(m, d)
                cur = abs(tip_sdf(tb))
                if cur < 0.003 or finger_pen(tb) >= 0.0015: break
                best = None
                for i in acts:
                    for dq in (0.06, -0.06):
                        test = fq_grip.copy(); test[i] = np.clip(test[i] + dq, lo[i], hi[i])
                        seat(eef_s[gf], test); set_obj_rest(); mujoco.mj_forward(m, d)
                        if finger_pen(tb) < 0.004:
                            val = abs(tip_sdf(tb))
                            if best is None or val < best[0]: best = (val, i, test[i])
                if best is None or best[0] >= cur - 1e-4: break
                fq_grip[best[1]] = best[2]

    def total_off():
        seat(eef_s[gf], fq_grip); set_obj_rest(); mujoco.mj_forward(m, d)
        return sum(abs(tip_sdf(t)) for t in tips), pen_split()

    fq_grip = open_q.copy()
    if grip_mode == "pinch":
        # TWO-FINGER PINCH using the RETARGETED pose (thumb-index gap already tracks the
        # human). Place the cube at the thumb-index MIDPOINT so it sits between them, then
        # squeeze the two fingers together until each presses a face. Friction from the
        # two opposed contacts is the hold -- exactly the grasp you described.
        th_tb = tips[thumb_idx]
        idx_tb = tips[0] if thumb_idx != 0 else tips[1]
        th_acts, idx_acts = finger_acts[th_tb], finger_acts[idx_tb]
        fq_grip = open_q.copy()                               # start the pinch fingers OPEN
        seat(eef_s[gf], fq_grip); set_obj_rest(); mujoco.mj_forward(m, d)
        M = 0.5 * (d.xpos[th_tb] + d.xpos[idx_tb])
        eef_s[gf][:3, 3] += (C - M)                            # cube center at the open midpoint
        # close each finger toward the cube only until its TIP reaches the face (no overshoot)
        for tb, acts in [(th_tb, th_acts), (idx_tb, idx_acts)]:
            base = fq_grip.copy()
            for a in np.linspace(0.0, 1.8, 60):
                for i in acts: fq_grip[i] = np.clip(base[i] + a * close_dir[i], lo[i], hi[i])
                seat(eef_s[gf], fq_grip); set_obj_rest(); mujoco.mj_forward(m, d)
                if tip_sdf(tb) <= 0.001 or finger_pen(tb) >= 0.0015: break
        for tb, acts in finger_acts.items():                  # retract any idle finger that touches
            if tb in (th_tb, idx_tb): continue
            for _ in range(20):
                seat(eef_s[gf], fq_grip); set_obj_rest(); mujoco.mj_forward(m, d)
                if finger_pen(tb) < 0.001: break
                for i in acts: fq_grip[i] = np.clip(fq_grip[i] - 0.08 * close_dir[i], lo[i], hi[i])
        seat(eef_s[gf], fq_grip); set_obj_rest(); mujoco.mj_forward(m, d)
    else:
        for tb, acts in finger_acts.items():                   # (1) baseline wrap
            if not acts: continue
            for a in np.linspace(0.0, 1.8, 55):
                for i in acts: fq_grip[i] = np.clip(open_q[i] + a * close_dir[i], lo[i], hi[i])
                seat(eef_s[gf], fq_grip); set_obj_rest(); mujoco.mj_forward(m, d)
                if finger_pen(tb) >= 0.0015: break
        finger_refine()
        # (2) JOINT GRASP SYNTHESIS: alternately move the WRIST in toward the cube and
        # re-curl the fingers, minimizing total fingertip->face distance while keeping the
        # palm clear and penetration small. Joints alone can't reach; the wrist must come in.
        for _ in range(14):
            base_off, (bf, bp) = total_off()
            moved = False
            for axis in range(3):
                for dW in (0.006, -0.006):
                    cand = eef_s[gf].copy(); cand[axis, 3] += dW
                    keep = eef_s[gf].copy(); eef_s[gf] = cand
                    off, (ff_, pp_) = total_off()
                    if pp_ <= 0.0008 and ff_ < 0.006 and off < base_off - 1e-4:
                        base_off = off; moved = True
                    else:
                        eef_s[gf] = keep
            finger_refine()
            if not moved: break
    seat(eef_s[gf], fq_grip); set_obj_rest(); mujoco.mj_forward(m, d)
    m.geom_rgba[obj_geom] = [0.91, 0.39, 0.16, 0.9]
    fpen, ppen = pen_split()
    ncon_obj = sum(1 for c in range(d.ncon)
                   if (d.contact[c].geom1 in OBJ) ^ (d.contact[c].geom2 in OBJ))
    tipd = [abs(tip_sdf(t)) for t in tips]
    print(f"  [build] obj_half={obj_half*1000:.0f}mm grip={grip_mode}+refine "
          f"contacts={ncon_obj} PD finger={fpen*1000:.1f}mm palm={ppen*1000:.1f}mm "
          f"tips_off={[round(x*1000) for x in tipd]}mm")
    rel_obj = se3.pose_inv(palm()) @ se3.make_pose(rest_xyz, np.eye(3))
    HANDG = [g for g in range(m.ngeom) if g not in OBJ
             and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    def hand_low():
        return min(d.geom_xpos[g][2] - float(np.max(m.geom_size[g])) for g in HANDG)

    # ---- the SHOWPIECE: a full, legible pick the eye can read, identical on both
    # panels. The hand REACHES down (open / pre-grasp), CLOSES on the cube, LIFTS.
    # All motion is clean vertical -- open fingers descending straight down never
    # sweep through the cube or table.
    n_app, n_close, n_lift = 30, 28, 90
    app_h, lift_h = 0.11, 0.16
    pre_grip = open_q + 0.30 * (fq_grip - open_q)        # mostly open: clears cube
    base_root = eef_s[gf].copy()

    kp_world = np.einsum('tij,tkj->tki', eef[:T, :3, :3], kp[:T]) + eef[:T, :3, 3][:, None]
    hum_grasp = kp_world[gf].copy()
    hum_rest = kp_world[gf, [4, 8, 12, 16, 20]].mean(0)   # cube in the HUMAN grip

    mp, mq, ff, op_, oq_, kp_seq, hum_obj = [], [], [], [], [], [], []
    lows = []
    def push(wp, finger, obj_p, obj_q, kpf, ho):
        seat(wp, finger)
        mp.append(d.mocap_pos[sc.mocap_id].copy()); mq.append(d.mocap_quat[sc.mocap_id].copy())
        ff.append(finger.copy()); op_.append(obj_p.copy()); oq_.append(obj_q)
        kp_seq.append(kpf); hum_obj.append(ho.copy()); lows.append(hand_low())

    rq = np.array([1.0, 0, 0, 0])
    for k in range(n_app):                               # phase 0: reach DOWN, open hand
        dz = app_h * (1 - k / max(n_app - 1, 1))
        wp = base_root.copy(); wp[2, 3] += dz
        kpf = hum_grasp.copy(); kpf[:, 2] += dz
        push(wp, pre_grip, rest_xyz, rq, kpf, hum_rest)
    for k in range(n_close):                             # phase 1: CLOSE on the cube
        a = k / max(n_close - 1, 1)
        finger = np.clip(pre_grip + (fq_grip - pre_grip) * a, lo, hi)
        push(base_root, finger, rest_xyz, rq, hum_grasp.copy(), hum_rest)
    for k in range(n_lift):                              # phase 2: LIFT straight up
        dz = lift_h * (k / max(n_lift - 1, 1))
        wp = base_root.copy(); wp[2, 3] += dz
        seat(wp, fq_grip)
        p, Rm = se3.unmake_pose(palm() @ rel_obj)
        kpf = hum_grasp.copy(); kpf[:, 2] += dz
        ho = hum_rest.copy(); ho[2] += dz
        mp.append(d.mocap_pos[sc.mocap_id].copy()); mq.append(d.mocap_quat[sc.mocap_id].copy())
        ff.append(fq_grip.copy()); op_.append(p); oq_.append(se3.mat2quat(Rm))
        kp_seq.append(kpf); hum_obj.append(ho); lows.append(hand_low())

    # table just below the lowest the hand EVER gets (across reach+close+lift)
    plane = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE][0]
    new_table = min(rest_xyz[2] - obj_half - 0.002, min(lows) - 0.004)
    m.geom_pos[plane] = [0, 0, new_table]; sc.table_z = new_table

    return (sc, fadr, tips, np.array(mp), np.array(mq), np.array(ff),
            np.array(op_), np.array(oq_), np.array(kp_seq), np.array(hum_obj),
            gf, obj_half, new_table, T)


def clean_trajectory_mimic(demo, kp, hand_name="leap", obj_half_mm=None):
    """FAITHFUL replay: the robot reproduces the human's retargeted hand pose frame by
    frame, and the cube follows its REAL recorded trajectory. No synthesized grasp, no
    glue -- the robot does exactly what the human did. Same tuple shape as clean_trajectory."""
    spec = hands.load_hand(hand_name)
    from forge.retarget import CurlRetargeter
    fq_full = CurlRetargeter(spec).retarget_sequence(kp)   # natural finger-bend (no contortion, frame-free)
    eef = demo.eef_poses
    obj0 = demo.object_poses[list(demo.object_poses)[0]]
    objw = obj0 if obj0.ndim == 3 else np.tile(obj0, (len(eef), 1, 1))
    T = min(len(eef), len(fq_full), len(objw))
    gf = int(np.argmin(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1)))
    real_obj = obj0.ndim == 3 and float(np.ptp(objw[:T, :3, 3], 0).max()) > 0.01

    pr = ReplayScene(spec.xml_path, spec.palm, table_z=-1.0)
    pfa = [pr.model.jnt_qposadr[pr.model.actuator_trnid[i, 0]] for i in range(pr.model.nu)]
    fq = np.asarray(fq_full)[:, pfa]                       # actuated joints only (nq may exceed nu, e.g. Shadow)
    pti = [pr.model.body(t).id for t in spec.tips]
    rt = eef[gf] @ se3.pose_inv(pr.root_to_palm); wp, wR = se3.unmake_pose(rt)
    pr.data.mocap_pos[pr.mocap_id] = wp; pr.data.mocap_quat[pr.mocap_id] = se3.mat2quat(wR)
    pr.data.qpos[pfa] = fq[gf]; mujoco.mj_forward(pr.model, pr.data)
    txyz = np.array([pr.data.xpos[b] for b in pti]); cen0 = txyz.mean(0)
    obj_half = float(np.clip(np.median(np.linalg.norm(txyz - cen0, axis=1)), 0.016, 0.024))
    if obj_half_mm is not None: obj_half = float(obj_half_mm) / 1000.0
    obj_rest_z = float(np.min(objw[:T, 2, 3])) if real_obj else obj_half

    sc = ReplayScene(spec.xml_path, spec.palm, obj_size=(obj_half,)*3,
                     table_z=obj_rest_z - obj_half - 0.002)
    m, d = sc.model, sc.data
    m.vis.headlight.ambient[:] = 0.7; m.vis.headlight.diffuse[:] = 0.9
    fadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    tips = [m.body(t).id for t in spec.tips]
    obj_geom = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_bid)
    m.geom_size[obj_geom] = [obj_half]*3; m.geom_rgba[obj_geom] = [0.91, 0.39, 0.16, 0.95]
    fadr2 = fadr; thumb_idx = getattr(spec, "thumb_idx", len(tips) - 1)
    def _n(v): return v / (np.linalg.norm(v) + 1e-9)
    def seat(wp_, fqq):
        root = wp_ @ se3.pose_inv(sc.root_to_palm); p, R = se3.unmake_pose(root)
        d.mocap_pos[sc.mocap_id] = p; d.mocap_quat[sc.mocap_id] = se3.mat2quat(R)
        d.qpos[fadr2] = fqq; mujoco.mj_forward(m, d)
    kp_world = np.einsum('tij,tkj->tki', eef[:T, :3, :3], kp[:T]) + eef[:T, :3, 3][:, None, :]

    mp = np.zeros((T, 3)); mq = np.zeros((T, 4)); ff = np.zeros((T, m.nu))
    op_ = np.zeros((T, 3)); oq_ = np.zeros((T, 4))

    # ONE constant orientation correction, measured at the grasp frame, so the robot is
    # oriented like the human AND stays at the real wrist gripping the REAL moving cube
    # (per-frame de-flip would swing the hand off the cube, hiding the grip).
    def palm_frames(t):
        kw = kp_world[t]; w = kw[0]; mcp = kw[[5, 9, 13, 17]]
        fh = _n(mcp.mean(0) - w); sh = _n(kw[17] - kw[5])
        nh = _n(np.cross(fh, sh)); sh = np.cross(nh, fh)
        Rh = np.column_stack([fh, sh, nh])
        seat(eef[t], fq[t])
        pp = d.xpos[sc.palm_bid].copy(); tp = np.array([d.xpos[b] for b in tips])
        nt = np.array([tp[i] for i in range(len(tips)) if i != thumb_idx])
        frd = _n(nt.mean(0) - pp); srd = _n(nt[-1] - nt[0])
        nrd = _n(np.cross(frd, srd)); srd = np.cross(nrd, frd)
        return Rh, np.column_stack([frd, srd, nrd])
    # the cube stays put on its rest spot and only starts moving once the hand closes on it
    # (time-gated at the grasp frame), then it rides the hand through the lift.
    W = max(8, int(0.02 * T))
    g_ramp = np.clip((np.arange(T) - gf) / float(W), 0.0, 1.0)
    Rh_g, Rr_g = palm_frames(gf); R_corr_g = Rh_g @ Rr_g.T
    aligned_g = R_corr_g @ eef[gf, :3, :3]
    root_g = se3.make_pose(eef[gf, :3, 3], aligned_g) @ se3.pose_inv(sc.root_to_palm)
    pg, Rg = se3.unmake_pose(root_g)
    d.mocap_pos[sc.mocap_id] = pg; d.mocap_quat[sc.mocap_id] = se3.mat2quat(Rg)
    d.qpos[fadr2] = fq[gf]; mujoco.mj_forward(m, d)
    twg = np.array([d.xpos[b] for b in tips])
    gc_gf = 0.5 * (twg[thumb_idx] + np.array([twg[i] for i in range(len(tips)) if i != thumb_idx]).mean(0))
    cube_delta = gc_gf - objw[gf, :3, 3]
    robot_min_z = np.inf; angs = []
    for t in range(T):
        Rh, Rr = palm_frames(t)
        R_corr = Rh @ Rr.T
        angs.append(np.degrees(np.arccos(np.clip((np.trace(R_corr) - 1) / 2, -1, 1))))
        aligned = R_corr @ eef[t, :3, :3]
        root = se3.make_pose(eef[t, :3, 3], aligned) @ se3.pose_inv(sc.root_to_palm)
        p, R = se3.unmake_pose(root)
        mp[t] = p; mq[t] = se3.mat2quat(R); ff[t] = fq[t]
        d.mocap_pos[sc.mocap_id] = p; d.mocap_quat[sc.mocap_id] = se3.mat2quat(R)
        d.qpos[fadr2] = fq[t]; mujoco.mj_forward(m, d)
        robot_min_z = min(robot_min_z, float(d.xpos[1:, 2].min()))
        op_[t] = objw[t, :3, 3] + g_ramp[t] * cube_delta   # stationary until grasp, then held
        oq_[t] = se3.mat2quat(objw[t, :3, :3])
    print(f"  [mimic] per-frame orientation match (median {np.median(angs):.0f} deg); "
          f"cube held only from the grasp frame onward")
    hum_obj = objw[:T, :3, 3]
    table_z = min(obj_rest_z - obj_half, robot_min_z - 0.003)   # below robot so it never clips the table
    plane_g = next((g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE), None)
    if plane_g is not None: m.geom_pos[plane_g, 2] = table_z
    return (sc, fadr, tips, mp, mq, ff, op_, oq_, kp_world, hum_obj, gf, obj_half, table_z, T)


def render_pick(demo, kp, hand_name="leap", out="forge_demo.mp4", stride=3,
                W=480, H=480, fps=30, mode="kinematic", grip_mode="human",
                azim=-120, elev=-12, zoom=1.0, orbit=False):
    if mode == "physics":
        return _render_physics(demo, kp, hand_name, out, stride, W, H, fps)

    # ---------- build the trajectory: faithful mimic, or synthesized grasp ----------
    if mode in ("mimic", "overlay") or grip_mode == "mimic":
        (sc, fadr, tips, mp, mq, ff, op_, oq_, kp_seq, hum_obj,
         gf, obj_half, table_z, T) = clean_trajectory_mimic(demo, kp, hand_name)
    else:
        (sc, fadr, tips, mp, mq, ff, op_, oq_, kp_seq, hum_obj,
         gf, obj_half, table_z, T) = clean_trajectory(demo, kp, hand_name, grip_mode=grip_mode)
    m, d = sc.model, sc.data

    # ---- studio look: soft lighting, light floor, clean light hand, warm cube ----
    m.vis.headlight.ambient[:] = 0.5; m.vis.headlight.diffuse[:] = 0.9; m.vis.headlight.specular[:] = 0.3
    _plane = next((g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE), None)
    _obj = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_bid)
    if _plane is not None: m.geom_rgba[_plane] = [0.90, 0.91, 0.94, 1.0]
    m.geom_rgba[_obj] = [0.96, 0.45, 0.13, 1.0]
    for g in range(m.ngeom):
        if g not in (_plane, _obj) and m.geom_rgba[g, 3] > 0:
            m.geom_rgba[g, :3] = [0.92, 0.93, 0.96]      # clean off-white robot

    # robot static camera framing the whole rest->lift volume (angle/zoom user-controllable)
    AZ, EL = azim, elev
    top = op_[:, 2].max(); bot = op_[:, 2].min(); span = max(top - bot, 0.10)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [op_[:, 0].mean(), op_[:, 1].mean(), 0.5 * (bot + top)]
    cam.distance = (span + 0.20) * 2.4 / max(zoom, 0.1); cam.azimuth = AZ; cam.elevation = EL
    ren = mujoco.Renderer(m, height=H, width=W)

    frames = []
    sel = list(range(0, len(mp), stride))

    # ---- skeleton drawing helpers (used by overlay mode) ----
    BONES = ([(0, 1), (1, 2), (2, 3), (3, 4)] + [(0, 5), (5, 6), (6, 7), (7, 8)] +
             [(0, 9), (9, 10), (10, 11), (11, 12)] + [(0, 13), (13, 14), (14, 15), (15, 16)] +
             [(0, 17), (17, 18), (18, 19), (19, 20)])
    H_TIPS = [8, 12, 16, 4]
    GREEN = np.float32([0.10, 0.85, 0.30, 1]); RED = np.float32([0.95, 0.20, 0.20, 1])
    WHITE = np.float32([0.92, 0.95, 1.0, 1]); SKEL = np.float32([0.45, 0.62, 0.95, 1])
    I3 = np.eye(3).flatten()

    def add(scene, gtype, size, pos, rgba):
        if scene.ngeom >= scene.maxgeom: return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype,
                            np.asarray(size, float), np.asarray(pos, float), I3, rgba)
        scene.ngeom += 1

    def connect(scene, p1, p2, rgba, w=0.004):
        if scene.ngeom >= scene.maxgeom: return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), I3, rgba)
        try:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w,
                                 np.asarray(p1, float), np.asarray(p2, float))
        except Exception:
            mujoco.mjv_makeConnector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w,
                                     p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
        scene.ngeom += 1

    def draw_skeleton(scene, kw, tip_color=GREEN, robot_tips=None):
        for a, b in BONES: connect(scene, kw[a], kw[b], SKEL)
        for j in range(21): add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.005, 0, 0], kw[j], WHITE)
        for ht in H_TIPS: add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.011, 0, 0], kw[ht], tip_color)
        if robot_tips is not None:
            for tb in robot_tips: add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.011, 0, 0], d.xpos[tb], RED)

    # ---------- OVERLAY: human skeleton drawn INTO the robot's scene/camera ----------
    if mode == "overlay":
        obj_gid = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_bid)
        for g in range(m.ngeom):
            if g != obj_gid: m.geom_rgba[g] = [0.55, 0.57, 0.62, 0.45]
        pts = np.concatenate([kp_seq.reshape(-1, 3), hum_obj], 0)
        cam.lookat[:] = pts.mean(0)
        cam.distance = max(np.ptp(pts, 0).max(), 0.12) * 3.2 / max(zoom, 0.1)
        for n, i in enumerate(sel):
            if orbit: cam.azimuth = AZ + 360.0 * n / max(len(sel) - 1, 1)
            d.mocap_pos[sc.mocap_id] = mp[i]; d.mocap_quat[sc.mocap_id] = mq[i]
            d.qpos[fadr] = ff[i]
            d.qpos[sc.obj_qadr:sc.obj_qadr+3] = op_[i]; d.qpos[sc.obj_qadr+3:sc.obj_qadr+7] = oq_[i]
            mujoco.mj_forward(m, d)
            ren.update_scene(d, cam)
            draw_skeleton(ren.scene, kp_seq[i], robot_tips=tips)
            frames.append(ren.render())
            if n % 25 == 0: print(f"  rendered {n+1}/{len(sel)} frames")
        imageio.mimsave(out, frames, fps=fps, quality=8)
        print(f"\nwrote {out}  ({len(frames)} frames, {hand_name}, OVERLAY "
              f"blue=human red=robot-tips, {W}x{H})")
        return out

    # ---------- ROBOT pick render (single clean panel) ----------
    for n, i in enumerate(sel):
        if orbit: cam.azimuth = AZ + 360.0 * n / max(len(sel) - 1, 1)
        d.mocap_pos[sc.mocap_id] = mp[i]; d.mocap_quat[sc.mocap_id] = mq[i]; d.qpos[fadr] = ff[i]
        d.qpos[sc.obj_qadr:sc.obj_qadr+3] = op_[i]; d.qpos[sc.obj_qadr+3:sc.obj_qadr+7] = oq_[i]
        mujoco.mj_forward(m, d)
        ren.update_scene(d, cam)
        frames.append(ren.render())
        if n % 25 == 0: print(f"  rendered {n+1}/{len(sel)} frames")
    imageio.mimsave(out, frames, fps=fps, quality=8)
    print(f"\nwrote {out}  ({len(frames)} frames, {hand_name}, robot pick, {W}x{H})")
    return out


def _render_physics(demo, kp, hand_name, out, stride, W, H, fps):
    spec = hands.load_hand(hand_name)
    fq = DexPilotRetargeter(spec).retarget_sequence(kp)
    eef = demo.eef_poses
    obj0 = demo.object_poses[list(demo.object_poses)[0]]
    objw = obj0 if obj0.ndim == 3 else np.tile(obj0, (len(eef), 1, 1))
    T = min(len(eef), len(fq), len(objw))
    gf = int(np.argmin(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1)))
    real_obj = obj0.ndim == 3 and float(np.ptp(objw[:T, :3, 3], 0).max()) > 0.01
    obj_rest_z = float(np.min(objw[:T, 2, 3])) if real_obj else 0.02
    sc = ReplayScene(spec.xml_path, spec.palm, table_z=obj_rest_z - 0.02)
    m, d = sc.model, sc.data
    m.vis.headlight.ambient[:] = 0.7; m.vis.headlight.diffuse[:] = 0.9
    fadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    tips = [m.body(t).id for t in spec.tips]
    kp_world = np.einsum('tij,tkj->tki', eef[:T, :3, :3], kp[:T]) + eef[:T, :3, 3][:, None]
    hand_rad = max(float(np.median(np.linalg.norm(kp_world - kp_world[:, :1], axis=2).max(1))) * 1.25, 0.05)
    if real_obj:
        obj_rel = np.linalg.inv(eef[gf]) @ objw[gf]
    else:
        root = eef[gf] @ se3.pose_inv(sc.root_to_palm); wp, wR = se3.unmake_pose(root)
        d.mocap_pos[sc.mocap_id] = wp; d.mocap_quat[sc.mocap_id] = se3.mat2quat(wR)
        d.qpos[fadr] = fq[gf]; mujoco.mj_forward(m, d)
        cen = np.mean([d.xpos[b] for b in tips], axis=0)
        obj_rel = np.linalg.inv(eef[gf]) @ se3.make_pose(cen, np.eye(3))
    print("  physics: establishing grip then lifting ...")
    res = sc.dynamic_lift(eef, fq, obj_rel, g=gf, lift_h=0.12)
    print(f"  physics result: lift={res['lift']*100:+.1f}cm -> "
          f"{'HELD' if res['success'] else f'slipped {res['final_slip']*100:.1f}cm'}")
    rw, rfq, ro = res["wrist"], res["fingers"], res["object"]; L = len(rw)
    hidx = np.linspace(gf, T - 1, L).round().astype(int)
    cam = mujoco.MjvCamera(); cam.lookat[:] = ro[:, :3, 3].mean(0)
    cam.distance = max(np.ptp(ro[:, :3, 3], 0).max(), 0.18) * 4.0
    cam.azimuth = -120; cam.elevation = -14
    ren = mujoco.Renderer(m, height=H, width=W)
    frames = []
    for i in range(0, L, max(1, stride // 2)):
        root = rw[i] @ se3.pose_inv(sc.root_to_palm); wp, wR = se3.unmake_pose(root)
        d.mocap_pos[sc.mocap_id] = wp; d.mocap_quat[sc.mocap_id] = se3.mat2quat(wR)
        d.qpos[fadr] = rfq[i]
        op, oR = se3.unmake_pose(ro[i])
        d.qpos[sc.obj_qadr:sc.obj_qadr+3] = op; d.qpos[sc.obj_qadr+3:sc.obj_qadr+7] = se3.mat2quat(oR)
        mujoco.mj_forward(m, d)
        ren.update_scene(d, cam); robot = ren.render()
        human = human_frame(kp_world[hidx[i]], None, kp_world[hidx[i], 0], hand_rad, W, H)
        hh = min(human.shape[0], robot.shape[0]); ww = min(human.shape[1], robot.shape[1])
        frames.append(np.concatenate([human[:hh, :ww], robot[:hh, :ww]], axis=1))
    imageio.mimsave(out, frames, fps=fps, quality=8)
    print(f"\nwrote {out}  ({len(frames)} frames, {hand_name}, PHYSICS lift, {W*2}x{H})")
    return out


def _label_panel(img, text, accent=(244, 100, 26)):
    """Burn a small caption bar onto the top of a rendered panel."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(np.ascontiguousarray(img)); dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
        except Exception:
            font = ImageFont.load_default()
        dr.rectangle([0, 0, im.width, 36], fill=(16, 22, 38))
        dr.rectangle([0, 34, im.width, 36], fill=accent)
        dr.text((11, 7), text, fill=(234, 242, 255), font=font)
        return np.asarray(im)
    except Exception:
        return img


def render_trio(demo, kp, hands_list=("leap", "allegro", "shadow"),
                out="forge_demo.mp4", stride=3, W=360, H=440, fps=30,
                azim=-120, elev=-14, zoom=1.0):
    """The cross-embodiment shot: ONE human demo retargeted onto several robot
    hands, rendered side by side from a single shared camera. Drives home the
    core claim -- one demonstration in, many validated robot hands out."""
    panels, labels, nf = [], [], None
    for hn in hands_list:
        try:
            (sc, fadr, tips, mp, mq, ff, op_, oq_, kp_seq, hum_obj,
             gf, obj_half, table_z, T) = clean_trajectory_mimic(demo, kp, hn)
        except Exception as e:
            print(f"  [trio] {hn:8s} skipped ({type(e).__name__})")
            continue
        m, d = sc.model, sc.data
        # studio look: soft lighting, light floor, clean hand, warm cube
        m.vis.headlight.ambient[:] = 0.5; m.vis.headlight.diffuse[:] = 0.9; m.vis.headlight.specular[:] = 0.3
        _plane = next((g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE), None)
        _obj = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_bid)
        if _plane is not None: m.geom_rgba[_plane] = [0.90, 0.91, 0.94, 1.0]
        m.geom_rgba[_obj] = [0.96, 0.45, 0.13, 1.0]
        for g in range(m.ngeom):
            if g not in (_plane, _obj) and m.geom_rgba[g, 3] > 0:
                m.geom_rgba[g, :3] = [0.92, 0.93, 0.96]
        top = op_[:, 2].max(); bot = op_[:, 2].min(); span = max(top - bot, 0.10)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [op_[:, 0].mean(), op_[:, 1].mean(), 0.5 * (bot + top)]
        cam.distance = (span + 0.20) * 2.4 / max(zoom, 0.1); cam.azimuth = azim; cam.elevation = elev
        ren = mujoco.Renderer(m, height=H, width=W)
        sel = list(range(0, len(mp), stride))
        frames = []
        for i in sel:
            d.mocap_pos[sc.mocap_id] = mp[i]; d.mocap_quat[sc.mocap_id] = mq[i]; d.qpos[fadr] = ff[i]
            d.qpos[sc.obj_qadr:sc.obj_qadr+3] = op_[i]; d.qpos[sc.obj_qadr+3:sc.obj_qadr+7] = oq_[i]
            mujoco.mj_forward(m, d); ren.update_scene(d, cam); frames.append(ren.render().copy())
        try: ren.close()
        except Exception: pass
        panels.append(frames); labels.append(hn.upper())
        nf = len(frames) if nf is None else min(nf, len(frames))
        print(f"  [trio] {hn:8s} {len(frames)} frames")
    out_frames = []
    for k in range(nf):
        row = [_label_panel(panels[j][k], labels[j]) for j in range(len(panels))]
        hh = min(r.shape[0] for r in row)
        out_frames.append(np.concatenate([r[:hh] for r in row], axis=1))
    imageio.mimsave(out, out_frames, fps=fps, quality=8)
    print(f"\nwrote {out}  ({len(out_frames)} frames, trio {'+'.join(hands_list)}, "
          f"{W*len(hands_list)}x{H})")
    return out


def render_skeleton_grid(kp, out="demo_grid.png", n=12, cols=4, elev=16, azim=-72,
                         title="human demonstration  (the input we convert)"):
    """A contact-sheet of the human hand demo: the skeleton sampled across the clip
    in a clean grid. Static, fast, and reads as data rather than a shaky video."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    BG = "#0b0f1a"
    FINGERS = [[(0, 1), (1, 2), (2, 3), (3, 4)],          # thumb
               [(0, 5), (5, 6), (6, 7), (7, 8)],          # index
               [(0, 9), (9, 10), (10, 11), (11, 12)],     # middle
               [(0, 13), (13, 14), (14, 15), (15, 16)],   # ring
               [(0, 17), (17, 18), (18, 19), (19, 20)]]   # pinky
    COLORS = ["#4fd6c4", "#5aa9ff", "#9b8cff", "#ff7eb6", "#ffb454"]
    kp = np.asarray(kp, float)
    T = len(kp)
    idx = np.linspace(0, T - 1, n).round().astype(int)
    rows = int(np.ceil(n / cols))
    # consistent scale across all cells
    span = max(float(np.abs(kp - kp.mean(1, keepdims=True)).max()), 1e-3) * 1.05
    fig = plt.figure(figsize=(cols * 2.15, rows * 2.35), facecolor=BG)
    for c, fr in enumerate(idx):
        ax = fig.add_subplot(rows, cols, c + 1, projection="3d")
        P = kp[fr] - kp[fr].mean(0)
        for fi, bones in enumerate(FINGERS):
            for a, b in bones:
                ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], [P[a, 2], P[b, 2]],
                        color=COLORS[fi], lw=2.1, solid_capstyle="round")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="#eaf2ff", s=9, depthshade=False, edgecolors="none")
        ax.scatter(*P[0], c="#ffffff", s=26, depthshade=False, edgecolors="none")  # wrist
        ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_zlim(-span, span)
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
        ax.set_facecolor(BG); ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)
        ax.text2D(0.04, 0.04, f"t={fr}", transform=ax.transAxes, color="#5f7a9e", fontsize=8)
    fig.suptitle(title, color="#cdd9ec", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}  ({n} skeleton poses, {rows}x{cols} grid)")
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_factory import load_input
    args = sys.argv[2:]
    grip_mode = ("pinch" if "pinch" in args else
                 ("retarget" if "retarget" in args else "human"))   # default: human
    mode = ("physics" if "physics" in args else
            ("overlay" if "overlay" in args else
             ("kinematic" if "kinematic" in args else "mimic")))   # default: faithful mimic
    orbit = "orbit" in args
    az, el, zoom = -120.0, -12.0, 1.0
    for a in args:                                   # az=-90  el=10  zoom=1.4
        if a.startswith("az="): az = float(a[3:])
        elif a.startswith("el="): el = float(a[3:])
        elif a.startswith("zoom="): zoom = float(a[5:])
    consumed = ("human", "retarget", "pinch", "mimic", "physics", "kinematic", "overlay", "orbit", "trio")
    rest = [a for a in args if a not in consumed and not a.startswith(("az=", "el=", "zoom="))]
    nums = [a for a in rest if a.lstrip("-").isdigit()]
    stride = int(nums[0]) if nums else 3
    hand = next((a for a in rest if not a.lstrip("-").isdigit()), "leap")
    hd = load_input(sys.argv[1]); demo, kp = hd.to_demo()
    if "trio" in args:
        print(f"Loaded {demo.T} frames, object '{hd.object_name}'. Rendering TRIO "
              f"(leap+allegro+shadow, az={az} el={el} zoom={zoom}) ...")
        render_trio(demo, kp, out="forge_demo.mp4", stride=stride, azim=az, elev=el, zoom=zoom)
        return
    print(f"Loaded {demo.T} frames, object '{hd.object_name}'. Rendering '{hand}' "
          f"(grip={grip_mode}, {mode}, az={az} el={el} zoom={zoom}{' orbit' if orbit else ''}) ...")
    render_pick(demo, kp, hand_name=hand, stride=stride, mode=mode, grip_mode=grip_mode,
                azim=az, elev=el, zoom=zoom, orbit=orbit)


if __name__ == "__main__":
    main()
