"""pinch.py -- contact-targeted antipodal pinch.

Instead of copying the human hand's GESTURE and squeezing all fingers (what the
full pick does), this reads the human's actual fingertip CONTACT POINTS at the
grasp frame, finds the antipodal pair -- the two human contacts most directly
opposed across the object -- and drives only the robot's two matching fingers
(thumb + the best-opposed finger) onto the object, force-capped, then lifts.

Why this can beat the full grasp on a small hand: a two/three-contact antipodal
pinch is a force-closure theorem (antipodal contacts with friction span the
wrench origin), and it needs only TWO fingers to reach, so it sidesteps both the
finger-count mismatch (5 human contacts -> 4 robot fingers) and the reach problem
that sinks a small hand like LEAP on a full wrap. The result is verified against
the SAME Ferrari-Canny gate the rest of forge uses -- no special-casing.
"""
import numpy as np
import mujoco
import forge.se3 as se3
from forge.replay import ReplayScene, force_closure_metric
from forge.retarget import DexPilotRetargeter

# 21-keypoint hand layout (MediaPipe / MANO ordering): fingertip indices.
HUMAN_TIP = {"index": 8, "middle": 12, "ring": 16, "pinky": 20, "thumb": 4}


def _chain_bodies(model, tip_body, palm_body):
    """Body ids on the kinematic path tip -> ... -> palm (inclusive)."""
    out = []
    b = model.body(tip_body).id
    palm = model.body(palm_body).id
    while b != 0 and b != palm:
        out.append(b)
        b = model.body_parentid[b]
    out.append(palm)
    return set(out)


def _finger_actuators(model, palm_body, tip_body):
    """Actuator indices that move the finger ending at tip_body."""
    chain = _chain_bodies(model, tip_body, palm_body)
    acts = []
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        if int(model.jnt_bodyid[jid]) in chain:
            acts.append(i)
    return acts


def choose_antipodal(kp_gf, eef_gf, obj_c, candidates):
    """Pick the (thumb, finger) human pair most opposed across the object center.

    Returns (finger_name, thumb_unit, finger_unit, opposition) where opposition in
    [-1, 1] is the dot of the two object->tip directions (-1 = perfectly opposed)."""
    kw = (eef_gf[:3, :3] @ kp_gf.T).T + eef_gf[:3, 3]
    def u(p): 
        v = p - obj_c
        return v / (np.linalg.norm(v) + 1e-9)
    th = u(kw[HUMAN_TIP["thumb"]])
    best, best_dot = None, 2.0
    for f in candidates:
        d = float(th @ u(kw[HUMAN_TIP[f]]))   # most negative = most opposed
        if d < best_dot:
            best_dot, best = d, f
    return best, th, u(kw[HUMAN_TIP[best]]), best_dot


def _static_fc(sc, grasp_q, open_q, obj_xy, meet_ids, active_act, finger_bodies,
               overclose=0.5, mu=1.8, grip_force=1.2, grip_gain=70.0, steps=250):
    """Seat the hand so the two pinch tips meet the object, close ONLY those
    fingers, settle, then read the hand<->object contacts ON THOSE TWO FINGERS
    and score force closure. Static (no lift) -- the analytic gate applied to the
    pinch alone, NOT the whole hand resting on the object."""
    m, d = sc.model, sc.data
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    for i in range(m.nu):
        m.actuator_gainprm[i, 0] = grip_gain; m.actuator_biasprm[i, 1] = -grip_gain
        m.actuator_forcelimited[i] = 1; m.actuator_forcerange[i] = [-grip_force, grip_force]
    obj_gid = [gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == sc.obj_bid]
    for gi in obj_gid:
        m.geom_friction[gi] = [mu, 0.1, 0.005]
    half = sc.obj_half_z
    cube = np.array([obj_xy[0], obj_xy[1], sc.table_z + half])
    amask = np.zeros(m.nu, bool); amask[list(active_act)] = True
    fadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]

    def set_wrist(p):
        root = se3.make_pose(p, np.eye(3)) @ se3.pose_inv(sc.root_to_palm)
        wp, wR = se3.unmake_pose(root)
        d.mocap_pos[sc.mocap_id] = wp; d.mocap_quat[sc.mocap_id] = se3.mat2quat(wR)

    mujoco.mj_resetData(m, d)
    d.qpos[sc.obj_qadr:sc.obj_qadr + 3] = cube
    d.qpos[sc.obj_qadr + 3:sc.obj_qadr + 7] = [1, 0, 0, 0]
    probe = np.array([0.0, 0.0, 0.3]); set_wrist(probe)
    d.qpos[fadr] = grasp_q; mujoco.mj_forward(m, d)
    off = np.mean([d.xpos[b] for b in meet_ids], axis=0) - probe
    set_wrist(cube - off); d.qpos[fadr] = open_q; mujoco.mj_forward(m, d)
    tgt = np.clip(grasp_q + overclose * (grasp_q - open_q), lo, hi)
    for _ in range(steps):
        c = open_q.copy(); c[amask] = tgt[amask]; d.ctrl[:] = c
        mujoco.mj_step(m, d)
    # hand<->object contacts: position + inward normal (toward object COM)
    com = d.xpos[sc.obj_bid].copy()
    pos, nrm = [], []
    for c in range(d.ncon):
        con = d.contact[c]
        on1 = con.geom1 in obj_gid; on2 = con.geom2 in obj_gid
        if on1 ^ on2:
            hand_geom = con.geom2 if on1 else con.geom1
            if int(m.geom_bodyid[hand_geom]) not in finger_bodies:
                continue                          # only the two pinch fingers count
            p = con.pos.copy(); n = con.frame[:3].copy()
            if (com - p) @ n < 0: n = -n          # point inward
            pos.append(p); nrm.append(n)
    if len(pos) < 2:
        return False, 0.0, len(pos)
    fc, eps, nc = force_closure_metric(np.array(pos), np.array(nrm), com, mu=mu)
    return fc, eps, nc


def antipodal_pinch(spec, demo, kp, gf=None, obj_xy=(0.2, 0.0), verbose=False):
    """Run the contact-targeted antipodal pinch for one hand on the real demo.

    Returns dict(finger, opposition, held_cm, final_speed, end_contacts, success,
    fc, eps, fc_contacts, active_act, pinch_tips)."""
    eef = demo.eef_poses
    objw = demo.object_poses[list(demo.object_poses)[0]]
    objw = objw if objw.ndim == 3 else np.tile(objw, (len(eef), 1, 1))
    if gf is None:
        gf = int(np.argmin(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1)))
    obj_c = objw[gf, :3, 3]

    m0 = spec.build_model()
    nu = m0.nu

    # robot finger lineup: non-thumb tips map to [index, middle, ring(, pinky)]
    th_idx = spec.thumb_idx if getattr(spec, "thumb_idx", -1) >= 0 else len(spec.tips) - 1
    nonthumb = [t for i, t in enumerate(spec.tips) if i != th_idx]
    names = ["index", "middle", "ring", "pinky"][:len(nonthumb)]
    name2tip = {names[i]: nonthumb[i] for i in range(len(nonthumb))}
    thumb_tip = spec.tips[th_idx]

    finger, th_u, f_u, opp = choose_antipodal(kp[gf], eef[gf], obj_c, names)
    pinch_tips = [thumb_tip, name2tip[finger]]

    # retargeted grasp (DexPilot matches the human grip) + open pose
    dp = DexPilotRetargeter(spec)
    fq = dp.retarget_sequence(kp)
    grasp_q = fq[gf].copy(); open_q = fq[0].copy()
    if grasp_q.shape[0] != nu:                       # e.g. shadow nq!=nu
        return {"finger": finger, "opposition": opp, "skipped": "nq!=nu"}

    # real object size from a quick probe, then a scene sized to it
    probe = ReplayScene(spec.xml_path, spec.palm, table_z=0.0)
    try:
        info = probe.grasp_probe(eef[gf], grasp_q, objw[gf], open_q=open_q)
        obj_half = float(info["obj_half"])
    except Exception:
        obj_half = 0.02

    # actuators for the two pinch fingers (others parked open)
    sc = ReplayScene(spec.xml_path, spec.palm, obj_size=(obj_half,) * 3, table_z=0.0)
    act_t = _finger_actuators(sc.model, spec.palm, thumb_tip)
    act_f = _finger_actuators(sc.model, spec.palm, name2tip[finger])
    active = sorted(set(act_t) | set(act_f))
    meet_ids = [sc.model.body(t).id for t in pinch_tips]

    res = sc.pick(grasp_q, open_q, obj_xy, list(spec.tips),
                  active_act=active, meet_tips=pinch_tips,
                  overclose=0.5, grip_force=1.3, grip_gain=75.0, mu=1.8)

    sc2 = ReplayScene(spec.xml_path, spec.palm, obj_size=(obj_half,) * 3, table_z=0.0)
    meet_ids2 = [sc2.model.body(t).id for t in pinch_tips]
    finger_bodies = (_chain_bodies(sc2.model, thumb_tip, spec.palm)
                     | _chain_bodies(sc2.model, name2tip[finger], spec.palm))
    finger_bodies -= {sc2.model.body(spec.palm).id}        # palm doesn't count as a fingertip
    fc, eps, fcn = _static_fc(sc2, grasp_q, open_q, obj_xy, meet_ids2, active, finger_bodies)

    out = {"finger": finger, "opposition": opp, "obj_half": obj_half,
           "held_cm": res["final_height"] * 100.0, "final_speed": res["final_speed"],
           "end_contacts": res["end_contacts"], "success": res["success"],
           "fc": fc, "eps": eps, "fc_contacts": fcn,
           "active_act": active, "pinch_tips": pinch_tips}
    if verbose:
        print(f"[pinch] {spec.name if hasattr(spec,'name') else '?'} "
              f"thumb+{finger} opp={opp:+.2f} held={out['held_cm']:+.1f}cm "
              f"fc={fc} eps={eps:.3f}")
    return out
