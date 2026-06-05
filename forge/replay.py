"""MuJoCo CPU physics for forge, plus the analytic force-closure gate.

`force_closure_metric` is the heart of the pipeline: a closed-form Ferrari-Canny
Q1 grasp-quality test that screens candidate grasps WITHOUT any dynamic rollout,
which is what lets the whole factory run on a laptop CPU. `ReplayScene` provides
the full dynamic replay (wrist position-controlled via a mocap body, fingers
position-actuated, object free under gravity on a table) for studying grasps
under contact dynamics.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco
from . import se3


@dataclass
class ReplayResult:
    success: bool
    lift_height: float
    max_obj_speed: float
    object_path: np.ndarray




def force_closure_metric(contact_pos, contact_normal, com, mu=0.8, n_edges=6, lam=0.05):
    """Ferrari-Canny Q1 grasp-quality metric -- the force-closure gate.

    Closed form, CPU, no dynamics. Given the contact points and surface normals of
    a squeezed grasp, this answers the core stability question analytically: can the
    grasp resist an external wrench pushing in ANY direction?

    Method:
      1. Linearize each contact's Coulomb friction cone into `n_edges` unit force
         rays (friction coefficient `mu`).
      2. Map each force ray to a 6D wrench [force ; torque/lam] about the object
         COM, giving the grasp's primitive wrench set.
      3. Take the convex hull of those wrenches (the Grasp Wrench Space).
      4. The grasp achieves force closure iff the hull strictly contains the origin;
         the margin `eps` is the distance from the origin to the nearest hull facet,
         i.e. the largest-magnitude disturbance the grasp is guaranteed to resist
         (the Ferrari-Canny Q1 quality).

    Returns (force_closure: bool, eps: float, n_contacts: int).
    """
    from scipy.spatial import ConvexHull
    contact_pos = np.asarray(contact_pos, float); contact_normal = np.asarray(contact_normal, float)
    N = len(contact_pos)
    if N < 3:
        return False, 0.0, N
    W = []
    for i in range(N):
        n = contact_normal[i] / (np.linalg.norm(contact_normal[i]) + 1e-9)
        t0 = np.cross(n, [1, 0, 0])
        if np.linalg.norm(t0) < 1e-4: t0 = np.cross(n, [0, 1, 0])
        t0 /= np.linalg.norm(t0) + 1e-9
        t1 = np.cross(n, t0)
        r = contact_pos[i] - com
        for k in range(n_edges):
            a = 2 * np.pi * k / n_edges
            f = n + mu * (np.cos(a) * t0 + np.sin(a) * t1)
            f /= np.linalg.norm(f) + 1e-9
            W.append(np.concatenate([f, np.cross(r, f) / lam]))
    W = np.array(W)
    try:
        hull = ConvexHull(W)
    except Exception:
        return False, 0.0, N
    off = hull.equations[:, -1]; nrm = np.linalg.norm(hull.equations[:, :6], axis=1)
    if np.all(off <= 1e-9):
        eps = float(np.min(-off / (nrm + 1e-12)))
        return eps > 1e-4, eps, N
    return False, 0.0, N


class ReplayScene:
    def __init__(self, hand_xml_path, palm_body, obj_size=(0.02, 0.02, 0.02),
                 obj_mass=0.05, table_z=0.0, obj_type="box"):
        spec = mujoco.MjSpec.from_file(hand_xml_path)
        # The mocap "wrist controller" must be a direct child of world. For hands
        # whose palm is buried under a forearm/wrist chain (e.g. Shadow), drive
        # the chain's ROOT body and later correct for the fixed root->palm offset.
        tmp_m = mujoco.MjModel.from_xml_path(hand_xml_path)
        pid = tmp_m.body(palm_body).id
        root_id = pid
        while tmp_m.body_parentid[root_id] != 0:
            root_id = tmp_m.body_parentid[root_id]
        root_body = tmp_m.body(root_id).name
        tmp_d = mujoco.MjData(tmp_m)
        mujoco.mj_kinematics(tmp_m, tmp_d)
        root_pose = se3.make_pose(tmp_d.xpos[root_id], tmp_d.xmat[root_id].reshape(3, 3))
        palm_pose = se3.make_pose(tmp_d.xpos[pid], tmp_d.xmat[pid].reshape(3, 3))
        self.root_to_palm = se3.pose_in_A_to_pose_in_B(palm_pose, se3.pose_inv(root_pose))

        # make the root a kinematically-driven mocap body
        spec.body(root_body).mocap = True
        self._root_body = root_body
        # free object
        obj = spec.worldbody.add_body(name="object")
        obj.add_freejoint()
        g = obj.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE if obj_type == "sphere" else mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(obj_size); g.mass = obj_mass
        g.rgba = [0.85, 0.3, 0.2, 1]; g.condim = 4
        g.friction = [1.2, 0.02, 0.001]
        g.solref = [0.01, 1.0]; g.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
        # table
        t = spec.worldbody.add_geom()
        t.type = mujoco.mjtGeom.mjGEOM_PLANE
        t.size = [1, 1, 0.1]; t.pos = [0, 0, table_z]
        t.condim = 4; t.friction = [1.2, 0.02, 0.001]

        self.model = spec.compile()
        # numerical stability for stiff contact + firm grip on CPU
        self.model.opt.timestep = 5e-4
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        kp = 16.0
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = kp
            self.model.actuator_biasprm[i, 1] = -kp
        for j in range(self.model.njnt):
            adr = self.model.jnt_dofadr[j]
            self.model.dof_damping[adr] = max(self.model.dof_damping[adr], 0.05)
        self.data = mujoco.MjData(self.model)
        self.palm_bid = self.model.body(palm_body).id
        self.mocap_id = self.model.body(self._root_body).mocapid[0]
        self.obj_bid = self.model.body("object").id
        self.obj_qadr = self.model.jnt_qposadr[self.model.body("object").jntadr[0]]
        self.table_z = table_z
        self.obj_half_z = obj_size[2]

    def rollout(self, wrist_poses, finger_qpos, obj_pose, substeps=32,
                lift_thresh=0.03, settle=50, overclose=0.5):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        T = len(wrist_poses)
        nu = m.nu

        op, oR = se3.unmake_pose(obj_pose)
        op = op.copy(); op[2] = self.table_z + self.obj_half_z + 0.002
        d.qpos[self.obj_qadr:self.obj_qadr + 3] = op
        d.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = se3.mat2quat(oR)

        root0 = wrist_poses[0] @ se3.pose_inv(self.root_to_palm)
        p0, R0 = se3.unmake_pose(root0)
        d.mocap_pos[self.mocap_id] = p0
        d.mocap_quat[self.mocap_id] = se3.mat2quat(R0)
        # squeeze-to-contact: drive fingers further closed than the retargeted gap,
        # in the SAME direction the grasp was closing. Contact stops them, so they
        # wrap the object regardless of its exact size and press with real force.
        open_q = finger_qpos[0].copy()
        def squeezed(qf):
            tgt = qf + overclose * (qf - open_q)
            return np.clip(tgt, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        if finger_qpos.shape[1] == nu:
            d.ctrl[:] = squeezed(finger_qpos[0])
        mujoco.mj_forward(m, d)
        z0 = d.xpos[self.obj_bid][2]

        path = []; max_speed = 0.0
        for t in range(T + settle):
            ti = min(t, T - 1)
            roott = wrist_poses[ti] @ se3.pose_inv(self.root_to_palm)
            wp, wR = se3.unmake_pose(roott)
            d.mocap_pos[self.mocap_id] = wp
            d.mocap_quat[self.mocap_id] = se3.mat2quat(wR)
            if finger_qpos.shape[1] == nu:
                d.ctrl[:] = squeezed(finger_qpos[ti])
            for _ in range(substeps):
                mujoco.mj_step(m, d)
            path.append(d.xpos[self.obj_bid].copy())
            max_speed = max(max_speed, float(np.linalg.norm(d.cvel[self.obj_bid])))

        path = np.array(path)
        lift = float(d.xpos[self.obj_bid][2] - z0)
        sane = (max_speed < 12.0) and np.all(np.isfinite(path))
        return ReplayResult(bool(sane and lift > lift_thresh), lift, max_speed, path)


    def grasp_probe(self, wrist_pose, finger_qpos, obj_pose, open_q=None, overclose=0.5, hold=200):
        """Seat the hand at a single (wrist, finger) pose with the object at
        obj_pose, step in place, and report contact geometry. Diagnostic only."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        op, oR = se3.unmake_pose(obj_pose)
        op = op.copy(); op[2] = max(op[2], self.table_z + self.obj_half_z + 0.002)
        d.qpos[self.obj_qadr:self.obj_qadr + 3] = op
        d.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = se3.mat2quat(oR)
        root = wrist_pose @ se3.pose_inv(self.root_to_palm)
        wp, wR = se3.unmake_pose(root)
        d.mocap_pos[self.mocap_id] = wp; d.mocap_quat[self.mocap_id] = se3.mat2quat(wR)
        if finger_qpos.shape[0] == m.nu:
            tgt = finger_qpos if open_q is None else finger_qpos + overclose * (finger_qpos - open_q)
            d.ctrl[:] = np.clip(tgt, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_forward(m, d)
        for _ in range(hold):
            mujoco.mj_step(m, d)
        # contacts between any hand geom and the object geom
        obj_gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.obj_bid]
        n_contact = 0
        for c in range(d.ncon):
            con = d.contact[c]
            if (con.geom1 in obj_gid) ^ (con.geom2 in obj_gid):
                n_contact += 1
        # min distance from object center to any hand geom (rough, via body xpos)
        obj_c = d.xpos[self.obj_bid]
        dists = [np.linalg.norm(d.xpos[b] - obj_c) for b in range(m.nbody)
                 if b not in (0, self.obj_bid)]
        return {"n_contact": n_contact, "obj_z": float(d.xpos[self.obj_bid][2]),
                "min_body_obj_dist_mm": float(min(dists) * 1000) if dists else -1,
                "obj_half": self.obj_half_z}


    def grasp_lift(self, wrist_poses, finger_qpos, obj_world_traj, substeps=32,
                   lift_thresh=0.03, settle=120, overclose=0.5):
        """Initialize at the captured grasp (hand + object co-located as recorded),
        establish the grip in place, then roll out ONLY the lift portion. Avoids the
        open-loop approach knocking the object. obj_world_traj: [T,4,4] real object
        world poses (used to seat the object at the grasp frame)."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        T = len(wrist_poses); nu = m.nu
        open_q = finger_qpos[0].copy()
        def squeezed(qf):
            return np.clip(qf + overclose * (qf - open_q),
                           m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        g = int(np.argmax(np.linalg.norm(finger_qpos - open_q, axis=1)))  # tightest grasp

        # seat hand at grasp frame; object at its real grasp-frame world pose
        root = wrist_poses[g] @ se3.pose_inv(self.root_to_palm)
        wp, wR = se3.unmake_pose(root)
        d.mocap_pos[self.mocap_id] = wp; d.mocap_quat[self.mocap_id] = se3.mat2quat(wR)
        if finger_qpos.shape[1] == nu:
            d.ctrl[:] = squeezed(finger_qpos[g])
        op, oR = se3.unmake_pose(obj_world_traj[g])
        d.qpos[self.obj_qadr:self.obj_qadr + 3] = op
        d.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = se3.mat2quat(oR)
        mujoco.mj_forward(m, d)

        # establish grip in place (no wrist motion) so contact forms before lifting
        for _ in range(settle):
            if finger_qpos.shape[1] == nu:
                d.ctrl[:] = squeezed(finger_qpos[g])
            mujoco.mj_step(m, d)
        z_grip = float(d.xpos[self.obj_bid][2])

        # lift: replay wrist from grasp frame to end, grip held
        path = []; max_speed = 0.0
        for t in range(g, T):
            root = wrist_poses[t] @ se3.pose_inv(self.root_to_palm)
            wp, wR = se3.unmake_pose(root)
            d.mocap_pos[self.mocap_id] = wp; d.mocap_quat[self.mocap_id] = se3.mat2quat(wR)
            if finger_qpos.shape[1] == nu:
                d.ctrl[:] = squeezed(finger_qpos[t])
            for _ in range(substeps):
                mujoco.mj_step(m, d)
            path.append(d.xpos[self.obj_bid].copy())
            max_speed = max(max_speed, float(np.linalg.norm(d.cvel[self.obj_bid])))
        path = np.array(path) if path else np.zeros((1, 3))
        zf = float(d.xpos[self.obj_bid][2])
        lift = zf - z_grip
        held = zf > z_grip - 0.02     # didn't drop out of the hand
        sane = (max_speed < 12.0) and np.all(np.isfinite(path))
        return ReplayResult(bool(sane and held and lift > lift_thresh), lift, max_speed, path)

    def grasp_quality(self, wrist_poses, finger_qpos, obj_rel, g=None, overclose=0.5,
                      settle=120, mu=0.8):
        """CPU force-closure gate (Ferrari-Canny). Initialize at the captured grasp,
        squeeze to form contact, read hand<->object contacts, score force closure.
        No dynamic lift. Returns (force_closure, eps, n_contacts). `g` is the grasp
        frame; if None, falls back to the frame of maximum finger closure."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        nu = m.nu
        open_q = finger_qpos[0].copy()
        if g is None:
            g = int(np.argmax(np.linalg.norm(finger_qpos - open_q, axis=1)))
        g = int(np.clip(g, 0, len(wrist_poses) - 1))
        def squeezed(qf):
            return np.clip(qf + overclose * (qf - open_q),
                           m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        root = wrist_poses[g] @ se3.pose_inv(self.root_to_palm)
        wp, wR = se3.unmake_pose(root)
        d.mocap_pos[self.mocap_id] = wp; d.mocap_quat[self.mocap_id] = se3.mat2quat(wR)
        if finger_qpos.shape[1] == nu:
            d.ctrl[:] = squeezed(finger_qpos[g])
        obj_world = wrist_poses[g] @ obj_rel               # co-locate as captured
        op, oR = se3.unmake_pose(obj_world)
        d.qpos[self.obj_qadr:self.obj_qadr + 3] = op
        d.qpos[self.obj_qadr + 3:self.obj_qadr + 7] = se3.mat2quat(oR)
        mujoco.mj_forward(m, d)
        for _ in range(settle):
            if finger_qpos.shape[1] == nu:
                d.ctrl[:] = squeezed(finger_qpos[g])
            mujoco.mj_step(m, d)
        # extract hand<->object contacts (position + inward normal)
        obj_gid = set(gi for gi in range(m.ngeom) if m.geom_bodyid[gi] == self.obj_bid)
        com = d.xpos[self.obj_bid].copy()
        pts, nrms = [], []
        for c in range(d.ncon):
            con = d.contact[c]
            if (con.geom1 in obj_gid) ^ (con.geom2 in obj_gid):
                pos = con.pos.copy()
                n = con.frame[:3].copy()                       # contact normal
                if np.dot(com - pos, n) < 0: n = -n            # orient INTO object
                pts.append(pos); nrms.append(n)
        if len(pts) < 3:
            return False, 0.0, len(pts)
        return force_closure_metric(np.array(pts), np.array(nrms), com, mu=mu)



    def dynamic_lift(self, wrist_poses, finger_qpos, obj_rel, g, lift_h=0.12,
                     establish=220, lift_steps=520, hold=120, overclose=0.8,
                     mu=1.6, slip_tol=0.015):
        """CLOSED-LOOP dynamic lift. Establish the grip at the captured grasp, then
        raise the wrist straight up while the object is held ONLY by simulated
        contact friction. A slip-adaptive controller tightens the grip when the
        object starts sliding out of the hand. Records the full physics rollout.
        Returns dict with success, lift, and recorded wrist/finger/object trajs."""
        m, d = self.model, self.data
        for gi in range(m.ngeom):                       # rubber-like fingertip friction
            if m.geom_bodyid[gi] == self.obj_bid:
                m.geom_friction[gi] = [mu, 0.05, 0.002]
        mujoco.mj_resetData(m, d)
        nu = m.nu
        open_q = finger_qpos[0].copy()
        g = int(np.clip(g, 0, len(wrist_poses) - 1))

        def seat(wrist_pose):
            root = wrist_pose @ se3.pose_inv(self.root_to_palm)
            wp, wR = se3.unmake_pose(root)
            d.mocap_pos[self.mocap_id] = wp; d.mocap_quat[self.mocap_id] = se3.mat2quat(wR)

        def squeezed(extra):
            return np.clip(finger_qpos[g] + extra * (finger_qpos[g] - open_q),
                           m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])

        seat(wrist_poses[g])
        if finger_qpos.shape[1] == nu:
            d.ctrl[:] = squeezed(overclose)
        obj_world = wrist_poses[g] @ obj_rel
        op, oR = se3.unmake_pose(obj_world)
        d.qpos[self.obj_qadr:self.obj_qadr+3] = op
        d.qpos[self.obj_qadr+3:self.obj_qadr+7] = se3.mat2quat(oR)
        mujoco.mj_forward(m, d)

        rec_w, rec_f, rec_o = [], [], []
        # establish grip in place
        extra = overclose
        for _ in range(establish):
            if finger_qpos.shape[1] == nu: d.ctrl[:] = squeezed(extra)
            mujoco.mj_step(m, d)
        palm0 = d.xpos[self.palm_bid][2]
        obj0 = d.xpos[self.obj_bid][2]
        rel0 = obj0 - palm0                              # object height relative to palm

        base_p = wrist_poses[g][:3, 3].copy()
        base_R = wrist_poses[g][:3, :3].copy()
        def record():
            rec_w.append(se3.make_pose(d.mocap_pos[self.mocap_id].copy(),
                                       se3.quat2mat(d.mocap_quat[self.mocap_id].copy())))
            rec_f.append(d.qpos[[m.jnt_qposadr[m.actuator_trnid[i,0]] for i in range(nu)]].copy())
            oq = d.qpos[self.obj_qadr:self.obj_qadr+7].copy()
            P = se3.make_pose(oq[:3], se3.quat2mat(oq[3:7])); rec_o.append(P)

        # lift straight up with ease-in; tighten grip if the object slips down
        for s in range(lift_steps):
            a = 0.5 - 0.5*np.cos(np.pi*(s+1)/lift_steps)   # smooth 0->1
            tgt = base_p + np.array([0, 0, lift_h*a])
            seat(se3.make_pose(tgt, base_R))
            slip = (d.xpos[self.palm_bid][2] + rel0) - d.xpos[self.obj_bid][2]
            if slip > slip_tol:                            # object lagging -> grip harder
                extra = min(extra + 0.05, 3.0)
            if finger_qpos.shape[1] == nu: d.ctrl[:] = squeezed(extra)
            mujoco.mj_step(m, d)
            if s % 8 == 0: record()
        for _ in range(hold):                              # hold at top
            if finger_qpos.shape[1] == nu: d.ctrl[:] = squeezed(extra)
            mujoco.mj_step(m, d)
        record()

        obj_lift = float(d.xpos[self.obj_bid][2] - obj0)
        final_slip = float((d.xpos[self.palm_bid][2] + rel0) - d.xpos[self.obj_bid][2])
        held = final_slip < 0.04 and np.all(np.isfinite(d.qpos))
        success = held and obj_lift > 0.6 * lift_h
        return {"success": bool(success), "lift": obj_lift, "final_slip": final_slip,
                "grip_extra": float(extra),
                "wrist": np.array(rec_w), "fingers": np.array(rec_f), "object": np.array(rec_o)}
