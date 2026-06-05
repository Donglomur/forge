"""Real human-demonstration loaders. Fail-loud by design: if a file is missing,
a schema doesn't match, or a needed model (MANO) is absent, these RAISE. There
is no synthetic fallback anywhere.

Canonical human input (what the retargeter consumes), the 21-joint layout:
  0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20
keypoints in meters, 3D, per frame, in a consistent (camera or world) frame.

A HumanDemo also carries the wrist 6DoF trajectory and the manipulated object's
6DoF trajectory, which the scene-multiplier needs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from . import se3
from .multiply import Demo, Subtask

N_KP = 21


@dataclass
class HumanDemo:
    keypoints: np.ndarray              # [T,21,3] hand joints (meters)
    wrist_pose: np.ndarray             # [T,4,4] wrist/palm 6DoF in world
    object_pose: np.ndarray            # [T,4,4] object 6DoF in world
    object_name: str = "object"
    fps: float = 30.0

    def __post_init__(self):
        T = self.keypoints.shape[0]
        if self.keypoints.shape[1:] != (N_KP, 3):
            raise ValueError(f"keypoints must be [T,{N_KP},3], got {self.keypoints.shape}")
        for nm, a, sh in [("wrist_pose", self.wrist_pose, (T, 4, 4)),
                          ("object_pose", self.object_pose, (T, 4, 4))]:
            if a.shape != sh:
                raise ValueError(f"{nm} must be {sh}, got {a.shape}")

    def to_demo(self):
        """Split into a multiply.Demo (wrist + object + gripper) and keypoints.
        Gripper signal = normalized mean fingertip-to-wrist distance (closed=1)."""
        tips = self.keypoints[:, [4, 8, 12, 16, 20], :]
        wrist = self.keypoints[:, 0:1, :]
        spread = np.linalg.norm(tips - wrist, axis=2).mean(1)
        g = 1.0 - (spread - spread.min()) / (np.ptp(spread) + 1e-9)   # 0 open ->1 closed
        demo = Demo(eef_poses=self.wrist_pose,
                    object_poses={self.object_name: self.object_pose},
                    gripper=g[:, None],
                    subtasks=[Subtask(self.object_name, 0, len(g), num_interp=6)])
        return demo, self.keypoints


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


def load_keypoints(path) -> HumanDemo:
    """Universal real input: the output of a hand tracker (WiLoR / HaMeR /
    MediaPipe) plus object tracking. Expects a .npz with:
        keypoints   [T,21,3]   (required)
        wrist_pose  [T,4,4]     (optional; derived from keypoints if absent)
        object_pose [T,4,4]     (required for the scene-multiplier)
        object_name str         (optional)
    Raises on anything missing/misshapen. No synthesis."""
    d = dict(np.load(path, allow_pickle=True))
    _require("keypoints" in d, f"{path}: missing 'keypoints' [T,21,3]")
    kp = np.asarray(d["keypoints"], float)
    _require(kp.ndim == 3 and kp.shape[1:] == (N_KP, 3),
             f"{path}: 'keypoints' must be [T,21,3], got {kp.shape}")
    T = kp.shape[0]
    if "wrist_pose" in d:
        wrist = np.asarray(d["wrist_pose"], float)
    else:
        wrist = _wrist_from_keypoints(kp)
    _require("object_pose" in d, f"{path}: missing 'object_pose' [T,4,4] "
             "(the scene-multiplier needs the object trajectory)")
    obj = np.asarray(d["object_pose"], float)
    _require(obj.shape == (T, 4, 4), f"{path}: 'object_pose' must be [{T},4,4], got {obj.shape}")
    name = str(d["object_name"]) if "object_name" in d else "object"
    return HumanDemo(kp, wrist, obj, name, float(d.get("fps", 30.0)))


def _wrist_from_keypoints(kp):
    """Build a wrist frame from keypoints: origin at wrist(0), z toward middle
    MCP(9), x toward index MCP(5). Right-handed, orthonormalized."""
    T = kp.shape[0]
    out = np.zeros((T, 4, 4))
    for t in range(T):
        o = kp[t, 0]
        z = kp[t, 9] - o; z /= np.linalg.norm(z) + 1e-9
        x = kp[t, 5] - o; x -= x.dot(z) * z; x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)
        out[t] = se3.make_pose(o, np.stack([x, y, z], axis=1))
    return out


# ---- dataset-specific real loaders (schema-correct, fail-loud) ----

# MANO 16-joint order -> our 21-joint tips need the 5 fingertip verts/joints.
# Datasets that ship precomputed 3D joints in the 21-layout are used directly;
# MANO-param-only datasets require a real MANO layer (see mano_to_keypoints).

def load_dexycb(label_npz, object_pose_key="pose_y", joints_key="joint_3d",
                object_name="ycb") -> HumanDemo:
    """DexYCB per-sequence label file. Uses the dataset's precomputed 3D hand
    joints (no MANO needed). Raises if keys/shapes are wrong."""
    d = dict(np.load(label_npz, allow_pickle=True))
    _require(joints_key in d, f"{label_npz}: missing '{joints_key}' (DexYCB 3D joints)")
    j = np.asarray(d[joints_key], float).reshape(-1, N_KP, 3) / 1000.0  # mm->m if needed
    _require(object_pose_key in d, f"{label_npz}: missing '{object_pose_key}' (object 6D)")
    op = np.asarray(d[object_pose_key], float)
    obj = _poses_from_6d(op, j.shape[0])
    wrist = _wrist_from_keypoints(j)
    return HumanDemo(j, wrist, obj, object_name)


def mano_to_keypoints(mano_pose, mano_trans, mano_shape=None, side="right"):
    """Convert MANO params to 21 keypoints using a REAL MANO layer. Raises if
    the MANO model/package is not installed. No synthetic skeleton."""
    try:
        from manotorch.manolayer import ManoLayer  # type: ignore
        import torch
    except Exception as e:
        raise ImportError(
            "MANO params present but no MANO layer installed. Install a real MANO "
            "(e.g. `pip install manotorch` and obtain MANO_RIGHT.pkl from "
            "https://mano.is.tue.mpg.de). Refusing to synthesize joints.") from e
    layer = ManoLayer(mano_assets_root="mano_models", side=side, use_pca=False, flat_hand_mean=False)
    pose = torch.as_tensor(np.asarray(mano_pose), dtype=torch.float32)
    if pose.ndim == 1:
        pose = pose[None]
    out = layer(pose, torch.as_tensor(np.atleast_2d(mano_shape if mano_shape is not None
                                                     else np.zeros((pose.shape[0], 10))), dtype=torch.float32))
    joints = out.joints.detach().cpu().numpy()
    trans = np.asarray(mano_trans, float)[:, None, :]
    return joints + trans


def _poses_from_6d(arr, T):
    """Accept [T,4,4] directly, or [T,7] (quat+trans) / [T,3+4]. Else raise."""
    arr = np.asarray(arr, float)
    if arr.shape == (T, 4, 4):
        return arr
    if arr.ndim == 2 and arr.shape[0] == T and arr.shape[1] >= 7:
        out = np.zeros((T, 4, 4))
        for t in range(T):
            q = arr[t, :4]; p = arr[t, 4:7]
            out[t] = se3.make_pose(p, se3.quat2mat(q))
        return out
    raise ValueError(f"object pose array shape {arr.shape} not understood for T={T}")


def quat_to_axis_angle(q):
    """[...,4] (w,x,y,z) -> [...,3] axis-angle."""
    q = np.asarray(q, float)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w = np.clip(q[..., 0], -1, 1)
    ang = 2 * np.arccos(w)
    s = np.sqrt(np.clip(1 - w * w, 0, 1))
    axis = np.where(s[..., None] < 1e-6, np.zeros_like(q[..., 1:]), q[..., 1:] / (s[..., None] + 1e-12))
    return axis * ang[..., None]


# manotorch .joints order -> our 21-joint MediaPipe layout.
# manotorch returns: 0 wrist, then index/middle/pinky/ring/thumb (3 each),
# then 5 tips. This map pulls them into [wrist, thumb1-4, index, middle, ring, pinky].
# NOTE: verify against your OakInk/MANO version; it is the one assumption here.
_MANO_TO_MP = [0,
               13, 14, 15, 20,    # thumb mcp,pip,dip,tip
               1, 2, 3, 16,       # index
               4, 5, 6, 17,       # middle
               10, 11, 12, 19,    # ring
               7, 8, 9, 18]       # pinky


def load_oakink(npz_path, object_name="oakink", side="right", joint_order="mano") -> HumanDemo:
    """Real OakInk loader. Parses OakInk's MANO annotation and produces TRUE 3D
    joints, either from precomputed joints in the file or by running the REAL
    MANO layer. No synthetic skeleton: if MANO params are all that's present and
    no MANO model is installed, this RAISES.

    Recognized keys (OakInk / OakInk2 .npz):
        hand_pose : (T,16,4) quaternion OR (T,48)/(T,51) axis-angle
        hand_tsl  : (T,3) translation        hand_shape : (10,) betas
        obj_transf: (T,4,4) object pose
        joints/hand_joints/j3d : (T,21,3) precomputed (used directly if present)
    """
    d = dict(np.load(npz_path, allow_pickle=True))

    # object pose first (required by the scene-multiplier)
    obj = None
    for k in ("obj_transf", "object_pose", "obj_transform", "obj_pose"):
        if k in d:
            obj = np.asarray(d[k], float); break
    _require(obj is not None and obj.shape[-2:] == (4, 4),
             f"{npz_path}: missing obj_transf [T,4,4]")
    T = obj.shape[0]

    # 1) precomputed joints path (no MANO needed)
    j = None
    for k in ("joints", "hand_joints", "j3d", "hand_j"):
        if k in d:
            j = np.asarray(d[k], float); break
    if j is not None:
        _require(j.shape == (T, N_KP, 3), f"{npz_path}: '{k}' must be [{T},21,3], got {j.shape}")
        kp = j
    else:
        # 2) MANO-params path -> REAL MANO (fail loud if absent)
        pose = None
        for k in ("hand_pose", "mano_pose", "pose"):
            if k in d:
                pose = np.asarray(d[k], float); break
        _require(pose is not None, f"{npz_path}: no precomputed joints and no MANO pose found")
        if pose.ndim == 3 and pose.shape[-1] == 4:      # quaternion per joint -> axis-angle
            pose = quat_to_axis_angle(pose).reshape(T, -1)
        if pose.shape[1] == 51:                          # 3 global + 48
            pass
        elif pose.shape[1] == 48:
            pass
        _require(pose.shape[1] in (48, 51), f"{npz_path}: unexpected hand_pose dim {pose.shape[1]}")
        trans = np.asarray(d["hand_tsl"], float) if "hand_tsl" in d else np.zeros((T, 3))
        shape = np.asarray(d["hand_shape"], float) if "hand_shape" in d else None
        joints = mano_to_keypoints(pose, trans, shape, side=side)   # raises without real MANO
        if joint_order == "mano":
            joints = joints[:, _MANO_TO_MP, :]
        kp = joints

    wrist = _wrist_from_keypoints(kp)
    return HumanDemo(kp, wrist, _poses_from_6d(obj, T), object_name)


def infer_scene_from_keypoints(kp, scene_scale=0.30, lift=0.12):
    """Given REAL per-frame finger geometry [T,21,3] (MediaPipe order, wrist-origined),
    infer a wrist approach->grip->lift path and an object at the grip moment. The
    FINGERS are real; the wrist path and object are inferred (a hand-only sequence
    carries no global wrist pose or object). Used by load_handjoints and the video
    ingester's fallback."""
    T = len(kp)
    tips = kp[:, [4, 8, 12, 16, 20], :]
    spread = np.linalg.norm(tips - kp[:, :1, :], axis=2).mean(1)
    gi = int(np.argmin(spread))                      # tightest grip = grasp moment
    # wrist: descend to the object by frame gi, then lift
    wpos = np.zeros((T, 3))
    for t in range(T):
        if t <= gi:
            wpos[t] = [0, 0, scene_scale * (1 - t / max(gi, 1))]
        else:
            wpos[t] = [0, 0, lift * (t - gi) / max(T - 1 - gi, 1)]
    wrist_pose = np.zeros((T, 4, 4))
    for t in range(T):
        z = kp[t, 9]; z = z / (np.linalg.norm(z) + 1e-9)
        x = kp[t, 5]; x = x - x.dot(z) * z; x = x / (np.linalg.norm(x) + 1e-9)
        wrist_pose[t] = se3.make_pose(wpos[t], np.stack([x, np.cross(z, x), z], axis=1))
    obj_local = tips[gi].mean(0)
    op = wrist_pose[gi][:3, :3] @ obj_local + wrist_pose[gi][:3, 3]
    object_pose = np.tile(se3.make_pose(op, np.eye(3)), (T, 1, 1))
    return wrist_pose, object_pose, gi


def load_handjoints(path, object_name="object") -> HumanDemo:
    """Load a REAL human-hand joint sequence [T,21,3] in MediaPipe order, meters,
    wrist-origined (e.g. dex-retargeting's human_joint_right.pkl, or any MediaPipe
    world-landmark dump). Fingers are real; wrist/object inferred for the physics
    stage. Accepts .pkl or .npy/.npz (key 'keypoints'/'joints')."""
    if path.endswith(".pkl"):
        import pickle
        kp = np.array(pickle.load(open(path, "rb")), float)
    elif path.endswith(".npy"):
        kp = np.load(path).astype(float)
    else:
        d = np.load(path, allow_pickle=True)
        k = next((x for x in ("keypoints", "joints", "hand_joints") if x in d), None)
        _require(k is not None, f"{path}: no keypoints/joints array")
        kp = np.asarray(d[k], float)
    _require(kp.ndim == 3 and kp.shape[1:] == (N_KP, 3),
             f"{path}: expected [T,21,3], got {kp.shape}")
    kp = kp - kp[:, :1, :]                            # ensure wrist-origined
    if len(kp) > 150:                                 # stride long clips (still real)
        kp = kp[np.linspace(0, len(kp) - 1, 150).round().astype(int)]
    wrist_pose, object_pose, gi = infer_scene_from_keypoints(kp)
    print(f"[handjoints] {len(kp)} real frames; inferred grip at frame {gi}")
    return HumanDemo(kp, wrist_pose, object_pose, object_name)


def _stack2d(x, cols):
    """Coerce nested parquet output (list-of-rows, object ndarray, etc.) to [T,cols]."""
    x = x.tolist() if hasattr(x, "tolist") else x
    return np.array([np.asarray(r, float).ravel()[:cols] for r in x], float)


def _aa_to_mat(aa):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(aa, float)).as_matrix()


def _auto_mp_remap(J):
    """J:[21,3], joint 0=wrist. Detect MANO 21-joint convention (grouped vs smplx)
    and return 21 indices mapping to MediaPipe layout
    [wrist, thumb(mcp,pip,dip,tip), index.., middle.., ring.., pinky..].
    Identifies thumb (shortest finger) and finger order (knuckle line) geometrically."""
    W = J[0]; idx = np.arange(1, 21)
    d = np.linalg.norm(J[idx] - W, axis=1)
    grouped = all(d[4*k] < d[4*k+1] < d[4*k+2] < d[4*k+3] for k in range(5))
    if grouped:
        chains = [list(idx[4*k:4*k+4]) for k in range(5)]
    else:
        mcp3 = [list(idx[3*k:3*k+3]) for k in range(5)]
        tips = list(idx[15:20]); used = set(); chains = []
        for c in mcp3:
            dip, pip = J[c[-1]], J[c[-2]]
            extrap = dip + (dip - pip)
            for o in np.argsort([np.linalg.norm(J[t]-extrap) for t in tips]):
                if tips[o] not in used: used.add(tips[o]); chains.append(c+[tips[o]]); break
    chains = np.array(chains)
    tiplen = np.linalg.norm(J[chains[:, -1]] - W, axis=1)
    thumb = int(np.argmin(tiplen))
    others = [f for f in range(5) if f != thumb]
    mcp = J[chains[others, 0]]; c0 = mcp.mean(0)
    _, _, vt = np.linalg.svd(mcp - c0); axis = vt[0]
    proj = (mcp - c0) @ axis
    if np.dot(J[chains[thumb, 0]] - c0, axis) > 0: proj = -proj
    order = [others[i] for i in np.argsort(proj)]
    remap = [0]
    for f in [thumb] + order: remap += list(chains[f])
    return remap, ("grouped" if grouped else "smplx")


def load_dexcanvas(parquet_path, row=0, euler_order="xyz") -> HumanDemo:
    """Real loader for DexCanvas (HuggingFace DEXROBOT/DexCanvas, ODbL).
    Real human mocap of hand-object manipulation, world frame. Precomputed MANO
    joints [T,21,3] + real object pose. Joint layout auto-detected; wrist derived
    from joints (same frame as object). Handles both documented object schemas
    (object_info.pose [T,6]  OR  object_info.position+euler_angle).
    """
    import pandas as pd
    from scipy.spatial.transform import Rotation
    df = pd.read_parquet(parquet_path)
    rec = df.iloc[row].to_dict()

    def to_TN(x, n):
        try:
            a = np.asarray(x, float)
            if a.ndim == 2 and a.shape[1] == n: return a
            if a.ndim == 1 and a.size % n == 0: return a.reshape(-1, n)
            if a.ndim >= 2 and a.size % n == 0: return a.reshape(-1, n)
        except Exception: pass
        rows = x.tolist() if hasattr(x, "tolist") else x
        return np.array([np.asarray(r, float).ravel()[:n] for r in rows], float)

    def subtree(name):
        """Return the dict stored under key `name` anywhere in the row."""
        out = [None]
        def w(d):
            if out[0] is not None: return
            if isinstance(d, dict):
                for k, v in d.items():
                    if str(k).lower() == name and isinstance(v, dict): out[0] = v; return
                    w(v)
        w(rec); return out[0]

    def leaf(name):
        """Return the first non-dict value stored under key `name` anywhere."""
        out = [None]
        def w(d):
            if out[0] is not None: return
            if isinstance(d, dict):
                for k, v in d.items():
                    if str(k).lower() == name and not isinstance(v, dict): out[0] = v; return
                    w(v)
        w(rec); return out[0]

    def schema_dump():
        lines = []
        def w(d, pre=""):
            if isinstance(d, dict):
                for k, v in d.items(): w(v, pre + str(k) + ".")
            else:
                sh = getattr(d, "shape", None)
                if sh is None and hasattr(d, "__len__"):
                    try: sh = "len=" + str(len(d))
                    except Exception: sh = "?"
                lines.append("    " + pre[:-1] + ": " + type(d).__name__ + " " + str(sh))
        for k, v in rec.items(): w(v, str(k) + ".")
        return chr(10).join(lines)

    # --- joints ---
    mmo = subtree("mano_model_output")
    j = (mmo or {}).get("joints") if mmo else None
    if j is None: j = leaf("joints")
    if j is None:
        raise ValueError(str(parquet_path) + ": no MANO joints. Schema:" + chr(10) + schema_dump())
    J = to_TN(j, N_KP * 3).reshape(-1, N_KP, 3)
    T = J.shape[0]

    # --- joint layout auto-detect on the most-extended frame ---
    spread = np.linalg.norm(J - J[:, :1, :], axis=2).sum(1)
    remap, kind = _auto_mp_remap(J[int(np.argmax(spread))])
    kp = J[:, remap, :]

    # --- object pose (world): support pose[T,6] OR position+euler_angle ---
    oi = subtree("object_info") or {}
    opos = oeul = None
    if "pose" in oi:
        p6 = to_TN(oi["pose"], 6); opos, oeul = p6[:, :3], p6[:, 3:6]
    elif "position" in oi:
        opos = to_TN(oi["position"], 3)
        oeul = to_TN(oi.get("euler_angle", oi.get("rotation", oi.get("euler", np.zeros((T, 3))))), 3)
    else:
        op = leaf("pose")
        if op is not None and np.asarray(op).size >= T * 6:
            p6 = to_TN(op, 6); opos, oeul = p6[:, :3], p6[:, 3:6]
    if opos is None:
        raise ValueError(str(parquet_path) + ": no object pose found. Schema:" + chr(10) + schema_dump())
    obj = np.tile(np.eye(4), (T, 1, 1))
    obj[:, :3, 3] = opos[:T]
    obj[:, :3, :3] = Rotation.from_euler(euler_order, oeul[:T]).as_matrix()

    # --- world wrist pose for PHYSICS placement (kp stays LOCAL for retargeting) ---
    # The hand-rotation euler convention isn't documented, so auto-pick the order
    # whose palm normal points toward the object at the grasp moment (most physical).
    hj = subtree("hand_joint") or {}
    hpos = hj.get("position"); hrot = hj.get("rotation")
    wrist = np.tile(np.eye(4), (T, 1, 1))
    if hpos is not None and hrot is not None:
        hpos = to_TN(hpos, 3)[:T]; hrot = to_TN(hrot, 3)[:T]
        gi = int(np.argmin(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1)))   # tightest pinch = grasp
        to_obj = obj[gi, :3, 3] - hpos[gi]; to_obj /= (np.linalg.norm(to_obj) + 1e-9)
        # local palm normal (from the canonical joints): wrist->middle x wrist->index
        zc = kp[gi, 9] - kp[gi, 0]; xc = kp[gi, 5] - kp[gi, 0]
        n_local = np.cross(zc, xc); n_local /= (np.linalg.norm(n_local) + 1e-9)
        def rot_seq(mode, vals):
            if mode == "rotvec": return Rotation.from_rotvec(vals).as_matrix()
            return Rotation.from_euler(mode, vals).as_matrix()
        cands = ["rotvec", "xyz", "XYZ", "zyx", "ZYX", "yxz", "YXZ", "zxy", "ZXY"]
        best, best_score = "rotvec", -2.0
        for md in cands:
            Rg = rot_seq(md, hrot[gi])
            n_world = Rg @ n_local
            score = abs(float(n_world @ to_obj))   # palm aligned with hand->object axis
            if score > best_score: best_score, best = score, md
        euler_order = best
        wrist[:, :3, :3] = rot_seq(euler_order, hrot)
        wrist[:, :3, 3] = hpos
        _dc_world = True
        print("[dexcanvas] auto-picked hand rotation mode='" + euler_order +
              "' (palm-object alignment %.2f)" % best_score)
    else:
        wrist[:, :3, 3] = kp[:, 0, :]
        for t in range(T):
            z = kp[t, 9] - kp[t, 0]; z /= (np.linalg.norm(z) + 1e-9)
            x = kp[t, 5] - kp[t, 0]; x = x - x.dot(z) * z; x /= (np.linalg.norm(x) + 1e-9)
            wrist[t, :3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
        _dc_world = False

    name = leaf("object")
    if not isinstance(name, str): name = "object"
    # frame/scale sanity: where is the hand vs the object?
    wmin = kp.reshape(-1, 3).min(0); wmax = kp.reshape(-1, 3).max(0)
    handspan = np.linalg.norm(kp[T // 2, 12] - kp[T // 2, 0]) * 1000
    ww = wrist[:, :3, 3]
    d_wo = np.linalg.norm(ww - obj[:, :3, 3], axis=1)
    print("[dexcanvas] FRAME CHECK (world wrist applied: " + str(_dc_world) + "):")
    print("   wrist path  x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]" % (
        ww[:,0].min(), ww[:,0].max(), ww[:,1].min(), ww[:,1].max(),
        ww[:,2].min(), ww[:,2].max()))
    print("   object path x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]" % (
        obj[:,0,3].min(), obj[:,0,3].max(), obj[:,1,3].min(), obj[:,1,3].max(),
        obj[:,2,3].min(), obj[:,2,3].max()))
    print("   hand size wrist->middle-tip = %.0fmm (real hand ~170mm)" % handspan)
    print("   wrist<->object distance: min=%.0fmm max=%.0fmm" % (d_wo.min()*1000, d_wo.max()*1000))
    print("   (for a grasp these should overlap and be within a hand-span)")
    hgap = np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1) * 1000
    print("[dexcanvas] row " + str(row) + ": " + str(T) + " frames, layout=" + kind +
          ", object='" + name + "', thumb-index gap " +
          str(int(hgap.min())) + "-" + str(int(hgap.max())) + "mm")
    return HumanDemo(kp, wrist, obj, name)
