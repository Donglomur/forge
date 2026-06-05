"""Cross-embodiment retargeting, CPU. Reimplementation of dex-retargeting's
VectorOptimizer idea: match robot keypoint *vectors* (tip - palm) to human
keypoint vectors via per-frame nonlinear optimization with temporal smoothness.

Targets are MuJoCo *frames* identified by name. Real Menagerie hands have no
fingertip sites, only bodies, so we resolve each target to a site OR a body and
use the matching Jacobian (mj_jacSite / mj_jacBody). Pure CPU.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import mujoco
from scipy.optimize import minimize


@dataclass
class HandSpec:
    """How to retarget onto a particular embodiment.

    `palm` and `tips` are MuJoCo frame names (site or body). For real Menagerie
    hands these are body names (e.g. 'palm', 'ff_tip'). `xml_path` makes mesh
    assets resolve; pass it for real models. `tip_offset` optionally shifts the
    target point along the body's local frame to reach the actual fingertip.
    """
    xml: Optional[str] = None
    xml_path: Optional[str] = None
    palm: str = "palm"
    tips: List[str] = field(default_factory=list)
    human_origin: int = 0
    human_tips: List[int] = field(default_factory=list)
    scaling: float = 1.0
    tip_offset: tuple = (0.0, 0.0, 0.0)   # local offset added to each tip frame
    thumb_idx: int = -1                    # which entry in `tips` is the thumb (-1 = last)

    def build_model(self):
        if self.xml_path is not None:
            return mujoco.MjModel.from_xml_path(self.xml_path)
        if self.xml is not None:
            return mujoco.MjModel.from_xml_string(self.xml)
        raise ValueError("HandSpec needs xml_path or xml")


# frame kinds
_SITE, _BODY = 0, 1

def _resolve_frame(model, name):
    """Return (kind, id) for a MuJoCo frame given by name. Sites win if ambiguous."""
    for i in range(model.nsite):
        if model.site(i).name == name:
            return (_SITE, i)
    for i in range(model.nbody):
        if model.body(i).name == name:
            return (_BODY, i)
    raise KeyError(f"frame '{name}' not found as site or body. bodies: "
                   f"{[model.body(i).name for i in range(model.nbody)]}")


class VectorRetargeter:
    def __init__(self, spec: HandSpec, huber_delta=0.1, smooth=1e-4):
        self.spec = spec
        self.model = spec.build_model()
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.delta = huber_delta
        self.smooth = smooth
        self._loss_scale = 1e4
        self.palm = self._resolve(spec.palm)
        self.tip = [self._resolve(t) for t in spec.tips]
        self.offset = np.asarray(spec.tip_offset, float)
        lo = self.model.jnt_range[:, 0].copy(); hi = self.model.jnt_range[:, 1].copy()
        unlimited = ~self.model.jnt_limited.astype(bool)
        lo[unlimited] = -np.pi; hi[unlimited] = np.pi
        self.bounds = list(zip(lo, hi))
        self._jacp = np.zeros((3, self.model.nv))

    def _resolve(self, name):
        return _resolve_frame(self.model, name)

    def _frame_pos(self, kind_id, local_offset=None):
        kind, fid = kind_id
        if kind == _SITE:
            p = self.data.site_xpos[fid]
            R = self.data.site_xmat[fid].reshape(3, 3)
        else:
            p = self.data.xpos[fid]
            R = self.data.xmat[fid].reshape(3, 3)
        if local_offset is not None and np.any(local_offset):
            return p + R @ local_offset
        return p.copy()

    def _frame_jac(self, kind_id, local_offset=None):
        kind, fid = kind_id
        if kind == _SITE:
            mujoco.mj_jacSite(self.model, self.data, self._jacp, None, fid)
        else:
            if local_offset is not None and np.any(local_offset):
                R = self.data.xmat[fid].reshape(3, 3)
                point = self.data.xpos[fid] + R @ local_offset
                mujoco.mj_jac(self.model, self.data, self._jacp, None, point, fid)
            else:
                mujoco.mj_jacBody(self.model, self.data, self._jacp, None, fid)
        return self._jacp[:, :self.nq].copy()

    def _fk_vectors(self, q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        palm = self._frame_pos(self.palm)
        tips = np.array([self._frame_pos(t, self.offset) for t in self.tip])
        return palm, tips

    def _objective(self, q, target_vecs, q_last):
        palm, tips = self._fk_vectors(q)
        err = (tips - palm[None]) - target_vecs
        d = np.linalg.norm(err, axis=1) + 1e-9
        huber = np.where(d < self.delta, 0.5 * d * d, self.delta * (d - 0.5 * self.delta))
        S = self._loss_scale
        loss = S * (huber.sum() + self.smooth * np.sum((q - q_last) ** 2))
        Jpalm = self._frame_jac(self.palm)
        g = np.zeros(self.nq)
        for k in range(len(self.tip)):
            Jk = self._frame_jac(self.tip[k], self.offset) - Jpalm
            hp = d[k] if d[k] < self.delta else self.delta
            g += hp * (err[k] / d[k]) @ Jk
        g += 2 * self.smooth * (q - q_last)
        return float(loss), S * g

    def retarget_frame(self, target_vecs, q_init):
        res = minimize(self._objective, q_init, args=(target_vecs * self.spec.scaling, q_init),
                       jac=True, method="SLSQP", bounds=self.bounds,
                       options=dict(maxiter=200, ftol=1e-10))
        return res.x

    def retarget_sequence(self, human_keypoints):
        """human_keypoints [T, J, 3] -> robot qpos [T, nq]."""
        T = human_keypoints.shape[0]
        out = np.zeros((T, self.nq))
        q = np.clip(np.zeros(self.nq),
                    [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        for t in range(T):
            kp = human_keypoints[t]
            origin = kp[self.spec.human_origin]
            tgt = np.array([kp[i] - origin for i in self.spec.human_tips])
            q = self.retarget_frame(tgt, q)
            out[t] = q
        return out


# ---------------------------------------------------------------------------
# DexPilot-style retargeter. Reimplemented and cleaned from dex-retargeting's
# DexPilotOptimizer (MIT, (c) 2023 Yuzhe Qin); algorithm from DexPilot
# (https://arxiv.org/abs/1910.03135). Adds inter-finger pinch projection so the
# robot thumb actually closes on precision grasps. Improvements over the
# original: analytic MuJoCo Jacobians (no per-link python FK loop), consistent
# use of project_dist/escape_dist (the original hardcodes 0.03 in one branch),
# explicit thumb anchor instead of relying on finger-1 ordering, vectorized
# weighted Huber.
# ---------------------------------------------------------------------------
class DexPilotRetargeter:
    def __init__(self, spec: HandSpec, thumb_idx=None,
                 project_dist=0.03, escape_dist=0.05, eta=8e-3,
                 huber_delta=0.03, smooth=4e-4, pinch_weight=50.0, align=False):
        self.spec = spec
        self.model = spec.build_model()
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.delta = huber_delta
        self.smooth = smooth
        self._loss_scale = 1e4
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta = eta                      # target gap when pinching (meters)
        self.pinch_weight = pinch_weight
        self.offset = np.asarray(spec.tip_offset, float)
        self.R_align = np.eye(3)            # human-frame -> robot-frame calibration (set in retarget_sequence)
        self.align = align                  # frame calibration: ON for correspondence (render/trace), OFF for grasp synthesis

        self.palm = _resolve_frame(self.model, spec.palm)
        self.tip = [_resolve_frame(self.model, t) for t in spec.tips]
        n = len(self.tip)
        ti = spec.thumb_idx if thumb_idx is None else thumb_idx
        self.thumb = (n - 1) if ti < 0 else ti
        # pinch pairs: thumb -> every other finger (indices into self.tip)
        self.pairs = [(self.thumb, f) for f in range(n) if f != self.thumb]
        self.projected = np.zeros(len(self.pairs), dtype=bool)
        # human keypoint ids
        self.h_origin = spec.human_origin
        self.h_tips = list(spec.human_tips)
        self.h_thumb = self.h_tips[self.thumb]

        lo = self.model.jnt_range[:, 0].copy(); hi = self.model.jnt_range[:, 1].copy()
        unlimited = ~self.model.jnt_limited.astype(bool)
        lo[unlimited] = -np.pi; hi[unlimited] = np.pi
        self.bounds = list(zip(lo, hi))
        self._jacp = np.zeros((3, self.model.nv))

    # FK helpers (shared logic with VectorRetargeter, kept local for clarity)
    def _fk(self, q):
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def _pos(self, kind_id, off=None):
        kind, fid = kind_id
        p = self.data.site_xpos[fid] if kind == _SITE else self.data.xpos[fid]
        if off is not None and np.any(off):
            R = (self.data.site_xmat[fid] if kind == _SITE else self.data.xmat[fid]).reshape(3, 3)
            return p + R @ off
        return p.copy()

    def _jac(self, kind_id, off=None):
        kind, fid = kind_id
        if kind == _SITE:
            mujoco.mj_jacSite(self.model, self.data, self._jacp, None, fid)
        elif off is not None and np.any(off):
            R = self.data.xmat[fid].reshape(3, 3)
            mujoco.mj_jac(self.model, self.data, self._jacp, None, self.data.xpos[fid] + R @ off, fid)
        else:
            mujoco.mj_jacBody(self.model, self.data, self._jacp, None, fid)
        return self._jacp[:, :self.nq].copy()

    def _build_targets(self, kp):
        """Human reference vectors + per-vector weights for one frame.
        Vector set = [wrist->tip_i for all i] ++ [thumb->finger_j pinch pairs].
        Human vectors are rotated by R_align into the robot's frame first."""
        s = self.spec.scaling
        Ra = self.R_align
        origin = kp[self.h_origin]
        # wrist->tip targets (base tracking), rotated into robot frame
        wrist_vecs = np.array([Ra @ (kp[t] - origin) * s for t in self.h_tips])  # [n,3]
        wrist_w = np.ones(len(self.h_tips))
        # pinch pair targets with projection + hysteresis
        pair_vecs, pair_w = [], []
        for k, (a, b) in enumerate(self.pairs):
            raw = Ra @ (kp[self.h_tips[b]] - kp[self.h_tips[a]])                # thumb->finger (human), in robot frame
            d_human = np.linalg.norm(raw)                                      # UNSCALED gap decides pinch
            hv = raw * s
            if d_human < self.project_dist:
                self.projected[k] = True
            elif d_human > self.escape_dist:
                self.projected[k] = False
            if self.projected[k]:
                pair_vecs.append(hv / (np.linalg.norm(hv) + 1e-9) * self.eta)  # snap robot gap to eta
                pair_w.append(self.pinch_weight)
            else:
                pair_vecs.append(hv)
                pair_w.append(1.0)
        return wrist_vecs, np.array(pair_vecs), wrist_w, np.array(pair_w)

    def _objective(self, q, wrist_vecs, pair_vecs, wrist_w, pair_w, q_last):
        self._fk(q)
        palm = self._pos(self.palm)
        tips = np.array([self._pos(t, self.offset) for t in self.tip])
        Jpalm = self._jac(self.palm)
        Jtip = [self._jac(t, self.offset) for t in self.tip]

        loss = 0.0; g = np.zeros(self.nq)
        # wrist->tip terms
        for i in range(len(self.tip)):
            err = (tips[i] - palm) - wrist_vecs[i]
            d = np.linalg.norm(err) + 1e-9
            hub = 0.5 * d * d if d < self.delta else self.delta * (d - 0.5 * self.delta)
            loss += wrist_w[i] * hub
            hp = d if d < self.delta else self.delta
            g += wrist_w[i] * hp * (err / d) @ (Jtip[i] - Jpalm)
        # thumb->finger pinch terms
        for k, (a, b) in enumerate(self.pairs):
            err = (tips[b] - tips[a]) - pair_vecs[k]
            d = np.linalg.norm(err) + 1e-9
            hub = 0.5 * d * d if d < self.delta else self.delta * (d - 0.5 * self.delta)
            loss += pair_w[k] * hub
            hp = d if d < self.delta else self.delta
            g += pair_w[k] * hp * (err / d) @ (Jtip[b] - Jtip[a])
        loss += self.smooth * np.sum((q - q_last) ** 2)
        g += 2 * self.smooth * (q - q_last)
        S = self._loss_scale
        return S * loss, S * g

    def retarget_frame(self, kp, q_init):
        wv, pv, ww, pw = self._build_targets(kp)
        res = minimize(self._objective, q_init, args=(wv, pv, ww, pw, q_init),
                       jac=True, method="SLSQP", bounds=self.bounds,
                       options=dict(maxiter=200, ftol=1e-10))
        return res.x

    def _frame_from_vecs(self, palm, tips):
        """Right-handed palm frame (cols: forward, side, normal) from tip cloud."""
        nt = [tips[i] for i in range(len(tips)) if i != self.thumb]
        fwd = np.array(nt).mean(0) - palm
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        side = nt[-1] - nt[0]; side = side / (np.linalg.norm(side) + 1e-9)
        nrm = np.cross(fwd, side); nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        side = np.cross(nrm, fwd)
        return np.column_stack([fwd, side, nrm])

    def _calibrate(self, human_keypoints):
        """Compute R_align: rotation taking the human hand frame onto the robot's
        rest frame, so human finger targets land in the robot's reachable space.
        Uses the robot rest pose and the clip-mean human geometry (robust to curl)."""
        q0 = np.clip(np.zeros(self.nq), [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        self._fk(q0)
        palm_r = self._pos(self.palm)
        tips_r = [self._pos(t, self.offset) for t in self.tip]
        R_robot = self._frame_from_vecs(palm_r, tips_r)
        # clip-mean human tips/origin (mean over frames cancels per-frame curl jitter)
        kp = human_keypoints
        origin = kp[:, self.h_origin].mean(0)
        tips_h = [kp[:, i].mean(0) for i in self.h_tips]
        R_human = self._frame_from_vecs(origin, tips_h)
        R = R_robot @ R_human.T
        # guard: must be a proper rotation
        if np.linalg.det(R) < 0:
            R_human2 = R_human.copy(); R_human2[:, 1] *= -1
            R = R_robot @ R_human2.T
        self.R_align = R
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        print(f"[retarget] frame calibration: rotating human targets {ang:.0f} deg into robot frame")
        return R

    def retarget_sequence(self, human_keypoints):
        T = human_keypoints.shape[0]
        out = np.zeros((T, self.nq))
        if self.align:
            self._calibrate(human_keypoints)
        q = np.clip(np.zeros(self.nq), [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        self.projected[:] = False
        for t in range(T):
            q = self.retarget_frame(human_keypoints[t], q)
            out[t] = q
        return out


# ---------------------------------------------------------------------------
# CurlRetargeter -- joint-space (flexion) retargeting for FAITHFUL VISUALIZATION.
# Instead of matching fingertip *positions* (which contorts the hand when the
# human frame is rotated vs the robot), it copies how *bent* each human finger is
# onto the robot's flex joints. A bend angle has no frame, so this never contorts
# and needs no calibration. Not for grasp synthesis -- it doesn't aim at contacts;
# it makes the robot visibly do the human's gesture.
# ---------------------------------------------------------------------------
class CurlRetargeter:
    # robot qpos indices per finger: flex chain (knuckle->pip->dip) + spread joint
    LEAP = {
        "if": dict(flex=[0, 2, 3], spread=1),
        "mf": dict(flex=[4, 6, 7], spread=5),
        "rf": dict(flex=[8, 10, 11], spread=9),
        "th": dict(flex=[12, 14, 15], spread=13),
    }
    # human MANO/MediaPipe chains: [base, mcp, pip, dip, tip]
    HUMAN = {
        "if": [0, 5, 6, 7, 8],
        "mf": [0, 9, 10, 11, 12],
        "rf": [0, 13, 14, 15, 16],
        "th": [0, 1, 2, 3, 4],
    }
    ORDER = ["if", "mf", "rf", "th"]

    def __init__(self, spec, flex_gain=1.15, spread_gain=0.6):
        self.spec = spec
        self.model = spec.build_model()
        self.nq = self.model.nq
        lo = self.model.jnt_range[:, 0].copy(); hi = self.model.jnt_range[:, 1].copy()
        unlimited = ~self.model.jnt_limited.astype(bool)
        lo[unlimited] = -np.pi; hi[unlimited] = np.pi
        self.lo, self.hi = lo, hi
        self.flex_gain = flex_gain; self.spread_gain = spread_gain
        self.map = self.LEAP  # leap layout (allegro shares the if/mf/rf/th 4x4 order)

    @staticmethod
    def _angle(a, b):
        a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.arccos(np.clip(a @ b, -1, 1)))

    def retarget_frame(self, kp, q_prev=None):
        q = np.zeros(self.nq)
        # palm plane for spread: normal from wrist->index_mcp x wrist->ring_mcp
        w = kp[0]; n = np.cross(kp[5] - w, kp[13] - w); n /= (np.linalg.norm(n) + 1e-9)
        mid_dir = kp[9] - w; mid_dir = mid_dir - mid_dir @ n * n  # middle finger reference in palm plane
        for name in self.ORDER:
            rj = self.map[name]; hc = self.HUMAN[name]
            P = [kp[i] for i in hc]                     # base, mcp, pip, dip, tip
            segs = [P[i+1] - P[i] for i in range(4)]    # base->mcp, mcp->pip, pip->dip, dip->tip
            flex = [self._angle(segs[0], segs[1]),      # mcp bend
                    self._angle(segs[1], segs[2]),      # pip bend
                    self._angle(segs[2], segs[3])]      # dip bend
            for qi, fa in zip(rj["flex"], flex):
                q[qi] = np.clip(fa * self.flex_gain, self.lo[qi], self.hi[qi])
            # spread: signed angle of this finger's mcp direction vs middle, in palm plane
            fdir = P[1] - w; fdir = fdir - fdir @ n * n
            s = self._angle(fdir, mid_dir); sign = np.sign(np.cross(mid_dir, fdir) @ n)
            qi = rj["spread"]
            q[qi] = np.clip(sign * s * self.spread_gain, self.lo[qi], self.hi[qi])
        return q

    def retarget_sequence(self, human_keypoints):
        T = human_keypoints.shape[0]
        out = np.zeros((T, self.nq))
        for t in range(T):
            out[t] = self.retarget_frame(human_keypoints[t])
        return out
