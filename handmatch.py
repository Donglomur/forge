"""Quantify how closely the robot hand matches the human hand, part by part.

    python handmatch.py mocap.parquet [hand] [curl|dexpilot]

For each finger and each joint, reports the geometric BEND-ANGLE error between the
human finger and the retargeted robot finger (degrees), the per-finger tip
DIRECTION error (palm-relative), and the overall PALM-FRAME rotation between the
two hands (the single "the whole hand points differently" number). Bend angles
are frame-free, so this is a fair comparison across different hand morphologies.

Averaged over the whole clip; also printed at the grasp frame.
"""
import sys, os
import numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco

from forge import hands
from forge.retarget import CurlRetargeter, DexPilotRetargeter
from run_factory import load_input

# human MANO/MediaPipe finger chains: [wrist, mcp, pip, dip, tip]
HUMAN_CHAIN = {"index": [0, 5, 6, 7, 8], "middle": [0, 9, 10, 11, 12],
               "ring": [0, 13, 14, 15, 16], "thumb": [0, 1, 2, 3, 4]}
HUMAN_TIP = {"index": 8, "middle": 12, "ring": 16, "thumb": 4}
JOINTS = ["mcp", "pip", "dip"]


def _bends(points):
    """Bend angles (deg) between consecutive segments of a polyline of points."""
    segs = [points[i + 1] - points[i] for i in range(len(points) - 1)]
    out = []
    for i in range(len(segs) - 1):
        a = segs[i] / (np.linalg.norm(segs[i]) + 1e-9)
        b = segs[i + 1] / (np.linalg.norm(segs[i + 1]) + 1e-9)
        out.append(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
    return np.array(out)


def _robot_chain(model, tip_body, palm_body):
    """Body ids from palm..tip by walking the kinematic parent chain."""
    chain = []; bid = model.body(tip_body).id; palm = model.body(palm_body).id
    for _ in range(64):
        chain.append(bid)
        if bid == palm or model.body_parentid[bid] == 0:
            break
        bid = model.body_parentid[bid]
    return list(reversed(chain))


def _palm_frame(palm, tips):
    """Right-handed frame from a palm point and 4 fingertip points."""
    fwd = np.mean(tips[:-1], 0) - palm; fwd /= np.linalg.norm(fwd) + 1e-9
    side = tips[-2] - tips[0]; side /= np.linalg.norm(side) + 1e-9
    nrm = np.cross(fwd, side); nrm /= np.linalg.norm(nrm) + 1e-9
    side = np.cross(nrm, fwd)
    return np.column_stack([fwd, side, nrm])


def main():
    if len(sys.argv) < 2:
        print("usage: python handmatch.py <demo.parquet|.npz> [hand] [curl|dexpilot]"); return
    hand = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else "leap"
    method = "dexpilot" if "dexpilot" in sys.argv else "curl"
    hd = load_input(sys.argv[1]); demo, kp = hd.to_demo()
    T = len(kp)
    gf = int(np.argmin(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1)))
    spec = hands.load_hand(hand)
    rt = CurlRetargeter(spec) if method == "curl" else DexPilotRetargeter(spec)
    q = rt.retarget_sequence(kp)
    m = spec.build_model(); d = mujoco.MjData(m)
    fadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    palm_b = spec.palm
    # robot finger order matches the human index/middle/ring/thumb where possible
    fingers = ["index", "middle", "ring", "thumb"]
    tip_bodies = list(spec.tips)
    # thumb is usually last in spec.tips; map by spec.thumb_idx
    th = spec.thumb_idx if spec.thumb_idx >= 0 else len(tip_bodies) - 1
    order = [i for i in range(len(tip_bodies)) if i != th][:3] + [th]
    chains = {fingers[k]: _robot_chain(m, tip_bodies[order[k]], palm_b)
              for k in range(min(4, len(order)))}

    def robot_frame(t):
        if q.shape[1] == m.nu:
            d.qpos[fadr] = q[t]
        else:
            d.qpos[:min(len(q[t]), m.nq)] = q[t][:m.nq]
        mujoco.mj_forward(m, d)

    # accumulate per-joint bend error, per-finger tip-direction error, palm rotation
    bend_err = {f: [] for f in chains}
    dir_err = {f: [] for f in chains}
    palm_rot = []
    for t in range(T):
        robot_frame(t)
        # palm frames
        h_palm = kp[t, 0]
        h_tips = np.array([kp[t, HUMAN_TIP[f]] for f in fingers])
        Rh = _palm_frame(h_palm, h_tips)
        r_palm = d.xpos[m.body(palm_b).id]
        r_tips = np.array([d.xpos[m.body(tip_bodies[order[k]]).id] for k in range(4)])
        Rr = _palm_frame(r_palm, r_tips)
        R = Rr @ Rh.T
        palm_rot.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        for f in chains:
            hpts = kp[t, HUMAN_CHAIN[f]]
            hb = _bends(hpts)                                   # 3 human bends
            rpts = np.array([d.xpos[b] for b in chains[f]])
            rb = _bends(rpts)                                   # robot bends
            k = min(len(hb), len(rb))
            if k:
                bend_err[f].append(np.abs(hb[-k:] - rb[-k:]))   # distal-aligned
            # tip direction in each palm frame
            hd_ = (kp[t, HUMAN_TIP[f]] - h_palm); hd_ = Rh.T @ hd_; hd_ /= np.linalg.norm(hd_) + 1e-9
            rd_ = (r_tips[fingers.index(f)] - r_palm); rd_ = Rr.T @ rd_; rd_ /= np.linalg.norm(rd_) + 1e-9
            dir_err[f].append(np.degrees(np.arccos(np.clip(hd_ @ rd_, -1, 1))))

    print(f"\nHAND-MATCH  human vs {hand} ({method}),  {T} frames,  grasp frame {gf}")
    print(f"palm-frame rotation (whole-hand 'points differently'):  "
          f"mean {np.mean(palm_rot):5.1f} deg   at grasp {palm_rot[gf]:5.1f} deg\n")
    hdr = "finger    " + "  ".join(f"{j+'_bendErr':>11}" for j in JOINTS) + "   tip_dirErr"
    print(hdr); print("-" * len(hdr))
    allb = []
    for f in chains:
        be = np.array(bend_err[f]); de = np.array(dir_err[f])
        means = be.mean(0) if be.size else np.array([np.nan] * 3)
        allb.append(be.mean() if be.size else np.nan)
        cells = "  ".join(f"{means[j]:8.1f}deg" if j < len(means) else f"{'-':>11}"
                          for j in range(3))
        print(f"{f:9s} {cells}   {de.mean():6.1f}deg")
    print("-" * len(hdr))
    print(f"overall mean bend error: {np.nanmean(allb):.1f} deg   "
          f"(0 = robot finger bends exactly like the human)")
    print(f"overall mean tip-direction error: "
          f"{np.nanmean([np.mean(dir_err[f]) for f in chains]):.1f} deg")


if __name__ == "__main__":
    main()
