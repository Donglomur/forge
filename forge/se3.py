"""SE(3) pose math. Pure numpy, CPU. The algebra the whole factory rides on.

A pose is a 4x4 homogeneous matrix. Batched poses are [..., 4, 4].
This is a clean reimplementation of the pose ops that MimicGen buries in
pose_utils.py, plus quaternion <-> matrix and screw interpolation.
"""
from __future__ import annotations
import numpy as np


def make_pose(pos, rot):
    """pos [...,3], rot [...,3,3] -> [...,4,4]."""
    pos = np.asarray(pos, float)
    rot = np.asarray(rot, float)
    shape = rot.shape[:-2]
    out = np.zeros(shape + (4, 4))
    out[..., :3, :3] = rot
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


def unmake_pose(pose):
    return pose[..., :3, 3], pose[..., :3, :3]


def pose_inv(pose):
    """Inverse of a homogeneous transform, batched."""
    R = pose[..., :3, :3]
    p = pose[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(pose)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, p)
    out[..., 3, 3] = 1.0
    return out


def pose_in_A_to_pose_in_B(pose_in_A, pose_A_in_B):
    """Change frame of a (batched) pose. pose_A_in_B may be [4,4] or broadcastable."""
    return np.matmul(pose_A_in_B, pose_in_A)


def quat2mat(q):
    """quaternion (w,x,y,z) -> 3x3. Batched."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def mat2quat(R):
    """3x3 -> quaternion (w,x,y,z). Single matrix."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
    d = np.dot(q0, q1)
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(d); th = th0 * t
    q2 = q1 - q0 * d; q2 /= np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def interp_poses(pose_a, pose_b, n):
    """Linear in position, slerp in rotation. Returns [n,4,4] from a..b inclusive."""
    pa, Ra = unmake_pose(pose_a); pb, Rb = unmake_pose(pose_b)
    qa, qb = mat2quat(Ra), mat2quat(Rb)
    out = np.zeros((n, 4, 4))
    for i in range(n):
        t = i / max(n - 1, 1)
        p = (1 - t) * pa + t * pb
        R = quat2mat(slerp(qa, qb, t))
        out[i] = make_pose(p, R)
    return out
