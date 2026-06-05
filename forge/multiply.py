"""The factory floor: turn ONE demo into MANY by object-centric SE(3) retransform.

This is MimicGen's core idea, reimplemented clean and standalone (no robosuite).
Key operation (the entire multiplication trick):

    T_new_eef = T_new_obj @ inv(T_src_obj) @ T_src_eef

i.e. preserve the end-effector pose *relative to the object*, then drop the
object wherever you like in the new scene. A demo is segmented into
object-centric subtasks; each segment is retransformed to the new object pose,
and consecutive segments are bridged by interpolation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from . import se3


@dataclass
class Subtask:
    """One object-centric segment of a demo."""
    object_name: str            # which object this segment is 'about'
    start: int                  # inclusive frame index into the demo
    end: int                    # exclusive
    num_interp: int = 8         # interpolation steps to bridge into this segment


@dataclass
class Demo:
    """A source demonstration in the world frame."""
    eef_poses: np.ndarray                      # [T,4,4] end-effector / wrist target poses
    object_poses: dict                         # name -> [T,4,4] (or [4,4] if static)
    gripper: np.ndarray                        # [T,G] gripper / finger actuation
    subtasks: List[Subtask] = field(default_factory=list)

    @property
    def T(self):
        return self.eef_poses.shape[0]

    def obj_pose_at(self, name, t):
        op = self.object_poses[name]
        return op if op.ndim == 2 else op[t]


def transform_segment(new_obj_pose, src_eef_poses, src_obj_pose):
    """The one-liner. new_obj_pose [4,4], src_eef_poses [T,4,4], src_obj_pose [4,4]."""
    rel = se3.pose_in_A_to_pose_in_B(src_eef_poses, se3.pose_inv(src_obj_pose[None]))
    return se3.pose_in_A_to_pose_in_B(rel, new_obj_pose[None])


def multiply_demo(demo: Demo, new_scene: dict, start_eef: Optional[np.ndarray] = None):
    """Generate one new eef trajectory for a new object layout.

    Args:
        demo: source Demo with subtasks defined.
        new_scene: name -> 4x4 object pose in the NEW scene.
        start_eef: optional 4x4 pose the robot starts from (else demo's first eef).
    Returns:
        eef_traj [T',4,4], gripper_traj [T',G]
    """
    if not demo.subtasks:
        demo.subtasks = [Subtask(list(demo.object_poses)[0], 0, demo.T)]

    cur = start_eef if start_eef is not None else demo.eef_poses[0]
    eef_out, grip_out = [], []

    for st in demo.subtasks:
        seg_eef = demo.eef_poses[st.start:st.end]
        seg_grip = demo.gripper[st.start:st.end]
        src_obj = demo.obj_pose_at(st.object_name, st.start)
        new_obj = new_scene[st.object_name]

        transformed = transform_segment(new_obj, seg_eef, src_obj)

        # bridge: interpolate from current pose to the first transformed pose
        bridge = se3.interp_poses(cur, transformed[0], st.num_interp)
        eef_out.append(bridge[:-1])                       # drop last (== transformed[0])
        grip_out.append(np.repeat(seg_grip[0:1], st.num_interp - 1, axis=0))

        eef_out.append(transformed)
        grip_out.append(seg_grip)
        cur = transformed[-1]

    return np.concatenate(eef_out, 0), np.concatenate(grip_out, 0)


def sample_scene(demo: Demo, rng, pos_sigma=0.08, yaw_range=(-np.pi, np.pi),
                 workspace=((-0.25, 0.25), (-0.25, 0.25))):
    """Randomly relocate each object: a new layout = a new trajectory to generate."""
    scene = {}
    for name in demo.object_poses:
        base = demo.obj_pose_at(name, 0)
        p, R = se3.unmake_pose(base)
        dx = rng.uniform(*workspace[0]); dy = rng.uniform(*workspace[1])
        newp = p + np.array([dx, dy, 0.0]) + rng.normal(0, pos_sigma, 3) * np.array([1, 1, 0])
        yaw = rng.uniform(*yaw_range)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        scene[name] = se3.make_pose(newp, Rz @ R)
    return scene
