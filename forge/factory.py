"""Orchestrator: ONE human demo -> MANY robot trajectories, across embodiments.
Two independent multiplication axes:
  * scene axis  : multiply.multiply_demo retransforms the WRIST trajectory to
                  each new object layout (MimicGen-style).
  * embodiment  : retarget.VectorRetargeter maps the human FINGER keypoints to
                  each robot hand's joints (dex-retargeting-style).
Robot trajectory = multiplied wrist + retargeted fingers. All CPU.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import time
import numpy as np
from .multiply import Demo, multiply_demo, sample_scene
from .retarget import VectorRetargeter, DexPilotRetargeter, HandSpec
from .replay import ReplayScene


def feasible(qpos, model_bounds, max_jerk=50.0):
    """Fast kinematic PRE-filter (joint limits + jerk). Cheap reject before the
    real physics gate in ReplayScene. Not a physics stand-in anymore."""
    lo = np.array([b[0] for b in model_bounds]); hi = np.array([b[1] for b in model_bounds])
    if np.any(qpos < lo - 1e-3) or np.any(qpos > hi + 1e-3):
        return False
    if qpos.shape[0] > 3:
        jerk = np.abs(np.diff(qpos, n=3, axis=0)).max()
        if jerk > max_jerk:
            return False
    return True


@dataclass
class Trajectory:
    embodiment: str
    scene_id: int
    wrist: np.ndarray      # [T,4,4]
    finger_qpos: np.ndarray  # [T,nq]


def run_factory(demo: Demo, human_keypoints: np.ndarray, specs: Dict[str, HandSpec],
                n_scenes=20, seed=0, verbose=True, retargeter=DexPilotRetargeter,
                physics=True, object_pose=None, gate="force_closure"):
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # embodiment axis: retarget fingers ONCE per hand (grasp shape is scene-invariant)
    finger_traj, finger_bounds = {}, {}
    for name, spec in specs.items():
        rt = retargeter(spec)
        finger_traj[name] = rt.retarget_sequence(human_keypoints)
        finger_bounds[name] = rt.bounds

    # optional REAL physics gate: build one replay scene per embodiment.
    # Rest the table at the object's true resting height so the object sits
    # where the hand actually grasps it (no fake drop-to-z=0).
    obj0 = demo.object_poses[list(demo.object_poses)[0]]
    obj_rest_z = float(np.min(obj0[:, 2, 3])) if obj0.ndim == 3 else float(obj0[2, 3])
    obj_half = 0.02
    scenes = {}
    obj_rel = {}
    if physics:
        for name, spec in specs.items():
            scenes[name] = ReplayScene(spec.xml_path, spec.palm,
                                       table_z=obj_rest_z - obj_half)
            if gate == "force_closure":
                # invariant grasp at the tightest human thumb-index pinch frame
                fq = finger_traj[name]
                gf = int(np.argmin(np.linalg.norm(
                    human_keypoints[:, 4] - human_keypoints[:, 8], axis=1)))
                objw = obj0 if obj0.ndim == 3 else np.tile(obj0, (len(fq), 1, 1))
                gi = min(gf, len(objw) - 1, len(demo.eef_poses) - 1)
                obj_rel[name] = (np.linalg.inv(demo.eef_poses[gi]) @ objw[gi], gi)

    # scene axis: multiply the wrist trajectory to many object layouts
    out: List[Trajectory] = []
    kept = 0
    obj_name = list(demo.object_poses)[0]
    for s in range(n_scenes):
        scene = sample_scene(demo, rng)
        wrist, _ = multiply_demo(demo, scene)
        for name in specs:
            fq = finger_traj[name]
            if fq.shape[0] != wrist.shape[0]:
                idx = np.linspace(0, fq.shape[0] - 1, wrist.shape[0]).round().astype(int)
                fq_aligned = fq[idx]
            else:
                fq_aligned = fq
            # cheap kinematic pre-filter always; real physics gate if requested
            if not feasible(fq_aligned, finger_bounds[name]):
                continue
            if physics:
                if gate == "force_closure":
                    rel, gi = obj_rel[name]
                    fc, eps, ncon = scenes[name].grasp_quality(wrist, fq_aligned, rel, g=gi)
                    if not fc:
                        continue
                else:
                    res = scenes[name].rollout(wrist, fq_aligned, scene[obj_name])
                    if not res.success:
                        continue
            out.append(Trajectory(name, s, wrist, fq_aligned)); kept += 1

    dt = time.time() - t0
    if verbose:
        total = n_scenes * len(specs)
        print(f"  generated {kept}/{total} trajectories across {len(specs)} embodiments "
              f"x {n_scenes} scenes in {dt:.2f}s  ({1000*dt/max(total,1):.1f} ms/traj)")
    return out
