# Method

<sub>[Home](../README.md) · [Problem](PROBLEM.md) · **Method** · [Results](RESULTS.md)</sub>

---


forge is a four-stage pipeline. Each stage is a small, independently testable module; the diagnostics reproduce every number in [Results](RESULTS.md) from one command.

## 1. Relocate the demo — `forge/multiply.py`, `forge/se3.py`

One captured demonstration is turned into many by retransforming it around the object. The entire trick is one SE(3) identity:

```
T_new_eef = T_new_obj · inv(T_src_obj) · T_src_eef
```

In words: preserve the wrist pose *relative to the object*, then drop the object anywhere in a new scene. A demo is segmented into object-centric subtasks, each segment is retransformed to the new object pose, and consecutive segments are bridged by interpolation. This is the object-centric replay idea, reimplemented standalone (pure NumPy, no simulator dependency).

> **Correctness check:** the object-relative pose is invariant under relocation to **4.4e-16** (machine precision). See [Results](RESULTS.md).

## 2. Retarget onto each robot hand — `forge/retarget.py`

The human finger keypoints are mapped to each robot hand's joints by optimization-based inverse kinematics: minimize the discrepancy between human and robot fingertip vectors subject to the hand's joint limits. The pipeline runs across four hands (LEAP, Allegro, Shadow, and one more) from the bundled MuJoCo Menagerie models.

A second, joint-space **curl** retargeter is used only for *visualization*: it maps each human finger's bend angles onto the robot's flex joints, which is frame-invariant and never contorts, so the demo render faithfully mirrors the human's motion. (The validated grasps come from the IK retargeter, not the curl one.)

## 3. The force-closure gate — `forge/replay.py` (`force_closure_metric`)

This is the heart of forge: a closed-form test for whether a squeezed grasp can resist an external wrench from **any** direction, with no simulation.


The algorithm (Ferrari-Canny Q1):

1. **Linearize each contact's friction cone** into a set of unit force rays (Coulomb friction, coefficient μ).
2. **Lift each ray to a 6D wrench** `[force ; torque/λ]` about the object's center of mass, giving the grasp's primitive wrench set.
3. **Take the convex hull** of those wrenches — the *Grasp Wrench Space*.
4. **Force closure holds iff the origin is strictly inside the hull.** The quality `ε` is the distance from the origin to the nearest hull facet: the largest-magnitude disturbance the grasp is guaranteed to resist.

> **Why this is the whole game.** Building the wrench set is `O(contacts)` and the hull test is convex geometry. Screening a candidate is matrix algebra, not a rollout, which is exactly what lets the factory run on a CPU. See the cost argument in [Results](RESULTS.md).

## 4. The factory — `forge/factory.py`

The orchestrator ties it together along two independent axes: the **scene axis** (relocate the wrist trajectory to each object layout) and the **embodiment axis** (retarget the fingers to each hand). For every (hand, scene) pair it generates a candidate grasp trajectory and runs it through the gate, emitting only the validated ones.

## What runs the show

| Stage | Module | Output |
|---|---|---|
| Relocate | `multiply.py`, `se3.py` | wrist trajectory in each new scene |
| Retarget | `retarget.py` | robot finger joints per hand |
| Gate | `replay.py` `force_closure_metric` | force-closure verdict + ε margin |
| Factory | `factory.py` | validated trajectories (the yield) |

The numbers these produce, and what they mean, are in [Results](RESULTS.md).

---
<sub>[Home](../README.md) · [Problem](PROBLEM.md) · **Method** · [Results](RESULTS.md)</sub>
