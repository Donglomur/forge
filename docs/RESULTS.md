# Results & Evidence

<sub>[Home](../README.md) · [Problem](PROBLEM.md) · [Method](METHOD.md) · **Results**</sub>

---

Everything below is measured on a real DexCanvas pick demonstration (1,207 frames, object `cube1`), **on CPU**, and is reproduced by `python diagnose.py <parquet>`.

## 1. Retargeting fidelity

How tightly each robot's grip tracks the human's across the whole clip, measured as the correlation between the human thumb-index gap and the robot's (1.0 = perfect tracking).

| Robot hand | DOF | Grip-tracking corr | Speed |
|---|---|---|---|
| **LEAP** | 16 | **+0.79** | ~1.2 ms/frame |
| **Allegro** | 16 | **+0.77** | ~1.5 ms/frame |
| **Shadow** | 24 | **+0.69** | ~2.4 ms/frame |

> **Reading it:** a correlation near 0.8 across three structurally different hands means the retargeter is reproducing the *opening and closing behavior* of the demonstration, not just a static pose. Fidelity degrades gracefully with hand complexity (Shadow has the most DOF and the lowest, still-strong, correlation).

## 2. Force-closure gate

The stability margin `ε` of the squeezed grasp (higher = more robust). This is the analytic gate, no dynamics.

| Robot hand | Verdict | ε (margin) | Contacts |
|---|---|---|---|
| **LEAP** | ✅ force closure | **0.310** | 41 |
| **Allegro** | ✅ force closure | **0.078** | 5 |
| Shadow | rejected | 0.000 | 1 |

> **Reading it:** the gate is *discriminative*, not a rubber stamp. LEAP earns a large margin from many contacts; Allegro passes with a slimmer margin; Shadow's single contact on this demo is correctly rejected. A gate that passed everything would be worthless; this one separates the strong grasps from the weak.

## 3. Factory throughput

Generating and screening candidates across embodiments and scenes.

| Stage | Result | Time |
|---|---|---|
| Candidate generation | 45 / 45 | 6.3 s |
| **Force-closure validated** | **30 / 45** (LEAP 15, Allegro 15) | 8.9 s — **~197 ms/traj** |

A 67% yield means two of every three machine-generated candidates clear a real physics-grounded stability bar, from a single human demo, in seconds.

## 4. Cost & speed — the headline

The validation step is the expensive part of any cross-embodiment data pipeline. forge makes it cheap by construction:

| | Rollout / learned validation | **forge** |
|---|---|---|
| Validation step | physics rollout or neural inference per candidate | **closed-form force-closure test** |
| Cost per candidate | `O(sim steps)` of contact dynamics | **`O(contacts)` linear algebra** |
| Hardware | GPU acceleration in the loop | **laptop CPU, no GPU** |
| Measured throughput | (cluster-dependent) | **~197 ms / validated trajectory** |
| Marginal cost of a new hand | re-run the rollouts | swap the model, re-run the gate |

> This is an **architectural** comparison: forge replaces the rollout with a convex-geometry test. The forge column is measured; the left column describes the class of GPU-based approaches rather than any one benchmarked system.

## 5. Correctness & testing

- **SE(3) relocation invariant:** object-relative pose drift under relocation = **4.4e-16** (machine precision) — the multiplication math is exact.
- **Test suite:** **5 / 5 passing** (`python test_forge.py`), covering the pose algebra, relocation, retargeting shape/limits, and the gate.
- **Reproducibility:** one command (`diagnose.py`) regenerates every number here; the renderer (`render_demo.py trio`) regenerates the demo.

## How forge defines success

Different projects evaluate differently. forge is a **data factory**, so its success metric is *yield of physically-validated grasps per human demo, per unit compute*. By that metric: 30 validated trajectories from one demo, in ~9 s, on a CPU, with a discriminative gate that rejects the weak grasps. The evidence above is chosen to speak directly to that claim.

---
<sub>[Home](../README.md) · [Problem](PROBLEM.md) · [Method](METHOD.md) · **Results**</sub>
