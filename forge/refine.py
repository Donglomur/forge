"""Force-closure grasp refinement.

A retargeted grasp lands the robot fingers *near* a good grip, but morphology
differences leave the contacts imperfect: a fingertip grazes the object, the
wrench balance is lopsided, the force-closure margin is thinner than it could be.
This module locally refines the retargeted finger configuration to MAXIMIZE the
analytic Ferrari-Canny force-closure margin (the same metric forge's gate uses),
turning an approximate retarget into a physically sound grasp.

Method note: the idea of synthesizing grasps by optimizing a force-closure
objective comes from the Differentiable Force Closure energy (Liu et al., 2021)
and the accelerated energy-minimization synthesis of DexGraspNet (Wang et al.,
2023). Those minimize a *differentiable proxy* (frictionless, equal-force) and
then validate survivors in a GPU simulator. forge does the opposite where it counts:
it optimizes the EXACT analytic Grasp Wrench Space margin directly, on CPU, with a
derivative-free search (CEM). No proxy mismatch, no GPU. This is an original
implementation; only the high-level objective is shared with that prior work.
"""
import numpy as np


def refine_grasp(scene, eef_poses, finger_q, obj_rel, g,
                 iters=6, npop=16, elite=4, sd0=0.06, seed=0,
                 search_settle=70, final_settle=120, verbose=False):
    """Refine the grasp finger config at frame `g` to maximize force-closure eps.

    scene       : ReplayScene for the hand
    eef_poses   : [T,4,4] wrist trajectory (fixed; only frame g is used)
    finger_q    : [T,nu] retargeted finger sequence (row 0 = open, row g = grasp)
    obj_rel     : object pose relative to the wrist at the grasp frame
    g           : grasp-frame index
    Returns (refined_finger_q, eps_before, eps_after): a COPY of finger_q with
    row g replaced by the refined config, plus the margins before/after.

    CEM is warm-started from the retargeted grasp (so it only has to improve a
    decent starting point) and early-stops once it stops gaining, which keeps it
    to a few CPU-seconds rather than a from-scratch synthesis.
    """
    m = scene.model
    lo, hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
    base = np.asarray(finger_q, float).copy()
    nu = m.nu
    if base.shape[1] != nu:               # nothing to refine (no actuated fingers)
        return base, 0.0, 0.0
    q0 = base[g].copy()

    def eps_of(q, settle):
        seq = base.copy(); seq[g] = q
        _, eps, _ = scene.grasp_quality(eef_poses, seq, obj_rel, g=g, settle=settle)
        return float(eps)

    rng = np.random.default_rng(seed)
    mu = q0.copy(); sd = np.full(nu, sd0)
    eps0 = eps_of(q0, final_settle)
    best_eps, best_q, stall = eps0, q0.copy(), 0
    for it in range(iters):
        pop = np.clip(np.vstack([mu, mu + rng.normal(0, 1, (npop - 1, nu)) * sd]), lo, hi)
        scores = np.array([eps_of(c, search_settle) for c in pop])
        order = np.argsort(scores)[::-1]
        elites = pop[order[:elite]]
        top = float(scores[order[0]])
        if top > best_eps + 1e-4:
            best_eps, best_q, stall = top, pop[order[0]].copy(), 0
        else:
            stall += 1
        mu = elites.mean(0)
        sd = 0.6 * sd + 0.4 * elites.std(0) + 1e-3
        if verbose:
            print(f"[refine] iter {it}: best eps={best_eps:.4f}")
        if stall >= 2:
            break

    eps_after = eps_of(best_q, final_settle)   # confirm at full fidelity
    if eps_after < eps0:                        # never return a worse grasp
        best_q, eps_after = q0, eps0
    out = base.copy(); out[g] = best_q
    return out, eps0, eps_after
