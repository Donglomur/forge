"""Auto-discovered pick control.

Hand-tuning the executor (grip gain, squeeze, force cap, pre-grasp spread) for
each hand and object is a dead end: a value that works for one hand and one cube
breaks the moment either changes. Instead, treat those knobs as a low-dimensional
ACTION and search them, this is the executor's policy. Given a hand and a grasp,
`tune_pick` runs CEM over the action space, rewarded by how high the object is
held at rest, and returns the parameters that achieve a stable pick. New hand or
new object: re-run the search, no human in the loop.

This is the lightweight, derivative-free form of the learned grasp policies in the
video-to-manipulation literature (e.g. PKDA's residual policy, VIDEOMANIP's grasp
model). The action space here is small enough (4-5 numbers) that CEM finds a
working pick in seconds on CPU, no GPU rollouts required.
"""
import numpy as np
from forge.replay import ReplayScene


def tune_pick(spec, grasp_q, open_q, obj_xy=(0.2, 0.0), obj_half=None,
              table_z=0.0, iters=6, npop=12, elite=4, seed=1,
              search_obj_size=True, compliant_wrist=False, verbose=False):
    """Search pick-controller parameters that achieve a stable held pick.

    spec        : HandSpec (uses spec.xml_path, spec.palm, spec.tips)
    grasp_q     : closed (force-closure) finger config, length nu
    open_q      : open finger config
    obj_xy      : (x,y) of the object on the table
    obj_half    : fixed object half-size, or None to let the search pick one
    search_obj_size : if True, the object half-size is part of the action

    Returns (params, result): params is a dict of the discovered controller
    settings; result is the pick() output dict for those settings.
    """
    # action = [grip_gain, overclose, grip_force, pre_splay (, obj_half)]
    lo = [30.0, 0.20, 0.6, 0.30]
    hi = [95.0, 0.70, 2.5, 0.95]
    fixed_half = None
    if search_obj_size or obj_half is None:
        lo.append(0.012); hi.append(0.030)
    else:
        fixed_half = float(obj_half)
    lo = np.array(lo); hi = np.array(hi)
    dim = len(lo)

    def evaluate(a):
        gg, oc, gf, spl = a[0], a[1], a[2], a[3]
        half = a[4] if dim == 5 else fixed_half
        sc = ReplayScene(spec.xml_path, spec.palm, obj_size=(half,) * 3, table_z=table_z,
                         compliant_wrist=compliant_wrist)
        r = sc.pick(grasp_q, open_q, obj_xy, spec.tips,
                    grip_gain=float(gg), overclose=float(oc),
                    grip_force=float(gf), pre_splay=float(spl))
        reward = r["final_height"] if r["success"] else r["final_height"] - 0.5
        r["obj_half"] = float(half)
        return float(reward), r

    rng = np.random.default_rng(seed)
    mu = 0.5 * (lo + hi)
    sd = 0.25 * (hi - lo)
    best = (-9.0, None, None)
    for it in range(iters):
        pop = np.clip(np.vstack([mu, mu + rng.normal(0, 1, (npop - 1, dim)) * sd]), lo, hi)
        scored = [evaluate(a) for a in pop]
        rewards = np.array([s[0] for s in scored])
        order = np.argsort(rewards)[::-1][:elite]
        if rewards[order[0]] > best[0]:
            best = (float(rewards[order[0]]), pop[order[0]].copy(), scored[order[0]][1])
        mu = pop[order].mean(0)
        sd = 0.55 * sd + 0.45 * pop[order].std(0) + 1e-3
        if verbose:
            print(f"[tune] iter {it}: best reward={best[0]:+.3f} "
                  f"(success={best[2]['success'] if best[2] else False})")
        if best[2] is not None and best[2]["success"]:
            break                                       # stop as soon as it holds

    a = best[1]
    params = {"grip_gain": float(a[0]), "overclose": float(a[1]),
              "grip_force": float(a[2]), "pre_splay": float(a[3]),
              "obj_half": float(a[4]) if dim == 5 else fixed_half}
    return params, best[2]
