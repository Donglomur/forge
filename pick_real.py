"""Real-data pick test: does forge actually pick up YOUR object?

    python pick_real.py your_dexcanvas.parquet
    python pick_real.py seq.npz

Loads a real human demo, retargets and refines the grasp per hand, then
AUTO-TUNES the executor's controller (grip gain, squeeze, force cap, pre-grasp
spread) for the REAL object size taken from the data, no object-size fudging,
and runs the pick. Reports, per hand, whether the object ends up held aloft at
rest, and the controller the search discovered.

This is the honest loop: the object is fixed by reality, only the controller is
searched. If a hand can't pick the real object, it says so.
"""
import sys, os, time
import numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco

from forge import hands
from forge.retarget import DexPilotRetargeter
from forge.replay import ReplayScene
from forge.refine import refine_grasp
from forge.tune import tune_pick
from run_factory import load_input


def main():
    if len(sys.argv) < 2:
        print("usage: python pick_real.py <demo.parquet|.npz|.mp4>")
        return
    t0 = time.time()
    hd = load_input(sys.argv[1])
    demo, kp = hd.to_demo()
    gf = int(np.argmin(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1)))
    objp = demo.object_poses[list(demo.object_poses)[0]]
    obj_rest_z = float(np.min(objp[:, 2, 3])) if objp.ndim == 3 else float(objp[2, 3])
    tbl = obj_rest_z - 0.02
    print(f"demo: frames={demo.T}  object='{hd.object_name}'  grasp_frame={gf}/{demo.T}")
    print(f"object rest z={obj_rest_z:.3f}m   table z={tbl:.3f}m\n")

    for n in hands.available():
        try:
            sp = hands.load_hand(n)
            nu = sp.build_model().nu
            dp = DexPilotRetargeter(sp)
            fq = dp.retarget_sequence(kp)
            if fq.shape[1] != nu:
                print(f"{n:8s} SKIP  (retarget dim {fq.shape[1]} != actuators {nu}; "
                      f"needs nq->nu mapping)")
                continue

            # REAL object size, measured from the demo (not chosen by us)
            sc0 = ReplayScene(sp.xml_path, sp.palm, table_z=tbl)
            info = sc0.grasp_probe(demo.eef_poses[gf], fq[gf], objp[gf], open_q=fq[0])
            real_half = float(info["obj_half"])

            # refine the grasp on the analytic force-closure metric
            obj_rel = np.linalg.inv(demo.eef_poses[gf]) @ objp[gf]
            fq_ref, e0, e1 = refine_grasp(sc0, demo.eef_poses, fq, obj_rel, gf,
                                          iters=5, npop=12, elite=4)
            qg, qo = fq_ref[gf], fq_ref[0]

            # auto-tune ONLY the controller for the REAL object (size is fixed)
            t1 = time.time()
            params, r = tune_pick(sp, qg, qo, obj_xy=(0.2, 0.0),
                                  obj_half=real_half, table_z=0.0,
                                  search_obj_size=False, iters=6, npop=12)
            dt = time.time() - t1

            verdict = "PICKED" if r["success"] else "could NOT pick"
            print(f"{n:8s} real object {2*real_half*100:.1f}cm  fc-margin {e0:.3f}->{e1:.3f}")
            print(f"         held {r['final_height']*100:+5.1f}cm  final_speed {r['final_speed']:.2f}  "
                  f"contacts {r['end_contacts']}  ->  {verdict}   ({dt:.0f}s search)")
            print(f"         controller: grip_gain={params['grip_gain']:.0f} "
                  f"overclose={params['overclose']:.2f} grip_force={params['grip_force']:.2f} "
                  f"pre_splay={params['pre_splay']:.2f}\n")
        except Exception as e:
            import traceback
            print(f"{n:8s} FAILED: {e}")
            traceback.print_exc()
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
