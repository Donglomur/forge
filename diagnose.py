"""Full forge self-diagnostic: runs the whole pipeline with heavy instrumentation
and prints a dense, readable report.

    python diagnose.py data.parquet   # your DexCanvas mocap (a real input is required)
"""
import sys, os, time, platform, traceback
import numpy as np

# ---- terminal styling (auto-disabled when piped or NO_COLOR is set) ----
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def _c(code): return code if _COLOR else ""
RESET = _c("\033[0m"); BOLD = _c("\033[1m"); DIM = _c("\033[2m")
ORANGE = _c("\033[38;5;208m"); CYAN = _c("\033[38;5;44m"); GREEN = _c("\033[38;5;42m")
GREY = _c("\033[38;5;245m"); WHITE = _c("\033[97m"); AMBER = _c("\033[38;5;215m")

BANNER = r"""
  ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
  ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
  █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
  ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
  ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
  ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝"""

_step = [0]
def line(c="━"): print(f"{GREY}{c*74}{RESET}")
def hdr(t):
    _step[0] += 1
    print(f"\n{ORANGE}{'━'*74}{RESET}")
    print(f"  {CYAN}{BOLD}◆ STEP {_step[0]:02d}{RESET}   {BOLD}{WHITE}{t}{RESET}")
    print(f"{ORANGE}{'━'*74}{RESET}")
def banner():
    print(f"{ORANGE}{BANNER}{RESET}")
    print(f"  {DIM}human hand demos {RESET}{AMBER}→{RESET}{DIM} physically-validated robot grasps"
          f"   ·   CPU-native{RESET}\n")

def summary_box(SUMMARY, elapsed):
    print(f"\n{GREEN}{'━'*74}{RESET}")
    print(f"  {BOLD}{GREEN}✓  DIAGNOSTIC COMPLETE{RESET}    "
          f"{DIM}forge: one demo in, many validated robot grasps out{RESET}")
    print(f"{GREEN}{'━'*74}{RESET}")
    rt = SUMMARY.get("retarget", {})
    if rt:
        det = "   ".join(f"{WHITE}{k}{RESET} {CYAN}{v:+.2f}{RESET}" for k, v in rt.items())
        print(f"  {BOLD}retarget fidelity{RESET}    grip-tracking corr:   {det}")
    fc = SUMMARY.get("fc", {})
    if fc:
        passed = sum(1 for ok, _, _ in fc.values() if ok)
        det = "   ".join(f"{WHITE}{k}{RESET} ε={(GREEN if ok else GREY)}{eps:.3f}{RESET}"
                         for k, (ok, eps, _) in fc.items())
        print(f"  {BOLD}force-closure gate{RESET}   {det}")
        print(f"  {DIM}                     {passed}/{len(fc)} hands reach force closure{RESET}")
    y = SUMMARY.get("yield")
    if y:
        print(f"  {BOLD}factory yield{RESET}        {GREEN}{y[0]}{RESET}/{y[1]} "
              f"candidate grasps force-closure validated")
    print(f"  {BOLD}compute{RESET}              CPU only, no GPU    ·    "
          f"total runtime {AMBER}{elapsed:.1f}s{RESET}")
    print(f"{GREEN}{'━'*74}{RESET}\n")

def main():
    banner()
    t_start = time.time()
    SUMMARY = {"retarget": {}, "fc": {}, "yield": None}
    hdr("ENVIRONMENT")
    print("[env] platform :", platform.platform())
    print("[env] python   :", sys.version.split()[0], "|", sys.executable)
    print("[env] cwd       :", os.getcwd())
    print("[env] cpu_count :", os.cpu_count())
    for mod in ("mujoco", "scipy", "numpy"):
        try:
            m = __import__(mod); print(f"[env] {mod:7s} :", getattr(m, "__version__", "?"))
        except Exception as e:
            print(f"[env] {mod:7s} : MISSING ({e})")
    try:
        import mujoco
    except Exception as e:
        print("FATAL: mujoco import failed:", e); return

    from forge import se3, datasets, hands
    from forge.retarget import VectorRetargeter, DexPilotRetargeter
    from forge.multiply import transform_segment
    from forge.factory import run_factory
    from forge.replay import ReplayScene

    # ---- hands ----
    hdr("REAL HANDS")
    try:
        base = hands.menagerie_dir()
        print("[hands] model dir:", base)
    except Exception as e:
        print("[hands] NO MODELS:", e); return
    avail = hands.available()
    print("[hands] available:", avail)
    specs = {}
    for n in avail:
        try:
            sp = hands.load_hand(n); specs[n] = sp
            m = sp.build_model()
            print(f"[hands] {n:8s} nq={m.nq:2d} nu={m.nu:2d} nbody={m.nbody:2d} "
                  f"palm='{sp.palm}' tips={sp.tips} scaling={sp.scaling}")
        except Exception as e:
            print(f"[hands] {n}: LOAD FAILED: {e}")

    # ---- input ----
    hdr("INPUT DEMO")
    try:
        if len(sys.argv) < 2:
            print("[input] ERROR: real input required. Usage:")
            print("        python diagnose.py hand_video.mp4   (a video of a hand)")
            print("        python diagnose.py seq.npz          (keypoints/oakink/dexycb)")
            return
        from run_factory import load_input
        path = sys.argv[1]
        print("[input] loading REAL input:", path)
        hd = load_input(path)
        demo, kp = hd.to_demo()
        print(f"[input] frames={demo.T} keypoints={kp.shape} object='{hd.object_name}'")
        print(f"[input] gripper range=[{demo.gripper.min():.2f},{demo.gripper.max():.2f}] "
              f"wrist0 pos={np.round(demo.eef_poses[0,:3,3],3)}")
        print(f"[input] human thumb-index gap over time (mm): "
              f"{np.round(np.linalg.norm(kp[:,4]-kp[:,8],axis=1)*1000,1)}")
    except Exception as e:
        print("[input] FAILED:", e); traceback.print_exc(); return

    # ---- SE3 invariant ----
    hdr("SE(3) MULTIPLY INVARIANT")
    rng = np.random.default_rng(0)
    src = np.stack([se3.make_pose(rng.normal(size=3), se3.quat2mat(rng.normal(size=4))) for _ in range(6)])
    so = se3.make_pose(rng.normal(size=3), se3.quat2mat(rng.normal(size=4)))
    no = se3.make_pose(rng.normal(size=3), se3.quat2mat(rng.normal(size=4)))
    new = transform_segment(no, src, so)
    rs = se3.pose_in_A_to_pose_in_B(src, se3.pose_inv(so[None]))
    rn = se3.pose_in_A_to_pose_in_B(new, se3.pose_inv(no[None]))
    print(f"[multiply] object-relative drift = {np.abs(rs-rn).max():.2e} (want ~1e-15)")

    # ---- retargeting per hand: plain vs DexPilot, residuals, timing ----
    hdr("RETARGETING (plain vector  vs  DexPilot pinch)")
    for n, sp in specs.items():
        try:
            vr = VectorRetargeter(sp); t0 = time.time(); qv = vr.retarget_sequence(kp)
            tv = (time.time()-t0)/len(kp)*1000
            dp = DexPilotRetargeter(sp); t0 = time.time(); qd = dp.retarget_sequence(kp)
            td = (time.time()-t0)/len(kp)*1000
            # measure the TRUE thumb-index gap on every hand (handles thumb-first
            # hands like Shadow), not tip[last]-tip[first].
            def thumb_index_gap(rt, q):
                th = rt.spec.thumb_idx if rt.spec.thumb_idx >= 0 else len(rt.spec.tips) - 1
                try: ix = rt.spec.human_tips.index(8)   # human INDEX fingertip
                except ValueError: ix = 0
                if hasattr(rt, "_pos"):
                    rt._fk(q); a = rt._pos(rt.tip[th], rt.offset); b = rt._pos(rt.tip[ix], rt.offset)
                else:
                    rt._fk_vectors(q); a = rt._frame_pos(rt.tip[th], rt.offset); b = rt._frame_pos(rt.tip[ix], rt.offset)
                return np.linalg.norm(a - b)
            hg = np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1) * 1000           # human gap per frame
            rg = np.array([thumb_index_gap(dp, q) for q in qd]) * 1000          # robot gap per frame
            corr = float(np.corrcoef(hg, rg)[0, 1]) if hg.std() > 1e-6 else float("nan")
            SUMMARY["retarget"][n] = corr
            print(f"[retarget] {n:8s} nq={vr.nq:2d}  plain {tv:5.1f}ms/frame  DexPilot {td:5.1f}ms/frame")
            print(f"[retarget] {n:8s} thumb-index gap over clip: min={rg.min():5.1f}mm max={rg.max():5.1f}mm "
                  f"| human min={hg.min():5.1f} max={hg.max():5.1f} | corr(human,robot)={corr:+.2f}")
        except Exception as e:
            print(f"[retarget] {n}: FAILED: {e}"); traceback.print_exc()

    # ---- physics replay: sample rollouts per hand ----
    hdr("GRASP REACH CHECK (real trajectory, grasp frame)")
    # seat each hand at the tightest-pinch frame of the REAL (un-multiplied) demo
    gf = int(np.argmin(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1)))
    objp = demo.object_poses[list(demo.object_poses)[0]]
    obj_rest_z = float(np.min(objp[:, 2, 3])) if objp.ndim == 3 else float(objp[2, 3])
    tbl = obj_rest_z - 0.02
    print(f"[grasp] grasp frame = {gf}/{demo.T}; object at "
          f"{np.round(objp[gf,:3,3],3)}, wrist at {np.round(demo.eef_poses[gf,:3,3],3)}")
    for n, sp in specs.items():
        try:
            sc = ReplayScene(sp.xml_path, sp.palm, table_z=tbl)
            dp = DexPilotRetargeter(sp); fq = dp.retarget_sequence(kp)
            info = sc.grasp_probe(demo.eef_poses[gf], fq[gf], objp[gf], open_q=fq[0])
            print(f"[grasp] {n:8s} hand-object contacts={info['n_contact']:2d}  "
                  f"nearest-link->obj={info['min_body_obj_dist_mm']:.0f}mm  "
                  f"obj_half={info['obj_half']*1000:.0f}mm  obj_z_after_hold={info['obj_z']*100:.1f}cm")
        except Exception as e:
            print(f"[grasp] {n}: FAILED: {e}")

    hdr("DYNAMIC EXECUTION  —  real trajectory (research frontier)")
    objp_full = demo.object_poses[list(demo.object_poses)[0]]
    for n, sp in specs.items():
        try:
            sc = ReplayScene(sp.xml_path, sp.palm, table_z=tbl)
            dp = DexPilotRetargeter(sp); fq = dp.retarget_sequence(kp)
            fqa = fq if fq.shape[0]==demo.T else fq[np.linspace(0,fq.shape[0]-1,demo.T).round().astype(int)]
            r = sc.grasp_lift(demo.eef_poses, fqa, objp_full)
            print(f"[real] {n:8s} success={r.success!s:5s} lift={r.lift_height*100:+5.1f}cm "
                  f"max_obj_speed={r.max_obj_speed:5.2f}")
        except Exception as e:
            print(f"[real] {n}: FAILED: {e}"); traceback.print_exc()

    hdr("FORCE CLOSURE GATE (Ferrari-Canny / Q1, CPU, no dynamics)")
    # invariant grasp: object pose relative to the wrist at the grasp frame.
    # SAME grasp frame as the reach check: tightest human thumb-index pinch.
    objw = demo.object_poses[list(demo.object_poses)[0]]
    gf = int(np.argmin(np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1)))
    for n, sp in specs.items():
        try:
            sc = ReplayScene(sp.xml_path, sp.palm, table_z=tbl)
            dp = DexPilotRetargeter(sp); fq = dp.retarget_sequence(kp)
            obj_rel = np.linalg.inv(demo.eef_poses[gf]) @ objw[gf]
            fc, eps, ncon = sc.grasp_quality(demo.eef_poses, fq, obj_rel, g=gf)
            SUMMARY["fc"][n] = (bool(fc), float(eps), int(ncon))
            verdict = "FORCE CLOSURE" if fc else "not closed"
            print(f"[fc] {n:8s} {verdict:14s} eps={eps:.4f}  contacts={ncon}")
        except Exception as e:
            print(f"[fc] {n}: FAILED: {e}"); traceback.print_exc()

    hdr("DYNAMIC EXECUTION  —  sampled relocations (research frontier)")
    from forge.multiply import multiply_demo, sample_scene
    rng = np.random.default_rng(1)
    obj_name = list(demo.object_poses)[0]
    for n, sp in specs.items():
        try:
            sc = ReplayScene(sp.xml_path, sp.palm, table_z=tbl)
            print(f"[replay] {n:8s} scene compiled: nq={sc.model.nq} nu={sc.model.nu} "
                  f"nbody={sc.model.nbody} dt={sc.model.opt.timestep} integrator={sc.model.opt.integrator}")
            dp = DexPilotRetargeter(sp); fq = dp.retarget_sequence(kp)
            for s in range(3):
                scene = sample_scene(demo, rng)
                wrist, _ = multiply_demo(demo, scene)
                fqa = fq if fq.shape[0]==wrist.shape[0] else fq[np.linspace(0,fq.shape[0]-1,wrist.shape[0]).round().astype(int)]
                t0=time.time(); r = sc.rollout(wrist, fqa, scene[obj_name]); dt=time.time()-t0
                print(f"[replay] {n:8s} scene{s}: success={r.success!s:5s} lift={r.lift_height*100:+5.1f}cm "
                      f"max_obj_speed={r.max_obj_speed:5.2f} finite={np.all(np.isfinite(r.object_path))} {dt*1000:.0f}ms")
        except Exception as e:
            print(f"[replay] {n}: FAILED: {e}"); traceback.print_exc()

    # ---- factory yields ----
    hdr("FACTORY YIELD  (kinematic pre-filter vs force-closure physics gate)")
    try:
        k = run_factory(demo, kp, specs, n_scenes=15, seed=0, physics=False, verbose=True)
        p = run_factory(demo, kp, specs, n_scenes=15, seed=0, physics=True,
                        gate="force_closure", verbose=True)
        SUMMARY["yield"] = (len(p), len(k))
        from collections import Counter
        print("[yield] kinematic    per-hand:", dict(Counter(t.embodiment for t in k)))
        print("[yield] force-closure per-hand:", dict(Counter(t.embodiment for t in p)))
        print("[yield] gate = Ferrari-Canny Q1 force closure (analytic, CPU, no dynamic")
        print("        rollout). A scene passes when the squeezed grasp's contact wrenches")
        print("        span the origin, i.e. it can resist arbitrary external wrench.")
        if len(p) == 0:
            print("[yield] NOTE: 0 here means no embodiment reached force closure on the")
            print("        co-located grasp. Check the [fc] lines above for eps/contacts.")
    except Exception as e:
        print("[yield] FAILED:", e); traceback.print_exc()

    summary_box(SUMMARY, time.time() - t_start)

if __name__ == "__main__":
    main()
