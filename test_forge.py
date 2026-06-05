import tempfile, numpy as np
from forge import se3, datasets
from forge.multiply import transform_segment
from forge.retarget import VectorRetargeter
from forge.hands import load_hand, available


def test_se3_inverse():
    rng = np.random.default_rng(1)
    P = se3.make_pose(rng.normal(size=3), se3.quat2mat(rng.normal(size=4)))
    assert np.allclose(se3.pose_in_A_to_pose_in_B(P, se3.pose_inv(P)), np.eye(4), atol=1e-9)
    print("PASS se3 inverse")


def test_object_relative_invariant():
    rng = np.random.default_rng(2); T = 8
    src = np.stack([se3.make_pose(rng.normal(size=3), se3.quat2mat(rng.normal(size=4))) for _ in range(T)])
    so = se3.make_pose(rng.normal(size=3), se3.quat2mat(rng.normal(size=4)))
    no = se3.make_pose(rng.normal(size=3), se3.quat2mat(rng.normal(size=4)))
    new = transform_segment(no, src, so)
    rs = se3.pose_in_A_to_pose_in_B(src, se3.pose_inv(so[None]))
    rn = se3.pose_in_A_to_pose_in_B(new, se3.pose_inv(no[None]))
    assert np.allclose(rs, rn, atol=1e-9)
    print(f"PASS object-relative invariant (drift {np.abs(rs-rn).max():.1e})")


def test_real_hand_retarget_convergence():
    """Recover a known LEAP pose from its own FK targets (real model)."""
    if "leap" not in available():
        print("SKIP real-hand test (Menagerie not present)"); return
    rt = VectorRetargeter(load_hand("leap"), smooth=0.0)
    q_true = np.array([0.4, 0.1, 0.8, 0.7, 0.4, 0.1, 0.8, 0.7,
                       0.4, 0.1, 0.8, 0.7, 0.6, 0.3, 0.5, 0.5])[:rt.nq]
    palm, tips = rt._fk_vectors(q_true); target = tips - palm[None]
    # retarget by matching robot tips to these exact target vectors
    q = rt.retarget_frame(target / rt.spec.scaling, np.zeros(rt.nq))
    p2, t2 = rt._fk_vectors(q); res = np.linalg.norm((t2 - p2[None]) - target, axis=1)
    print(f"PASS real LEAP retarget: per-finger residual (mm) = {np.round(res*1000,2)}")
    assert res.mean() < 0.005


def test_replay_real_physics():
    """ReplayScene compiles a real hand+object+table, steps real MuJoCo, stays
    numerically stable, and correctly REJECTS a non-grasp (object not lifted)."""
    if "leap" not in available():
        print("SKIP replay test (Menagerie not present)"); return
    import numpy as np
    from forge.replay import ReplayScene
    from forge import se3
    spec = load_hand("leap")
    sc = ReplayScene(spec.xml_path, spec.palm)
    T = 20
    wp = np.tile(np.eye(4), (T, 1, 1)); wp[:, :3, 3] = [0, 0, 0.15]   # hover, never grasp
    fq = np.zeros((T, sc.model.nu))
    r = sc.rollout(wp, fq, np.eye(4))
    assert np.all(np.isfinite(r.object_path)), "physics went unstable (NaN)"
    assert not r.success, "non-grasp should not pass the lift gate"
    print(f"PASS replay real physics: stable, non-grasp rejected (lift {r.lift_height*100:.1f}cm)")


def test_datasets_fail_loud():
    bad = tempfile.NamedTemporaryFile(suffix=".npz", delete=False).name
    np.savez(bad, foo=np.zeros(3))
    for ok, fn in [(False, lambda: datasets.load_keypoints(bad)),
                   (False, lambda: datasets.mano_to_keypoints(np.zeros((2, 48)), np.zeros((2, 3))))]:
        try:
            fn(); raise AssertionError("should have raised")
        except (ValueError, ImportError, FileNotFoundError):
            pass
    print("PASS datasets fail loud (no synthetic fallback)")


if __name__ == "__main__":
    test_se3_inverse()
    test_object_relative_invariant()
    test_real_hand_retarget_convergence()
    test_datasets_fail_loud()
    test_replay_real_physics()
    print("\nALL TESTS PASSED")
