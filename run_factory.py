"""forge: real human demo -> many cross-embodiment robot trajectories. CPU.

REAL INPUT ONLY. No synthetic fallback.
    python run_factory.py hand_video.mp4        # a video of a hand (MediaPipe)
    python run_factory.py seq.npz               # keypoints [T,21,3] + object_pose
    python run_factory.py oakink_seq.npz        # OakInk MANO annotation
    python run_factory.py dexycb_label.npz      # DexYCB labels
"""
import sys
from forge import datasets
from forge.hands import load_hand, available
from forge.factory import run_factory

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")


def load_input(path):
    pl = path.lower()
    if pl.endswith(VIDEO_EXT):
        from forge.video import keypoints_from_video
        print(f"Ingesting hand video: {path}")
        return keypoints_from_video(path)
    if pl.endswith(".parquet") or "dexcanvas" in pl:
        return datasets.load_dexcanvas(path)
    if pl.endswith((".pkl", ".npy")):
        return datasets.load_handjoints(path)
    if pl.endswith(".npz"):
        if "oakink" in pl:   return datasets.load_oakink(path)
        if "dexycb" in pl:   return datasets.load_dexycb(path)
        if "joint"  in pl or "keypoint" in pl: return datasets.load_handjoints(path)
        return datasets.load_keypoints(path)
    raise ValueError(f"unrecognized input '{path}'. Give a video {VIDEO_EXT}, .parquet, .pkl/.npy joints, or .npz")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    hd = load_input(sys.argv[1])
    demo, keypoints = hd.to_demo()
    print(f"Loaded HumanDemo: {demo.T} frames, object '{hd.object_name}'")

    have = available()
    if not have:
        raise SystemExit("No real hand models found (forge/assets/menagerie).")
    specs = {n: load_hand(n) for n in have}
    print(f"Retargeting onto real hands: {list(specs)}\n")

    trajs = run_factory(demo, keypoints, specs, n_scenes=20, seed=0, physics=True)
    by = {}
    for tr in trajs:
        by.setdefault(tr.embodiment, []).append(tr)
    print("\nPhysics-passed trajectories per embodiment:")
    for e in specs:
        ts = by.get(e, [])
        print(f"  {e}: {len(ts)}" + (f"  (e.g. wrist {ts[0].wrist.shape}, qpos {ts[0].finger_qpos.shape})" if ts else ""))


if __name__ == "__main__":
    main()
