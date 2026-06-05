"""Zero-download real-data demo. Runs a REAL human grasp trajectory (mocap from
the MIT dex-retargeting repo, bundled) through forge's retargeting onto every
hand. Hand-only clip (no object), so it shows retargeting + grasp tracking, not
the physics lift. For the full lift, feed a DexCanvas parquet (has object pose).
"""
import os, time
import numpy as np
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from forge import hands
from forge.retarget import DexPilotRetargeter

DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "assets", "demos", "human_grasp_right.npz")


def main():
    kp = np.load(DEMO)["keypoints"][::8]
    hgap = np.linalg.norm(kp[:, 4] - kp[:, 8], axis=1) * 1000
    print(f"REAL human motion (bundled): {kp.shape[0]} frames, "
          f"human thumb-index gap {hgap.min():.0f}-{hgap.max():.0f}mm")
    print("retargeting onto every real hand (robot gap should TRACK the human's):\n")
    for n in hands.available():
        sp = hands.load_hand(n); dp = DexPilotRetargeter(sp)
        t0 = time.time(); q = dp.retarget_sequence(kp); dt = time.time() - t0
        rg = []
        for qi in q:
            dp._fk(qi)
            a = dp._pos(dp.tip[dp.thumb], dp.offset)
            b = dp._pos(dp.tip[dp.spec.human_tips.index(8)], dp.offset)
            rg.append(np.linalg.norm(a - b) * 1000)
        rg = np.array(rg); r = np.corrcoef(hgap, rg)[0, 1]
        print(f"  {n:8s} {dt/len(kp)*1000:4.1f}ms/frame  robot gap {rg.min():3.0f}-{rg.max():3.0f}mm  "
              f"tracks-human r={r:.2f}")


if __name__ == "__main__":
    main()
