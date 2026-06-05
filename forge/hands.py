"""Real robot hands from MuJoCo Menagerie. No synthetic fallback: if the model
files are not present, this raises. Get them with:

    git clone https://github.com/google-deepmind/mujoco_menagerie

and point MENAGERIE at the checkout (env var MUJOCO_MENAGERIE or pass dir=...).

Human keypoints use the 21-joint MediaPipe/OpenPose hand layout:
  0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20
(tips are 4,8,12,16,20). Specs below map each robot fingertip body to the
human tip it should track.
"""
from __future__ import annotations
import os
from .retarget import HandSpec

# tip indices in the 21-joint human layout
THUMB, INDEX, MIDDLE, RING, PINKY = 4, 8, 12, 16, 20

# name -> (relative model path, palm body, [tip bodies], [human tips], scaling)
_REGISTRY = {
    "leap": ("leap_hand/right_hand.xml", "palm",
             ["if_ds", "mf_ds", "rf_ds", "th_ds"], [INDEX, MIDDLE, RING, THUMB], 1.0),
    "allegro": ("wonik_allegro/right_hand.xml", "palm",
                ["ff_tip", "mf_tip", "rf_tip", "th_tip"], [INDEX, MIDDLE, RING, THUMB], 1.4),
    "shadow": ("shadow_hand/right_hand.xml", "rh_palm",
               ["rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"],
               [THUMB, INDEX, MIDDLE, RING, PINKY], 1.1),
}


def menagerie_dir(dir=None):
    d = dir or os.environ.get("MUJOCO_MENAGERIE")
    if d is None:
        # bundled real models ship inside the package: forge/assets/menagerie
        bundled = os.path.join(os.path.dirname(__file__), "..", "assets", "menagerie")
        for cand in (bundled, "./mujoco_menagerie", os.path.expanduser("~/mujoco_menagerie"),
                     "../refs/mujoco_menagerie", "./refs/mujoco_menagerie"):
            if os.path.isdir(cand):
                d = cand; break
    if d is None or not os.path.isdir(d):
        raise FileNotFoundError(
            "No robot hand models found. The package ships real models in "
            "forge/assets/menagerie; if that is missing, clone MuJoCo Menagerie and "
            "set MUJOCO_MENAGERIE. (No synthetic hand fallback by design.)")
    return d


def load_hand(name, dir=None, scaling=None) -> HandSpec:
    """Return a HandSpec backed by the REAL Menagerie model. Raises if missing."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown hand '{name}'. known: {list(_REGISTRY)}")
    rel, palm, tips, human_tips, scale = _REGISTRY[key]
    path = os.path.join(menagerie_dir(dir), rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"model for '{name}' not found at {path}. "
                                f"Run: git sparse-checkout add {rel.split('/')[0]}")
    thumb_idx = 0 if key == "shadow" else len(tips) - 1   # shadow lists thumb first
    return HandSpec(xml_path=path, palm=palm, tips=tips, human_origin=0,
                    human_tips=human_tips, scaling=scaling if scaling is not None else scale,
                    thumb_idx=thumb_idx)


def available(dir=None):
    """Which registered hands actually have model files present."""
    try:
        base = menagerie_dir(dir)
    except FileNotFoundError:
        return []
    return [n for n, v in _REGISTRY.items() if os.path.isfile(os.path.join(base, v[0]))]
