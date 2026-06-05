"""Real hand keypoints from an RGB video, via MediaPipe. Free, CPU, no gated data.
A video of a hand -> the 21-joint layout forge uses, in meters (world landmarks).

Tries the legacy `mp.solutions.hands` API first (model bundled in the wheel, no
download), then the Tasks `HandLandmarker` API (downloads a ~7MB model once to
~/.cache/forge). Finger geometry is real; wrist path + object are inferred.
"""
from __future__ import annotations
import os
import urllib.request
import numpy as np
from . import se3
from .datasets import HumanDemo, infer_scene_from_keypoints

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/1/hand_landmarker.task")


def _frames(path, every):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % every == 0:
            yield fi, fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fi += 1
    cap.release()


def _track_solutions(path, every):
    """Legacy mp.solutions.hands (bundled model, no download)."""
    import mediapipe as mp
    H = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)
    world, img, fps = [], [], 30.0
    for _, fps, rgb in _frames(path, every):
        r = H.process(rgb)
        if r.multi_hand_world_landmarks:
            world.append(np.array([[p.x, p.y, p.z] for p in r.multi_hand_world_landmarks[0].landmark]))
            img.append(np.array([[p.x, p.y, p.z] for p in r.multi_hand_landmarks[0].landmark]))
    H.close()
    return world, img, fps


def _track_tasks(path, every):
    """Tasks HandLandmarker (downloads model once)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    cache = os.path.expanduser("~/.cache/forge"); os.makedirs(cache, exist_ok=True)
    model = os.path.join(cache, "hand_landmarker.task")
    if not os.path.isfile(model):
        print("[video] downloading MediaPipe hand model (one-time, ~7MB)...")
        urllib.request.urlretrieve(_MODEL_URL, model)
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    lm = vision.HandLandmarker.create_from_options(opts)
    world, img, fps = [], [], 30.0
    for fi, fps, rgb in _frames(path, every):
        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        r = lm.detect_for_video(mpimg, int(1000 * fi / fps))
        if r.hand_world_landmarks:
            world.append(np.array([[p.x, p.y, p.z] for p in r.hand_world_landmarks[0]]))
            img.append(np.array([[p.x, p.y, p.z] for p in r.hand_landmarks[0]]))
    lm.close()
    return world, img, fps


def keypoints_from_video(path, every=1, scene_scale=0.4) -> HumanDemo:
    """Track one hand through a video -> HumanDemo. Works on any RGB clip of a hand
    (your own, a dataset sample, or a downloaded YouTube clip)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        import mediapipe  # noqa
        import cv2         # noqa
    except Exception as e:
        raise ImportError("video ingest needs: pip install mediapipe opencv-python") from e

    try:
        world, img, fps = _track_solutions(path, every)
    except (AttributeError, ModuleNotFoundError):
        world, img, fps = _track_tasks(path, every)

    if len(world) < 3:
        raise RuntimeError(f"only {len(world)} frames had a detected hand in {path}; "
                           "use a clearer/closer view")
    world = np.array(world); img = np.array(img)
    kp = world - world[:, :1, :]                       # real metric finger geometry

    # prefer the real image-based wrist motion when available
    wi = img[:, 0, :]
    if np.ptp(wi[:, :2]) > 1e-3:                        # the hand actually moves in-frame
        wpos = np.stack([(wi[:, 0] - 0.5) * scene_scale,
                         -(wi[:, 1] - 0.5) * scene_scale,
                         0.20 - wi[:, 2] * scene_scale], axis=1)
        wrist_pose = np.zeros((len(kp), 4, 4))
        for t in range(len(kp)):
            z = kp[t, 9]; z = z / (np.linalg.norm(z) + 1e-9)
            x = kp[t, 5]; x = x - x.dot(z) * z; x = x / (np.linalg.norm(x) + 1e-9)
            wrist_pose[t] = se3.make_pose(wpos[t], np.stack([x, np.cross(z, x), z], axis=1))
        tips = kp[:, [4, 8, 12, 16, 20], :]
        gi = int(np.argmin(np.linalg.norm(tips - kp[:, :1, :], axis=2).mean(1)))
        op = wrist_pose[gi][:3, :3] @ tips[gi].mean(0) + wrist_pose[gi][:3, 3]
        object_pose = np.tile(se3.make_pose(op, np.eye(3)), (len(kp), 1, 1))
    else:
        wrist_pose, object_pose, _ = infer_scene_from_keypoints(kp)

    print(f"[video] tracked {len(kp)} frames @ {fps:.0f}fps")
    return HumanDemo(kp, wrist_pose, object_pose, "object", fps=fps)
