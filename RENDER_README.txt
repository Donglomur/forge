SHOW-OFF VIDEO  (human hand  |  robot hand in the MuJoCo simulator)

One-time deps in your forge venv:
    source ~/.venvs/forge/bin/activate
    pip install imageio imageio-ffmpeg matplotlib

Render:
    cd ~/Downloads/forge
    python render_demo.py ~/dexcanvas/mocap_ver0.1.parquet              # leap, kinematic
    python render_demo.py ~/dexcanvas/mocap_ver0.1.parquet allegro      # other hand
    python render_demo.py ~/dexcanvas/mocap_ver0.1.parquet leap physics # closed-loop dynamic lift
    python render_demo.py ~/dexcanvas/mocap_ver0.1.parquet leap 2       # denser/smoother (stride 2)

Works on ANY input load_input understands: a hand .mp4 video (MediaPipe), a
.parquet (DexCanvas), or .npz/.pkl joints. The object is NOT a hardcoded cube:
it is sized to your grasp aperture, and if the dataset has no tracked object
(e.g. a phone video) one is inferred at the grasp point. Feed a different
human video and it renders that grasp instead.

Two modes:
  kinematic (default) - faithful pose replay; the object follows its captured
      track (or rides the grasp if inferred). Always clean. Caption honestly as
      "retargeted human manipulation replayed on a <hand> in MuJoCo."
  physics             - establishes the grip and lifts with the object held
      ONLY by simulated contact friction (slip-adaptive grip). This is the real
      closed-loop attempt; it may hold or slip depending on the grasp.

Output: forge_demo.mp4   (LEFT = human hand input, RIGHT = robot in sim)

----------------------------------------------------------------
AUDIT (quantify clipping / motion / slip, don't eyeball stills)

    python audit_demo.py ~/dexcanvas/mocap_ver0.1.parquet            # kinematic
    python audit_demo.py ~/dexcanvas/mocap_ver0.1.parquet leap physics

Prints PASS/FAIL on the field's metrics and writes forge_audit.png:
  PD  hand->object penetration   (INVALID > 5 mm)
  TP  table penetration          (INVALID > 10 mm)
  SD  object slip vs the palm after the grasp
  jit/jerk  motion spikes & smoothness
These are the standard dexterous-grasp metrics (penetration depth /
penetration volume / simulation displacement).
