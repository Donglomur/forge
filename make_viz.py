"""Render the montage: a full 21-joint human grasp + LEAP/Allegro/Shadow retargets."""
import numpy as np, mujoco
from forge import datasets
from forge.hands import load_hand, available
from forge.retarget import VectorRetargeter
from forge.viz import render_montage


def full_human_grasp(T=30):
    """A full 21-joint hand performing a pinch-close (open->closed)."""
    # rest pose: wrist at origin, fingers extended +z, spread in x; thumb to +x
    knuck = {1:(.035,-.01,.015),5:(.025,0,.04),9:(.008,0,.045),13:(-.01,0,.043),17:(-.028,0,.038)}
    tipz_open = {4:(.05,-.02,.05),8:(.03,0,.105),12:(.01,0,.11),16:(-.012,0,.105),20:(-.032,0,.092)}
    tipz_clos = {4:(.028,.03,.045),8:(.026,.045,.07),12:(.008,.05,.073),16:(-.014,.045,.07),20:(-.03,.038,.06)}
    kp = np.zeros((T,21,3))
    for t in range(T):
        a = np.clip((t/(T-1)-.15)/.6,0,1)
        kp[t,0]=(0,0,0)
        for fi,mcp in zip([1,5,9,13,17],[1,5,9,13,17]):
            kp[t,fi]=knuck[fi]
        for tip,base in zip([4,8,12,16,20],[1,5,9,13,17]):
            o=np.array(tipz_open[tip]); c=np.array(tipz_clos[tip])
            tippos=(1-a)*o+a*c
            kp[t,tip]=tippos
            # two intermediate joints between knuckle and tip
            kk=np.array(knuck[base])
            kp[t,base+1]=kk+(tippos-kk)*0.4
            kp[t,base+2]=kk+(tippos-kk)*0.72
    return kp

kp = full_human_grasp()
robots={}
for name in available():
    spec=load_hand(name)
    rt=VectorRetargeter(spec)
    q=rt.retarget_sequence(kp)
    robots[name]=(rt.model, rt.data, q)
    print(f"retargeted {name}: qpos {q.shape}")

frames=[2, 12, 21, 29]
out=render_montage(kp, robots, frames, "/mnt/user-data/outputs/forge_montage.png")
print("wrote", out)
