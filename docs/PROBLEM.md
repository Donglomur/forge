# The Problem

<sub>[Home](../README.md) · **Problem** · [Method](METHOD.md) · [Results](RESULTS.md)</sub>

---


## Dexterous manipulation has a data-economics problem

The field is not short on architectures. It is short on **grasp data**, and specifically on grasp data for the *particular* robot hand in front of you. The reasons compound:

- **Robot hands do not share a language.** A grasp recorded on a LEAP hand does not run on an Allegro or a Shadow. Different finger counts, joint limits, link lengths, and actuation. Every embodiment is its own dialect, and demonstrations do not port between them for free.
- **Human demonstrations are the cheap, abundant source, in the wrong language.** Mocap and hand-tracking give an effectively unlimited supply of dexterous behavior, but recorded on a five-fingered, ~20-DOF human hand that no robot exactly matches.
- **The expensive part is earning trust.** Converting a demo onto a robot is only half the job. You also have to know which converted grasps would actually *hold*. The standard way to find out is to roll each candidate through physics simulation or a learned model, and that verification step is what pushes these pipelines onto GPU clusters.

> So the demonstrations are abundant and the robots are starving, and the wall between them is **translation plus trust**: convert the demo onto each hand, then prove which conversions are any good.

## Why this framing matters

Most "more data" efforts attack the supply of demonstrations. forge argues the binding constraint is elsewhere: it is the **unit cost of validating a candidate grasp**. If validating a grasp requires a simulation rollout, then the cost of manufacturing cross-embodiment data scales with GPU time, and adding the next robot hand means another batch of rollouts.

Change the cost of *that one step* and the whole economics moves. That is the opening forge goes after.

## The bet

Whether a grasp can hold is, at bottom, a question about forces: *do the contact forces span the space of disturbances the grasp might face?* That is a closed-form, linear-algebra question, the **Ferrari-Canny force-closure test**, not a simulation. forge's bet is that you can earn trust analytically, screen candidates on a CPU in milliseconds, and turn cross-embodiment grasp generation into something you run on a laptop.

The next doc shows exactly how. → [Method](METHOD.md)

---
<sub>[Home](../README.md) · **Problem** · [Method](METHOD.md) · [Results](RESULTS.md)</sub>
