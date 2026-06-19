# DexHand Lab

DexHand Lab is a hand-only MuJoCo dexterous manipulation benchmark built around a human-like five-finger robotic hand. The project demonstrates object-specific grasping, in-hand rotation, a signature 224-degree cap/knob twist, tactile/contact evidence, slip recovery, load hold, and judge-readable evaluation artifacts.

## Project Summary

DexHand Lab focuses on dexterous hand manipulation rather than a full robot arm. The default demo shows a five-finger hand opening, grasping a sphere, cube, cylinder, stylus, and cap/knob, rotating objects after contact verification, pressing a button with the index finger, and writing reproducible robot-learning data.

## Human-Like Skeletal Hand Model

The hand is a custom MuJoCo primitive model with palm, wrist mount, thumb, index, middle, ring, and little fingers. The thumb is mounted on the side of the palm for opposition instead of behaving as a fifth parallel finger.

## Finger Anatomy and Joint Model

Each long finger has MCP abduction/flexion, PIP flexion, DIP flexion, visible phalanx capsules, joint caps, and fingertip pads. The thumb has CMC-like opposition/abduction, MCP flexion, IP flexion, and a fingertip pad. Finger proportions are intentionally varied: middle is longest, ring and index are shorter, little is shortest among long fingers, and thumb is shorter/thicker.

## Object-Specific Human-Like Grasps

The controller uses simulation-native object pose perception and a hybrid contact-aware dexterous manipulation routine. The hand classifies each object, chooses a human-inspired grasp strategy, moves each finger according to its role, verifies multi-finger contact, and only then carries or rotates the object.

## Sphere Enclosure Grasp

The sphere task uses thumb opposition plus index/middle support and ring/little lower support. The sphere is treated as enclosed by a finger cage before it is held.

## Cube Opposing-Face Grasp

The cube task uses opposing-face contact: thumb on one face, index/middle on the opposing face, and ring support below. The controller logs face-centered contact evidence and rejects one-face-only/corner-style contact.

## Cylinder Side-Body Grasp

The cylinder task uses a lateral body wrap, not a top-down grasp. Thumb contacts one side of the cylinder body while index/middle/ring/little support the opposite and lower sides.

## In-Hand Rotation with Finger Gaiting

The cylinder is rotated in-hand by a contact-aware hybrid routine. The index finger acts as a tangential push finger while thumb/middle/ring provide counterhold and support. Rotation is gradual and logged with achieved angle, rotation error, active rotation finger, and support fingers.

## Signature Cap/Knob Rotation

The 90+ evidence upgrade adds `CAP_KNOB_ROTATION_224`. A visible marker on the cap shows a 224-degree twist after five-finger contact verification. The cap task logs target angle, achieved angle, angle error, active/counterhold fingers, cap slip, contact balance, pressure targets, and whether hybrid rotation was used.

## Tactile Evidence

Five fingertip channels are logged: thumb, index, middle, ring, and little. Each channel includes contact active, contact object, normal force proxy, shear slip proxy, friction margin, contact confidence, pressure target, fingertip position, and role. These are deterministic MuJoCo/contact-controller proxies, not physical tactile sensor readings.

## Minimum-Jerk Tactile Control

`minimum_jerk_controller.py` generates deterministic tactile-inspired minimum-jerk segments for approach, preshape, contact seek, grasp closure, cap twist, slip recovery, load hold, and release. It writes `dataset/minimum_jerk_report.json` and `dataset/minimum_jerk_trace.csv`.

## No-Snap Verified Grasp Routine

Objects are not attached or moved until the required finger contacts are active and stable verification passes. Hybrid carry/rotation preserves the relative transform from the verified contact moment and moves objects gradually.

## Slip Recovery and Load Hold

The cap task includes a mild slip disturbance proxy, recovery by increasing thumb opposition and support pressure, and a 9x load-hold marker. The demo logs final slip, max slip, recovery action, pressure boost, load hold multiplier, and object drop count.

## Stylus Tripod Grasp

The stylus is held near the handle center using thumb, index, and middle. Ring and little remain curled/clear. The stylus tip is then moved to the checkpoint.

## Index-Only Button Press

The button task uses the index fingertip only. Non-index button contacts and palm contact are logged as failure evidence; the intended default is zero.

## Contact Timeline

`outputs/contact_timeline.json` records phase, target object, grasp type, per-finger contacts, per-finger roles, contact points, active finger count, balance score, multi-side score, tactile proxies, cap angle, slip, load hold, and button state.

## Stress Evaluation

`run_stress_eval.py --seeds 32` runs deterministic perturbations over object pose, mass, friction, lateral shove, cap angle, cap friction, contact loss, and rotation target. It writes `outputs/stress_eval.json`, `dataset/stress_eval.json`, `outputs/baseline_vs_feedback.json`, and `outputs/stress_eval_summary.csv`.

## Judge Evidence Pack

The project includes `JUDGE_BRIEF.md`, `rubric_scorecard.json`, `submission_manifest.json`, `media/keyframes.png`, `media/demo.mp4`, `outputs/narration.srt`, tactile reports, task-gate reports, stress reports, and hardware replay audit files.

## Hardware Replay Audit

The hardware audit maps simulated joints to LEAP/Shadow-style channels and generates a bounded 50 Hz command stream. This is a simulation-to-hardware replay safety audit, not a real hardware trial.

## Demo Video

The default demo is intended to be judge-readable, roughly 75-120 seconds after rendering. It shows the hand skeleton, sphere enclosure, cube opposing-face grasp, cylinder side-body grasp, cylinder in-hand rotation, cap 224-degree twist, slip/load-hold evidence, stylus tripod grasp, index-only button press, and final evidence pose.

## How to Run

```bash
python submissions/dexhand_lab/run_demo.py
python submissions/dexhand_lab/run_demo.py --episodes 3 --seed 42 --no-video --difficulty medium
python submissions/dexhand_lab/run_stress_eval.py --seeds 32
python submissions/dexhand_lab/arena_task_suite.py
python submissions/dexhand_lab/minimum_jerk_controller.py
python submissions/dexhand_lab/contact_feedback_audit.py
python submissions/dexhand_lab/hardware_adaptation_audit.py
python submissions/dexhand_lab/validate_submission.py
```

## Outputs

Key outputs include `outputs/demo.mp4`, `media/demo.mp4`, `media/keyframes.png`, `outputs/summary.json`, `outputs/contact_timeline.json`, `outputs/final_report.txt`, `dataset/task_suite_report.json`, `dataset/tactile_feedback_report.json`, `dataset/minimum_jerk_report.json`, `dataset/stress_eval.json`, and `dataset/hardware_adaptation_report.json`.

## Limitations

DexHand Lab does not claim real-world deployment, real camera vision, perfect contact physics, learned reinforcement learning, or physical tactile sensors. Hybrid carry and cap rotation are used only after verified contact to keep the benchmark deterministic and reproducible.

## Future Improvements

Future work could replace proxy tactile channels with MuJoCo touch sensors, tune contacts with richer dynamics, add learned residual controllers, and replay the bounded command stream on a real dexterous hand platform.
