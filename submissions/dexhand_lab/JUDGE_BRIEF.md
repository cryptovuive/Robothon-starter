# DexHand Lab Judge Brief

## What This Project Is

DexHand Lab is a hand-only MuJoCo dexterous manipulation benchmark. It replaces the earlier robot-arm direction with a human-like 5-finger skeletal robot hand focused on object-specific grasping, in-hand cylinder rotation, stylus precision interaction, and index-only button pressing.

## Why A 5-Finger Hand

The task is designed to show manipulation behaviors that a simple parallel gripper cannot express: thumb opposition, independent finger timing, lower support fingers, tripod precision grasping, and finger-gaited in-hand rotation.

## Hand Model

The hand is a custom MJCF model built from stable primitives. It includes:

- Palm and wrist mount.
- Thumb CMC opposition, thumb abduction, MCP flexion, and IP flexion.
- Index, middle, ring, and little MCP abduction, MCP flexion, PIP flexion, and DIP flexion.
- Visible phalanx capsules, joint caps, and fingertip pad collision geoms.
- Human-inspired finger length ordering: middle longest, ring slightly shorter, index shorter, little clearly shorter, thumb shorter and opposable.

## Object-Specific Grasp Strategies

- Sphere: `SPHERICAL_ENCLOSURE_GRASP`, using thumb opposition and a multi-finger cage.
- Cube: `OPPOSING_FACE_CUBE_GRASP`, using thumb and index/middle on opposing face centers.
- Cylinder: `LATERAL_CYLINDER_BODY_GRASP`, using side-body contacts around the cylinder midpoint, not top-down.
- Stylus: `TRIPOD_PRECISION_GRASP`, using thumb, index, and middle near the handle center.
- Button: `INDEX_FINGERTIP_PRESS`, using only the index fingertip.

## D-Grasp-Inspired Pipeline

The controller follows a dynamic grasp pipeline:

1. Classify object and select grasp reference.
2. Preshape the hand.
3. Approach the object.
4. Seek fingertip contact regions.
5. Soft-close fingers asynchronously.
6. Estimate contacts.
7. Verify stability.
8. Secure grasp only after verification.
9. Manipulate, rotate, press, or release.

## Fingertip Envelope Alignment

The current controller explicitly opens the thumb and long fingers before grasping, then moves through a contact-seek pose before any object is attached. For sphere, cube, and cylinder tasks, the logs verify that the object center lies inside the fingertip envelope before `SECURE_GRASP`.

Relevant fields in `trajectory.json`, `contact_timeline.json`, and `summary.json` include:

- `object_center_inside_finger_envelope`
- `object_center_between_fingers_rate`
- `grasp_centroid_error_m`
- `finger_envelope_x_span_m`
- `finger_envelope_y_span_m`
- `thumb_to_fingers_opposition_valid`

## No-Snap Hybrid Routine

The project uses simulation-native object pose perception and a hybrid contact-aware dexterous manipulation routine. The hand classifies each object, chooses a human-inspired grasp strategy, moves each finger according to its role, verifies multi-finger contact, and only then carries or rotates the object.

For robustness, object carry and in-hand rotation preserve the relative transform from the verified grasp moment. The project does not claim perfect contact physics, real RL, camera vision, or real-world deployment.

## Contact Timeline

Inspect `outputs/contact_timeline.json` first for sampled evidence of:

- Per-finger contacts.
- Per-finger contact points.
- Finger roles.
- Active finger count.
- Multi-side contact score.
- Fingertip envelope alignment and centroid error.
- One-face-only cube contact checks.
- Cylinder side-body contact count.
- Stylus tripod contacts.
- Index-only button press state.

## Stress Evaluation

`run_stress_eval.py` provides a deterministic fixed-seed stress summary over pose offsets, friction/mass variation, lateral shove, rotation target variation, and simulated contact loss. It compares an open-loop baseline with the contact-aware controller and writes:

- `outputs/stress_eval.json`
- `outputs/baseline_vs_feedback.json`

## Files To Inspect First

- `submissions/dexhand_lab/README.md`
- `submissions/dexhand_lab/rubric_scorecard.json`
- `submissions/dexhand_lab/submission_manifest.json`
- `submissions/dexhand_lab/scene.xml`
- `submissions/dexhand_lab/human_grasp_library.py`
- `submissions/dexhand_lab/object_classifier.py`
- `submissions/dexhand_lab/dexhand_controller.py`
- `submissions/dexhand_lab/run_demo.py`
- `submissions/dexhand_lab/outputs/summary.json`
- `submissions/dexhand_lab/outputs/contact_timeline.json`
- `submissions/dexhand_lab/outputs/policy_card.json`
- `submissions/dexhand_lab/outputs/sensor_manifest.json`
- `submissions/dexhand_lab/outputs/stress_eval.json`
- `submissions/dexhand_lab/outputs/baseline_vs_feedback.json`
- `submissions/dexhand_lab/outputs/final_report.txt`
- `submissions/dexhand_lab/outputs/demo.mp4`

## Honest Limitations

- The controller is heuristic and contact-aware, not learned RL.
- Contact data is a combination of MuJoCo state and controller contact proxies for reproducible visual stability.
- Hybrid carry/rotation is used only after grasp verification.
- The project does not perform camera-based perception.
