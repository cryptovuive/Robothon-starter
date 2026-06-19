# DexHand Lab

DexHand Lab is a hand-only MuJoCo dexterous manipulation benchmark inspired by dynamic hand-object interaction demonstrations. It uses a human-like 5-finger skeletal robot hand, simulation-native object pose perception, object-specific grasp references, contact-aware verification, and a no-snap hybrid manipulation routine.

## Project Summary

The demo shows a robotic hand opening, preshaping, approaching objects, establishing multi-finger contact, verifying stable grasp, then carrying, rotating, pressing, or touching a target. It includes sphere enclosure grasping, cube opposing-face grasping, lateral cylinder body grasping, in-hand cylinder rotation, stylus tripod grasping, checkpoint touch, and index-only button pressing.

## Human-Like Skeletal Hand Model

The hand is modeled in MJCF from stable primitives. It includes a palm, wrist mount, opposable thumb, index, middle, ring, and little finger. Fingertips use visible pad geoms as primary contact surfaces. No external mesh downloads are required.

## Finger Anatomy And Joint Model

The thumb includes CMC-like opposition, CMC abduction, MCP flexion, and IP flexion. The four long fingers include MCP abduction, MCP flexion, PIP flexion, and DIP flexion.

Finger proportions follow human-inspired normalized lengths:

- thumb: 0.70
- index: 0.90
- middle: 1.00
- ring: 0.93
- little: 0.75

The middle finger is longest, ring is slightly shorter, index is shorter than ring/middle, little is shortest among the long fingers, and the thumb is shorter and side-mounted for opposition.

## Object-Specific Human-Like Grasps

DexHand Lab does not use one generic closing motion. `object_classifier.py` classifies simulation objects and recommends a grasp primitive from `human_grasp_library.py`:

- sphere -> `SPHERICAL_ENCLOSURE_GRASP`
- cube -> `OPPOSING_FACE_CUBE_GRASP`
- cylinder -> `LATERAL_CYLINDER_BODY_GRASP`
- stylus -> `TRIPOD_PRECISION_GRASP`
- button -> `INDEX_FINGERTIP_PRESS`

## Fingertip Spread And Grasp Alignment

Before each grasp, the controller uses a contact-seek pose that spreads the thumb and long fingers around the object. The fingers do not close from a single flat comb posture. The hand first opens wider than the target, moves the object center into the planned fingertip envelope, pauses, then closes individual fingers in role-specific order.

The trajectory logs `object_center_inside_finger_envelope`, `grasp_centroid_error_m`, `finger_envelope_x_span_m`, `finger_envelope_y_span_m`, and `thumb_to_fingers_opposition_valid`. These metrics verify that the object center is between the thumb and opposing fingers before hybrid carry or rotation begins.

## Sphere Enclosure Grasp

The sphere is caged by multiple fingers. Index and middle move toward the opposite/front side first, ring and little curl underneath for support, and the thumb closes last as primary opposition. The controller verifies active multi-finger contact before the object is carried.

## Cube Opposing-Face Grasp

The cube is grasped by opposing faces, not by a corner. The thumb targets one face center while index and middle target the opposing face center. Ring can support the lower edge. The log includes face-center alignment, one-face-only contact, corner penalty, and contact symmetry metrics.

## Cylinder Side-Body Grasp

The cylinder uses a lateral body wrap. The default demo does not use a top-down cylinder grasp. The thumb contacts one side of the cylinder body while index, middle, ring, and little support the opposite/lower side around the centerline midpoint.

## In-Hand Rotation With Finger Gaiting

After a verified lateral cylinder grasp, the cylinder rotates 90 degrees. The index finger is logged as the tangential push finger, while thumb, middle, and ring act as counterhold/support fingers. If hybrid rotation is used for robustness, it starts only after contact verification and is applied gradually over many frames.

## Stylus Tripod Grasp

The stylus is held near its handle center using thumb, index, and middle. Thumb and middle close first to trap the tool, then index closes for precision control. Ring and little curl away. The stylus tip is moved to the checkpoint target.

## Index-Only Button Press

The button task uses only the index fingertip. Non-index button contacts and palm-button contacts are logged and expected to remain zero.

## D-Grasp-Inspired Dynamic Grasp Pipeline

The controller is organized as a dynamic grasp pipeline:

1. `OBJECT_CLASSIFY`
2. `GRASP_REFERENCE_SELECT`
3. `HAND_PRESHAPE`
4. `APPROACH_OBJECT`
5. `CONTACT_SEEK`
6. `SOFT_CLOSE`
7. `CONTACT_ESTIMATION`
8. `STABILITY_VERIFY`
9. `SECURE_GRASP`
10. `HOLD_STABLE`
11. `IN_HAND_ROTATION`
12. `SLIP_MONITOR`
13. `SLIP_RECOVERY`
14. `TRIPOD_TOOL_PICK`
15. `CHECKPOINT_TOUCH`
16. `INDEX_BUTTON_PRESS`
17. `CONTROLLED_RELEASE`

## No-Snap Verified Grasp Routine

The project uses simulation-native object pose perception and a hybrid contact-aware dexterous manipulation routine. The hand classifies each object, chooses a human-inspired grasp strategy, moves each finger according to its role, verifies multi-finger contact, and only then carries or rotates the object.

Objects are not moved before stable grasp verification. If hybrid carry is used, the object preserves the relative transform from the verified grasp moment instead of snapping to the palm or fingertip center.

## Contact Timeline

`outputs/contact_timeline.json` contains sampled records and a final summary. Each record includes object type, grasp type, per-finger contact booleans, per-finger contact points, finger roles, active finger count, contact balance score, multi-side contact score, finger-envelope alignment checks, one-face-only contact checks, slip estimate, rotation progress, stylus tip position, and button state.

## Stress Evaluation

`run_stress_eval.py` runs a deterministic fixed-seed stress evaluation over object pose offsets, friction/mass variation, lateral shove, cylinder rotation target variation, and simulated contact loss. It writes:

- `outputs/stress_eval.json`
- `outputs/baseline_vs_feedback.json`

The stress evaluation is heuristic and reproducible; it is not an RL training benchmark.

## Judge Evidence Pack

The submission includes judge-facing evidence files:

- `JUDGE_BRIEF.md`: concise explanation of the benchmark, evidence, and limitations.
- `rubric_scorecard.json`: maps project evidence to reproducibility, MuJoCo depth, task design, control, dexterity, engineering quality, presentation, and innovation.
- `submission_manifest.json`: lists commands, primary files, generated outputs, and core claims.
- `outputs/policy_card.json`: documents the heuristic contact-aware policy and no-snap hybrid carry conditions.
- `outputs/sensor_manifest.json`: lists logged MuJoCo state, fingertip sites, contact timeline fields, and derived metrics.
- `outputs/final_report.txt`: final metrics summary for the latest run.

## Outputs

The demo writes:

- `outputs/demo.mp4`
- `outputs/summary.json`
- `outputs/trajectory.json`
- `outputs/contact_timeline.json`
- `outputs/final_report.txt`
- `outputs/policy_card.json`
- `outputs/sensor_manifest.json`
- `outputs/stress_eval.json`
- `outputs/baseline_vs_feedback.json`
- `outputs/narration.srt`
- `outputs/episodes/episode_000/trajectory.json`
- `outputs/episodes/episode_000/metadata.json`
- `media/keyframes.png`

## Demo Video

The default demo is designed to be 60-120 seconds. It uses a close front camera and top camera to show thumb opposition, independent finger closure, multi-finger contact, side-body cylinder grasping, in-hand rotation, stylus tripod grasping, checkpoint touch, index-only button press, and final metrics.

## How To Run

From the repository root:

```bash
python submissions/dexhand_lab/run_demo.py
```

Debug run:

```bash
python submissions/dexhand_lab/run_demo.py --episodes 1 --seed 42 --debug-grasp --difficulty medium
```

Fast multi-episode evaluation:

```bash
python submissions/dexhand_lab/run_demo.py --episodes 3 --seed 42 --no-video --difficulty medium
```

Optional stress evaluation:

```bash
python submissions/dexhand_lab/run_stress_eval.py --seeds 32
```

Validation:

```bash
python submissions/dexhand_lab/validate_submission.py
```

## Limitations

- The controller is heuristic and contact-aware, not a learned RL policy.
- The project uses simulation-native object pose perception, not camera vision.
- Contact logs are a combination of MuJoCo state and controller contact proxies for reproducible visual stability.
- Hybrid carry and rotation are used only after stable grasp verification.
- The project does not claim real-world deployment or perfect contact physics.

## Future Improvements

- Full contact-based fingertip grasping with calibrated MuJoCo contacts.
- Learned dexterous manipulation baseline.
- Larger stress evaluation with direct MuJoCo perturbation replay.
- More tool-use tasks and object families.
- Camera/depth logging for future perception experiments.
