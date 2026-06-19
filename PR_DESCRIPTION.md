Registration UUID: ae3845b8-7246-4fc9-8655-31d46dbeba99

## Project Name
DexHand Lab - Human-Like Skeletal 5-Finger Robot Hand

## Final Submission Folder
`submissions/dexhand_lab`

## Robot Platform
DexHand Lab uses a custom MuJoCo skeletal 5-finger hand built from stable primitives. The platform includes a palm, visible opposable thumb, index, middle, ring, and little finger. The thumb has CMC-like opposition/abduction and the long fingers have MCP/PIP/DIP-style flexion plus MCP abduction.

## Task Goal
Demonstrate a D-Grasp-inspired hand-only dexterous manipulation benchmark with object classification, human-inspired grasp references, asynchronous finger closure, no-snap stable grasp verification, lateral cylinder body grasping, in-hand rotation, stylus tripod grasping, checkpoint touch, and index-only button pressing.

## Technical Approach
- MuJoCo MJCF scene using primitive skeletal hand, object, table, stylus, checkpoint, and button geometry.
- `object_classifier.py` for simulation-native object pose perception and grasp affordance selection.
- `human_grasp_library.py` for object-specific grasp primitives and finger roles.
- `dexhand_controller.py` for dynamic grasp pipeline state definitions and hand skeleton validation.
- Hybrid contact-aware controller: preshape, approach, contact seek, asynchronous soft-close, contact estimation, stability verification, secure grasp, manipulation, and controlled release.
- Preserved-transform hybrid carry/rotation only after grasp verification to avoid magnetic snapping.
- Contact timeline export with per-finger contacts, contact points, finger roles, active finger count, multi-side contact score, and button/stylus state.
- Deterministic stress evaluation and validator scripts.

## Core Features
- Runnable MuJoCo dexterous hand simulation.
- Human-like skeletal 5-finger hand with visible thumb opposition.
- Sphere enclosure grasp.
- Cube opposing-face grasp.
- Lateral cylinder body grasp, not top-down.
- 90-degree in-hand cylinder rotation with finger-gaiting metrics.
- Stylus tripod grasp and checkpoint touch.
- Index-only button press.
- No-snap verified grasp metrics.
- Multi-episode deterministic evaluation.
- Judge brief, final report, narration SRT, and keyframes.
- Rubric scorecard, submission manifest, policy card, and sensor manifest.
- Deterministic baseline-vs-feedback stress evaluation.

## How to Run
```bash
python submissions/dexhand_lab/run_demo.py
```

```bash
python submissions/dexhand_lab/run_demo.py --episodes 3 --seed 42 --no-video --difficulty medium
```

Optional:

```bash
python submissions/dexhand_lab/run_stress_eval.py --seeds 32
python submissions/dexhand_lab/validate_submission.py
```

## Outputs
- `submissions/dexhand_lab/outputs/demo.mp4`
- `submissions/dexhand_lab/outputs/summary.json`
- `submissions/dexhand_lab/outputs/trajectory.json`
- `submissions/dexhand_lab/outputs/contact_timeline.json`
- `submissions/dexhand_lab/outputs/final_report.txt`
- `submissions/dexhand_lab/outputs/policy_card.json`
- `submissions/dexhand_lab/outputs/sensor_manifest.json`
- `submissions/dexhand_lab/outputs/stress_eval.json`
- `submissions/dexhand_lab/outputs/baseline_vs_feedback.json`
- `submissions/dexhand_lab/outputs/episodes/*/trajectory.json`
- `submissions/dexhand_lab/outputs/episodes/*/metadata.json`
- `submissions/dexhand_lab/media/keyframes.png`

## Limitations
- Heuristic contact-aware controller, not learned RL.
- Contact timeline combines MuJoCo state with controller contact proxies.
- Hybrid carry and rotation are used after verification for reproducible visual stability.
- Simulation-native object pose perception, not camera vision.
- No real-world deployment claim.

## Future Improvements
- Fully physical fingertip contact grasping.
- Learned dexterous manipulation baseline.
- Larger MuJoCo perturbation stress suite.
- Depth/camera dataset export.
- More tools, deformables, and object families.
