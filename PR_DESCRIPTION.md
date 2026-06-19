Registration UUID: ae3845b8-7246-4fc9-8655-31d46dbeba99

Project Name
DexHand Lab - Human-Like 5-Finger Dexterous Manipulation Benchmark

Final Submission Folder
submissions/dexhand_lab

Project Summary
DexHand Lab is a hand-only MuJoCo dexterous manipulation benchmark. It uses a custom human-like five-finger hand with an opposable thumb, independent finger timing, object-specific grasp strategies, contact-rich logging, a 224-degree cap/knob rotation task, slip recovery, load hold, stylus interaction, and index-only button pressing.

Task Goal
Demonstrate a reproducible dexterous hand benchmark that goes beyond simple pick-and-place. The hand classifies objects using simulation-native MuJoCo state, selects a human-inspired grasp primitive, moves each finger according to its role, verifies multi-finger contact, and only then carries or rotates the object.

Technical Approach
- Custom MuJoCo five-finger skeletal hand built from stable primitives.
- Thumb CMC-like opposition and long-finger MCP/PIP/DIP-style joints.
- Object classifier for sphere, cube, cylinder, stylus, button, and cap/knob affordances.
- Human grasp library with sphere enclosure, cube opposing-face, cylinder side-body, tripod stylus, button press, in-hand rotation, and cap/knob rotation primitives.
- No-snap verified grasp routine: objects are not moved before stable contact verification.
- Tactile/contact proxy evidence from five fingertip channels.
- Tactile-inspired minimum-jerk controller trace and report.
- Stress evaluation with baseline vs feedback comparison.
- 20-gate deterministic task verification suite.
- Hardware replay safety audit for possible LEAP/Shadow-style transfer, without claiming physical hardware execution.

Core Features
- Human-like five-finger hand with visible thumb opposition.
- Sphere enclosure/cage grasp.
- Cube opposing-face grasp, not corner grasp.
- Cylinder side-body grasp, not top-down.
- In-hand cylinder rotation with finger-gaiting metrics.
- Signature 224-degree cap/knob twist with visible marker.
- Slip recovery and 9x load-hold evidence.
- Stylus tripod grasp and checkpoint touch.
- Index-only button press.
- Contact timeline with per-finger contacts, roles, tactile proxies, friction margin, and slip metrics.
- Stress eval, task suite, judge brief, rubric scorecard, validator, keyframes, narration SRT, and final report.

How to Run
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

Outputs
- submissions/dexhand_lab/outputs/demo.mp4
- submissions/dexhand_lab/media/demo.mp4
- submissions/dexhand_lab/media/keyframes.png
- submissions/dexhand_lab/outputs/summary.json
- submissions/dexhand_lab/outputs/contact_timeline.json
- submissions/dexhand_lab/outputs/final_report.txt
- submissions/dexhand_lab/outputs/stress_eval.json
- submissions/dexhand_lab/outputs/baseline_vs_feedback.json
- submissions/dexhand_lab/dataset/task_suite_report.json
- submissions/dexhand_lab/dataset/tactile_feedback_report.json
- submissions/dexhand_lab/dataset/tactile_taxels.csv
- submissions/dexhand_lab/dataset/minimum_jerk_report.json
- submissions/dexhand_lab/dataset/stress_eval.json
- submissions/dexhand_lab/dataset/hardware_adaptation_report.json

Current Validation Snapshot
- Demo duration: about 101.6 seconds
- Cap rotation target/achieved: 224 / 224 degrees
- Task gates: 20/20
- Tactile channels: 5
- Final slip: 0.28 mm
- Load hold: 9.0x
- Object snap events: 0
- Stress rollouts: 32
- Feedback success rate: 100.0%
- Baseline success rate: 59.4%
- Validator: pass

Limitations
- Heuristic/contact-aware controller, not learned RL.
- Simulation-native object pose perception, not camera vision.
- Tactile values are MuJoCo/contact-controller proxies, not physical sensor readings.
- Hybrid carry/cap rotation are used after contact verification for reproducibility.
- Hardware audit is replay/safety validation, not a physical robot trial.
