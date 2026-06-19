# DexHand Lab Judge Brief

## One-Sentence Summary

DexHand Lab is a human-like five-finger MuJoCo hand benchmark for object-specific grasping, cylinder in-hand rotation, a 224-degree cap/knob twist, tactile/contact evidence, slip recovery, load hold, stylus interaction, and index-only button pressing.

## Why This Targets 90+

The submission focuses on dexterity evidence instead of a simple pick-and-place animation: five independent fingers, thumb opposition, object-specific grasp strategies, no-snap verification before object motion, tactile proxy streams, signature cap rotation, load-hold recovery, stress evaluation, and a multi-gate judge checklist.

## Inspect First

1. `media/demo.mp4`
2. `media/keyframes.png`
3. `outputs/summary.json`
4. `dataset/task_suite_report.json`
5. `dataset/tactile_feedback_report.json`
6. `dataset/tactile_taxels.csv`
7. `dataset/minimum_jerk_report.json`
8. `dataset/stress_eval.json`
9. `outputs/baseline_vs_feedback.json`
10. `dataset/hardware_adaptation_report.json`
11. `rubric_scorecard.json`
12. `validate_submission.py`

## Quantitative Evidence

The main demo and evidence scripts report:

- 20-gate deterministic dexterity suite with actual passed count in `dataset/task_suite_report.json`
- cap rotation target: 224 degrees
- cap rotation achieved: saved as `cap_rotation_achieved_deg`
- final slip: saved as `final_slip_mm`
- load hold: saved as `load_hold_x`
- tactile channels: 5 fingertip streams
- stress success rate and baseline-vs-feedback comparison
- object snap events: expected 0
- average active fingers for dexterous grasps
- average multi-side contact score for dexterous grasps
- hardware replay audit status

## Rubric Mapping

- Reproducibility: fixed seed CLI, deterministic task suite, validator, stress evaluation.
- MuJoCo depth: articulated hand MJCF, named geoms/sites, contact timeline, object state logs, cap hinge joint.
- Task design: sphere, cube, cylinder, cap rotation, slip/load-hold, stylus checkpoint, button press.
- Control: contact-aware verified grasp routine, minimum-jerk tactile-inspired segments, no-snap policy.
- Dexterity: thumb opposition, independent finger roles, multi-side contact, cylinder rotation, cap twist.
- Engineering quality: JSON/CSV evidence pack, validator, manifest, final report, structured modules.
- Presentation: long demo video, keyframes, narration SRT, final evidence report.
- Innovation: cap/knob 224-degree marker task, tactile proxy audit, hardware replay safety audit.

## Honest Scope

The project uses simulation-native object pose perception and a hybrid contact-aware dexterous manipulation routine. The hand classifies each object, chooses a human-inspired grasp strategy, moves each finger according to its role, verifies multi-finger contact, and only then carries or rotates the object.

This is not a learned RL policy, not real camera vision, not a claim of perfect contact physics, and not a real hardware execution. The tactile channels are MuJoCo/contact-controller proxies. The hardware adaptation audit is a replay/safety validation artifact for possible LEAP/Shadow-style transfer.

## Key Files

- `run_demo.py`: main deterministic demo and data writer.
- `scene.xml`: custom five-finger MJCF hand and task board objects.
- `human_grasp_library.py`: grasp primitives and finger roles.
- `object_classifier.py`: simulation-native affordance classifier.
- `minimum_jerk_controller.py`: tactile-inspired trajectory evidence.
- `contact_feedback_audit.py`: five-fingertip tactile evidence.
- `arena_task_suite.py`: 20-gate verification suite.
- `run_stress_eval.py`: fixed-seed stress evaluation and baseline comparison.
- `hardware_adaptation_audit.py`: simulated hardware replay audit.
- `validate_submission.py`: final evidence and metric validator.
