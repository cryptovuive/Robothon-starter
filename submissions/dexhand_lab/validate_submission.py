from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_DIR = PROJECT_DIR / "dataset"


def valid_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def main() -> int:
    required_files = [
        PROJECT_DIR / "registration.json",
        PROJECT_DIR / "README.md",
        PROJECT_DIR / "JUDGE_BRIEF.md",
        PROJECT_DIR / "run_demo.py",
        PROJECT_DIR / "run_stress_eval.py",
        PROJECT_DIR / "arena_task_suite.py",
        PROJECT_DIR / "minimum_jerk_controller.py",
        PROJECT_DIR / "contact_feedback_audit.py",
        PROJECT_DIR / "hardware_adaptation_audit.py",
        PROJECT_DIR / "scene.xml",
        PROJECT_DIR / "human_grasp_library.py",
        PROJECT_DIR / "object_classifier.py",
        PROJECT_DIR / "dexhand_controller.py",
        PROJECT_DIR / "rubric_scorecard.json",
        PROJECT_DIR / "submission_manifest.json",
        OUTPUT_DIR / "demo.mp4",
        OUTPUT_DIR / "summary.json",
        OUTPUT_DIR / "trajectory.json",
        OUTPUT_DIR / "contact_timeline.json",
        OUTPUT_DIR / "final_report.txt",
        OUTPUT_DIR / "narration.srt",
        OUTPUT_DIR / "policy_card.json",
        OUTPUT_DIR / "sensor_manifest.json",
        OUTPUT_DIR / "stress_eval.json",
        OUTPUT_DIR / "baseline_vs_feedback.json",
        OUTPUT_DIR / "stress_eval_summary.csv",
        PROJECT_DIR / "media" / "keyframes.png",
        PROJECT_DIR / "media" / "demo.mp4",
        DATASET_DIR / "task_suite_report.json",
        DATASET_DIR / "task_suite.csv",
        DATASET_DIR / "tactile_feedback_report.json",
        DATASET_DIR / "tactile_taxels.csv",
        DATASET_DIR / "minimum_jerk_report.json",
        DATASET_DIR / "minimum_jerk_trace.csv",
        DATASET_DIR / "stress_eval.json",
        DATASET_DIR / "hardware_adaptation_report.json",
        DATASET_DIR / "hardware_command_stream.csv",
        DATASET_DIR / "sim2real_safety_case.json",
        PROJECT_DIR / "hardware_transfer.json",
        PROJECT_DIR / "HARDWARE_ADAPTATION.md",
        OUTPUT_DIR / "episodes" / "episode_000" / "trajectory.json",
        OUTPUT_DIR / "episodes" / "episode_000" / "metadata.json",
    ]
    json_files = [
        PROJECT_DIR / "registration.json",
        PROJECT_DIR / "rubric_scorecard.json",
        PROJECT_DIR / "submission_manifest.json",
        OUTPUT_DIR / "summary.json",
        OUTPUT_DIR / "trajectory.json",
        OUTPUT_DIR / "contact_timeline.json",
        OUTPUT_DIR / "policy_card.json",
        OUTPUT_DIR / "sensor_manifest.json",
        OUTPUT_DIR / "stress_eval.json",
        OUTPUT_DIR / "baseline_vs_feedback.json",
        DATASET_DIR / "task_suite_report.json",
        DATASET_DIR / "tactile_feedback_report.json",
        DATASET_DIR / "minimum_jerk_report.json",
        DATASET_DIR / "stress_eval.json",
        DATASET_DIR / "hardware_adaptation_report.json",
        DATASET_DIR / "sim2real_safety_case.json",
        PROJECT_DIR / "hardware_transfer.json",
        OUTPUT_DIR / "episodes" / "episode_000" / "trajectory.json",
        OUTPUT_DIR / "episodes" / "episode_000" / "metadata.json",
    ]
    missing = [path for path in required_files if not path.exists()]
    invalid = [path for path in json_files if path.exists() and not valid_json(path)]
    if missing or invalid:
        print("DexHand validation failed")
        if missing:
            print("Missing files:")
            for path in missing:
                print(f"- {path}")
        if invalid:
            print("Invalid JSON:")
            for path in invalid:
                print(f"- {path}")
        return 1
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    required_metrics = [
        "hand_skeleton_valid",
        "five_fingers_present",
        "thumb_opposition_joint_present",
        "object_snap_events",
        "attach_before_verification_count",
        "verified_grasp_before_attach_rate",
        "sphere_enclosure_grasp_success",
        "cube_opposing_face_grasp_success",
        "cylinder_side_body_grasp_success",
        "top_down_cylinder_grasp_count",
        "in_hand_rotation_success",
        "achieved_rotation_deg",
        "rotation_error_deg",
        "stylus_tripod_success",
        "checkpoint_touch_success",
        "index_only_button_press_success",
        "stress_eval_available",
        "tactile_channels",
        "cap_rotation_target_deg",
        "cap_rotation_achieved_deg",
        "cap_rotation_error_deg",
        "cap_rotation_success",
        "final_slip_mm",
        "max_slip_mm",
        "slip_recovery_success",
        "load_hold_x",
        "load_hold_success",
        "object_drop_count",
        "task_gate_count",
        "task_gates_passed",
        "task_gate_success_rate",
        "stress_rollouts",
        "stress_success_rate",
        "baseline_success_rate",
        "feedback_success_rate",
        "improvement_percentage",
        "average_active_fingers_dexterous_grasps",
        "average_multi_side_contact_score_dexterous_grasps",
        "minimum_jerk_controller_pass",
        "hardware_audit_pass",
        "object_center_between_fingers_rate",
        "contact_timeline_path",
        "overall_task_success",
    ]
    missing_metrics = [metric for metric in required_metrics if metric not in summary]
    if missing_metrics:
        print("DexHand validation failed")
        print("Missing summary metrics:")
        for metric in missing_metrics:
            print(f"- {metric}")
        return 1
    expected_values = {
        "hand_skeleton_valid": True,
        "five_fingers_present": True,
        "thumb_opposition_joint_present": True,
        "sphere_enclosure_grasp_success": True,
        "cube_opposing_face_grasp_success": True,
        "cylinder_side_body_grasp_success": True,
        "in_hand_rotation_success": True,
        "stylus_tripod_success": True,
        "checkpoint_touch_success": True,
        "index_only_button_press_success": True,
        "cap_rotation_success": True,
        "slip_recovery_success": True,
        "load_hold_success": True,
        "minimum_jerk_controller_pass": True,
        "hardware_audit_pass": True,
        "overall_task_success": True,
    }
    bad_values = [
        f"{metric} expected {expected!r}, got {summary.get(metric)!r}"
        for metric, expected in expected_values.items()
        if summary.get(metric) != expected
    ]
    if int(summary.get("object_snap_events", 1)) != 0:
        bad_values.append("object_snap_events expected 0")
    if int(summary.get("attach_before_verification_count", 1)) != 0:
        bad_values.append("attach_before_verification_count expected 0")
    if int(summary.get("top_down_cylinder_grasp_count", 1)) != 0:
        bad_values.append("top_down_cylinder_grasp_count expected 0")
    if float(summary.get("verified_grasp_before_attach_rate", 0.0)) < 0.99:
        bad_values.append("verified_grasp_before_attach_rate expected >= 0.99")
    if float(summary.get("object_center_between_fingers_rate", 0.0)) < 0.99:
        bad_values.append("object_center_between_fingers_rate expected >= 0.99")
    if not bool(summary.get("stress_eval_available", False)):
        bad_values.append("stress_eval_available expected true; run run_stress_eval.py --seeds 32")
    if int(summary.get("tactile_channels", 0)) != 5:
        bad_values.append("tactile_channels expected 5")
    if float(summary.get("cap_rotation_target_deg", 0.0)) != 224.0:
        bad_values.append("cap_rotation_target_deg expected 224")
    if float(summary.get("cap_rotation_achieved_deg", 0.0)) < 214.0:
        bad_values.append("cap_rotation_achieved_deg expected >= 214")
    if float(summary.get("final_slip_mm", 999.0)) > 0.5:
        bad_values.append("final_slip_mm expected <= 0.5")
    if float(summary.get("load_hold_x", 0.0)) < 5.0:
        bad_values.append("load_hold_x expected >= 5.0")
    if float(summary.get("task_gate_success_rate", 0.0)) < 0.90:
        bad_values.append("task_gate_success_rate expected >= 0.90")
    if float(summary.get("feedback_success_rate", 0.0)) < float(summary.get("baseline_success_rate", 0.0)):
        bad_values.append("feedback_success_rate expected >= baseline_success_rate")
    if float(summary.get("average_active_fingers_dexterous_grasps", 0.0)) < 4.0:
        bad_values.append("average_active_fingers_dexterous_grasps expected >= 4.0")
    if float(summary.get("average_multi_side_contact_score_dexterous_grasps", 0.0)) < 0.80:
        bad_values.append("average_multi_side_contact_score_dexterous_grasps expected >= 0.80")
    if bad_values:
        print("DexHand validation failed")
        print("Unexpected summary values:")
        for item in bad_values:
            print(f"- {item}")
        return 1
    print("DexHand validation passed")
    print(f"Summary: {OUTPUT_DIR / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
