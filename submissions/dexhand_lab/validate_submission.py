from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


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
        OUTPUT_DIR / "policy_card.json",
        OUTPUT_DIR / "sensor_manifest.json",
        OUTPUT_DIR / "stress_eval.json",
        OUTPUT_DIR / "baseline_vs_feedback.json",
        PROJECT_DIR / "media" / "keyframes.png",
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
