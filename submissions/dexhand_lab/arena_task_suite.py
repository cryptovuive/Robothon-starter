from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATASET_DIR = PROJECT_DIR / "dataset"


def read_summary() -> dict:
    path = OUTPUT_DIR / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def bool_metric(summary: dict, key: str) -> bool:
    return bool(summary.get(key, False))


def build_gates(summary: dict) -> list[dict]:
    return [
        {"gate": 1, "name": "hand_skeleton_valid", "passed": bool_metric(summary, "hand_skeleton_valid")},
        {"gate": 2, "name": "five_fingers_present", "passed": bool_metric(summary, "five_fingers_present")},
        {"gate": 3, "name": "thumb_opposition_visible", "passed": bool_metric(summary, "thumb_opposition_visible")},
        {"gate": 4, "name": "fingertip_pads_present", "passed": bool_metric(summary, "fingertip_pads_present")},
        {"gate": 5, "name": "sphere_classified", "passed": bool_metric(summary, "sphere_enclosure_grasp_success")},
        {"gate": 6, "name": "sphere_enclosure_grasp_success", "passed": bool_metric(summary, "sphere_enclosure_grasp_success")},
        {"gate": 7, "name": "cube_classified", "passed": bool_metric(summary, "cube_opposing_face_grasp_success")},
        {"gate": 8, "name": "cube_opposing_face_grasp_success", "passed": bool_metric(summary, "cube_opposing_face_grasp_success")},
        {"gate": 9, "name": "cylinder_classified", "passed": bool_metric(summary, "cylinder_side_body_grasp_success")},
        {"gate": 10, "name": "cylinder_side_body_grasp_success", "passed": bool_metric(summary, "cylinder_side_body_grasp_success")},
        {"gate": 11, "name": "top_down_cylinder_grasp_count_zero", "passed": int(summary.get("top_down_cylinder_grasp_count", 1)) == 0},
        {"gate": 12, "name": "stylus_classified", "passed": bool_metric(summary, "stylus_tripod_success")},
        {"gate": 13, "name": "stylus_tripod_success", "passed": bool_metric(summary, "stylus_tripod_success")},
        {"gate": 14, "name": "checkpoint_touch_success", "passed": bool_metric(summary, "checkpoint_touch_success")},
        {"gate": 15, "name": "index_only_button_press_success", "passed": bool_metric(summary, "index_only_button_press_success")},
        {"gate": 16, "name": "cap_object_classified", "passed": bool_metric(summary, "cap_marker_visible")},
        {"gate": 17, "name": "cap_rotation_224_success", "passed": bool_metric(summary, "cap_rotation_success") and float(summary.get("cap_rotation_achieved_deg", 0.0)) >= 214.0},
        {"gate": 18, "name": "slip_recovery_success", "passed": bool_metric(summary, "slip_recovery_success")},
        {"gate": 19, "name": "load_hold_success", "passed": bool_metric(summary, "load_hold_success") and float(summary.get("load_hold_x", 0.0)) >= 5.0},
        {"gate": 20, "name": "no_snap_verified", "passed": int(summary.get("object_snap_events", 1)) == 0 and int(summary.get("attach_before_verification_count", 1)) == 0},
    ]


def update_summary(report: dict) -> None:
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "task_gate_count": int(report["gate_count"]),
            "task_gates_passed": int(report["gates_passed"]),
            "task_gate_success_rate": float(report["success_rate"]),
            "task_suite_report_path": "submissions/dexhand_lab/dataset/task_suite_report.json",
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_summary()
    gates = build_gates(summary)
    passed = sum(1 for gate in gates if gate["passed"])
    report = {
        "project": "DexHand Lab",
        "suite": "20-gate deterministic dexterity verification",
        "gate_count": len(gates),
        "gates_passed": passed,
        "success_rate": round(passed / len(gates), 5),
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "max_pose_error_m": float(summary.get("average_grasp_centroid_error_m", 0.0)),
        "max_rotation_error_deg": float(summary.get("cap_rotation_error_deg", 0.0)),
        "final_task_success": passed >= 18,
        "gates": gates,
    }
    (DATASET_DIR / "task_suite_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (DATASET_DIR / "task_suite.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate", "name", "passed"])
        writer.writeheader()
        writer.writerows(gates)
    update_summary(report)
    print("DexHand task suite")
    print("------------------")
    print(f"Gates passed: {passed}/{len(gates)}")
    if report["failed_gates"]:
        print("Failed gates: " + ", ".join(report["failed_gates"]))
    return 0 if passed >= 18 else 1


if __name__ == "__main__":
    raise SystemExit(main())
