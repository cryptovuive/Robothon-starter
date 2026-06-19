from __future__ import annotations

from dataclasses import dataclass

from human_grasp_library import FINGER_JOINTS, NORMALIZED_FINGER_LENGTHS


PIPELINE_STATES = (
    "RESET",
    "SHOW_HAND_OPEN_CLOSE",
    "OBJECT_CLASSIFY",
    "GRASP_REFERENCE_SELECT",
    "HAND_PRESHAPE",
    "APPROACH_OBJECT",
    "CONTACT_SEEK",
    "SOFT_CLOSE",
    "CONTACT_ESTIMATION",
    "STABILITY_VERIFY",
    "SECURE_GRASP",
    "HOLD_STABLE",
    "IN_HAND_ROTATION",
    "SLIP_MONITOR",
    "SLIP_RECOVERY",
    "TRIPOD_TOOL_PICK",
    "CHECKPOINT_TOUCH",
    "INDEX_BUTTON_PRESS",
    "CONTROLLED_RELEASE",
    "FINAL_REPORT",
    "FINISH",
)


@dataclass(frozen=True)
class DynamicGraspPipeline:
    grasp_reference: str
    object_name: str
    object_type: str
    states: tuple[str, ...] = PIPELINE_STATES
    hybrid_carry_used: bool = True
    contact_aware: bool = True


def _has_joint(model, mujoco, joint_name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0


def _has_geom(model, mujoco, geom_name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) >= 0


def _joint_has_limit(model, mujoco, joint_name: str) -> bool:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return joint_id >= 0 and bool(model.jnt_limited[joint_id])


def validate_hand_skeleton(model, mujoco) -> dict:
    fingers = ("thumb", "index", "middle", "ring", "little")
    finger_joints = {
        "thumb": ("thumb_cmc_opposition", "thumb_cmc_abduction", "thumb_mcp_flexion", "thumb_ip_flexion"),
        "index": ("index_mcp_flexion", "index_mcp_abduction", "index_pip_flexion", "index_dip_flexion"),
        "middle": ("middle_mcp_flexion", "middle_mcp_abduction", "middle_pip_flexion", "middle_dip_flexion"),
        "ring": ("ring_mcp_flexion", "ring_mcp_abduction", "ring_pip_flexion", "ring_dip_flexion"),
        "little": ("little_mcp_flexion", "little_mcp_abduction", "little_pip_flexion", "little_dip_flexion"),
    }
    joint_presence = {
        finger: all(_has_joint(model, mujoco, joint_name) for joint_name in joints)
        for finger, joints in finger_joints.items()
    }
    fingertip_pads = {
        finger: _has_geom(model, mujoco, f"{finger}_tip_pad")
        for finger in fingers
    }
    safe_limits = all(_joint_has_limit(model, mujoco, joint_name) for joint_name in FINGER_JOINTS)
    length_order_ok = (
        NORMALIZED_FINGER_LENGTHS["middle"] > NORMALIZED_FINGER_LENGTHS["ring"]
        > NORMALIZED_FINGER_LENGTHS["index"] > NORMALIZED_FINGER_LENGTHS["little"]
        and NORMALIZED_FINGER_LENGTHS["thumb"] < NORMALIZED_FINGER_LENGTHS["index"]
    )
    five_fingers = all(joint_presence.values())
    fingertip_pads_present = all(fingertip_pads.values())
    result = {
        "five_fingers_present": bool(five_fingers),
        "thumb_opposition_joint_present": _has_joint(model, mujoco, "thumb_cmc_opposition"),
        "thumb_abduction_joint_present": _has_joint(model, mujoco, "thumb_cmc_abduction"),
        "index_joints": "MCP/PIP/DIP" if joint_presence["index"] else "missing",
        "middle_joints": "MCP/PIP/DIP" if joint_presence["middle"] else "missing",
        "ring_joints": "MCP/PIP/DIP" if joint_presence["ring"] else "missing",
        "little_joints": "MCP/PIP/DIP" if joint_presence["little"] else "missing",
        "finger_length_order_ok": bool(length_order_ok),
        "fingertip_pads_present": bool(fingertip_pads_present),
        "all_joints_limited": bool(safe_limits),
    }
    result["hand_skeleton_valid"] = all(
        bool(result[key])
        for key in (
            "five_fingers_present",
            "thumb_opposition_joint_present",
            "thumb_abduction_joint_present",
            "finger_length_order_ok",
            "fingertip_pads_present",
            "all_joints_limited",
        )
    )
    return result


def format_skeleton_check(check: dict) -> str:
    return "\n".join(
        [
            "[HAND SKELETON CHECK]",
            f"five_fingers_present: {str(bool(check.get('five_fingers_present'))).lower()}",
            f"thumb_opposition_joint: {str(bool(check.get('thumb_opposition_joint_present'))).lower()}",
            f"index_joints: {check.get('index_joints')}",
            f"middle_joints: {check.get('middle_joints')}",
            f"ring_joints: {check.get('ring_joints')}",
            f"little_joints: {check.get('little_joints')}",
            f"finger_length_order_ok: {str(bool(check.get('finger_length_order_ok'))).lower()}",
            f"fingertip_pads_present: {str(bool(check.get('fingertip_pads_present'))).lower()}",
            f"hand_skeleton_valid: {str(bool(check.get('hand_skeleton_valid'))).lower()}",
        ]
    )
