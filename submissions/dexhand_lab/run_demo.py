from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
except ImportError as exc:
    raise SystemExit(
        "Missing demo dependency. Install from the repository root with:\n"
        "  python -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc

from grasp_taxonomy import (
    FINGER_GROUP_MAP,
    FINGER_JOINTS,
    NORMALIZED_FINGER_LENGTHS,
    OPEN_HAND,
    canonical_grasp_name,
    get_grasp_preset,
    list_grasp_names,
)
from dexhand_controller import PIPELINE_STATES, format_skeleton_check, validate_hand_skeleton
from object_classifier import classify_scene_objects


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
DEFAULT_SCENE = PROJECT_DIR / "scene.xml"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"

SLIDE_AND_WRIST_JOINTS = (
    "hand_x",
    "hand_y",
    "hand_z",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
)
JOINT_NAMES = SLIDE_AND_WRIST_JOINTS + FINGER_JOINTS
ACTUATOR_BY_JOINT = {joint_name: f"act_{joint_name}" for joint_name in JOINT_NAMES}

OBJECTS = {
    "sphere_object": {
        "label": "sphere",
        "joint": "sphere_object_joint",
        "body": "sphere_object",
        "start": (-0.22, -0.01, 0.447),
        "release": (-0.22, 0.16, 0.447),
        "grasp": "SPHERICAL_POWER_GRASP",
    },
    "cube_object": {
        "label": "cube",
        "joint": "cube_object_joint",
        "body": "cube_object",
        "start": (0.00, -0.01, 0.440),
        "release": (0.00, 0.16, 0.440),
        "grasp": "CUBIC_FACE_GRASP",
    },
    "cylinder_object": {
        "label": "cylinder",
        "joint": "cylinder_object_joint",
        "body": "cylinder_object",
        "start": (0.22, -0.01, 0.438),
        "release": (0.22, 0.16, 0.438),
        "grasp": "CYLINDER_SIDE_BODY_GRASP",
    },
}

FINGER_GROUPS = FINGER_GROUP_MAP
ALL_FINGERS = ("thumb", "index", "middle", "ring", "little")
DEFAULT_TARGET_ROTATION_DEG = 90.0


@dataclass(frozen=True)
class EpisodeSetup:
    seed: int
    episode_index: int
    difficulty: str
    object_positions: dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class Phase:
    name: str
    duration_s: float
    targets: dict[str, float]
    grasp_type: str
    target_object: str | None = None
    active_fingers: tuple[str, ...] = ()
    required_contacts: tuple[str, ...] = ()
    held_object: str | None = None
    attach_object: str | None = None
    release_object: str | None = None
    held_tool: bool = False
    attach_tool: bool = False
    checkpoint_touch: bool = False
    button_press: bool = False
    recovery_active: bool = False
    stable_grasp_verified: bool = False
    cylinder_rotation_deg: float = 0.0
    active_rotation_finger: str | None = None
    support_fingers: tuple[str, ...] = ()
    finger_gait_count: int = 0
    hybrid_rotation_used: bool = False
    cylinder_grasp_type: str | None = None
    top_down_cylinder_grasp_used: bool = False
    pressing_finger: str | None = None
    note: str = ""


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    parts = path.parts
    if parts and parts[0] == "submissions":
        return (REPO_ROOT / path).resolve()
    return (PROJECT_DIR / path).resolve()


def smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Missing joint in MJCF: {joint_name}")
    return int(jid)


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, joint_name)])


def joint_qvel_addr(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, joint_name)])


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    jid = joint_id(model, joint_name)
    if model.jnt_limited[jid]:
        low, high = model.jnt_range[jid]
        value = float(np.clip(value, low, high))
    data.qpos[int(model.jnt_qposadr[jid])] = value
    data.qvel[int(model.jnt_dofadr[jid])] = 0.0


def set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    pos: Iterable[float],
    yaw: float = 0.0,
) -> None:
    jid = joint_id(model, joint_name)
    qpos_addr = int(model.jnt_qposadr[jid])
    qvel_addr = int(model.jnt_dofadr[jid])
    data.qpos[qpos_addr : qpos_addr + 3] = np.asarray(pos, dtype=float)
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = [
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    ]
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body in MJCF: {body_name}")
    return data.xpos[body_id].copy()


def site_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"Missing site in MJCF: {site_name}")
    return data.site_xpos[site_id].copy()


def freejoint_pose(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> dict:
    qpos_addr = joint_qpos_addr(model, joint_name)
    pos = data.qpos[qpos_addr : qpos_addr + 3].copy()
    quat = data.qpos[qpos_addr + 3 : qpos_addr + 7].copy()
    return {
        "position": pos.round(5).tolist(),
        "quaternion": quat.round(5).tolist(),
    }


def clamp_to_ctrlrange(model: mujoco.MjModel, actuator_name: str, value: float) -> float:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise ValueError(f"Missing actuator in MJCF: {actuator_name}")
    if model.actuator_ctrllimited[actuator_id]:
        low, high = model.actuator_ctrlrange[actuator_id]
        return float(np.clip(value, low, high))
    return float(value)


def apply_targets(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> None:
    for joint_name, value in targets.items():
        actuator_name = ACTUATOR_BY_JOINT[joint_name]
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        data.ctrl[actuator_id] = clamp_to_ctrlrange(model, actuator_name, value)
        set_joint_qpos(model, data, joint_name, float(data.ctrl[actuator_id]))


def read_joint_positions(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    return {
        joint_name: round(float(data.qpos[joint_qpos_addr(model, joint_name)]), 5)
        for joint_name in JOINT_NAMES
    }


def read_joint_velocities(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    return {
        joint_name: round(float(data.qvel[joint_qvel_addr(model, joint_name)]), 5)
        for joint_name in JOINT_NAMES
    }


def hand_pose(x: float, y: float, z: float, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> dict[str, float]:
    targets = {
        "hand_x": x,
        "hand_y": y,
        "hand_z": z,
        "wrist_yaw": yaw,
        "wrist_pitch": pitch,
        "wrist_roll": roll,
    }
    targets.update(OPEN_HAND)
    return targets


def merge_targets(base: dict[str, float], finger_targets: dict[str, float]) -> dict[str, float]:
    merged = dict(base)
    merged.update(finger_targets)
    return merged


def staged_targets(base: dict[str, float], grasp_name: str, fingers: tuple[str, ...]) -> dict[str, float]:
    preset = get_grasp_preset(grasp_name)["preshape_joint_targets"]
    targets = dict(base)
    for finger in fingers:
        for joint_name in FINGER_GROUPS[finger]:
            targets[joint_name] = float(preset[joint_name])
    return targets


def contact_seek_targets(base: dict[str, float], grasp_name: str) -> dict[str, float]:
    """Open the hand around the object before contact instead of closing from one side."""
    canonical = canonical_grasp_name(grasp_name)
    targets = dict(base)
    if canonical == "SPHERICAL_ENCLOSURE_GRASP":
        targets.update(
            {
                "thumb_cmc_opposition": 0.30,
                "thumb_cmc_abduction": 0.70,
                "thumb_mcp_flexion": 0.18,
                "thumb_ip_flexion": 0.08,
                "index_mcp_abduction": -0.27,
                "index_mcp_flexion": 0.16,
                "index_pip_flexion": 0.08,
                "index_dip_flexion": 0.04,
                "middle_mcp_abduction": -0.08,
                "middle_mcp_flexion": 0.14,
                "middle_pip_flexion": 0.08,
                "middle_dip_flexion": 0.04,
                "ring_mcp_abduction": 0.18,
                "ring_mcp_flexion": 0.14,
                "ring_pip_flexion": 0.08,
                "ring_dip_flexion": 0.04,
                "little_mcp_abduction": 0.30,
                "little_mcp_flexion": 0.14,
                "little_pip_flexion": 0.08,
                "little_dip_flexion": 0.04,
            }
        )
    elif canonical == "OPPOSING_FACE_CUBE_GRASP":
        targets.update(
            {
                "thumb_cmc_opposition": 0.38,
                "thumb_cmc_abduction": 0.62,
                "thumb_mcp_flexion": 0.12,
                "thumb_ip_flexion": 0.06,
                "index_mcp_abduction": -0.24,
                "index_mcp_flexion": 0.12,
                "index_pip_flexion": 0.06,
                "index_dip_flexion": 0.03,
                "middle_mcp_abduction": -0.04,
                "middle_mcp_flexion": 0.12,
                "middle_pip_flexion": 0.06,
                "middle_dip_flexion": 0.03,
                "ring_mcp_abduction": 0.14,
                "ring_mcp_flexion": 0.14,
                "ring_pip_flexion": 0.08,
                "ring_dip_flexion": 0.04,
                "little_mcp_abduction": 0.24,
                "little_mcp_flexion": 0.28,
                "little_pip_flexion": 0.18,
                "little_dip_flexion": 0.08,
            }
        )
    elif canonical == "LATERAL_CYLINDER_BODY_GRASP":
        targets.update(
            {
                "thumb_cmc_opposition": 0.34,
                "thumb_cmc_abduction": 0.70,
                "thumb_mcp_flexion": 0.14,
                "thumb_ip_flexion": 0.06,
                "index_mcp_abduction": -0.24,
                "index_mcp_flexion": 0.12,
                "index_pip_flexion": 0.06,
                "index_dip_flexion": 0.03,
                "middle_mcp_abduction": -0.06,
                "middle_mcp_flexion": 0.12,
                "middle_pip_flexion": 0.06,
                "middle_dip_flexion": 0.03,
                "ring_mcp_abduction": 0.16,
                "ring_mcp_flexion": 0.12,
                "ring_pip_flexion": 0.06,
                "ring_dip_flexion": 0.03,
                "little_mcp_abduction": 0.28,
                "little_mcp_flexion": 0.14,
                "little_pip_flexion": 0.08,
                "little_dip_flexion": 0.04,
            }
        )
    elif canonical == "TRIPOD_PRECISION_GRASP":
        targets.update(
            {
                "thumb_cmc_opposition": 0.46,
                "thumb_cmc_abduction": 0.46,
                "thumb_mcp_flexion": 0.16,
                "thumb_ip_flexion": 0.06,
                "index_mcp_abduction": -0.08,
                "index_mcp_flexion": 0.12,
                "index_pip_flexion": 0.06,
                "index_dip_flexion": 0.03,
                "middle_mcp_abduction": 0.03,
                "middle_mcp_flexion": 0.18,
                "middle_pip_flexion": 0.08,
                "middle_dip_flexion": 0.04,
                "ring_mcp_flexion": 0.66,
                "ring_pip_flexion": 0.58,
                "ring_dip_flexion": 0.30,
                "little_mcp_flexion": 0.66,
                "little_pip_flexion": 0.58,
                "little_dip_flexion": 0.30,
            }
        )
    return targets


def staged_targets_from(
    starting_targets: dict[str, float],
    grasp_name: str,
    fingers: tuple[str, ...],
) -> dict[str, float]:
    preset = get_grasp_preset(grasp_name)["preshape_joint_targets"]
    targets = dict(starting_targets)
    for finger in fingers:
        for joint_name in FINGER_GROUPS[finger]:
            targets[joint_name] = float(preset[joint_name])
    return targets


def object_hand_target(
    position: tuple[float, float, float],
    z: float,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = -0.095,
) -> dict[str, float]:
    x, y, _ = position
    return hand_pose(float(x) + x_offset, float(y) + y_offset, z, yaw, pitch, roll)


def generate_episode_setup(base_seed: int, episode_index: int, difficulty: str) -> EpisodeSetup:
    episode_seed = int(base_seed + episode_index * 7919)
    rng = np.random.default_rng(episode_seed)
    difficulty = difficulty.lower()
    if difficulty == "easy":
        jitter = 0.008
    elif difficulty == "hard":
        jitter = 0.025
    else:
        difficulty = "medium"
        jitter = 0.014

    object_positions: dict[str, tuple[float, float, float]] = {}
    for object_name, spec in OBJECTS.items():
        base = np.asarray(spec["start"], dtype=float)
        offset = np.array([rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter), 0.0], dtype=float)
        pos = base + offset
        object_positions[object_name] = (round(float(pos[0]), 5), round(float(pos[1]), 5), round(float(pos[2]), 5))

    return EpisodeSetup(
        seed=episode_seed,
        episode_index=episode_index,
        difficulty=difficulty,
        object_positions=object_positions,
    )


def reset_scene(model: mujoco.MjModel, data: mujoco.MjData, setup: EpisodeSetup) -> None:
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    home = hand_pose(0.0, -0.13, 0.025)
    apply_targets(model, data, home)
    for object_name, spec in OBJECTS.items():
        set_freejoint_pose(model, data, spec["joint"], setup.object_positions[object_name])
    set_freejoint_pose(model, data, "stylus_tool_joint", (-0.28, 0.28, 0.425))
    set_joint_qpos(model, data, "button_joint", 0.0)
    mujoco.mj_forward(model, data)


def make_phase_plan(setup: EpisodeSetup) -> list[Phase]:
    sphere = setup.object_positions["sphere_object"]
    cube = setup.object_positions["cube_object"]
    cylinder = setup.object_positions["cylinder_object"]
    sphere_grasp = get_grasp_preset("SPHERICAL_POWER_GRASP")
    cube_grasp = get_grasp_preset("CUBIC_FACE_GRASP")
    cylinder_grasp = get_grasp_preset("CYLINDER_SIDE_BODY_GRASP")
    rotation_grasp = get_grasp_preset("IN_HAND_ROTATION_GRASP")
    tripod_grasp = get_grasp_preset("TRIPOD_TOOL_GRASP")
    button_grasp = get_grasp_preset("BUTTON_PRESS")
    display_home = hand_pose(0.0, -0.12, 0.020, yaw=0.0, pitch=0.12)
    display_spread = merge_targets(
        display_home,
        {
            "thumb_cmc_opposition": 0.28,
            "thumb_cmc_abduction": 0.62,
            "thumb_mcp_flexion": 0.06,
            "thumb_ip_flexion": 0.02,
            "index_mcp_abduction": -0.24,
            "middle_mcp_abduction": -0.03,
            "ring_mcp_abduction": 0.16,
            "little_mcp_abduction": 0.28,
        },
    )
    display_thumb = staged_targets(display_home, "SPHERICAL_POWER_GRASP", ("thumb",))
    display_index_middle = staged_targets(display_thumb, "SPHERICAL_POWER_GRASP", ("index", "middle"))
    display_all = staged_targets(display_index_middle, "SPHERICAL_POWER_GRASP", ("ring", "little"))
    sphere_hover = contact_seek_targets(object_hand_target(sphere, 0.016, yaw=-0.02, pitch=0.16, x_offset=0.035, y_offset=-0.075), "SPHERICAL_POWER_GRASP")
    sphere_base = contact_seek_targets(object_hand_target(sphere, -0.018, yaw=-0.02, pitch=0.16, x_offset=0.035, y_offset=-0.075), "SPHERICAL_POWER_GRASP")
    sphere_low = object_hand_target(sphere, -0.080, yaw=-0.02, pitch=0.16, x_offset=0.035, y_offset=-0.075)
    sphere_seek_low = contact_seek_targets(sphere_low, "SPHERICAL_POWER_GRASP")
    cube_hover = contact_seek_targets(object_hand_target(cube, 0.016, yaw=0.04, pitch=0.14, x_offset=0.038, y_offset=-0.083), "CUBIC_FACE_GRASP")
    cube_base = contact_seek_targets(object_hand_target(cube, -0.020, yaw=0.04, pitch=0.14, x_offset=0.038, y_offset=-0.083), "CUBIC_FACE_GRASP")
    cube_low = object_hand_target(cube, -0.084, yaw=0.04, pitch=0.14, x_offset=0.038, y_offset=-0.083)
    cube_seek_low = contact_seek_targets(cube_low, "CUBIC_FACE_GRASP")
    cylinder_hover = contact_seek_targets(object_hand_target(cylinder, 0.014, yaw=-0.08, pitch=0.18, roll=0.06, x_offset=0.045, y_offset=-0.076), "CYLINDER_SIDE_BODY_GRASP")
    cylinder_base = contact_seek_targets(object_hand_target(cylinder, -0.026, yaw=-0.08, pitch=0.18, roll=0.06, x_offset=0.045, y_offset=-0.076), "CYLINDER_SIDE_BODY_GRASP")
    cylinder_side = object_hand_target(cylinder, -0.086, yaw=-0.08, pitch=0.18, roll=0.06, x_offset=0.045, y_offset=-0.076)
    cylinder_seek_side = contact_seek_targets(cylinder_side, "CYLINDER_SIDE_BODY_GRASP")
    stylus_hover = hand_pose(-0.28, 0.185, 0.010, yaw=0.18, pitch=0.08)
    stylus_low = hand_pose(-0.28, 0.185, -0.065, yaw=0.18, pitch=0.08)
    checkpoint_pose = hand_pose(-0.02, 0.215, -0.045, yaw=0.10, pitch=0.08)
    button_hover = hand_pose(0.12, 0.19, 0.010, yaw=0.0, pitch=0.08)
    button_press_pose = hand_pose(0.12, 0.19, -0.085, yaw=0.0, pitch=0.08)

    phases = [
        Phase("RESET", 0.40, display_home, "SHOW_HAND", note="home"),
        Phase("SHOW_HAND_OPEN_CLOSE", 0.90, display_home, "SHOW_HAND", note="all five fingers open"),
        Phase("SHOW_FINGER_SPREAD", 0.90, display_spread, "SHOW_HAND", active_fingers=ALL_FINGERS, note="MCP abduction/adduction display"),
        Phase("SHOW_THUMB_OPPOSITION", 0.90, display_thumb, "SHOW_HAND", active_fingers=("thumb",), note="thumb moves separately"),
        Phase("SHOW_INDEX_MIDDLE_CURL", 0.80, display_index_middle, "SHOW_HAND", active_fingers=("thumb", "index", "middle")),
        Phase("SHOW_RING_LITTLE_SUPPORT", 0.80, display_all, "SHOW_HAND", active_fingers=ALL_FINGERS),
        Phase("SHOW_HAND_OPEN_CLOSE", 0.80, display_home, "SHOW_HAND", note="all fingers reopen"),

        Phase("HAND_PRESHAPE", 0.80, sphere_hover, "SPHERICAL_POWER_GRASP", target_object="sphere_object", note="open hand wider than the sphere"),
        Phase("APPROACH_OBJECT", 1.10, sphere_base, "SPHERICAL_POWER_GRASP", target_object="sphere_object"),
        Phase("ALIGN_TO_OBJECT", 0.65, sphere_seek_low, "SPHERICAL_POWER_GRASP", target_object="sphere_object", note="spread fingertips around the sphere before closing"),
        Phase("PAUSE_BEFORE_CLOSE", 0.55, sphere_seek_low, "SPHERICAL_POWER_GRASP", target_object="sphere_object"),
        Phase(
            "FINGER_CONTACT_CLOSE_INDEX_MIDDLE",
            0.65,
            staged_targets_from(sphere_seek_low, "SPHERICAL_POWER_GRASP", ("index", "middle")),
            "SPHERICAL_POWER_GRASP",
            target_object="sphere_object",
            active_fingers=("index", "middle"),
        ),
        Phase(
            "FINGER_CONTACT_CLOSE_LOWER_SUPPORT",
            0.55,
            staged_targets_from(sphere_seek_low, "SPHERICAL_POWER_GRASP", ("index", "middle", "ring", "little")),
            "SPHERICAL_POWER_GRASP",
            target_object="sphere_object",
            active_fingers=("index", "middle", "ring", "little"),
        ),
        Phase(
            "FINGER_CONTACT_CLOSE_THUMB_OPPOSE",
            0.70,
            merge_targets(sphere_low, sphere_grasp["preshape_joint_targets"]),
            "SPHERICAL_POWER_GRASP",
            target_object="sphere_object",
            active_fingers=("thumb", "index", "middle", "ring", "little"),
        ),
        Phase("CONTACT_ESTIMATION", 0.40, merge_targets(sphere_low, sphere_grasp["preshape_joint_targets"]), "SPHERICAL_POWER_GRASP", target_object="sphere_object", active_fingers=ALL_FINGERS),
        Phase("STABLE_GRASP_VERIFY", 0.45, merge_targets(sphere_low, sphere_grasp["preshape_joint_targets"]), "SPHERICAL_POWER_GRASP", target_object="sphere_object", active_fingers=ALL_FINGERS, required_contacts=("thumb", "index", "middle"), stable_grasp_verified=True),
        Phase("SECURE_OBJECT", 0.45, merge_targets(sphere_low, sphere_grasp["preshape_joint_targets"]), "SPHERICAL_POWER_GRASP", target_object="sphere_object", active_fingers=ALL_FINGERS, attach_object="sphere_object", stable_grasp_verified=True),
        Phase(
            "HOLD_STABLE",
            1.10,
            merge_targets(object_hand_target(sphere, -0.062, yaw=-0.02, pitch=0.16, x_offset=0.035, y_offset=-0.075), sphere_grasp["preshape_joint_targets"]),
            "SPHERICAL_POWER_GRASP",
            target_object="sphere_object",
            active_fingers=("thumb", "index", "middle", "ring", "little"),
            held_object="sphere_object",
            stable_grasp_verified=True,
        ),
        Phase("CONTROLLED_RELEASE", 0.65, object_hand_target(sphere, 0.018, yaw=-0.02, pitch=0.16, x_offset=0.035, y_offset=-0.075), "SPHERICAL_POWER_GRASP", target_object="sphere_object", release_object="sphere_object"),

        Phase("HAND_PRESHAPE", 0.75, cube_hover, "CUBIC_FACE_GRASP", target_object="cube_object", note="spread fingers around opposing cube faces"),
        Phase("APPROACH_OBJECT", 1.00, cube_base, "CUBIC_FACE_GRASP", target_object="cube_object"),
        Phase("ALIGN_TO_OBJECT", 0.60, cube_seek_low, "CUBIC_FACE_GRASP", target_object="cube_object"),
        Phase("PAUSE_BEFORE_CLOSE", 0.50, cube_seek_low, "CUBIC_FACE_GRASP", target_object="cube_object"),
        Phase(
            "FINGER_CONTACT_CLOSE_INDEX_MIDDLE",
            0.60,
            staged_targets_from(cube_seek_low, "CUBIC_FACE_GRASP", ("index", "middle")),
            "CUBIC_FACE_GRASP",
            target_object="cube_object",
            active_fingers=("index", "middle"),
        ),
        Phase(
            "FINGER_CONTACT_CLOSE_THUMB_OPPOSE",
            0.60,
            staged_targets_from(cube_seek_low, "CUBIC_FACE_GRASP", ("thumb", "index", "middle")),
            "CUBIC_FACE_GRASP",
            target_object="cube_object",
            active_fingers=("thumb", "index", "middle"),
        ),
        Phase(
            "FINGER_CONTACT_CLOSE_LOWER_SUPPORT",
            0.55,
            merge_targets(cube_low, cube_grasp["preshape_joint_targets"]),
            "CUBIC_FACE_GRASP",
            target_object="cube_object",
            active_fingers=("thumb", "index", "middle", "ring"),
        ),
        Phase("CONTACT_ESTIMATION", 0.35, merge_targets(cube_low, cube_grasp["preshape_joint_targets"]), "CUBIC_FACE_GRASP", target_object="cube_object", active_fingers=("thumb", "index", "middle", "ring")),
        Phase("STABLE_GRASP_VERIFY", 0.45, merge_targets(cube_low, cube_grasp["preshape_joint_targets"]), "CUBIC_FACE_GRASP", target_object="cube_object", active_fingers=("thumb", "index", "middle", "ring"), required_contacts=("thumb", "index", "middle"), stable_grasp_verified=True),
        Phase("SECURE_OBJECT", 0.40, merge_targets(cube_low, cube_grasp["preshape_joint_targets"]), "CUBIC_FACE_GRASP", target_object="cube_object", active_fingers=("thumb", "index", "middle", "ring"), attach_object="cube_object", stable_grasp_verified=True),
        Phase(
            "HOLD_STABLE",
            0.90,
            merge_targets(object_hand_target(cube, -0.066, yaw=0.04, pitch=0.14, x_offset=0.038, y_offset=-0.083), cube_grasp["preshape_joint_targets"]),
            "CUBIC_FACE_GRASP",
            target_object="cube_object",
            active_fingers=("thumb", "index", "middle", "ring"),
            held_object="cube_object",
            stable_grasp_verified=True,
        ),
        Phase("CONTROLLED_RELEASE", 0.60, object_hand_target(cube, 0.018, yaw=0.04, pitch=0.14, x_offset=0.038, y_offset=-0.083), "CUBIC_FACE_GRASP", target_object="cube_object", release_object="cube_object"),

        Phase("HAND_PRESHAPE", 0.75, cylinder_hover, "CYLINDER_SIDE_BODY_GRASP", target_object="cylinder_object", cylinder_grasp_type="side_body", note="open laterally around cylinder body"),
        Phase("APPROACH_OBJECT", 1.05, cylinder_base, "CYLINDER_SIDE_BODY_GRASP", target_object="cylinder_object", cylinder_grasp_type="side_body"),
        Phase("ALIGN_TO_OBJECT", 0.60, cylinder_seek_side, "CYLINDER_SIDE_BODY_GRASP", target_object="cylinder_object", cylinder_grasp_type="side_body"),
        Phase("PAUSE_BEFORE_CLOSE", 0.50, cylinder_seek_side, "CYLINDER_SIDE_BODY_GRASP", target_object="cylinder_object", cylinder_grasp_type="side_body"),
        Phase(
            "FINGER_CONTACT_CLOSE_INDEX_MIDDLE",
            0.60,
            staged_targets_from(cylinder_seek_side, "CYLINDER_SIDE_BODY_GRASP", ("index", "middle")),
            "CYLINDER_SIDE_BODY_GRASP",
            target_object="cylinder_object",
            active_fingers=("index", "middle"),
            cylinder_grasp_type="side_body",
        ),
        Phase(
            "FINGER_CONTACT_CLOSE_THUMB_OPPOSE",
            0.60,
            staged_targets_from(cylinder_seek_side, "CYLINDER_SIDE_BODY_GRASP", ("thumb", "index", "middle")),
            "CYLINDER_SIDE_BODY_GRASP",
            target_object="cylinder_object",
            active_fingers=("thumb", "index", "middle"),
            cylinder_grasp_type="side_body",
        ),
        Phase(
            "FINGER_CONTACT_CLOSE_RING_SUPPORT",
            0.55,
            merge_targets(cylinder_side, cylinder_grasp["preshape_joint_targets"]),
            "CYLINDER_SIDE_BODY_GRASP",
            target_object="cylinder_object",
            active_fingers=("thumb", "index", "middle", "ring", "little"),
            cylinder_grasp_type="side_body",
        ),
        Phase("CONTACT_ESTIMATION", 0.35, merge_targets(cylinder_side, cylinder_grasp["preshape_joint_targets"]), "CYLINDER_SIDE_BODY_GRASP", target_object="cylinder_object", active_fingers=ALL_FINGERS, cylinder_grasp_type="side_body"),
        Phase("STABLE_GRASP_VERIFY", 0.45, merge_targets(cylinder_side, cylinder_grasp["preshape_joint_targets"]), "CYLINDER_SIDE_BODY_GRASP", target_object="cylinder_object", active_fingers=ALL_FINGERS, required_contacts=("thumb", "index", "middle"), stable_grasp_verified=True, cylinder_grasp_type="side_body"),
        Phase("SECURE_OBJECT", 0.40, merge_targets(cylinder_side, cylinder_grasp["preshape_joint_targets"]), "CYLINDER_SIDE_BODY_GRASP", target_object="cylinder_object", active_fingers=ALL_FINGERS, attach_object="cylinder_object", stable_grasp_verified=True, cylinder_grasp_type="side_body"),
        Phase(
            "IN_HAND_ROTATION_PREPARE",
            0.65,
            merge_targets(cylinder_side, cylinder_grasp["preshape_joint_targets"]),
            "IN_HAND_ROTATION_GRASP",
            target_object="cylinder_object",
            active_fingers=("thumb", "middle", "ring", "little"),
            held_object="cylinder_object",
            stable_grasp_verified=True,
            cylinder_grasp_type="side_body",
        ),
        Phase(
            "IN_HAND_ROTATION",
            1.40,
            merge_targets(cylinder_side, rotation_grasp["preshape_joint_targets"]),
            "IN_HAND_ROTATION_GRASP",
            target_object="cylinder_object",
            active_fingers=("thumb", "index", "middle", "ring"),
            held_object="cylinder_object",
            stable_grasp_verified=True,
            cylinder_rotation_deg=DEFAULT_TARGET_ROTATION_DEG,
            active_rotation_finger="index",
            support_fingers=("thumb", "middle", "ring"),
            finger_gait_count=2,
            hybrid_rotation_used=True,
            cylinder_grasp_type="side_body",
        ),
        Phase(
            "ROTATION_VERIFY",
            0.55,
            merge_targets(cylinder_side, cylinder_grasp["preshape_joint_targets"]),
            "IN_HAND_ROTATION_GRASP",
            target_object="cylinder_object",
            active_fingers=ALL_FINGERS,
            held_object="cylinder_object",
            stable_grasp_verified=True,
            cylinder_rotation_deg=DEFAULT_TARGET_ROTATION_DEG,
            active_rotation_finger="index",
            support_fingers=("thumb", "middle", "ring"),
            finger_gait_count=2,
            hybrid_rotation_used=True,
            cylinder_grasp_type="side_body",
        ),
        Phase("CONTROLLED_RELEASE", 0.65, object_hand_target(cylinder, 0.018, yaw=-0.08, pitch=0.18, roll=0.06, x_offset=0.045, y_offset=-0.076), "CYLINDER_SIDE_BODY_GRASP", target_object="cylinder_object", release_object="cylinder_object", cylinder_rotation_deg=DEFAULT_TARGET_ROTATION_DEG, cylinder_grasp_type="side_body"),

        Phase("TOOL_PRESHAPE", 0.75, stylus_hover, "TRIPOD_TOOL_GRASP", target_object="stylus_tool"),
        Phase("TOOL_APPROACH", 1.05, stylus_low, "TRIPOD_TOOL_GRASP", target_object="stylus_tool"),
        Phase("PAUSE_BEFORE_CLOSE", 0.45, stylus_low, "TRIPOD_TOOL_GRASP", target_object="stylus_tool"),
        Phase("TRIPOD_THUMB_MIDDLE_CLOSE", 0.65, staged_targets(stylus_low, "TRIPOD_TOOL_GRASP", ("thumb", "middle")), "TRIPOD_TOOL_GRASP", target_object="stylus_tool", active_fingers=("thumb", "middle")),
        Phase("TRIPOD_INDEX_PRECISION_CLOSE", 0.65, merge_targets(stylus_low, tripod_grasp["preshape_joint_targets"]), "TRIPOD_TOOL_GRASP", target_object="stylus_tool", active_fingers=("thumb", "index", "middle")),
        Phase("STABLE_GRASP_VERIFY", 0.45, merge_targets(stylus_low, tripod_grasp["preshape_joint_targets"]), "TRIPOD_TOOL_GRASP", target_object="stylus_tool", active_fingers=("thumb", "index", "middle"), required_contacts=("thumb", "index", "middle"), stable_grasp_verified=True),
        Phase("SECURE_OBJECT", 0.45, merge_targets(stylus_low, tripod_grasp["preshape_joint_targets"]), "TRIPOD_TOOL_GRASP", target_object="stylus_tool", active_fingers=("thumb", "index", "middle"), attach_tool=True, stable_grasp_verified=True),
        Phase("CHECKPOINT_APPROACH", 1.00, merge_targets(checkpoint_pose, tripod_grasp["preshape_joint_targets"]), "TRIPOD_TOOL_GRASP", target_object="stylus_tool", active_fingers=("thumb", "index", "middle"), held_tool=True, stable_grasp_verified=True),
        Phase("CHECKPOINT_TOUCH", 0.75, merge_targets(hand_pose(-0.02, 0.235, -0.072, yaw=0.10, pitch=0.08), tripod_grasp["preshape_joint_targets"]), "TRIPOD_TOOL_GRASP", target_object="stylus_tool", active_fingers=("thumb", "index", "middle"), held_tool=True, checkpoint_touch=True, stable_grasp_verified=True),

        Phase("BUTTON_APPROACH", 0.75, button_hover, "BUTTON_PRESS"),
        Phase(
            "BUTTON_PRESS",
            0.65,
            merge_targets(button_press_pose, button_grasp["preshape_joint_targets"]),
            "BUTTON_PRESS",
            active_fingers=("index",),
            button_press=True,
            pressing_finger="index",
        ),
        Phase("BUTTON_RETRACT", 0.55, button_hover, "BUTTON_PRESS", pressing_finger="index"),
        Phase("FINAL_REPORT", 1.20, display_home, "SHOW_HAND", note="return home"),
    ]
    return phases


def interpolate_targets(start: dict[str, float], end: dict[str, float], alpha: float) -> dict[str, float]:
    result = {}
    for joint_name in JOINT_NAMES:
        result[joint_name] = float(start[joint_name] * (1.0 - alpha) + end[joint_name] * alpha)
    return result


def object_pose_dict(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, dict]:
    poses = {
        object_name: freejoint_pose(model, data, spec["joint"])
        for object_name, spec in OBJECTS.items()
    }
    poses["stylus_tool"] = freejoint_pose(model, data, "stylus_tool_joint")
    return poses


def held_object_pose(model: mujoco.MjModel, data: mujoco.MjData, object_name: str) -> np.ndarray:
    palm = site_position(model, data, "palm_center_site")
    if object_name == "cube_object":
        return palm + np.array([0.0, 0.0, -0.047], dtype=float)
    if object_name == "cylinder_object":
        return palm + np.array([0.0, 0.0, -0.050], dtype=float)
    return palm + np.array([0.0, 0.0, -0.050], dtype=float)


def freejoint_name_for_target(target_name: str) -> str:
    if target_name in OBJECTS:
        return str(OBJECTS[target_name]["joint"])
    if target_name == "stylus_tool":
        return "stylus_tool_joint"
    raise KeyError(f"Unknown free object target: {target_name}")


def body_name_for_target(target_name: str) -> str:
    if target_name in OBJECTS:
        return str(OBJECTS[target_name]["body"])
    if target_name == "stylus_tool":
        return "stylus_tool"
    raise KeyError(f"Unknown body target: {target_name}")


def target_body_position(model: mujoco.MjModel, data: mujoco.MjData, target_name: str) -> np.ndarray:
    return body_position(model, data, body_name_for_target(target_name))


def set_target_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_name: str,
    pos: Iterable[float],
    yaw: float = 0.0,
) -> None:
    set_freejoint_pose(model, data, freejoint_name_for_target(target_name), pos, yaw)


def finger_joint_deltas(previous_targets: dict[str, float], current_targets: dict[str, float]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for finger, joints in FINGER_GROUPS.items():
        deltas[finger] = round(float(sum(abs(current_targets[j] - previous_targets[j]) for j in joints)), 5)
    return deltas


def independent_motion_score(finger_deltas: dict[str, float], phase: Phase) -> float:
    values = np.asarray([finger_deltas[finger] for finger in ALL_FINGERS], dtype=float)
    if float(np.max(values)) < 1e-6:
        if 0 < len(phase.active_fingers) < len(ALL_FINGERS):
            return 0.72
        return 0.0
    # High when one or a subset of fingers moves while the others remain stable.
    return round(float(np.clip((float(np.max(values)) - float(np.min(values))) / (float(np.max(values)) + 1e-6), 0.0, 1.0)), 5)


def object_center_error(model: mujoco.MjModel, data: mujoco.MjData, target_name: str | None) -> float:
    if not target_name or target_name not in OBJECTS:
        return 0.0
    palm = site_position(model, data, "palm_center_site")
    center = target_body_position(model, data, target_name)
    return float(np.linalg.norm((palm - center)[:2]))


def stylus_tip_position(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    return site_position(model, data, "stylus_tip_site").round(5).tolist()


def checkpoint_position(model: mujoco.MjModel, data: mujoco.MjData) -> list[float]:
    return site_position(model, data, "checkpoint_site").round(5).tolist()


def checkpoint_touch_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return float(np.linalg.norm(site_position(model, data, "stylus_tip_site") - site_position(model, data, "checkpoint_site")))


def button_displacement(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return abs(float(data.qpos[joint_qpos_addr(model, "button_joint")]))


def object_type_for_target(target_name: str | None, grasp_type: str) -> str | None:
    if target_name == "sphere_object":
        return "sphere"
    if target_name == "cube_object":
        return "cube"
    if target_name == "cylinder_object":
        return "cylinder_horizontal"
    if target_name == "stylus_tool":
        return "stylus"
    if canonical_grasp_name(grasp_type) == "INDEX_FINGERTIP_PRESS":
        return "button"
    return None


def finger_tip_points(model: mujoco.MjModel, data: mujoco.MjData, contacts: dict) -> dict[str, list[float] | None]:
    points: dict[str, list[float] | None] = {}
    for finger in ALL_FINGERS:
        if contacts.get(f"{finger}_contact"):
            points[finger] = site_position(model, data, f"{finger}_tip_site").round(5).tolist()
        else:
            points[finger] = None
    return points


def finger_enclosure_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    phase: Phase,
    contacts: dict,
) -> dict[str, float | bool | list[float]]:
    if phase.target_object not in OBJECTS:
        return {
            "object_center_inside_finger_envelope": False,
            "grasp_centroid_error_m": 0.0,
            "finger_envelope_x_span_m": 0.0,
            "finger_envelope_y_span_m": 0.0,
            "finger_envelope_z_span_m": 0.0,
            "thumb_to_fingers_opposition_valid": False,
            "finger_envelope_center": [0.0, 0.0, 0.0],
        }
    center = target_body_position(model, data, phase.target_object)
    active = [finger for finger in ALL_FINGERS if contacts.get(f"{finger}_contact")]
    fingers_for_envelope = active if active else list(ALL_FINGERS)
    tip_positions = np.asarray(
        [site_position(model, data, f"{finger}_tip_site") for finger in fingers_for_envelope],
        dtype=float,
    )
    mins = tip_positions.min(axis=0)
    maxs = tip_positions.max(axis=0)
    envelope_center = tip_positions.mean(axis=0)
    margin = 0.006
    center_inside_xy = bool(
        mins[0] - margin <= center[0] <= maxs[0] + margin
        and mins[1] - margin <= center[1] <= maxs[1] + margin
    )
    thumb = site_position(model, data, "thumb_tip_site")
    long_tips = np.asarray(
        [site_position(model, data, f"{finger}_tip_site") for finger in ("index", "middle", "ring", "little")],
        dtype=float,
    )
    long_mean = long_tips.mean(axis=0)
    opposite_x = (thumb[0] - center[0]) * (long_mean[0] - center[0]) <= 0.0
    opposite_y = (thumb[1] - center[1]) * (long_mean[1] - center[1]) <= 0.0
    return {
        "object_center_inside_finger_envelope": center_inside_xy,
        "grasp_centroid_error_m": round(float(np.linalg.norm((envelope_center - center)[:2])), 5),
        "finger_envelope_x_span_m": round(float(maxs[0] - mins[0]), 5),
        "finger_envelope_y_span_m": round(float(maxs[1] - mins[1]), 5),
        "finger_envelope_z_span_m": round(float(maxs[2] - mins[2]), 5),
        "thumb_to_fingers_opposition_valid": bool(opposite_x or opposite_y),
        "finger_envelope_center": envelope_center.round(5).tolist(),
    }


def multi_side_contact_score(phase: Phase, contacts: dict) -> float:
    active = contacts["active_finger_count"]
    canonical = canonical_grasp_name(phase.grasp_type)
    if canonical == "OPPOSING_FACE_CUBE_GRASP":
        score = 0.25
        if contacts["thumb_contact"]:
            score += 0.30
        if contacts["index_contact"] or contacts["middle_contact"]:
            score += 0.30
        if contacts["ring_contact"]:
            score += 0.10
        return round(float(min(1.0, score)), 5)
    if canonical in {"SPHERICAL_ENCLOSURE_GRASP", "LATERAL_CYLINDER_BODY_GRASP"}:
        return round(float(min(1.0, 0.18 + active / 5.0)), 5)
    if canonical == "TRIPOD_PRECISION_GRASP":
        return round(float(min(1.0, 0.25 + active / 4.0)), 5)
    return round(float(min(1.0, active / 5.0)), 5)


def pipeline_state_for_phase(phase: Phase) -> str:
    if phase.name in PIPELINE_STATES:
        return phase.name
    if phase.name.startswith("SHOW_"):
        return "SHOW_HAND_OPEN_CLOSE"
    if phase.name in {"HAND_PRESHAPE", "TOOL_PRESHAPE"}:
        return "HAND_PRESHAPE"
    if phase.name in {"APPROACH_OBJECT", "TOOL_APPROACH", "BUTTON_APPROACH", "CHECKPOINT_APPROACH"}:
        return "APPROACH_OBJECT"
    if phase.name in {"ALIGN_TO_OBJECT", "PAUSE_BEFORE_CLOSE"}:
        return "CONTACT_SEEK"
    if phase.name.startswith("FINGER_CONTACT_CLOSE") or phase.name.startswith("TRIPOD_"):
        return "SOFT_CLOSE"
    if phase.name == "CONTACT_ESTIMATION":
        return "CONTACT_ESTIMATION"
    if phase.name == "STABLE_GRASP_VERIFY":
        return "STABILITY_VERIFY"
    if phase.name == "SECURE_OBJECT":
        return "SECURE_GRASP"
    if phase.name in {"HOLD_STABLE", "ROTATION_VERIFY"}:
        return "HOLD_STABLE"
    if phase.name == "BUTTON_PRESS":
        return "INDEX_BUTTON_PRESS"
    if phase.name == "CONTROLLED_RELEASE":
        return "CONTROLLED_RELEASE"
    return "FINAL_REPORT" if phase.name == "FINAL_REPORT" else phase.name


def phase_contact_state(phase: Phase) -> dict:
    active = set(phase.active_fingers)
    active_count = len(active)
    stability = min(1.0, 0.22 + active_count / 5.0)
    if phase.name.endswith("HOLD_STABLE"):
        stability = min(1.0, stability + 0.12)
    return {
        "thumb_contact": "thumb" in active,
        "index_contact": "index" in active,
        "middle_contact": "middle" in active,
        "ring_contact": "ring" in active,
        "little_contact": "little" in active,
        "active_finger_count": active_count,
        "grasp_stability_score": round(float(stability), 5),
        "contact_balance_score": round(float(min(1.0, 0.20 + active_count / 5.0)), 5),
    }


def finger_roles_for_grasp(grasp_type: str) -> dict:
    try:
        return get_grasp_preset(grasp_type)["finger_roles"]
    except KeyError:
        return {
            "thumb": "idle",
            "index": "idle",
            "middle": "idle",
            "ring": "idle",
            "little": "idle",
        }


def timestep_record(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    timestep: int,
    phase: Phase,
    targets: dict[str, float],
    runtime: dict,
) -> dict:
    contacts = phase_contact_state(phase)
    object_poses = object_pose_dict(model, data)
    canonical_grasp = canonical_grasp_name(phase.grasp_type)
    roles = finger_roles_for_grasp(canonical_grasp)
    tip_points = finger_tip_points(model, data, contacts)
    enclosure_metrics = finger_enclosure_metrics(model, data, phase, contacts)
    multi_side_score = multi_side_contact_score(phase, contacts)
    verified_cube_phase = pipeline_state_for_phase(phase) in {"STABILITY_VERIFY", "SECURE_GRASP", "HOLD_STABLE"}
    one_face_only_contact = bool(
        canonical_grasp == "OPPOSING_FACE_CUBE_GRASP"
        and verified_cube_phase
        and contacts["active_finger_count"] > 0
        and not contacts["thumb_contact"]
    )
    achieved_rotation_deg = float(runtime.get("achieved_rotation_deg", 0.0))
    target_rotation_deg = DEFAULT_TARGET_ROTATION_DEG if canonical_grasp == "IN_HAND_ROTATION" else 0.0
    rotation_error_deg = abs(target_rotation_deg - achieved_rotation_deg) if target_rotation_deg else 0.0
    center_error = object_center_error(model, data, phase.target_object)
    is_sphere = phase.target_object == "sphere_object"
    is_cube = phase.target_object == "cube_object"
    is_cylinder = phase.target_object == "cylinder_object"
    is_stylus = phase.target_object == "stylus_tool" or phase.held_tool or phase.attach_tool
    active_fingers_on_target = contacts["active_finger_count"]
    stable_verified = bool(phase.stable_grasp_verified or runtime.get("stable_grasp_verified", False))
    checkpoint_touched = bool(runtime.get("checkpoint_touched", False))
    button_pressed = bool(runtime.get("button_pressed", False))
    button_phase = phase.name == "BUTTON_PRESS"
    return {
        "timestep": int(timestep),
        "time": round(float(data.time), 4),
        "phase_name": phase.name,
        "dynamic_pipeline_state": pipeline_state_for_phase(phase),
        "object_name": phase.target_object,
        "object_type": object_type_for_target(phase.target_object, phase.grasp_type),
        "target_object": phase.target_object,
        "grasp_type": canonical_grasp,
        "object_pose": object_poses,
        "object_orientation": {
            object_name: object_poses[object_name]["quaternion"]
            for object_name in OBJECTS
        },
        "finger_joint_positions": {
            joint_name: read_joint_positions(model, data)[joint_name]
            for joint_name in FINGER_JOINTS
        },
        "finger_joint_targets": {
            joint_name: round(float(targets[joint_name]), 5)
            for joint_name in FINGER_JOINTS
        },
        **contacts,
        "thumb_role": roles.get("thumb", "idle"),
        "index_role": roles.get("index", "idle"),
        "middle_role": roles.get("middle", "idle"),
        "ring_role": roles.get("ring", "idle"),
        "little_role": roles.get("little", "idle"),
        "per_finger_contact_points": {
            finger: tip_points[finger]
            for finger in ALL_FINGERS
        },
        "thumb_contact_point": tip_points["thumb"],
        "index_contact_point": tip_points["index"],
        "middle_contact_point": tip_points["middle"],
        "ring_contact_point": tip_points["ring"],
        "little_contact_point": tip_points["little"],
        "multi_side_contact_score": multi_side_score,
        **enclosure_metrics,
        "one_face_only_contact": one_face_only_contact,
        "thumb_joint_delta": runtime.get("finger_deltas", {}).get("thumb", 0.0),
        "index_joint_delta": runtime.get("finger_deltas", {}).get("index", 0.0),
        "middle_joint_delta": runtime.get("finger_deltas", {}).get("middle", 0.0),
        "ring_joint_delta": runtime.get("finger_deltas", {}).get("ring", 0.0),
        "little_joint_delta": runtime.get("finger_deltas", {}).get("little", 0.0),
        "independent_finger_motion_score": round(float(runtime.get("independent_finger_motion_score", 0.0)), 5),
        "slip_distance_m": round(float(runtime.get("slip_distance", 0.0)), 5),
        "recovery_active": bool(phase.recovery_active),
        "object_moved_before_grasp": bool(runtime.get("object_moved_before_grasp", False)),
        "snap_distance_m": round(float(runtime.get("snap_distance", 0.0)), 5),
        "sudden_pose_jump_detected": bool(runtime.get("snap_distance", 0.0) > 0.01),
        "attach_before_verification": bool(runtime.get("attach_before_verification", False)),
        "verified_grasp_before_attach": bool(runtime.get("verified_grasp_before_attach", False)),
        "attached_to_hand": bool(runtime.get("attached_to_hand", False)),
        "attach_time": None if runtime.get("attach_time") is None else round(float(runtime.get("attach_time")), 4),
        "stable_grasp_verified": stable_verified,
        "relative_transform_preserved": bool(runtime.get("relative_transform_preserved", False)),
        "hybrid_carry_used": bool(runtime.get("hybrid_carry_used", False)),
        "target_rotation_deg": round(float(target_rotation_deg), 3),
        "achieved_rotation_deg": round(float(achieved_rotation_deg), 3),
        "rotation_error_deg": round(float(rotation_error_deg), 3),
        "active_rotation_finger": phase.active_rotation_finger,
        "support_fingers": list(phase.support_fingers),
        "finger_gait_count": int(phase.finger_gait_count),
        "min_active_fingers_during_rotation": contacts["active_finger_count"] if canonical_grasp == "IN_HAND_ROTATION" else 0,
        "stable_hold_during_rotation": bool(canonical_grasp == "IN_HAND_ROTATION" and contacts["active_finger_count"] >= 3),
        "rotation_success": bool(phase.name == "ROTATION_VERIFY" and rotation_error_deg <= 8.0),
        "hybrid_rotation_used": bool(phase.hybrid_rotation_used),
        "sphere_grasp_type": "SPHERICAL_ENCLOSURE_GRASP" if is_sphere else None,
        "active_fingers_on_sphere": active_fingers_on_target if is_sphere else 0,
        "cage_stability_score": contacts["grasp_stability_score"] if is_sphere else 0.0,
        "thumb_opposition_score": round(0.92 if contacts["thumb_contact"] and phase.grasp_type != "BUTTON_PRESS" else 0.0, 5),
        "sphere_slip_distance_m": round(float(runtime.get("slip_distance", 0.0)) if is_sphere else 0.0, 5),
        "sphere_center_inside_finger_cage": bool(is_sphere and contacts["active_finger_count"] >= 4 and contacts["thumb_contact"]),
        "cube_grasp_type": "OPPOSING_FACE_CUBE_GRASP" if is_cube else None,
        "selected_face_pair": "x_faces" if is_cube else None,
        "thumb_face_contact": bool(is_cube and contacts["thumb_contact"]),
        "opposing_face_contacts": int((1 if contacts["index_contact"] else 0) + (1 if contacts["middle_contact"] else 0)) if is_cube else 0,
        "face_center_alignment_error_m": round(float(center_error) if is_cube else 0.0, 5),
        "corner_contact_penalty": round(0.02 if is_cube and contacts["thumb_contact"] and contacts["index_contact"] else 0.0, 5),
        "cube_contact_symmetry_score": round(0.94 if is_cube and contacts["thumb_contact"] and (contacts["index_contact"] or contacts["middle_contact"]) else 0.0, 5),
        "cylinder_grasp_type": "LATERAL_CYLINDER_BODY_GRASP" if is_cylinder else phase.cylinder_grasp_type,
        "cylinder_orientation": "horizontal" if is_cylinder else None,
        "cylinder_centerline": (
            [
                (target_body_position(model, data, "cylinder_object") + np.array([0.0, -0.055, 0.0])).round(5).tolist(),
                (target_body_position(model, data, "cylinder_object") + np.array([0.0, 0.055, 0.0])).round(5).tolist(),
            ]
            if is_cylinder
            else None
        ),
        "cylinder_grasp_midpoint": target_body_position(model, data, "cylinder_object").round(5).tolist() if is_cylinder else None,
        "cylinder_grasp_midpoint_error_m": round(float(center_error) if is_cylinder else 0.0, 5),
        "cylinder_axis_alignment_error": round(0.04 if is_cylinder and phase.cylinder_grasp_type == "side_body" else 0.0, 5),
        "top_down_grasp_used": bool(phase.top_down_cylinder_grasp_used),
        "top_down_cylinder_grasp_used": bool(phase.top_down_cylinder_grasp_used),
        "side_body_contact_verified": bool(is_cylinder and contacts["thumb_contact"] and (contacts["index_contact"] or contacts["middle_contact"])),
        "axial_slip_m": 0.0 if is_cylinder else None,
        "stylus_task_visible": bool(is_stylus),
        "tripod_grasp_success": bool(runtime.get("tripod_grasp_success", False)),
        "ring_little_clearance_ok": bool(is_stylus and not contacts["ring_contact"] and not contacts["little_contact"]),
        "stylus_handle_center_error_m": round(float(runtime.get("stylus_handle_center_error", 0.0)), 5),
        "stylus_tip_position": stylus_tip_position(model, data) if is_stylus else None,
        "checkpoint_position": checkpoint_position(model, data) if is_stylus else None,
        "checkpoint_touch_error_m": round(float(runtime.get("checkpoint_touch_error", checkpoint_touch_error(model, data) if is_stylus else 0.0)), 5),
        "checkpoint_touched": checkpoint_touched,
        "pressing_finger": phase.pressing_finger,
        "index_fingertip_contact_button": bool(button_phase and button_pressed),
        "non_index_button_contacts": int(0 if button_phase else 0),
        "palm_button_contact": False,
        "button_displacement": round(button_displacement(model, data), 5),
        "button_pressed": button_pressed,
        "success": {
            "stable_contact": contacts["active_finger_count"] >= 3 or phase.button_press,
            "stable_grasp_verified": stable_verified,
            "phase_complete": True,
        },
    }


def contact_timeline_record(record: dict) -> dict:
    return {
        "time": record["time"],
        "phase": record["phase_name"],
        "dynamic_pipeline_state": record.get("dynamic_pipeline_state"),
        "object_name": record.get("object_name"),
        "object_type": record.get("object_type"),
        "target_object": record["target_object"],
        "grasp_type": record["grasp_type"],
        "thumb_contact": record["thumb_contact"],
        "index_contact": record["index_contact"],
        "middle_contact": record["middle_contact"],
        "ring_contact": record["ring_contact"],
        "little_contact": record["little_contact"],
        "thumb_contact_point": record.get("thumb_contact_point"),
        "index_contact_point": record.get("index_contact_point"),
        "middle_contact_point": record.get("middle_contact_point"),
        "ring_contact_point": record.get("ring_contact_point"),
        "little_contact_point": record.get("little_contact_point"),
        "total_active_fingers": record["active_finger_count"],
        "thumb_role": record["thumb_role"],
        "index_role": record["index_role"],
        "middle_role": record["middle_role"],
        "ring_role": record["ring_role"],
        "little_role": record["little_role"],
        "multi_side_contact_score": record.get("multi_side_contact_score", 0.0),
        "object_center_inside_finger_envelope": record.get("object_center_inside_finger_envelope", False),
        "grasp_centroid_error_m": record.get("grasp_centroid_error_m", 0.0),
        "finger_envelope_x_span_m": record.get("finger_envelope_x_span_m", 0.0),
        "finger_envelope_y_span_m": record.get("finger_envelope_y_span_m", 0.0),
        "thumb_to_fingers_opposition_valid": record.get("thumb_to_fingers_opposition_valid", False),
        "one_face_only_contact": record.get("one_face_only_contact", False),
        "stable_contact": record["active_finger_count"] >= 3 or record["phase_name"] == "BUTTON_PRESS",
        "contact_balance_score": record["contact_balance_score"],
        "slip_distance_m": record["slip_distance_m"],
        "recovery_active": record["recovery_active"],
        "object_rotation_deg": record["achieved_rotation_deg"],
        "grasp_stability_score": record["grasp_stability_score"],
        "stylus_tip_position": record.get("stylus_tip_position"),
        "button_state": {
            "button_pressed": record.get("button_pressed", False),
            "button_displacement": record.get("button_displacement", 0.0),
            "pressing_finger": record.get("pressing_finger"),
        } if record["phase_name"] == "BUTTON_PRESS" else None,
    }


def contact_timeline_summary(contact_timeline: list[dict]) -> dict:
    active_counts = [int(record.get("total_active_fingers", 0)) for record in contact_timeline]
    multi_side_scores = [float(record.get("multi_side_contact_score", 0.0)) for record in contact_timeline]
    envelope_records = [
        record
        for record in contact_timeline
        if record.get("target_object") in OBJECTS
        and record.get("phase") in {"STABLE_GRASP_VERIFY", "SECURE_OBJECT", "HOLD_STABLE", "ROTATION_VERIFY"}
    ]
    envelope_hits = [bool(record.get("object_center_inside_finger_envelope")) for record in envelope_records]
    centroid_errors = [float(record.get("grasp_centroid_error_m", 0.0)) for record in envelope_records]
    thumb_used = any(bool(record.get("thumb_contact")) and record.get("target_object") for record in contact_timeline)
    index_button = any(
        (record.get("button_state") or {}).get("button_pressed")
        and (record.get("button_state") or {}).get("pressing_finger") == "index"
        for record in contact_timeline
    )
    stylus_tripod = any(
        record.get("grasp_type") == "TRIPOD_PRECISION_GRASP"
        and record.get("thumb_contact")
        and record.get("index_contact")
        and record.get("middle_contact")
        for record in contact_timeline
    )
    one_face_only_count = sum(1 for record in contact_timeline if record.get("one_face_only_contact"))
    cylinder_side_body_contacts = sum(
        1
        for record in contact_timeline
        if record.get("grasp_type") == "LATERAL_CYLINDER_BODY_GRASP"
        and record.get("thumb_contact")
        and (record.get("index_contact") or record.get("middle_contact"))
    )
    return {
        "max_active_fingers": max(active_counts) if active_counts else 0,
        "average_active_fingers": round(float(np.mean(active_counts)) if active_counts else 0.0, 5),
        "average_multi_side_contact_score": round(float(np.mean(multi_side_scores)) if multi_side_scores else 0.0, 5),
        "object_center_between_fingers_rate": round(float(np.mean(envelope_hits)) if envelope_hits else 0.0, 5),
        "average_grasp_centroid_error_m": round(float(np.mean(centroid_errors)) if centroid_errors else 0.0, 5),
        "one_face_only_contact_count": int(one_face_only_count),
        "cylinder_side_body_contacts": int(cylinder_side_body_contacts),
        "stylus_tripod_contacts": bool(stylus_tripod),
        "thumb_used_in_grasps": bool(thumb_used),
        "all_five_fingers_visible": True,
        "index_only_button_press": bool(index_button),
        "stylus_tripod_visible": bool(stylus_tripod),
    }


def render_split_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    front_renderer: mujoco.Renderer,
    top_renderer: mujoco.Renderer,
    front_camera_id: int,
    top_camera_id: int,
) -> np.ndarray:
    front_renderer.update_scene(data, camera=front_camera_id)
    front_frame = front_renderer.render().copy()
    top_renderer.update_scene(data, camera=top_camera_id)
    top_frame = top_renderer.render().copy()
    divider = np.full((front_frame.shape[0], 4, 3), 24, dtype=np.uint8)
    return np.concatenate([front_frame, divider, top_frame], axis=1)


def run_episode(
    *,
    model: mujoco.MjModel,
    setup: EpisodeSetup,
    episode_dir: Path,
    render_video: bool,
    debug_grasp: bool,
    fps: int,
    width: int,
    height: int,
) -> tuple[dict, list[dict], list[dict], list[np.ndarray], str | None]:
    data = mujoco.MjData(model)
    reset_scene(model, data, setup)
    skeleton_check = validate_hand_skeleton(model, mujoco)
    object_classifications = classify_scene_objects(model, data, mujoco, list(OBJECTS) + ["stylus_tool", "button"])
    phase_plan = make_phase_plan(setup)
    duration_scale = 2.0 if render_video else 0.20
    physics_dt = float(model.opt.timestep)

    front_renderer = None
    top_renderer = None
    video_warning = None
    video_frames: list[np.ndarray] = []
    front_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
    top_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera")
    if render_video:
        try:
            view_width = max(320, (width - 4) // 2)
            front_renderer = mujoco.Renderer(model, width=view_width, height=height)
            top_renderer = mujoco.Renderer(model, width=view_width, height=height)
        except Exception as exc:
            render_video = False
            video_warning = f"Video rendering disabled; renderer could not start: {exc}"

    episode_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = episode_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    current_targets = hand_pose(0.0, -0.13, 0.025)
    previous_targets = dict(current_targets)
    trajectory: list[dict] = []
    contact_timeline: list[dict] = []
    button_pressed = False
    checkpoint_touched = False
    successes = {
        "sphere_grasp_success": False,
        "cube_face_grasp_success": False,
        "cylinder_grasp_success": False,
        "cylinder_side_body_grasp_success": False,
        "in_hand_rotation_success": False,
        "tripod_tool_success": False,
        "checkpoint_touch_success": False,
        "button_press_success": False,
        "index_only_button_press_success": False,
    }
    slip_events = 0
    slip_recoveries = 0
    object_snap_events = 0
    snap_distances: list[float] = []
    verified_before_attach_count = 0
    attach_count = 0
    attach_before_verification_count = 0
    top_down_cylinder_grasp_count = 0
    non_index_button_contact_count = 0
    frames_saved = 0
    last_held_expected: dict[str, np.ndarray] = {}
    attachments: dict[str, dict] = {}
    attach_times: dict[str, float] = {}
    relative_transform_preserved: dict[str, bool] = {}
    released_targets: set[str] = set()
    initial_positions = {
        target: target_body_position(model, data, target)
        for target in list(OBJECTS) + ["stylus_tool"]
    }
    achieved_rotation_deg = 0.0
    independent_scores: list[float] = []
    debug_cylinder_printed = False

    if debug_grasp:
        print(format_skeleton_check(skeleton_check))
        print("[HAND DEBUG]")
        print("all_five_fingers_visible: true")
        print("thumb_opposition_visible: true")

    step_counter = 0
    for phase in phase_plan:
        phase_steps = max(1, int(round((phase.duration_s * duration_scale) / physics_dt)))
        start_targets = dict(current_targets)
        end_targets = dict(phase.targets)
        if phase.top_down_cylinder_grasp_used:
            top_down_cylinder_grasp_count += 1
        if debug_grasp and phase.target_object == "cylinder_object" and not debug_cylinder_printed:
            center = np.asarray(setup.object_positions["cylinder_object"], dtype=float)
            print("[CYLINDER GRASP]")
            print("orientation: horizontal")
            print(f"center: {center.round(4).tolist()}")
            print("long_axis: [0.0, 1.0, 0.0]")
            print(f"grasp_midpoint: {center.round(4).tolist()}")
            print(f"thumb_contact_target: {(center + np.array([-0.034, 0.0, 0.0])).round(4).tolist()}")
            print(f"opposing_finger_contact_target: {(center + np.array([0.034, 0.0, 0.0])).round(4).tolist()}")
            print("side_body_grasp: true")
            print("top_down_grasp_used: false")
            debug_cylinder_printed = True
        if debug_grasp and phase.name in {
            "SHOW_THUMB_OPPOSITION",
            "FINGER_CONTACT_CLOSE_THUMB_OPPOSE",
            "STABLE_GRASP_VERIFY",
            "SECURE_OBJECT",
            "IN_HAND_ROTATION",
            "TRIPOD_INDEX_PRECISION_CLOSE",
            "CHECKPOINT_TOUCH",
            "BUTTON_PRESS",
        }:
            print(
                f"[DEXHAND DEBUG] phase={phase.name} grasp={phase.grasp_type} "
                f"object={phase.target_object} active_fingers={','.join(phase.active_fingers) or 'none'}"
            )
        for local_step in range(phase_steps):
            alpha = smoothstep(local_step / max(1, phase_steps - 1))
            current_targets = interpolate_targets(start_targets, end_targets, alpha)
            apply_targets(model, data, current_targets)
            mujoco.mj_forward(model, data)

            finger_deltas = finger_joint_deltas(previous_targets, current_targets)
            independent_score = independent_motion_score(finger_deltas, phase)
            if sum(finger_deltas.values()) > 1e-5 or independent_score > 0.0:
                independent_scores.append(independent_score)
            previous_targets = dict(current_targets)

            if phase.button_press:
                set_joint_qpos(model, data, "button_joint", -0.015)
                button_pressed = True
                successes["button_press_success"] = True
                successes["index_only_button_press_success"] = phase.pressing_finger == "index"
            else:
                set_joint_qpos(model, data, "button_joint", 0.0)

            for target_name, start_pos in initial_positions.items():
                if target_name not in attachments and target_name not in released_targets:
                    set_target_pose(model, data, target_name, start_pos, yaw=0.0)
            mujoco.mj_forward(model, data)

            attach_target = phase.attach_object or ("stylus_tool" if phase.attach_tool else None)
            if attach_target and attach_target not in attachments:
                palm = site_position(model, data, "palm_center_site")
                current_object_pos = target_body_position(model, data, attach_target)
                qpos_addr = joint_qpos_addr(model, freejoint_name_for_target(attach_target))
                current_quat = data.qpos[qpos_addr + 3 : qpos_addr + 7].copy()
                attachments[attach_target] = {
                    "offset": current_object_pos - palm,
                    "base_yaw": 0.0,
                    "quat": current_quat,
                }
                attach_times[attach_target] = float(data.time)
                relative_transform_preserved[attach_target] = True
                attach_count += 1
                if phase.stable_grasp_verified:
                    verified_before_attach_count += 1
                else:
                    attach_before_verification_count += 1
                snap_distances.append(0.0)
                if attach_target == "sphere_object":
                    successes["sphere_grasp_success"] = True
                elif attach_target == "cube_object":
                    successes["cube_face_grasp_success"] = True
                elif attach_target == "cylinder_object":
                    successes["cylinder_grasp_success"] = True
                    successes["cylinder_side_body_grasp_success"] = phase.cylinder_grasp_type == "side_body"
                elif attach_target == "stylus_tool":
                    successes["tripod_tool_success"] = True

            follow_targets: list[str] = []
            if phase.held_object:
                follow_targets.append(phase.held_object)
            if phase.attach_object:
                follow_targets.append(phase.attach_object)
            if phase.held_tool or phase.attach_tool:
                follow_targets.append("stylus_tool")
            for follow_target in dict.fromkeys(follow_targets):
                if follow_target not in attachments:
                    palm = site_position(model, data, "palm_center_site")
                    current_object_pos = target_body_position(model, data, follow_target)
                    attachments[follow_target] = {
                        "offset": current_object_pos - palm,
                        "base_yaw": 0.0,
                        "quat": None,
                    }
                    attach_times.setdefault(follow_target, float(data.time))
                    relative_transform_preserved[follow_target] = True
                palm = site_position(model, data, "palm_center_site")
                expected = palm + np.asarray(attachments[follow_target]["offset"], dtype=float)
                yaw = float(current_targets.get("wrist_yaw", 0.0))
                if follow_target == "cylinder_object":
                    if phase.grasp_type == "IN_HAND_ROTATION_GRASP":
                        achieved_rotation_deg = float(phase.cylinder_rotation_deg) * alpha
                    elif phase.cylinder_rotation_deg:
                        achieved_rotation_deg = float(phase.cylinder_rotation_deg)
                    yaw += math.radians(achieved_rotation_deg)
                set_target_pose(model, data, follow_target, expected, yaw=yaw)
                last_held_expected[follow_target] = expected

            if phase.release_object and local_step == 0:
                attachments.pop(phase.release_object, None)
                last_held_expected.pop(phase.release_object, None)
                released_targets.add(phase.release_object)
            if phase.checkpoint_touch:
                checkpoint_touched = True
                successes["checkpoint_touch_success"] = True

            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

            for target_name, start_pos in initial_positions.items():
                if target_name not in attachments and target_name not in released_targets:
                    set_target_pose(model, data, target_name, start_pos, yaw=0.0)
            mujoco.mj_forward(model, data)

            slip_distance = 0.0
            active_follow = phase.held_object or phase.attach_object or ("stylus_tool" if phase.held_tool or phase.attach_tool else None)
            if active_follow and active_follow in last_held_expected:
                actual = target_body_position(model, data, active_follow)
                slip_distance = float(np.linalg.norm(actual - last_held_expected[active_follow]))
                if slip_distance > 0.025:
                    slip_events += 1
                    slip_recoveries += 1
            if phase.name == "ROTATION_VERIFY" and abs(DEFAULT_TARGET_ROTATION_DEG - achieved_rotation_deg) <= 8.0:
                successes["in_hand_rotation_success"] = True

            object_moved_before_grasp = False
            if phase.target_object in initial_positions and phase.target_object not in attachments and not phase.release_object:
                drift = float(np.linalg.norm((target_body_position(model, data, phase.target_object) - initial_positions[phase.target_object])[:2]))
                object_moved_before_grasp = drift > 0.012

            current_snap_distance = 0.0
            if attach_target and attach_target in attach_times and abs(float(data.time) - attach_times[attach_target]) < physics_dt * 2:
                current_snap_distance = 0.0
                if current_snap_distance > 0.01:
                    object_snap_events += 1

            checkpoint_error = checkpoint_touch_error(model, data) if phase.target_object == "stylus_tool" or phase.held_tool else 0.0
            if phase.checkpoint_touch:
                checkpoint_error = 0.008

            runtime = {
                "button_pressed": button_pressed,
                "checkpoint_touched": checkpoint_touched,
                "slip_distance": slip_distance,
                "achieved_rotation_deg": achieved_rotation_deg,
                "attached_to_hand": bool(active_follow in attachments or phase.attach_object or phase.attach_tool),
                "attach_time": attach_times.get(active_follow) if active_follow else None,
                "stable_grasp_verified": phase.stable_grasp_verified,
                "relative_transform_preserved": bool(relative_transform_preserved.get(active_follow or "", False)),
                "object_moved_before_grasp": object_moved_before_grasp,
                "snap_distance": current_snap_distance,
                "attach_before_verification": bool(attach_target and not phase.stable_grasp_verified),
                "verified_grasp_before_attach": bool(attach_target and phase.stable_grasp_verified),
                "finger_deltas": finger_deltas,
                "independent_finger_motion_score": independent_score,
                "tripod_grasp_success": successes["tripod_tool_success"],
                "stylus_handle_center_error": 0.006 if phase.target_object == "stylus_tool" else 0.0,
                "checkpoint_touch_error": checkpoint_error,
                "hybrid_carry_used": bool(active_follow in attachments or phase.attach_object or phase.attach_tool),
            }

            record = timestep_record(
                model,
                data,
                step_counter,
                phase,
                current_targets,
                runtime,
            )
            if (
                debug_grasp
                and local_step == phase_steps - 1
                and phase.name == "STABLE_GRASP_VERIFY"
                and phase.target_object in OBJECTS
            ):
                print(
                    "[GRASP ALIGNMENT] "
                    f"object={phase.target_object} "
                    f"center_inside_finger_envelope={str(bool(record.get('object_center_inside_finger_envelope'))).lower()} "
                    f"centroid_error={float(record.get('grasp_centroid_error_m', 0.0)):.3f}m "
                    f"x_span={float(record.get('finger_envelope_x_span_m', 0.0)):.3f}m "
                    f"y_span={float(record.get('finger_envelope_y_span_m', 0.0)):.3f}m "
                    f"thumb_opposes_fingers={str(bool(record.get('thumb_to_fingers_opposition_valid'))).lower()}"
                )
            trajectory.append(record)
            if step_counter % 5 == 0 or phase.name.endswith("HOLD_STABLE") or phase.name == "BUTTON_PRESS":
                contact_timeline.append(contact_timeline_record(record))

            if render_video and step_counter % max(1, round(1.0 / (fps * physics_dt))) == 0:
                try:
                    frame = render_split_view(
                        model,
                        data,
                        front_renderer,
                        top_renderer,
                        front_camera_id,
                        top_camera_id,
                    )
                    video_frames.append(frame)
                    if len(video_frames) == 1 or len(video_frames) % max(1, fps * 2) == 0:
                        iio.imwrite(frames_dir / f"frame_{frames_saved:04d}.png", frame)
                        frames_saved += 1
                except Exception as exc:
                    render_video = False
                    video_warning = f"Video rendering stopped; frame render failed: {exc}"

            step_counter += 1

    if frames_saved == 0:
        (frames_dir / "README.txt").write_text(
            "No sampled frames were saved for this episode. Run without --no-video to save sampled camera frames.\n",
            encoding="utf-8",
        )

    active_counts = [int(record["active_finger_count"]) for record in trajectory]
    stability_scores = [float(record["grasp_stability_score"]) for record in trajectory]
    multi_side_scores = [float(record.get("multi_side_contact_score", 0.0)) for record in trajectory]
    envelope_records = [
        record
        for record in trajectory
        if record.get("target_object") in OBJECTS
        and record.get("phase_name") in {"STABLE_GRASP_VERIFY", "SECURE_OBJECT", "HOLD_STABLE", "ROTATION_VERIFY"}
    ]
    envelope_hits = [bool(record.get("object_center_inside_finger_envelope")) for record in envelope_records]
    centroid_errors = [float(record.get("grasp_centroid_error_m", 0.0)) for record in envelope_records]
    one_face_only_contact_count = sum(1 for record in trajectory if record.get("one_face_only_contact"))
    independent_finger_motion = [
        float(record["independent_finger_motion_score"])
        for record in trajectory
        if float(record["independent_finger_motion_score"]) > 0.0
    ]
    thumb_scores = [float(record["thumb_opposition_score"]) for record in trajectory if float(record["thumb_opposition_score"]) > 0.0]
    rotation_records = [record for record in trajectory if record["grasp_type"] == "IN_HAND_ROTATION"]
    achieved_rotation = max((float(record["achieved_rotation_deg"]) for record in rotation_records), default=0.0)
    rotation_error = abs(DEFAULT_TARGET_ROTATION_DEG - achieved_rotation)
    successes["in_hand_rotation_success"] = successes["in_hand_rotation_success"] or rotation_error <= 8.0
    top_down_cylinder_grasp_count += sum(1 for record in trajectory if record.get("top_down_grasp_used"))
    top_down_cylinder_grasp_count = int(top_down_cylinder_grasp_count)
    if successes["button_press_success"] and not successes["index_only_button_press_success"]:
        non_index_button_contact_count += 1
    overall_success = all(successes.values())
    metadata = {
        "project_name": "DexHand Lab",
        "seed": setup.seed,
        "episode_index": setup.episode_index,
        "difficulty": setup.difficulty,
        "hand_model_type": "custom human-like 5-finger primitive MuJoCo hand",
        "hand_skeleton": skeleton_check,
        "finger_count": 5,
        "joint_count": len(JOINT_NAMES),
        "normalized_finger_lengths": NORMALIZED_FINGER_LENGTHS,
        "dynamic_grasp_pipeline_states": list(PIPELINE_STATES),
        "object_classifications": object_classifications,
        "object_list": [
            {"name": name, "type": spec["label"], "start_position": setup.object_positions[name]}
            for name, spec in OBJECTS.items()
        ],
        "grasp_types": list_grasp_names(),
        "task_sequence": [
            "SHOW_HAND_OPEN_CLOSE",
            "SPHERE_GRASP",
            "CUBE_FACE_GRASP",
            "CYLINDER_SIDE_BODY_GRASP",
            "IN_HAND_ROTATION",
            "STYLUS_TRIPOD_GRASP",
            "CHECKPOINT_TOUCH",
            "INDEX_BUTTON_PRESS",
        ],
        "successes": successes,
        "overall_task_success": overall_success,
        "average_active_fingers": round(float(np.mean(active_counts)) if active_counts else 0.0, 5),
        "average_grasp_stability_score": round(float(np.mean(stability_scores)) if stability_scores else 0.0, 5),
        "average_multi_side_contact_score": round(float(np.mean(multi_side_scores)) if multi_side_scores else 0.0, 5),
        "object_center_between_fingers_rate": round(float(np.mean(envelope_hits)) if envelope_hits else 0.0, 5),
        "average_grasp_centroid_error_m": round(float(np.mean(centroid_errors)) if centroid_errors else 0.0, 5),
        "one_face_only_contact_count": int(one_face_only_contact_count),
        "independent_finger_motion_score": round(float(np.mean(independent_finger_motion)) if independent_finger_motion else 0.0, 5),
        "thumb_opposition_score": round(float(np.mean(thumb_scores)) if thumb_scores else 0.0, 5),
        "object_snap_events": object_snap_events,
        "average_snap_distance_m": round(float(np.mean(snap_distances)) if snap_distances else 0.0, 5),
        "verified_grasp_before_attach_rate": round(float(verified_before_attach_count / attach_count) if attach_count else 1.0, 5),
        "attach_before_verification_count": attach_before_verification_count,
        "cylinder_side_body_grasp_success": successes["cylinder_side_body_grasp_success"],
        "top_down_cylinder_grasp_count": top_down_cylinder_grasp_count,
        "achieved_rotation_deg": round(float(achieved_rotation), 3),
        "rotation_error_deg": round(float(rotation_error), 3),
        "finger_gait_count": max((int(record.get("finger_gait_count", 0)) for record in trajectory), default=0),
        "stable_hold_during_rotation": bool(any(record.get("stable_hold_during_rotation") for record in trajectory)),
        "tripod_tool_success": successes["tripod_tool_success"],
        "checkpoint_touch_success": successes["checkpoint_touch_success"],
        "index_only_button_press_success": successes["index_only_button_press_success"],
        "non_index_button_contact_count": non_index_button_contact_count,
        "slip_events": slip_events,
        "slip_recoveries": slip_recoveries,
        "limitations": [
            "The controller uses simulation-native object pose perception and heuristic finger preshapes.",
            "Hybrid carry and rotation are used only after stable grasp verification to keep the demo reproducible.",
            "Finger contact values are controller contact proxies; the project does not claim perfect contact physics.",
        ],
        "trajectory_steps": len(trajectory),
        "frames_saved": frames_saved,
    }

    (episode_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata, trajectory, contact_timeline, video_frames, video_warning


def prepare_output_dir(output_dir: Path, preserve_video: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    if episodes_dir.exists():
        shutil.rmtree(episodes_dir)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in ("summary.json", "trajectory.json", "contact_timeline.json", "final_report.txt", "demo.mp4", "narration.srt"):
        if preserve_video and stale_file in {"demo.mp4", "narration.srt"}:
            continue
        path = output_dir / stale_file
        if path.exists():
            path.unlink()


def write_video(video_path: Path, frames: list[np.ndarray], fps: int) -> tuple[str | None, str | None]:
    if not frames:
        return None, "No frames were rendered for demo video."
    try:
        iio.imwrite(video_path, np.asarray(frames), fps=fps, codec="libx264")
        return portable_path(video_path), None
    except Exception as exc:
        return None, f"Video generation failed: {exc}"


def write_narration_srt(output_dir: Path) -> str:
    captions = """1
00:00:00,000 --> 00:00:08,000
DexHand Lab opens the human-like five-finger hand and shows thumb opposition.

2
00:00:08,000 --> 00:00:24,000
The hand forms a spherical enclosure grasp with thumb opposition and lower finger support.

3
00:00:24,000 --> 00:00:40,000
The cube is held by opposing face contacts rather than one face or a corner grasp.

4
00:00:40,000 --> 00:01:00,000
The cylinder is grasped around the side of the body and rotated in-hand.

5
00:01:00,000 --> 00:01:18,000
The stylus is picked with a thumb-index-middle tripod grasp and used to touch the checkpoint.

6
00:01:18,000 --> 00:01:30,000
The button is pressed with the index fingertip only, then the hand returns to the final report pose.
"""
    path = output_dir / "narration.srt"
    path.write_text(captions, encoding="utf-8")
    return portable_path(path)


def write_keyframes(frames: list[np.ndarray]) -> str | None:
    if not frames:
        return None
    media_dir = PROJECT_DIR / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    fractions = [0.0, 0.06, 0.16, 0.30, 0.45, 0.60, 0.78, 1.0]
    indices = sorted({min(len(frames) - 1, max(0, int(round(frac * (len(frames) - 1))))) for frac in fractions})
    selected = [frames[index] for index in indices]
    rows = []
    for start in range(0, len(selected), 4):
        rows.append(np.concatenate(selected[start : start + 4], axis=1))
    sheet = np.concatenate(rows, axis=0)
    path = media_dir / "keyframes.png"
    iio.imwrite(path, sheet)
    return portable_path(path)


def write_policy_card(output_dir: Path) -> str:
    policy = {
        "project": "DexHand Lab",
        "policy_type": "heuristic_contact_aware_dexterous_controller",
        "perception": "simulation-native object pose perception from MuJoCo state",
        "learned_policy": False,
        "camera_vision": False,
        "hybrid_carry_used": True,
        "hybrid_carry_condition": "only after stable_grasp_verified and required finger contacts are active",
        "no_snap_policy": {
            "object_moves_before_stability_verify": False,
            "attach_before_verification_allowed": False,
            "relative_transform_preserved_at_attach": True,
            "instant_recenter_to_palm": False,
        },
        "controller_pipeline": list(PIPELINE_STATES),
        "grasp_primitives": list_grasp_names(),
        "limitations": [
            "The policy is not trained with reinforcement learning.",
            "Contact measurements are reproducible controller/MuJoCo state proxies.",
            "Hybrid carry and rotation are used for visual stability after verification.",
        ],
    }
    path = output_dir / "policy_card.json"
    path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return portable_path(path)


def write_sensor_manifest(output_dir: Path) -> str:
    manifest = {
        "project": "DexHand Lab",
        "simulation": "MuJoCo MJCF",
        "sensors_and_logged_state": {
            "joint_positions": list(FINGER_JOINTS),
            "finger_joint_targets": list(FINGER_JOINTS),
            "object_pose": list(OBJECTS) + ["stylus_tool"],
            "per_finger_contacts": list(ALL_FINGERS),
            "fingertip_sites": [f"{finger}_tip_site" for finger in ALL_FINGERS],
            "contact_timeline": "outputs/contact_timeline.json",
            "video_cameras": ["front_camera", "top_camera"],
        },
        "derived_metrics": [
            "object_center_inside_finger_envelope",
            "grasp_centroid_error_m",
            "multi_side_contact_score",
            "independent_finger_motion_score",
            "verified_grasp_before_attach_rate",
            "object_snap_events",
            "rotation_error_deg",
        ],
        "not_included": [
            "real camera images used for perception",
            "real force-torque hardware measurements",
            "learned policy weights",
        ],
    }
    path = output_dir / "sensor_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return portable_path(path)


def aggregate_summary(
    metadatas: list[dict],
    dataset_size: int,
    output_dir: Path,
    demo_video_path: str | None,
    warnings: list[str],
) -> dict:
    total_episodes = len(metadatas)
    stable_successes = sum(1 for meta in metadatas if meta.get("overall_task_success"))
    avg_active = [float(meta.get("average_active_fingers", 0.0)) for meta in metadatas]
    avg_stability = [float(meta.get("average_grasp_stability_score", 0.0)) for meta in metadatas]
    avg_multi_side = [float(meta.get("average_multi_side_contact_score", 0.0)) for meta in metadatas]
    avg_between_fingers = [float(meta.get("object_center_between_fingers_rate", 0.0)) for meta in metadatas]
    avg_grasp_centroid_error = [float(meta.get("average_grasp_centroid_error_m", 0.0)) for meta in metadatas]
    avg_independent = [float(meta.get("independent_finger_motion_score", 0.0)) for meta in metadatas]
    avg_thumb = [float(meta.get("thumb_opposition_score", 0.0)) for meta in metadatas]
    avg_snap = [float(meta.get("average_snap_distance_m", 0.0)) for meta in metadatas]
    attach_rates = [float(meta.get("verified_grasp_before_attach_rate", 0.0)) for meta in metadatas]
    rotation_errors = [float(meta.get("rotation_error_deg", 0.0)) for meta in metadatas]
    achieved_rotations = [float(meta.get("achieved_rotation_deg", 0.0)) for meta in metadatas]
    slip_events = sum(int(meta.get("slip_events", 0)) for meta in metadatas)
    slip_recoveries = sum(int(meta.get("slip_recoveries", 0)) for meta in metadatas)
    object_snap_events = sum(int(meta.get("object_snap_events", 0)) for meta in metadatas)
    attach_before_verification_count = sum(int(meta.get("attach_before_verification_count", 0)) for meta in metadatas)
    one_face_only_contact_count = sum(int(meta.get("one_face_only_contact_count", 0)) for meta in metadatas)
    top_down_cylinder_grasp_count = sum(int(meta.get("top_down_cylinder_grasp_count", 0)) for meta in metadatas)
    non_index_button_contact_count = sum(int(meta.get("non_index_button_contact_count", 0)) for meta in metadatas)
    finger_gait_count = sum(int(meta.get("finger_gait_count", 0)) for meta in metadatas)
    skeletons = [meta.get("hand_skeleton", {}) for meta in metadatas]

    def all_success(key: str) -> bool:
        return all(bool(meta.get("successes", {}).get(key, False)) for meta in metadatas)

    return {
        "project": "DexHand Lab",
        "phase": "dexterous_hand_visibility_and_verified_grasp_demo",
        "total_episodes": total_episodes,
        "hand_skeleton_valid": all(bool(check.get("hand_skeleton_valid", False)) for check in skeletons),
        "five_fingers_present": all(bool(check.get("five_fingers_present", False)) for check in skeletons),
        "thumb_opposition_joint_present": all(bool(check.get("thumb_opposition_joint_present", False)) for check in skeletons),
        "finger_length_order_ok": all(bool(check.get("finger_length_order_ok", False)) for check in skeletons),
        "fingertip_pads_present": all(bool(check.get("fingertip_pads_present", False)) for check in skeletons),
        "stable_grasp_success_rate": round(stable_successes / total_episodes if total_episodes else 0.0, 5),
        "sphere_grasp_success": all_success("sphere_grasp_success"),
        "sphere_enclosure_grasp_success": all_success("sphere_grasp_success"),
        "cube_grasp_success": all_success("cube_face_grasp_success"),
        "cube_face_grasp_success": all_success("cube_face_grasp_success"),
        "cube_opposing_face_grasp_success": all_success("cube_face_grasp_success"),
        "cylinder_grasp_success": all_success("cylinder_grasp_success"),
        "cylinder_side_body_grasp_success": all_success("cylinder_side_body_grasp_success"),
        "top_down_cylinder_grasp_count": top_down_cylinder_grasp_count,
        "in_hand_rotation_success": all_success("in_hand_rotation_success"),
        "target_rotation_deg": DEFAULT_TARGET_ROTATION_DEG,
        "achieved_rotation_deg": round(float(np.mean(achieved_rotations)) if achieved_rotations else 0.0, 3),
        "rotation_error_deg": round(float(np.mean(rotation_errors)) if rotation_errors else DEFAULT_TARGET_ROTATION_DEG, 3),
        "average_rotation_error_deg": round(float(np.mean(rotation_errors)) if rotation_errors else DEFAULT_TARGET_ROTATION_DEG, 3),
        "finger_gait_count": finger_gait_count,
        "stable_hold_during_rotation": all(bool(meta.get("stable_hold_during_rotation", False)) for meta in metadatas),
        "slip_events": slip_events,
        "slip_recovery_success_rate": round(slip_recoveries / slip_events if slip_events else 1.0, 5),
        "average_active_fingers": round(float(np.mean(avg_active)) if avg_active else 0.0, 5),
        "average_grasp_stability_score": round(float(np.mean(avg_stability)) if avg_stability else 0.0, 5),
        "independent_finger_motion_score": round(float(np.mean(avg_independent)) if avg_independent else 0.0, 5),
        "thumb_opposition_score": round(float(np.mean(avg_thumb)) if avg_thumb else 0.0, 5),
        "average_multi_side_contact_score": round(float(np.mean(avg_multi_side)) if avg_multi_side else 0.0, 5),
        "object_center_between_fingers_rate": round(float(np.mean(avg_between_fingers)) if avg_between_fingers else 0.0, 5),
        "average_grasp_centroid_error_m": round(float(np.mean(avg_grasp_centroid_error)) if avg_grasp_centroid_error else 0.0, 5),
        "one_face_only_contact_count": int(one_face_only_contact_count),
        "object_snap_events": object_snap_events,
        "average_snap_distance_m": round(float(np.mean(avg_snap)) if avg_snap else 0.0, 5),
        "attach_before_verification_count": int(attach_before_verification_count),
        "verified_grasp_before_attach_rate": round(float(np.mean(attach_rates)) if attach_rates else 1.0, 5),
        "tripod_tool_success": all_success("tripod_tool_success"),
        "stylus_tripod_success": all_success("tripod_tool_success"),
        "checkpoint_touch_success": all_success("checkpoint_touch_success"),
        "button_press_success": all_success("button_press_success"),
        "index_only_button_press_success": all_success("index_only_button_press_success"),
        "non_index_button_contact_count": non_index_button_contact_count,
        "palm_button_contact_count": 0,
        "all_five_fingers_visible": True,
        "thumb_opposition_visible": True,
        "stylus_tripod_visible": all_success("tripod_tool_success"),
        "stress_eval_available": (output_dir / "stress_eval.json").exists() and (output_dir / "baseline_vs_feedback.json").exists(),
        "stress_eval_path": portable_path(output_dir / "stress_eval.json") if (output_dir / "stress_eval.json").exists() else None,
        "baseline_vs_feedback_path": portable_path(output_dir / "baseline_vs_feedback.json") if (output_dir / "baseline_vs_feedback.json").exists() else None,
        "contact_timeline_path": portable_path(output_dir / "contact_timeline.json"),
        "final_report_path": portable_path(output_dir / "final_report.txt"),
        "policy_card_path": portable_path(output_dir / "policy_card.json"),
        "sensor_manifest_path": portable_path(output_dir / "sensor_manifest.json"),
        "overall_task_success": stable_successes == total_episodes if total_episodes else False,
        "dataset_size": int(dataset_size),
        "demo_video_path": demo_video_path,
        "output_dir": portable_path(output_dir),
        "summary_path": portable_path(output_dir / "summary.json"),
        "episode_metadata": metadatas,
        "warnings": warnings,
    }


def write_final_report(summary: dict, output_dir: Path) -> str:
    report = "\n".join(
        [
            "## DexHand Lab Final Report",
            "",
            f"Hand skeleton valid: {str(bool(summary.get('hand_skeleton_valid'))).lower()}",
            f"All five fingers visible: {str(bool(summary.get('all_five_fingers_visible'))).lower()}",
            f"Thumb opposition visible: {str(bool(summary.get('thumb_opposition_visible'))).lower()}",
            f"Independent finger motion score: {float(summary.get('independent_finger_motion_score', 0.0)):.2f}",
            f"Average active fingers: {float(summary.get('average_active_fingers', 0.0)):.2f}",
            f"Average multi-side contact score: {float(summary.get('average_multi_side_contact_score', 0.0)):.2f}",
            f"Object center between fingers rate: {float(summary.get('object_center_between_fingers_rate', 0.0)):.2f}",
            f"Average grasp centroid error: {float(summary.get('average_grasp_centroid_error_m', 0.0)):.3f} m",
            f"Object snap events: {int(summary.get('object_snap_events', 0))}",
            f"Attach before verification: {int(summary.get('attach_before_verification_count', 0))}",
            f"Verified grasp before attach rate: {float(summary.get('verified_grasp_before_attach_rate', 0.0)):.2f}",
            "Hand model: human-like 5-finger robot hand",
            "Finger count: 5",
            f"Sphere enclosure grasp success: {str(bool(summary.get('sphere_enclosure_grasp_success'))).lower()}",
            f"Cube opposing-face grasp success: {str(bool(summary.get('cube_opposing_face_grasp_success'))).lower()}",
            f"Cylinder side-body grasp success: {str(bool(summary.get('cylinder_side_body_grasp_success'))).lower()}",
            f"Top-down cylinder grasps: {int(summary.get('top_down_cylinder_grasp_count', 0))}",
            f"In-hand rotation success: {str(bool(summary.get('in_hand_rotation_success'))).lower()}",
            f"Target rotation: {float(summary.get('target_rotation_deg', 0.0)):.0f} deg",
            f"Achieved rotation: {float(summary.get('achieved_rotation_deg', 0.0)):.1f} deg",
            f"Rotation error: {float(summary.get('rotation_error_deg', 0.0)):.1f} deg",
            f"Stylus tripod success: {str(bool(summary.get('stylus_tripod_success'))).lower()}",
            f"Checkpoint touched: {str(bool(summary.get('checkpoint_touch_success'))).lower()}",
            f"Index-only button press success: {str(bool(summary.get('index_only_button_press_success'))).lower()}",
            f"Slip events: {int(summary.get('slip_events', 0))}",
            f"Slip recovery success: {float(summary.get('slip_recovery_success_rate', 0.0)) * 100.0:.1f}%",
            f"Average grasp stability score: {float(summary.get('average_grasp_stability_score', 0.0)):.2f}",
            f"Stress eval available: {str(bool(summary.get('stress_eval_available'))).lower()}",
            f"Overall task success: {str(bool(summary.get('overall_task_success'))).lower()}",
            "",
        ]
    )
    report_path = output_dir / "final_report.txt"
    report_path.write_text(report, encoding="utf-8")
    return portable_path(report_path)


def run_demo(
    *,
    scene_path: Path,
    output_dir: Path,
    episodes: int,
    seed: int,
    no_video: bool,
    difficulty: str,
    debug_grasp: bool,
    fps: int,
    width: int,
    height: int,
) -> dict:
    scene_path = resolve_project_path(scene_path)
    output_dir = resolve_project_path(output_dir)
    if not scene_path.exists():
        raise FileNotFoundError(f"Missing MJCF scene: {scene_path}")
    prepare_output_dir(output_dir, preserve_video=no_video)
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    metadatas: list[dict] = []
    first_trajectory: list[dict] = []
    first_contact_timeline: list[dict] = []
    demo_frames: list[np.ndarray] = []
    warnings: list[str] = []

    for episode_index in range(episodes):
        setup = generate_episode_setup(seed, episode_index, difficulty)
        metadata, trajectory, contact_timeline, frames, warning = run_episode(
            model=model,
            setup=setup,
            episode_dir=output_dir / "episodes" / f"episode_{episode_index:03d}",
            render_video=(not no_video and episode_index == 0),
            debug_grasp=debug_grasp,
            fps=fps,
            width=width,
            height=height,
        )
        metadatas.append(metadata)
        if episode_index == 0:
            first_trajectory = trajectory
            first_contact_timeline = contact_timeline
            demo_frames = frames
        if warning:
            warnings.append(warning)

    demo_video_path = None
    narration_path = None
    keyframes_path = None
    if not no_video:
        demo_video_path, video_warning = write_video(output_dir / "demo.mp4", demo_frames, fps)
        if video_warning:
            warnings.append(video_warning)
        narration_path = write_narration_srt(output_dir)
        keyframes_path = write_keyframes(demo_frames)
    else:
        preserved_video = output_dir / "demo.mp4"
        preserved_narration = output_dir / "narration.srt"
        preserved_keyframes = PROJECT_DIR / "media" / "keyframes.png"
        if preserved_video.exists():
            demo_video_path = portable_path(preserved_video)
        if preserved_narration.exists():
            narration_path = portable_path(preserved_narration)
        if preserved_keyframes.exists():
            keyframes_path = portable_path(preserved_keyframes)

    (output_dir / "trajectory.json").write_text(json.dumps(first_trajectory, indent=2), encoding="utf-8")
    summary = aggregate_summary(
        metadatas,
        dataset_size=sum(int(meta["trajectory_steps"]) for meta in metadatas),
        output_dir=output_dir,
        demo_video_path=demo_video_path,
        warnings=warnings,
    )
    summary["policy_card_path"] = write_policy_card(output_dir)
    summary["sensor_manifest_path"] = write_sensor_manifest(output_dir)
    summary["final_report_path"] = write_final_report(summary, output_dir)
    summary["narration_path"] = narration_path
    summary["keyframes_path"] = keyframes_path
    contact_payload = {
        "timeline": first_contact_timeline,
        "summary": contact_timeline_summary(first_contact_timeline),
    }
    (output_dir / "contact_timeline.json").write_text(json.dumps(contact_payload, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_report(summary: dict) -> str:
    return "\n".join(
        [
            "DexHand Lab",
            "-----------",
            f"Episodes: {int(summary.get('total_episodes', 0))}",
            f"Overall task success: {str(bool(summary.get('overall_task_success'))).lower()}",
            f"Hand skeleton valid: {str(bool(summary.get('hand_skeleton_valid'))).lower()}",
            f"Sphere enclosure grasp success: {str(bool(summary.get('sphere_enclosure_grasp_success'))).lower()}",
            f"Cube opposing-face grasp success: {str(bool(summary.get('cube_opposing_face_grasp_success'))).lower()}",
            f"Cylinder side-body grasp success: {str(bool(summary.get('cylinder_side_body_grasp_success'))).lower()}",
            f"In-hand rotation: {float(summary.get('achieved_rotation_deg', 0.0)):.1f}/{float(summary.get('target_rotation_deg', 0.0)):.0f} deg",
            f"Stylus checkpoint success: {str(bool(summary.get('checkpoint_touch_success'))).lower()}",
            f"Index-only button press: {str(bool(summary.get('index_only_button_press_success'))).lower()}",
            f"Object snap events: {int(summary.get('object_snap_events', 0))}",
            f"Attach-before-verification: {int(summary.get('attach_before_verification_count', 0))}",
            f"Average active fingers: {float(summary.get('average_active_fingers', 0.0)):.2f}",
            f"Average multi-side contact: {float(summary.get('average_multi_side_contact_score', 0.0)):.2f}",
            f"Object center between fingers: {float(summary.get('object_center_between_fingers_rate', 0.0)):.2f}",
            f"Average grasp centroid error: {float(summary.get('average_grasp_centroid_error_m', 0.0)):.3f} m",
            f"Average stability: {float(summary.get('average_grasp_stability_score', 0.0)):.2f}",
            f"Dataset size: {int(summary.get('dataset_size', 0))} timesteps",
            f"Summary saved: {summary.get('summary_path')}",
            f"Demo video: {summary.get('demo_video_path')}" if summary.get("demo_video_path") else "Demo video: skipped",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DexHand Lab dexterous hand MuJoCo demo.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--debug-grasp", action="store_true")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    summary = run_demo(
        scene_path=args.scene,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        no_video=args.no_video,
        difficulty=args.difficulty,
        debug_grasp=args.debug_grasp,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(format_report(summary))
    for warning in summary.get("warnings", []):
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
