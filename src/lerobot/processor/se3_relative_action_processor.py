"""UMI-style relative end-effector actions for pose-target policies.

RelativeActionsProcessorStep (relative_action_processor.py) subtracts the state elementwise, which is right
for joint positions but not for an end-effector pose: subtracting Euler angles or 6D rotation vectors is not
a rotation. UMI (Chi et al., 2024) instead expresses every action of a chunk as the target pose IN THE FRAME
OF THE CURRENT END-EFFECTOR POSE:

    T_rel[k] = T_ref^-1 @ T_target[k]       T_ref = the end-effector pose at the latest observation

so a chunk is "where to go, seen from where I am now" -- invariant to where the task sits in the robot's
workspace, and (unlike per-step deltas) every action of the chunk refers to the same frame, so errors do not
accumulate along the chunk.

Layouts (fixed, documented here rather than configurable):
  action  (..., 9+): absolute target [x, y, z, r1(3), r2(3)] -- position, then the first two COLUMNS of the
                     rotation matrix (the 6D rotation of Zhou et al.); any further dims (e.g. auxiliary
                     channels) pass through unchanged.
  reference pose, where the current end-effector pose is read from (`reference_pose_key`):
    None (default)   the first 6 dims of observation.state, [x, y, z, rx, ry, rz]: position and EXTRINSIC xyz
                     Euler angles (scipy "xyz", the convention airo-spatial-algebra's SE3Container uses).
    a key, e.g. "observation.eef_pose"   that observation, [x, y, z, r1(3), r2(3)] (..., 9): position and 6D
                     rotation, like the actions. The policy does not read this key (diffusion conditions on
                     observation.state and the images only; if input_features are inferred from the dataset it is
                     listed there and normalized, which is harmless: this step uses the raw pose, before the
                     normalizer), and there are no Euler angles anywhere: no wrap-around at +-pi, no gimbal lock. The policy's state can then carry a
                     relative proprioception instead (UMI: past poses in the frame of the current one).

Only the training half lives in the processor pipeline. The inverse (relative -> absolute) is NOT done in a
postprocessor step: DiffusionPolicy queues a whole chunk and the postprocessor then runs once per POPPED
action, while a postprocessor reading the "last state" would see the state of the latest observation -- for
n_action_steps > 1 every action after the first of a chunk would be composed onto the wrong reference. The
caller converts back with `relative_to_absolute_pose_actions`, using the pose cached when the chunk was
predicted.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry

# the absolute end-effector pose [x, y, z, rot6d] of `reference_pose_key` datasets (only a convention: any key works)
OBS_EEF_POSE = "observation.eef_pose"
POSE_LAYOUTS = ("euler", "rot6d")

__all__ = [
    "OBS_EEF_POSE",
    "SE3RelativeActionsProcessorStep",
    "euler_xyz_to_matrix",
    "rotation_6d_to_matrix",
    "matrix_to_rotation_6d",
    "absolute_to_relative_pose_actions",
    "relative_to_absolute_pose_actions",
]


def euler_xyz_to_matrix(euler: Tensor) -> Tensor:
    """(..., 3) extrinsic xyz Euler angles -> (..., 3, 3); equals scipy Rotation.from_euler("xyz", e)."""
    cx, cy, cz = torch.cos(euler).unbind(-1)
    sx, sy, sz = torch.sin(euler).unbind(-1)
    # extrinsic x then y then z  ==  R = Rz @ Ry @ Rx
    row0 = torch.stack((cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz), dim=-1)
    row1 = torch.stack((cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz), dim=-1)
    row2 = torch.stack((-sy, sx * cy, cx * cy), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def rotation_6d_to_matrix(rot6d: Tensor) -> Tensor:
    """(..., 6) first two columns -> (..., 3, 3), Gram-Schmidt so a noisy prediction still gives a rotation."""
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)  # columns


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """(..., 3, 3) -> (..., 6): first column, then second column."""
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def _reference_pose(state: Tensor, pose_layout: str = "euler") -> tuple[Tensor, Tensor]:
    """(position (B, 3), rotation (B, 3, 3)) of the latest observation; state is (B, D) or (B, n_obs, D), with
    the pose in its first 6 ("euler": [xyz, extrinsic xyz euler]) or 9 ("rot6d": [xyz, rot6d]) dims."""
    if pose_layout not in POSE_LAYOUTS:
        raise ValueError(f"pose_layout must be one of {POSE_LAYOUTS}, got {pose_layout!r}")
    if state.ndim == 3:
        state = state[:, -1]
    if pose_layout == "rot6d":
        return state[..., :3], rotation_6d_to_matrix(state[..., 3:9])
    return state[..., :3], euler_xyz_to_matrix(state[..., 3:6])


def absolute_to_relative_pose_actions(actions: Tensor, state: Tensor, pose_layout: str = "euler") -> Tensor:
    """(B, T, 9+) or (B, 9+) absolute targets -> the same targets in the frame of the latest state pose.

    `state` holds the reference pose in the layout `pose_layout` ("euler" or "rot6d", see _reference_pose)."""
    squeeze = actions.ndim == 2
    if squeeze:
        actions = actions.unsqueeze(1)
    state = state.to(device=actions.device, dtype=actions.dtype)
    ref_position, ref_rotation = _reference_pose(state, pose_layout)
    ref_rotation_t = ref_rotation.transpose(-1, -2).unsqueeze(1)  # (B, 1, 3, 3)
    position = (ref_rotation_t @ (actions[..., :3] - ref_position.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    rotation = ref_rotation_t @ rotation_6d_to_matrix(actions[..., 3:9])
    out = torch.cat((position, matrix_to_rotation_6d(rotation), actions[..., 9:]), dim=-1)
    return out.squeeze(1) if squeeze else out


def relative_to_absolute_pose_actions(actions: Tensor, state: Tensor, pose_layout: str = "euler") -> Tensor:
    """Inverse of absolute_to_relative_pose_actions: relative targets + the reference state -> absolute."""
    squeeze = actions.ndim == 2
    if squeeze:
        actions = actions.unsqueeze(1)
    state = state.to(device=actions.device, dtype=actions.dtype)
    ref_position, ref_rotation = _reference_pose(state, pose_layout)
    ref_rotation = ref_rotation.unsqueeze(1)
    position = (ref_rotation @ actions[..., :3].unsqueeze(-1)).squeeze(-1) + ref_position.unsqueeze(1)
    rotation = ref_rotation @ rotation_6d_to_matrix(actions[..., 3:9])
    out = torch.cat((position, matrix_to_rotation_6d(rotation), actions[..., 9:]), dim=-1)
    return out.squeeze(1) if squeeze else out


@ProcessorStepRegistry.register("se3_relative_actions_processor")
@dataclass
class SE3RelativeActionsProcessorStep(ProcessorStep):
    """Absolute pose-target actions -> UMI-style relative ones (see the module docstring), before normalization.

    A no-op on transitions without an action, i.e. at inference: there the caller converts the policy's
    relative output back itself (relative_to_absolute_pose_actions), with the pose of the observation the
    chunk was predicted from.
    """

    enabled: bool = False
    # None: the reference pose is observation.state[..., :6] as [xyz, euler]; a key: that observation, [xyz, rot6d]
    reference_pose_key: str | None = None

    @property
    def pose_layout(self) -> str:
        return "euler" if self.reference_pose_key is None else "rot6d"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        action = transition.get(TransitionKey.ACTION)
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        if action is None:
            return transition
        reference = observation.get(self.reference_pose_key or OBS_STATE)
        if reference is None:
            if self.reference_pose_key is not None:
                raise KeyError(
                    f"relative actions: the reference pose {self.reference_pose_key!r} is not in the batch"
                )
            return transition
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = absolute_to_relative_pose_actions(
            action, reference, self.pose_layout
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        # the key only when set: a pipeline without one stays loadable by versions that predate the field
        config: dict[str, Any] = {"enabled": self.enabled}
        if self.reference_pose_key is not None:
            config["reference_pose_key"] = self.reference_pose_key
        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
