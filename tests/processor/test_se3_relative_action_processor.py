#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the UMI-style SE3 relative pose actions."""

import tempfile

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.processor import (
    DataProcessorPipeline,
    SE3RelativeActionsProcessorStep,
    absolute_to_relative_pose_actions,
    relative_to_absolute_pose_actions,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.processor.se3_relative_action_processor import euler_xyz_to_matrix, matrix_to_rotation_6d
from lerobot.utils.constants import ACTION, OBS_STATE

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


def _random_state(batch: int, generator: torch.Generator) -> torch.Tensor:
    """[xyz, extrinsic xyz euler, gripper] states."""
    position = torch.randn(batch, 3, generator=generator, dtype=torch.float64)
    euler = (torch.rand(batch, 3, generator=generator, dtype=torch.float64) - 0.5) * 2 * np.pi * 0.9
    return torch.cat(
        (position, euler, torch.rand(batch, 1, generator=generator, dtype=torch.float64)), dim=-1
    )


def _random_actions(batch: int, horizon: int, generator: torch.Generator, extra: int = 0) -> torch.Tensor:
    """absolute [xyz, rot6d, extra...] targets."""
    rotations = torch.from_numpy(Rotation.random(batch * horizon, random_state=0).as_matrix())
    rot6d = matrix_to_rotation_6d(rotations).view(batch, horizon, 6)
    position = torch.randn(batch, horizon, 3, generator=generator, dtype=torch.float64)
    return torch.cat(
        (position, rot6d, torch.randn(batch, horizon, extra, generator=generator, dtype=torch.float64)), -1
    )


def _pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, position
    return matrix


def test_euler_matches_scipy_extrinsic_xyz():
    euler = torch.tensor([[0.3, -1.2, 2.5], [-2.0, 0.4, 0.1]], dtype=torch.float64)
    expected = Rotation.from_euler("xyz", euler.numpy()).as_matrix()
    np.testing.assert_allclose(euler_xyz_to_matrix(euler).numpy(), expected, atol=1e-12)


def test_relative_action_is_target_in_reference_frame():
    generator = torch.Generator().manual_seed(0)
    state, actions = _random_state(2, generator), _random_actions(2, 4, generator)
    relative = absolute_to_relative_pose_actions(actions, state)

    for b in range(2):
        reference = _pose_matrix(
            state[b, :3].numpy(), Rotation.from_euler("xyz", state[b, 3:6].numpy()).as_matrix()
        )
        for k in range(4):
            target_rotation = np.stack((actions[b, k, 3:6], actions[b, k, 6:9]), axis=1)
            target_rotation = np.column_stack(
                (target_rotation, np.cross(target_rotation[:, 0], target_rotation[:, 1]))
            )
            expected = np.linalg.inv(reference) @ _pose_matrix(actions[b, k, :3].numpy(), target_rotation)
            np.testing.assert_allclose(relative[b, k, :3].numpy(), expected[:3, 3], atol=1e-10)
            np.testing.assert_allclose(
                relative[b, k, 3:9].numpy(), np.concatenate((expected[:3, 0], expected[:3, 1])), atol=1e-10
            )


def test_roundtrip_and_extra_dims_pass_through():
    generator = torch.Generator().manual_seed(1)
    state, actions = _random_state(3, generator), _random_actions(3, 5, generator, extra=2)
    relative = absolute_to_relative_pose_actions(actions, state)
    torch.testing.assert_close(relative[..., 9:], actions[..., 9:])
    torch.testing.assert_close(relative_to_absolute_pose_actions(relative, state), actions)
    # a single action (B, D) instead of a chunk (B, T, D)
    torch.testing.assert_close(
        relative_to_absolute_pose_actions(absolute_to_relative_pose_actions(actions[:, 0], state), state),
        actions[:, 0],
    )


def test_relative_actions_do_not_depend_on_where_the_task_is():
    # moving the whole scene (state and targets) by a rigid transform does not change the relative chunk
    generator = torch.Generator().manual_seed(2)
    state, actions = _random_state(1, generator), _random_actions(1, 3, generator)
    shift = Rotation.from_euler("xyz", [0.2, -0.5, 1.0])
    offset = np.array([0.4, -0.1, 0.3])

    moved_state = state.clone()
    reference = Rotation.from_euler("xyz", state[0, 3:6].numpy())
    moved_state[0, :3] = torch.from_numpy(shift.apply(state[0, :3].numpy()) + offset)
    moved_state[0, 3:6] = torch.from_numpy((shift * reference).as_euler("xyz"))
    moved_actions = actions.clone()
    moved_actions[0, :, :3] = torch.from_numpy(shift.apply(actions[0, :, :3].numpy()) + offset)
    rotation = shift.as_matrix()
    moved_actions[0, :, 3:6] = torch.from_numpy(actions[0, :, 3:6].numpy() @ rotation.T)
    moved_actions[0, :, 6:9] = torch.from_numpy(actions[0, :, 6:9].numpy() @ rotation.T)

    torch.testing.assert_close(
        absolute_to_relative_pose_actions(moved_actions, moved_state),
        absolute_to_relative_pose_actions(actions, state),
    )


def test_reference_is_the_latest_observation():
    generator = torch.Generator().manual_seed(3)
    states = torch.stack((_random_state(2, generator), _random_state(2, generator)), dim=1)  # (B, n_obs, D)
    actions = _random_actions(2, 4, generator)
    torch.testing.assert_close(
        absolute_to_relative_pose_actions(actions, states),
        absolute_to_relative_pose_actions(actions, states[:, -1]),
    )


def test_step_converts_actions_and_is_a_noop_without_action_or_when_disabled():
    generator = torch.Generator().manual_seed(4)
    state, actions = _random_state(2, generator), _random_actions(2, 4, generator)
    transition = create_transition({OBS_STATE: state}, actions)

    converted = SE3RelativeActionsProcessorStep(enabled=True)(transition)
    torch.testing.assert_close(converted[ACTION], absolute_to_relative_pose_actions(actions, state))
    assert SE3RelativeActionsProcessorStep(enabled=False)(transition)[ACTION] is actions
    inference = create_transition({OBS_STATE: state}, None)  # no action at inference
    assert SE3RelativeActionsProcessorStep(enabled=True)(inference)[ACTION] is None


def _diffusion_config(use_relative_actions: bool) -> DiffusionConfig:
    config = DiffusionConfig(use_relative_actions=use_relative_actions, device="cpu")
    config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))}
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(9,))}
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    return config


def test_diffusion_pipeline_only_has_the_step_when_enabled():
    preprocessor, _ = make_diffusion_pre_post_processors(_diffusion_config(False))
    assert not any(isinstance(step, SE3RelativeActionsProcessorStep) for step in preprocessor.steps)

    preprocessor, _ = make_diffusion_pre_post_processors(_diffusion_config(True))
    names = [type(step).__name__ for step in preprocessor.steps]
    # on the raw state, before normalization
    assert names.index("SE3RelativeActionsProcessorStep") == names.index("NormalizerProcessorStep") - 1


def test_diffusion_preprocessor_with_relative_actions_survives_save_and_load():
    generator = torch.Generator().manual_seed(5)
    # a training batch: (B, n_obs, D) states and (B, T, 9) action chunks
    state = torch.stack((_random_state(2, generator), _random_state(2, generator)), dim=1).float()
    actions = _random_actions(2, 4, generator).float()
    preprocessor, _ = make_diffusion_pre_post_processors(_diffusion_config(True))

    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessor.save_pretrained(tmpdir)
        loaded = DataProcessorPipeline.from_pretrained(tmpdir, config_filename="policy_preprocessor.json")

    batch = transition_to_batch(create_transition({OBS_STATE: state}, actions))
    processed = loaded(batch)
    expected = absolute_to_relative_pose_actions(actions, state)
    torch.testing.assert_close(processed[ACTION], expected)
