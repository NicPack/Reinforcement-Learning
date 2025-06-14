from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .utils import combined_shape


class BaseBuffer(ABC):
    @abstractmethod
    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ) -> None:
        self.device = device

        self.actions = torch.zeros(
            combined_shape(size, env.action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminations = torch.zeros(size, dtype=torch.float32, device=device)
        self.truncations = torch.zeros(size, dtype=torch.float32, device=device)
        self.infos = np.empty((size,), dtype=object)
        self._ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        action: NDArray,
        reward: float,
        next_observation: Union[NDArray, dict[str, NDArray]],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._store_observations(observation, next_observation)
        self.actions[self._ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self._ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.terminations[self._ptr] = torch.as_tensor(terminated, dtype=torch.float32)
        self.truncations[self._ptr] = torch.as_tensor(truncated, dtype=torch.float32)
        self.infos[self._ptr] = info
        self._ptr = (self._ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @abstractmethod
    def _store_observations(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        next_observation: Union[NDArray, dict[str, NDArray]],
    ) -> None: ...

    def sample_batch(
        self, batch_size: int = 32
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        idxs = torch.randint(0, self.size, size=(batch_size,))
        # idxs = np.random.randint(0, self.size, size=batch_size)
        return self.batch(idxs)

    def batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        data = dict(
            action=self.actions[idxs],
            reward=self.rewards[idxs],
            terminated=self.terminations[idxs],
            truncated=self.truncations[idxs],
            info=self.infos[idxs],
        )
        observations = self._observations_batch(idxs)
        data.update(observations)

        return data

    @abstractmethod
    def _observations_batch(
        self, idxs: Tensor
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]: ...

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def clear(self):
        self.actions.zero_()
        self.rewards.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.infos.fill(None)
        self._ptr, self.size = 0, 0


class DictReplayBuffer(BaseBuffer):
    """
    A dictionary experience replay buffer for off-policy agents.
    """

    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert isinstance(env, gym.Env)
        super().__init__(env=env, size=size, device=device)

        obs_space = {
            k: combined_shape(size, v.shape) for k, v in env.observation_space.items()
        }

        self.observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }
        self.next_observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }

    def _store_observations(
        self,
        observation: dict[str, NDArray],
        next_observation: dict[str, NDArray],
    ) -> None:
        for k in observation.keys():
            self.observations[k][self._ptr] = torch.as_tensor(
                observation[k], dtype=torch.float32
            )
        for k in next_observation.keys():
            self.next_observations[k][self._ptr] = torch.as_tensor(
                next_observation[k], dtype=torch.float32
            )

    def _observations_batch(self, idxs: Tensor) -> dict[str, dict[str, Tensor]]:
        return dict(
            observation={k: v[idxs] for k, v in self.observations.items()},
            next_observation={k: v[idxs] for k, v in self.next_observations.items()},
        )




class HerReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        env: gym.Env,
        size: int = 100000,
        device: Optional[torch.device] = None,
        n_sampled_goal: int = 1,
        goal_selection_strategy: str = "final",
    ):
        super().__init__(env=env, size=size, device=device)
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.selection_strategy = goal_selection_strategy
        # TODO: fill this in
        # You can put additional attributes here if needed.
        # Also: There is a number of methods in the base class that could be useful to override.
        self.current_episode = []  # Store the current episode transitions

   
    def store(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_observation: dict[str, torch.Tensor],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ):
        # Store in base replay buffer
        super().store(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        # Append transition to current episode for HER processing
        transition = (
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
            info,
        )
        self.current_episode.append(transition)

    def start_episode(self):
        self.current_episode = []
    
    def end_episode(self):
        if len(self.current_episode) == 0:
            return

        for idx, transition in enumerate(self.current_episode):
            obs, action, _, next_obs, terminated, truncated, info = transition

            # Sample relabeled goals using current strategy (e.g., future achieved goals)
            sampled_goals = self._sample_achieved_goals(idx)

            for goal in sampled_goals:
                # Deep copies to avoid modifying stored obs
                obs_ = obs.copy()
                next_obs_ = next_obs.copy()

                # Replace desired_goal with HER-goal
                obs_["desired_goal"] = goal
                next_obs_["desired_goal"] = goal

                # Recompute reward for relabeled goal
                reward = self.compute_reward(
                    achieved_goal=next_obs_["achieved_goal"],
                    desired_goal=goal,
                    info=info
                )

                # Store relabeled transition
                super().store(obs_, action, reward, next_obs_, terminated, truncated, info)

            # Optional: Store original transition as well
            # If using HER as augmentation, keep this
            super().store(obs, action, transition[2], next_obs, terminated, truncated, info)

        # Clear buffer for the next episode
        self.current_episode = []
    
    def _sample_achieved_goals(self, current_index: int) -> list[np.ndarray]:
        """Sample achieved goals according to strategy."""
        future_achieved_goals = []

        if self.selection_strategy == "final":
            final_transition = self.current_episode[-1]
            achieved_goal = final_transition[3]["achieved_goal"]
            future_achieved_goals = [achieved_goal] * self.n_sampled_goal

        elif self.selection_strategy == "future":
            future_transitions = self.current_episode[current_index + 1 :]
            if len(future_transitions) == 0:
                return []

            # Sample n goals from future transitions
            indices = np.random.choice(
                len(future_transitions),
                size=min(self.n_sampled_goal, len(future_transitions)),
                replace=False,
            )
            for i in indices:
                future_achieved_goal = future_transitions[i][3]["achieved_goal"]
                future_achieved_goals.append(future_achieved_goal)

        elif self.selection_strategy == "episode":
            indices = np.random.choice(
                len(self.current_episode),
                size=min(self.n_sampled_goal, len(self.current_episode)),
                replace=False,
            )
            for i in indices:
                random_achieved_goal = self.current_episode[i][3]["achieved_goal"]
                future_achieved_goals.append(random_achieved_goal)

        else:
            raise ValueError(f"Unknown goal selection strategy: {self.selection_strategy}")

        return future_achieved_goals
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        return -np.linalg.norm(achieved_goal - desired_goal)
