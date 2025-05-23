from __future__ import annotations

import collections
import random
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as skl_preprocessing
from problem import (
    Action,
    Corner,
    Driver,
    Environment,
    Experiment,
    State,
    available_actions,
)

ALMOST_INFINITE_STEP = 100000
MAX_LEARNING_STEPS = 500


class RandomDriver(Driver):
    def __init__(self):
        self.current_step: int = 0

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return random.choice(available_actions(state))

    def control(self, state: State, last_reward: int) -> Action:
        self.current_step += 1
        return random.choice(available_actions(state))

    def finished_learning(self) -> bool:
        return self.current_step > MAX_LEARNING_STEPS


class OffPolicyNStepSarsaDriver(Driver):
    def __init__(
        self,
        step_size: float,
        step_no: int,
        experiment_rate: float,
        discount_factor: float,
    ) -> None:
        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate
        self.discount_factor: float = discount_factor
        self.q: dict[tuple[State, Action], float] = collections.defaultdict(float)
        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = dict()
        self.actions: dict[int, Action] = dict()
        self.rewards: dict[int, int] = dict()
        self.visualization_mode: bool = False

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state
        action = self._select_action(
            self._select_policy(state, available_actions(state))
        )
        self.actions[self._access_index(self.current_step)] = action
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int) -> Action:
        if self.current_step < self.final_step:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.final_step == ALMOST_INFINITE_STEP and (
                last_reward == 0 or self.current_step == MAX_LEARNING_STEPS
            ):
                self.final_step = self.current_step
            action = self._select_action(
                self._select_policy(state, available_actions(state))
            )
            self.actions[self._access_index(self.current_step + 1)] = action
        else:
            action = Action(0, 0)

        update_step = self.current_step - self.step_no + 1
        if update_step >= 0:
            return_value_weight = self._return_value_weight(update_step)
            return_value = self._return_value(update_step)
            state_t = self.states[self._access_index(update_step)]
            action_t = self.actions[self._access_index(update_step)]
            self.q[state_t, action_t] += (
                self.step_size
                * return_value_weight
                * (return_value - self.q[state_t, action_t])
            )

        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1
        return action

    def _return_value(self, update_step):
        return_value = 0.0
        end_step = min(update_step + self.step_no, self.final_step)

        for i in range(update_step + 1, end_step):
            return_value += (
                self.discount_factor ** (i - update_step - 1)
            ) * self.rewards[self._access_index(i)]

        if update_step + self.step_no < self.final_step:
            next_state = self.states[self._access_index(update_step + self.step_no)]
            next_action = self.actions[self._access_index(update_step + self.step_no)]
            return_value += (self.discount_factor**self.step_no) * self.q[
                next_state, next_action
            ]

        return return_value

    def _return_value_weight(self, update_step):
        return_value_weight = 1.0
        end_step = min(update_step + self.step_no + 1, self.final_step - 1)

        for i in range(update_step + 1, end_step + 1):
            state_i = self.states[self._access_index(i)]
            action_i = self.actions[self._access_index(i)]
            pi_prob = self.greedy_policy(state_i, available_actions(state_i))[action_i]
            mu_prob = self.epsilon_greedy_policy(state_i, available_actions(state_i))[
                action_i
            ]
            if mu_prob == 0:
                return 0
            return_value_weight *= pi_prob / mu_prob

        return return_value_weight

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    @staticmethod
    def _select_action(actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        i = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[i]

    def epsilon_greedy_policy(
        self, state: State, actions: list[Action]
    ) -> dict[Action, float]:
        probabilities = np.ones(len(actions)) * (self.experiment_rate / len(actions))
        best_action_idx = np.argmax([self.q[state, a] for a in actions])
        probabilities[best_action_idx] += 1 - self.experiment_rate
        return {
            action: probability for action, probability in zip(actions, probabilities)
        }

    def greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        probabilities = self._greedy_probabilities(state, actions)
        return {
            action: probability for action, probability in zip(actions, probabilities)
        }

    def _greedy_probabilities(self, state: State, actions: list[Action]) -> np.ndarray:
        values = [self.q[state, action] for action in actions]
        maximal_spots = (values == np.max(values)).astype(float)
        return self._normalise(maximal_spots)

    def set_visualization_mode(self, enabled: bool):
        self.visualization_mode = enabled

    def _select_policy(self, state: State, actions: list[Action]):
        if self.visualization_mode:
            return self.greedy_policy(state, actions)
        return self.epsilon_greedy_policy(state, actions)

    @staticmethod
    def _random_probabilities(actions: list[Action]) -> np.ndarray:
        maximal_spots = np.array([1.0 for _ in actions])
        return OffPolicyNStepSarsaDriver._normalise(maximal_spots)

    @staticmethod
    def _normalise(probabilities: np.ndarray) -> np.ndarray:
        return skl_preprocessing.normalize(probabilities.reshape(1, -1), norm="l1")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_size', type=float, required=True)
    parser.add_argument('--step_no', type=int, required=True)
    parser.add_argument('--experiment_rate', type=float, default=0.05)
    parser.add_argument('--num_episodes', type=int, default=30000)
    parser.add_argument('--corner_name', type=str, default="corner_c")
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiment
    experiment = Experiment(
        environment=Environment(
            corner=Corner(name=args.corner_name),
            steering_fail_chance=0.01
        ),
        driver=OffPolicyNStepSarsaDriver(
            step_size=args.step_size,
            step_no=args.step_no,
            experiment_rate=args.experiment_rate,
            discount_factor=1.00
        ),
        number_of_episodes=args.num_episodes
    )
    experiment.run()

    # Save results
    results = {
        'params': vars(args),
        'penalties': experiment.penalties,
        'mean_error': np.mean(-np.array(experiment.penalties))
    }
    
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()

