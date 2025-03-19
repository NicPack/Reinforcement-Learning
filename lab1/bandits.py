import random
from abc import abstractmethod
from itertools import accumulate
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from scipy.stats import beta


class KArmedBandit(Protocol):
    @abstractmethod
    def arms(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: str) -> float:
        raise NotImplementedError


class BanditLearner(Protocol):
    name: str
    color: str

    @abstractmethod
    def reset(self, arms: list[str], time_steps: int):
        raise NotImplementedError

    @abstractmethod
    def pick_arm(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass


class BanditProblem:
    def __init__(self, time_steps: int, bandit: KArmedBandit, learner: BanditLearner):
        self.time_steps: int = time_steps
        self.bandit: KArmedBandit = bandit
        self.learner: BanditLearner = learner
        self.learner.reset(self.bandit.arms(), self.time_steps)

    def run(self) -> list[float]:
        rewards = []
        for _ in range(self.time_steps):
            arm = self.learner.pick_arm()
            reward = self.bandit.reward(arm)
            self.learner.acknowledge_reward(arm, reward)
            rewards.append(reward)
        return rewards


POTENTIAL_HITS = {
    "In Praise of Dreams": 0.8,
    "We Built This City": 0.9,
    "April Showers": 0.5,
    "Twenty Four Hours": 0.3,
    "Dirge for November": 0.1,
}


class TopHitBandit(KArmedBandit):
    def __init__(self, potential_hits: dict[str, float]):
        self.potential_hits: dict[str, float] = potential_hits

    def arms(self) -> list[str]:
        return list(self.potential_hits)

    def reward(self, arm: str) -> float:
        thumb_up_probability = self.potential_hits[arm]
        return 1.0 if random.random() <= thumb_up_probability else 0.0


class RandomLearner(BanditLearner):
    def __init__(self):
        self.name = "Random"
        self.color = "black"
        self.arms: list[str] = []

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms

    def pick_arm(self) -> str:
        return random.choice(self.arms)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass


class ExploreThenCommitLearner(BanditLearner):
    def __init__(self, m: int = 10):
        self.name = "ETC"
        self.color = "blue"
        self.arms: list[str] = []
        self.time_step = 0
        self.k_arms: int = None
        self.m = m
        self.arms_stats: dict[str] = {}

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.k_arms = len(arms)
        self.arms_stats = {arm: 0 for arm in self.arms}
        self.time_step = 0

    def pick_arm(self) -> str:
        if self.time_step < self.m * self.k_arms:
            self.time_step += 1
            return self.arms[(self.time_step % self.k_arms)]
        else:
            return max(self.arms_stats, key=self.arms_stats.get)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.arms_stats[arm] += reward


class UpperConfidenceBoundLearner(BanditLearner):
    def __init__(self, c: int = 2):
        self.name = "UCB1"
        self.color = "red"
        self.arms: list[str] = []
        self.time_step = 0
        self.k_arms: int = None
        self.c = c
        self.arms_pulls: dict[str] = {}
        self.arms_rewards: dict[str] = {}
        self.arms_means: dict[str] = {}
        self.arms_ubc: dict[str] = {}

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.k_arms = len(arms)
        self.arms_pulls = {arm: 0 for arm in self.arms}
        self.arms_rewards = {arm: 0 for arm in self.arms}
        self.arms_means = {arm: 0 for arm in self.arms}
        self.arms_ubc = {arm: 0 for arm in self.arms}
        self.time_step = 0

    def pick_arm(self) -> str:
        self.time_step += 1
        if self.time_step <= self.k_arms:
            return self.arms[self.time_step - 1]
        else:
            self.calculate_ubc()
            return max(self.arms_ubc, key=self.arms_ubc.get)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.arms_rewards[arm] += reward
        self.arms_pulls[arm] += 1
        self.arms_means[arm] = self.arms_rewards[arm] / self.arms_pulls[arm]

    def calculate_ubc(self):
        for arm in self.arms:
            if self.arms_pulls[arm] == 0:
                self.arms_ubc[arm] = float("inf")
            else:
                self.arms_ubc[arm] = self.arms_means[arm] + (
                    self.c * np.sqrt(np.log(self.time_step) / self.arms_pulls[arm])
                )


class GreedyLearner(BanditLearner):
    def __init__(self, strategy=None, Q=None, epsilon=None):
        self.color = "purple"
        self.arms: list[str] = []
        self.time_step = 0
        self.k_arms: int = None
        self.arms_stats: dict[dict[int, int]] = {}

        if not strategy:
            self.strategy = random.choice(["Greedy", "ε-Greedy", "Optimistic-Greedy"])
        else:
            self.strategy = strategy

        # Initialize based on the chosen strategy
        if self.strategy == "Greedy":
            self.init_pure_greedy()
        elif self.strategy == "ε-Greedy":
            self.init_epsilon_greedy()
        else:
            self.init_optimistic_greedy()

    def init_pure_greedy(self):
        self.name = self.strategy

    def init_epsilon_greedy(self, epsilon=np.random.rand()):
        self.name = self.strategy
        self.epsilon = epsilon

    def init_optimistic_greedy(self, Q=random.randint(1, 11)):
        self.name = self.strategy
        self.Q = Q

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.k_arms = len(arms)
        self.arms_stats = {
            arm: {
                "n_pulls": 0,
                "mean": self.Q if self.strategy == "Optimistic-Greedy" else 0,
            }
            for arm in self.arms
        }
        self.time_step = 0

    def pick_arm(self) -> str:
        if self.time_step < self.k_arms:
            self.time_step += 1
            return self.arms[(self.time_step - 1)]
        else:
            if self.strategy != "ε-Greedy":
                return max(
                    self.arms_stats, key=lambda arm: self.arms_stats[arm]["mean"]
                )
            else:
                return np.random.choice(
                    [
                        random.choice(self.arms),
                        max(
                            self.arms_stats,
                            key=lambda arm: self.arms_stats[arm]["mean"],
                        ),
                    ],
                    p=[self.epsilon, 1 - self.epsilon],
                )

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.arms_stats[arm]["n_pulls"] += 1
        self.arms_stats[arm]["mean"] += (1 / self.arms_stats[arm]["n_pulls"]) * (
            reward - self.arms_stats[arm]["mean"]
        )


class GradientLearner(BanditLearner):
    def __init__(self, alpha: float = 0.1, random_alpha: bool = False):
        self.name = "Gradient"
        self.color = "cyan"
        self.alpha = np.random.rand() if random_alpha else alpha
        self.arms: list[str] = []
        self.time_step: int = 0
        self.reward_baseline: float = 0.0
        self.preferences: dict[str, float] = {}

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.preferences = {arm: 0.0 for arm in self.arms}
        self.time_step = 0
        self.reward_baseline = 0.0

    def pick_arm(self) -> str:
        probabilities = softmax(
            list(self.preferences.values())
        )  # Compute action probabilities
        return np.random.choice(self.arms, p=probabilities)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        """Update preferences based on the received reward"""
        self.time_step += 1
        self.reward_baseline += (1 / self.time_step) * (
            reward - self.reward_baseline
        )  # Update baseline

        probabilities = softmax(
            list(self.preferences.values())
        )  # Compute probabilities before updating

        for i, a in enumerate(self.arms):
            if a == arm:
                self.preferences[a] += (
                    self.alpha
                    * (reward - self.reward_baseline)
                    * (1 - probabilities[i])
                )
            else:
                self.preferences[a] -= (
                    self.alpha * (reward - self.reward_baseline) * probabilities[i]
                )


class ThompsonLearner(BanditLearner):
    def __init__(self):
        self.name = "Thompson"
        self.color = "olive"
        self.arms: list[str] = []
        self.time_step = 0
        self.k_arms: int = None
        self.arms_stats: dict[str] = {}

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.k_arms = len(arms)
        self.arms_stats = {arm: {"a": 1, "b": 1} for arm in self.arms}

    def pick_arm(self) -> str:
        return max(
            self.arms,
            key=lambda arm: beta.rvs(
                self.arms_stats[arm]["a"], self.arms_stats[arm]["b"]
            ),
        )

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.arms_stats[arm]["a"] += reward
        self.arms_stats[arm]["b"] += 1 - reward


TIME_STEPS = 5000
TRIALS_PER_LEARNER = 200


def evaluate_learner(learner: BanditLearner) -> None:
    runs_results = []
    for _ in range(TRIALS_PER_LEARNER):
        bandit = TopHitBandit(POTENTIAL_HITS)
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner)
        rewards = problem.run()
        accumulated_rewards = list(accumulate(rewards))
        runs_results.append(accumulated_rewards)

    runs_results = np.array(runs_results)
    mean_accumulated_rewards = np.mean(runs_results, axis=0)
    std_accumulated_rewards = np.std(runs_results, axis=0)
    plt.plot(mean_accumulated_rewards, label=learner.name, color=learner.color)
    plt.fill_between(
        range(len(mean_accumulated_rewards)),
        mean_accumulated_rewards - std_accumulated_rewards,
        mean_accumulated_rewards + std_accumulated_rewards,
        color=learner.color,
        alpha=0.2,
    )


def main():
    learners = [
        RandomLearner(),
        ExploreThenCommitLearner(),
        GreedyLearner(),
        UpperConfidenceBoundLearner(),
        GradientLearner(),
        ThompsonLearner(),
    ]
    for learner in learners:
        evaluate_learner(learner)

    plt.xlabel("Czas")
    plt.ylabel("Suma uzyskanych nagród")
    plt.xlim(0, TIME_STEPS)
    plt.ylim(0, TIME_STEPS)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
