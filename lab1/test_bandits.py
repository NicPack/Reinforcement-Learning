import matplotlib.pyplot as plt
import numpy as np
import pytest
from bandits import (
    BanditProblem,
    GradientLearner,
    GreedyLearner,
    TopHitBandit,
    UpperConfidenceBoundLearner,
)


# Randomize potential hit probabilities for each test run/ Random Expected Values of Probabilities
def generate_potential_hits():
    return {f"Song {i}": np.random.uniform(0.1, 0.9) for i in range(5)}


# Define learners with specific parameter ranges/Parametric Case Study
bandit_configurations = [
    # Greedy Learners
    ("ε-Greedy", GreedyLearner, {"strategy": "ε-Greedy", "epsilon": 0.01}),
    ("ε-Greedy", GreedyLearner, {"strategy": "ε-Greedy", "epsilon": 0.05}),
    ("ε-Greedy", GreedyLearner, {"strategy": "ε-Greedy", "epsilon": 0.1}),
    ("ε-Greedy", GreedyLearner, {"strategy": "ε-Greedy", "epsilon": 0.15}),
    ("ε-Greedy", GreedyLearner, {"strategy": "ε-Greedy", "epsilon": 0.2}),
    ("ε-Greedy", GreedyLearner, {"strategy": "ε-Greedy", "epsilon": 0.25}),
    (
        "Optimistic-Greedy",
        GreedyLearner,
        {"strategy": "Optimistic-Greedy", "Q": 0.1},
    ),
    (
        "Optimistic-Greedy",
        GreedyLearner,
        {"strategy": "Optimistic-Greedy", "Q": 0.5},
    ),
    (
        "Optimistic-Greedy",
        GreedyLearner,
        {"strategy": "Optimistic-Greedy", "Q": 1},
    ),
    ("Optimistic-Greedy", GreedyLearner, {"strategy": "Optimistic-Greedy", "Q": 2}),
    ("Optimistic-Greedy", GreedyLearner, {"strategy": "Optimistic-Greedy", "Q": 4}),
    # Upper Confidence Bound (UCB)
    ("UCB", UpperConfidenceBoundLearner, {"c": 0.05}),
    ("UCB", UpperConfidenceBoundLearner, {"c": 0.1}),
    ("UCB", UpperConfidenceBoundLearner, {"c": 0.5}),
    ("UCB", UpperConfidenceBoundLearner, {"c": 1}),
    ("UCB", UpperConfidenceBoundLearner, {"c": 2}),
    ("UCB", UpperConfidenceBoundLearner, {"c": 4}),
    # Gradient Bandit
    ("Gradient", GradientLearner, {"alpha": 0.01}),
    ("Gradient", GradientLearner, {"alpha": 0.05}),
    ("Gradient", GradientLearner, {"alpha": 0.1}),
    ("Gradient", GradientLearner, {"alpha": 0.2}),
    ("Gradient", GradientLearner, {"alpha": 0.5}),
    ("Gradient", GradientLearner, {"alpha": 1}),
]

bandits = set(configuration[0] for configuration in bandit_configurations)
parameters = set(
    list(configuration[-1].values())[-1] for configuration in bandit_configurations
)

bandit_to_color = {
    "UCB": "blue",
    "Gradient": "orange",
    "ε-Greedy": "green",
    "Optimistic-Greedy": "red",
}

results_storage = {
    bandit: {"avg_reward": [], "avg_regret": [], "param": []} for bandit in bandits
}


@pytest.fixture(scope="session")
def store_results(request):
    yield
    # After test completes, check if we have all data
    generate_final_plot("reward")
    generate_final_plot("regret")


@pytest.mark.parametrize(
    "learner_name, learner_class, param_dict", bandit_configurations
)
def test_bandit_learner(learner_name, learner_class, param_dict, store_results):
    """Test and store results for final plotting"""
    time_steps = 1000
    num_runs = 200

    average_reward = 0
    average_regret = 0

    for n in range(num_runs):
        potential_hits = generate_potential_hits()
        bandit = TopHitBandit(potential_hits)
        optimal_prob = max(potential_hits.values())

        learner = learner_class(**param_dict)
        problem = BanditProblem(time_steps, bandit, learner)
        rewards = problem.run()

        rewards = np.array(rewards)
        regret = (optimal_prob * time_steps) - np.sum(rewards)

        average_reward += (1 / (n + 1)) * (np.mean(rewards) - average_reward)
        average_regret += (1 / (n + 1)) * (np.mean(regret) - average_regret)

    results_storage[learner_name]["avg_reward"].append(average_reward)
    results_storage[learner_name]["avg_regret"].append(average_regret)
    results_storage[learner_name]["param"].append(
        param_dict[list(param_dict.keys())[-1]]
    )


def generate_final_plot(measure: str = None):
    """Generate and save the final comparison plot"""

    if not (measure == "reward" or measure == "regret"):
        raise KeyError(
            "Inapropriate measure passed to the function. Must be either reward or regret"
        )

    measure = "avg_" + measure

    plt.figure(figsize=(14, 8))

    param_to_x = {}

    all_params = set()
    for learner in results_storage.keys():
        params = results_storage[learner]["param"]
        all_params.update(params)

    sorted_params = sorted(all_params)
    param_to_x = {param: i for i, param in enumerate(sorted_params)}

    for learner in sorted(results_storage.keys()):
        params = results_storage[learner]["param"]
        avg_rewards = results_storage[learner][measure]

        x_positions = [param_to_x[param] for param in params]

        plt.plot(
            x_positions,
            avg_rewards,
            label=learner,
            linewidth=2,
            alpha=0.7,
            marker="o",
            color=bandit_to_color[learner],
        )

    plt.xticks(
        range(len(sorted_params)),
        sorted_params,
        rotation=45,
        ha="right",
    )

    plt.xlabel("ε, α, C, Q", fontsize=16)
    plt.ylabel(f"Average {measure[4:].capitalize()}", fontsize=16)
    plt.title(f"Algorithm Performance vs {measure[4:].capitalize()}", fontsize=16)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Force legend even if empty
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    if measure[4:] == "regret":
        plt.savefig("lab1/regret_performance.png", bbox_inches="tight")
    else:
        plt.savefig("lab1/reward_performance.png", bbox_inches="tight")
