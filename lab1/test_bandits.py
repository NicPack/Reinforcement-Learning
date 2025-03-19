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
    ("Optimistic-Greedy", GreedyLearner, {"strategy": "Optimistic-Greedy", "Q": 5}),
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

results_storage = {bandit: {"avg_reward": [], "param": []} for bandit in bandits}


@pytest.fixture(scope="session")
def store_results(request):
    yield
    # After test completes, check if we have all data
    generate_final_plot()


@pytest.mark.parametrize(
    "learner_name, learner_class, param_dict", bandit_configurations
)
def test_bandit_learner(learner_name, learner_class, param_dict, store_results):
    """Test and store results for final plotting"""
    time_steps = 1000
    num_runs = 50

    average_reward = 0

    for n in range(num_runs):
        potential_hits = generate_potential_hits()
        bandit = TopHitBandit(potential_hits)
        optimal_prob = max(potential_hits.values())

        learner = learner_class(**param_dict)
        problem = BanditProblem(time_steps, bandit, learner)
        rewards = problem.run()
        rewards = np.array(rewards)
        average_reward += (1 / (n + 1)) * (np.mean(rewards) - average_reward)

    results_storage[learner_name]["avg_reward"].append(average_reward)
    results_storage[learner_name]["param"].append(
        param_dict[list(param_dict.keys())[-1]]
    )


def generate_final_plot():
    """Generate and save the final comparison plot"""
    plt.figure(figsize=(12, 6))

    print("\n=== Plot Data Verification ===")
    for learner in results_storage.keys():
        params = results_storage[learner]["param"]
        avg_rewards = results_storage[learner]["avg_reward"]

        x_indices = range(len(params))
        plt.plot(
            x_indices,
            avg_rewards,
            label=learner,
            linewidth=2,
            alpha=0.7,
            marker="o",
        )

        plt.xticks(x_indices, params, rotation=45, ha="right")

    plt.xlabel("ε, α, C, Q", fontsize=12)
    plt.ylabel("Average Reward over first 1000 steps", fontsize=12)
    plt.title("Bandit Algorithm Performance Comparison", fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Force legend even if empty
    handles, labels = plt.gca().get_legend_handles_labels()
    if not handles:
        plt.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=16)
    else:
        plt.legend(
            handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
        )

    # Save the plot
    plt.savefig("lab1/bandit_performance.png", bbox_inches="tight")
    print("\n--- Saved plots to bandit_performance.png ---")
