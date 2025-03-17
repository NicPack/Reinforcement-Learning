import matplotlib.pyplot as plt
import numpy as np
import pytest
from bandits import (
    BanditProblem,
    GradientLearner,
    GreedyLearner,
    ThompsonLearner,
    TopHitBandit,
    UpperConfidenceBoundLearner,
)


# Randomize potential hit probabilities for each test run
def generate_potential_hits():
    return {f"Song {i}": np.random.uniform(0.1, 0.9) for i in range(5)}


# Define learners with specific parameter ranges
bandit_configurations = [
    # Greedy Learners
    ("Greedy", GreedyLearner, {"strategy": "Greedy"}),
    ("ε-Greedy 0.01", GreedyLearner, {"strategy": "ε-Greedy", "epsilon": 0.01}),
    ("ε-Greedy 0.1", GreedyLearner, {"strategy": "ε-Greedy", "epsilon": 0.1}),
    ("ε-Greedy 0.2", GreedyLearner, {"strategy": "ε-Greedy", "epsilon": 0.2}),
    ("Optimistic-Greedy Q=1", GreedyLearner, {"strategy": "Optimistic-Greedy", "Q": 1}),
    ("Optimistic-Greedy Q=5", GreedyLearner, {"strategy": "Optimistic-Greedy", "Q": 5}),
    (
        "Optimistic-Greedy Q=10",
        GreedyLearner,
        {"strategy": "Optimistic-Greedy", "Q": 10},
    ),
    # Upper Confidence Bound (UCB)
    ("UCB c=0.1", UpperConfidenceBoundLearner, {"c": 0.1}),
    ("UCB c=1", UpperConfidenceBoundLearner, {"c": 1}),
    ("UCB c=2", UpperConfidenceBoundLearner, {"c": 2}),
    # Thompson Sampling
    ("Thompson Sampling", ThompsonLearner, {}),
    # Gradient Bandit
    ("Gradient α=0.01", GradientLearner, {"alpha": 0.01}),
    ("Gradient α=0.1", GradientLearner, {"alpha": 0.1}),
    ("Gradient α=0.5", GradientLearner, {"alpha": 0.5}),
]


results_storage = {}


@pytest.fixture(scope="function")
def store_results(request):
    yield
    # After test completes, check if we have all data
    if len(results_storage) == len(bandit_configurations):
        generate_final_plot()


@pytest.mark.parametrize(
    "learner_name, learner_class, param_dict", bandit_configurations
)
def test_bandit_learner(learner_name, learner_class, param_dict, store_results):
    """Test and store results for final plotting"""
    time_steps = 1000
    num_runs = 5  # Start with 5 runs for testing

    cumulative_regrets = []

    for _ in range(num_runs):
        potential_hits = generate_potential_hits()
        bandit = TopHitBandit(potential_hits)
        optimal_prob = max(potential_hits.values())

        try:
            learner = learner_class(**param_dict)
            problem = BanditProblem(time_steps, bandit, learner)
            rewards, actions = problem.run()

            # Verify actions were generated
            if not actions or len(actions) != time_steps:
                print(f"Warning: {learner_name} produced invalid actions")
                continue

            # Calculate proper regret
            arm_probs = potential_hits
            regret = [optimal_prob - arm_probs[action] for action in actions]
            cumulative_regret = np.cumsum(regret)
            cumulative_regrets.append(cumulative_regret)

        except Exception as e:
            print(f"Error in {learner_name}: {str(e)}")
            continue

    # Only store if we have valid data
    if cumulative_regrets:
        avg_regret = np.mean(cumulative_regrets, axis=0)
        results_storage[learner_name] = avg_regret
    else:
        print(f"Skipping {learner_name} - no valid data")
        results_storage[learner_name] = np.array([])


def generate_final_plot():
    """Generate and save the final comparison plot"""
    plt.figure(figsize=(12, 6))

    # Add debug prints to verify data
    print("\n=== Plot Data Verification ===")
    for learner_name, regret_curve in results_storage.items():
        if len(regret_curve) == 0:
            print(f"⚠️ Warning: {learner_name} has an empty regret curve!")
        print(f"{learner_name}:")
        print(f"  Data points: {len(regret_curve)}")
        print(f"  Min regret: {np.min(regret_curve):.2f}")
        print(f"  Max regret: {np.max(regret_curve):.2f}")
        print(f"  First 5 values: {regret_curve[:5]}")

        print(f"Plotting {learner_name}: {len(regret_curve)} points")
        # Explicit line styling
        plt.plot(
            regret_curve,
            label=learner_name,
            linewidth=2,
            alpha=0.7,
            marker="",  # Explicitly disable markers
        )

    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title("Bandit Algorithm Performance Comparison", fontsize=14)

    # Force legend even if empty
    handles, labels = plt.gca().get_legend_handles_labels()
    if not handles:
        plt.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=16)
    else:
        plt.legend(
            handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
        )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save in multiple formats for debugging
    plt.savefig("bandit_performance.png", bbox_inches="tight")
    plt.savefig(
        "bandit_performance.svg", bbox_inches="tight"
    )  # Vector format for inspection
    print("\n--- Saved plots to bandit_performance.png and .svg ---")
