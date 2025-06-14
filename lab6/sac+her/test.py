import argparse

import gymnasium as gym
import torch
from asdf.algos import SAC
from asdf.buffers import HerReplayBuffer
from asdf.extractors import DictExtractor
from asdf.policies import MlpPolicy
import panda_gym

def main(env_id: str) -> None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU")
    else:
        print("Using CPU")
        device = "cpu"

    env = gym.make(env_id)

    buffer = HerReplayBuffer(
        env=env,
        size=1_000_000,
        n_sampled_goal=3,
        goal_selection_strategy="future",
        device=device,
    )

    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=[64, 64],
        extractor_type=DictExtractor,
    )
    policy.to(device)
    algo = SAC(env=env, policy=policy, buffer=buffer)
    algo.load("models/panda_reach_auto_a.pth")

    policy.cpu()
    env = gym.make(env_id, render_mode="human")

    test_results = algo.test(env, n_episodes=10, sleep=1 / 5, render=True)
    print(f"Test results: {test_results}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="PandaReach-v3", help="Gym environment ID"
    )

    args = parser.parse_args()

    main(args.env)
