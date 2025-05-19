from pathlib import Path

import gymnasium as gym
import torch
from solution import ActorCriticController

MODELS_PATH = Path("lab4/ladownik/models")
MODEL_NAME = "LunarLanderV3_uplearning.pth"
MODEL = MODELS_PATH / MODEL_NAME
RENDER_MODE = "human"  # "human" or "rgb_array"
EXPERIMENT = "LunarLander-v2"  # "CartPole-v1" or "LunarLander-v2"
LEARNING_RATE = 0.000001
DISCOUNT_FACTOR = 0.99
HIDDEN1_SIZE = 1024
HIDDEN2_SIZE = 256


def evaluate():
    # Create environment
    environment = gym.make(EXPERIMENT, render_mode=RENDER_MODE)
    # Load model
    try:
        controller = ActorCriticController(
            environment, LEARNING_RATE, DISCOUNT_FACTOR, HIDDEN1_SIZE, HIDDEN2_SIZE, True
        )
        state_dict = torch.load(MODEL)
        controller.model.load_state_dict(state_dict)

    except TypeError:
        print(controller.model.state_dict())
        print(torch.load(MODEL))

    states = [[0,0.0,0,0,0,0,0,0], [1.5,2.5,0,0,0,0,0,0], [1.5,2.5,0,0,4,0,0,0]]

    for state in states:
        with torch.no_grad():
            state_tensor = controller.format_state(state)
            action, value = controller.model(state_tensor)
            print(f"State: {state}, Value: {value}")

    environment.close()


if __name__ == "__main__":
    evaluate()
