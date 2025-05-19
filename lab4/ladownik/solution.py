from pathlib import Path
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

HIDDEN1_SIZE = 1024
HIDDEN2_SIZE = 256
SEPARATE_LAYERS = True
MODELS_PATH = Path("lab4/ladownik/models")
MODELS_PATH.mkdir(parents=True, exist_ok=True)
PLOT_PATH = Path("lab4/ladownik/plots")
PLOT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "lunarLanderV3_uplearning2.pth"
MODEL = "lab4/ladownik/models/lunarLanderV3_uplearning.pth"
plot_dir = Path(PLOT_PATH) / MODEL_NAME[:-4]
plot_dir.mkdir(parents=True, exist_ok=True)


class ActorCriticController:
    def __init__(
        self,
        environment,
        learning_rate: float,
        discount_factor: float,
        hidden1_size: int = 1024,
        hidden2_size: int = 256,
        separate_layers: bool = True,
    ) -> None:
        self.environment = environment
        self.discount_factor: float = discount_factor
        self.learning_rate: float = learning_rate
        self.hidden1_size: int = hidden1_size
        self.hidden2_size: int = hidden2_size
        self.separate_layers: bool = separate_layers
        self.model: nn.Module = self.create_actor_critic_model()
        self.optimizer: torch.optim = (
            torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate
            ) 
        )
        self.action_log_prob: Optional[torch.tensor] = (
            None  # zmienna pomocnicza, przyda się do obliczania docelowej straty
        )
        self.last_error_squared: float = 0.0  # zmienna używana do wizualizacji wyników

    def create_actor_critic_model(self) -> torch.nn.Module:
        in_features = self.environment.observation_space.shape[0]
        hidden1_size = self.hidden1_size
        hidden2_size = self.hidden2_size
        number_actions = self.environment.action_space.n

        if self.separate_layers:
            # model z oddzielnymi warstwami dla aktora i krytyka
            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.actor = nn.Sequential(
                        nn.Linear(in_features, out_features=hidden1_size),
                        nn.ReLU(),
                        nn.Linear(hidden1_size, out_features=hidden2_size),
                        nn.ReLU(),
                        nn.Linear(hidden2_size, out_features=number_actions),
                        nn.Softmax(dim=-1),
                    )
                    self.critic = nn.Sequential(
                        nn.Linear(in_features, out_features=hidden1_size),
                        nn.ReLU(),
                        nn.Linear(hidden1_size, out_features=hidden2_size),
                        nn.ReLU(),
                        nn.Linear(hidden2_size, out_features=1),
                    )

                def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    action_probs = self.actor(x)
                    value = self.critic(x)
                    return action_probs, value

        else:
            # model z jedną wspólną warstwą dla aktora i krytyka
            class Model(nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.common_layer = nn.Sequential(
                        nn.Linear(in_features, out_features=hidden1_size),
                        nn.ReLU(),
                        nn.Linear(hidden1_size, out_features=hidden2_size),
                        nn.ReLU(),
                    )
                    self.actor = nn.Sequential(
                        nn.Linear(hidden2_size, out_features=number_actions),
                        nn.Softmax(dim=-1),
                    )
                    self.critic = nn.Linear(hidden2_size, out_features=1)

                def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    features = self.common_layer(x)
                    action_probs = self.actor(features)
                    value = self.critic(features)
                    return action_probs, value

        return Model()

    def choose_action(self, state: np.ndarray) -> int:
        state = self.format_state(state)
        action_probs, _ = self.model(state)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        self.action_log_prob = m.log_prob(action)
        return action.item()

    def learn(
        self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool
    ) -> None:
        state = self.format_state(state)
        new_state = self.format_state(new_state)

        self.model.train()

        # Compute the value of the current state
        # Compute state_value with gradients (needed for critic_loss)
        _, state_value = self.model(state)
        state_value = state_value.squeeze()

        # Compute next_state_value without gradients (for stability)
        with torch.no_grad():
            if terminal:
                next_state_value = torch.tensor(0.0)
            else:
                _, next_state_value = self.model(new_state)
                next_state_value = next_state_value.squeeze()

        # Compute the advantage
        advantage = reward + self.discount_factor * next_state_value - state_value

        # Compute actor and critic losses
        actor_loss = -self.action_log_prob * advantage.detach()
        target = reward + self.discount_factor * next_state_value
        critic_loss = nn.functional.mse_loss(state_value, target)

        common_loss = actor_loss + critic_loss
        self.last_error_squared = np.mean(critic_loss.detach().numpy())

        # Update actor and critic
        # optimizer zero grad
        self.optimizer.zero_grad()

        # Loss Backward
        common_loss.backward()

        # Optimizer Step
        self.optimizer.step()

    @staticmethod
    def format_state(state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float32)
        return state.reshape(-1)


def main() -> None:
    environment = gym.make(
        # "CartPole-v1", render_mode="ansi"
        "LunarLander-v2",
        render_mode="ansi",
    ) 
    # zmień lub usuń render_mode, by przeprowadzić trening bez wizualizacji środowiska
    controller = ActorCriticController(
        environment,
        0.0000005,
        0.99,
        HIDDEN1_SIZE,
        HIDDEN2_SIZE,
        SEPARATE_LAYERS,
    )
    state_dict = torch.load(MODEL)
    controller.model.load_state_dict(state_dict)
    past_rewards = []
    past_errors = []
    for i_episode in tqdm(range(3001)):  # tu decydujemy o liczbie epizodów
        done = False
        truncated = False
        state, info = environment.reset()
        reward_sum = 0.0
        errors_history = []

        while (not done) and (not truncated):
            # environment.render()  # tą linijkę możemy wykomentować, jeżeli nie chcemy mieć wizualizacji na żywo

            action = controller.choose_action(state)
            new_state, reward, done, truncated, info = environment.step(action)
            terminal = done or truncated
            controller.learn(state, reward, new_state, terminal)
            state = new_state
            reward_sum += reward
            errors_history.append(controller.last_error_squared)

        past_rewards.append(reward_sum)
        past_errors.append(np.mean(errors_history))

        window_size = 50  # tutaj o rozmiarze okienka od średniej kroczącej
        if i_episode % 100 == 0:  # tutaj o częstotliwości zrzucania wykresów
            if len(past_rewards) >= window_size:
                fig, axs = plt.subplots(2)
                axs[0].plot(
                    [
                        np.mean(past_errors[i : i + window_size])
                        for i in range(len(past_errors) - window_size)
                    ],
                    "tab:red",
                )
                axs[0].set_title("mean squared error")
                axs[1].plot(
                    [
                        np.mean(past_rewards[i : i + window_size])
                        for i in range(len(past_rewards) - window_size)
                    ],
                    "tab:green",
                )
                axs[1].set_title("sum of rewards")
                fig.tight_layout()
            plt.savefig(plot_dir / f"learning_{i_episode}.png")
            plt.close()
            plt.clf()

    environment.close()
    torch.save(
        controller.model.state_dict(), MODELS_PATH / MODEL_NAME
    )  # tu zapisujemy model


if __name__ == "__main__":
    main()
