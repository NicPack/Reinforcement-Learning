import numpy as np
import pandas as pd
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

MODEL_NAME = "grpo_220_b03_lr05_mcl50_crf"


def load_prompts(file_path: str) -> Dataset:
    """
    Load questions from a CSV file and return a Dataset object.

    Args:
        file_path (str): Path to the CSV file containing questions.

    Returns:
        Dataset: A Dataset object containing the questions.
    """
    # Read each line from file
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Create a DataFrame from the lines
    df = pd.DataFrame(lines, columns=["prompt"])

    # Convert the DataFrame to a list of dictionaries
    data_list = df.to_dict(orient="records")

    # Create a Dataset from the list of dictionaries
    dataset = Dataset.from_list(data_list)

    return dataset


def normalize_rewards(rewards: list[float]) -> list[float]:
    mean = np.mean(rewards)
    std = np.std(rewards) + 1e-8  # avoid division by zero
    return [1 / (1 + np.exp(-(r - mean) / std)) for r in rewards]


dataset = load_prompts("lab5/questions.txt")


def reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Calculate the reward based on the number of correct answers.
    A simple, useful proxy for coherence and readability is average word length or punctuation presence.
    Idea: Reward completions that are concise and contain punctuation.

    Args:
        questions (Dataset): Dataset containing questions.
        answers (Dataset): Dataset containing answers.

    Returns:
        float: Reward based on the number of correct answers.
    """
    rewards = []
    for completion in completions:
        words = completion.strip().split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0.0
        has_punctuation = any(p in completion for p in [".", "?", "!"])
        length_penalty = max(
            0.0, 1 - abs(len(words) - 10) / 10
        )  # best length ~10 words

        score = 0.5 * avg_word_length + 0.5 * has_punctuation + length_penalty
        rewards.append(score)
    return normalize_rewards(rewards)


training_args = GRPOConfig(
    output_dir=f"lab5/checkpoints/{MODEL_NAME}",
    logging_steps=10,
    beta=0.3,
    learning_rate=1e-5,
    logging_dir=f"lab5/logs/{MODEL_NAME}_logs",
    gradient_accumulation_steps=1,
    max_completion_length=50,
)

trainer = GRPOTrainer(
    model="gpt2",
    args=training_args,
    reward_funcs=reward,
    train_dataset=dataset,
)


def main():
    """
    Main function to train the GRPO model.
    """
    trainer.train()

    trainer.save_model(f"lab5/models/{MODEL_NAME}")


if __name__ == "__main__":
    main()
