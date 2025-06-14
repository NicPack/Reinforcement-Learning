import os

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "grpo_220_b03_lr05_mcl50_crf"


def generate_answer(prompt: str, model_name: str = MODEL_NAME):
    if model_name not in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        tokenizer = AutoTokenizer.from_pretrained(f"lab5/models/{model_name}")
        model = AutoModelForCausalLM.from_pretrained(f"lab5/models/{model_name}").to(
            "mps"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(f"{model_name}").to("mps")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to("mps")

    outputs = model.generate(**inputs)

    if not os.path.exists(f"lab5/answers/{model_name}_answers.txt"):
        open(f"lab5/answers/{model_name}_answers.txt", "w").close()

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output = output.replace(prompt, "").replace('"', "").replace("\n", " ").strip()

    with open(f"lab5/answers/{model_name}_answers.txt", "a") as f:
        f.write(f"{prompt.strip()}\n{output}\n\n")


prompts_list = [
    "What does a cow say?",
    "What's the weather like today?",
    "How do you feel like?",
    "What's your mom's name?",
    "What does the sun look like?",
    "Why is the pillow always tired?",
    "Can a snail win a race?",
    "Why does toast scream inside?",
    "Is the pen mightier than a duck?",
]

for prompt in prompts_list:
    generate_answer(prompt, MODEL_NAME)
