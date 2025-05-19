import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

with open("lab3/mcts_experiment_results.pkl", "rb") as f:
    results = pickle.load(f)

data = []
for (time_limit, c_coeff), scores in results.items():
    data.append(
        {
            "time_limit": time_limit,
            "c_coefficient": c_coeff,
            "win_rate": scores["blue"] / (scores["red"] + scores["blue"]),
        }
    )

df = pd.DataFrame(data)
# df["key"] = "time_limit : " + df["time_limit"].astype(str) + " | c_coefficient " + df["c_coefficient"].astype(str)
pivot = pd.pivot(
    data=df, columns=["time_limit"], index="c_coefficient", values="win_rate"
)
print(pivot.head())
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
plt.title("Win rate vs MCTS(0.5, 0.2) Agent")
plt.ylabel("time_limit")
plt.xlabel("c_coefficient")
plt.show()
