import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def visualize_results():
    results = defaultdict(list)
    
    # Load all results
    for json_file in glob.glob("results/**/results.json", recursive=True):
        with open(json_file) as f:
            data = json.load(f)
            step_no = data['params']['step_no']
            step_size = data['params']['step_size']
            results[step_no].append((step_size, data['mean_error']))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    for step_no in sorted(results.keys()):
        # Sort by step_size for proper plotting
        data = sorted(results[step_no], key=lambda x: x[0])
        x = [item[0] for item in data]
        y = [item[1] for item in data]
        plt.plot(x, y, 'o-', label=f'n={step_no}')
    
    plt.xlabel('Step Size (α)')
    plt.ylabel('Mean Error')
    plt.title('Performance Across Hyperparameters (Corner C)')
    plt.legend()
    plt.grid(True)
    plt.savefig('parametric_study.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    visualize_results()