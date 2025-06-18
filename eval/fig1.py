import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Models and embedding methods
models = ["Qwen2-audio", "Qwen-audio", "Salmonn-audio", "MusiLingo", "LTU", "LTU-AS"]
embeddings = ["Accurate RPC/PR", "distilbert-base", "bge-large-en-v1.5", "gte-Qwen2-7B-instruct"]
tasks = ["MTG-Emotion", "MTG-Instrument"]

# Data dictionary:
# For each task and each embedding, we have a list of tuples corresponding to models.
# Each tuple is (first_value, second_value) and we will interpret these as two numbers,
# drawing a line from min(first_value, second_value) to max(first_value, second_value).
data = {
    "MTG-Emotion": {
        "Accurate RPC/PR": [(55.83, 6.27), (55.70, 4.84), (50.22, 3.25), (50.36, 3.28), (50.15, 3.27), (50.55, 3.36)],
        "distilbert-base":            [(50.37, 3.92), (50.76, 3.91), (48.76, 3.65), (50.02, 3.45), (49.67, 3.59), (49.96, 3.49)],
        "bge-large-en-v1.5":             [(60.89, 7.85), (59.06, 6.09), (50.69, 3.65), (53.07, 3.95), (51.41, 3.98), (52.02, 3.72)],
        "gte-Qwen2-7B-instruct":        [(64.40, 9.65), (65.48, 8.23), (51.01, 3.83), (55.54, 4.12), (51.43, 4.24), (53.14, 4.31)]
    },
    "MTG-Instrument": {
        "Accurate RPC/PR": [(54.31, 8.94), (54.36, 8.26), (50.21, 6.32), (51.93, 7.07), (52.88, 7.70), (50.94, 6.75)],
        "distilbert-base":            [(49.67, 8.35), (53.08, 8.38), (50.06, 7.21), (52.38, 7.65), (52.66, 8.57), (50.09, 7.55)],
        "bge-large-en-v1.5":             [(58.90, 12.41), (56.95, 11.35), (48.78, 7.44), (55.63, 9.24), (55.34, 10.98), (53.02, 8.90)],
        "gte-Qwen2-7B-instruct":        [(58.73, 12.62), (61.45, 12.54), (51.37, 7.84), (56.65, 9.33), (55.66, 11.34), (52.72, 8.95)]
    }
}

def plot_task_rectangles(task_name, task_data):
    # Create one subplot per embedding method.
    fig, axes = plt.subplots(1, len(embeddings), figsize=(20, 6), sharey=True)
    fig.suptitle(task_name, fontsize=18, fontweight='bold')
    
    # X positions for the models
    x = np.arange(len(models))
    rect_width = 0.4  # width of the rectangle
    
    for ax, embed in zip(axes, embeddings):
        values = task_data[embed]
        # Calculate lower and upper bounds for each tuple
        lowers = [min(v[0], v[1]) for v in values]
        uppers = [max(v[0], v[1]) for v in values]
        
        # Plot rectangle for each model
        for i, (low, up) in enumerate(zip(lowers, uppers)):
            height = up - low
            # Center the rectangle at the x position
            rect = Rectangle((i - rect_width/2, low), rect_width, height,
                             edgecolor='black', facecolor='skyblue', alpha=0.6)
            ax.add_patch(rect)
            # Optionally mark the endpoints
            ax.plot(i, low, 'o', color='blue', markersize=6, label='PR-AUC' if i == 0 else "")
            ax.plot(i, up, 'o', color='red', markersize=6, label='ROC-AUC' if i == 0 else "")
        
        ax.set_title(embed, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{task_name}.pdf", dpi=300)

# Plot for each task
for task in tasks:
    plot_task_rectangles(task, data[task])