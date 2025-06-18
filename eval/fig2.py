import matplotlib.pyplot as plt
import numpy as np
from seaborn import set_style
from matplotlib import font_manager 
 
# download the font files and save in this fold
FONT_PATH = "/import/c4dm-04/siyoul/CMI-bench/eval/segoe-ui-this/segoeuithis.ttf"
font_manager.fontManager.addfont(FONT_PATH)
font_props=font_manager.FontProperties(fname=FONT_PATH)
plt.rcParams['font.family']=font_props.get_name()
plt.rcParams['mathtext.fontset'] = 'cm'  # 'cm' (Computer Modern)
#plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 12})
# Set the style for the plots
set_style("whitegrid")
# Define models, embedding methods, and tasks
models = ["Qwen2-audio", "Qwen-audio", "Salmonn-audio", "MusiLingo", "LTU", "LTU-AS"]
embeddings = ["RPC/PR", "dBERT", "BGE", "GTE"]
tasks = ["MTG-Emotion", "MTG-Instrument"]

# Data: each tuple contains two numbers which we treat as the two boundaries.
data = {
    "MTG-Emotion": {
        "RPC/PR": [(55.83, 6.27), (55.70, 4.84), (50.22, 3.25), (50.36, 3.28), (50.15, 3.27), (50.55, 3.36)],
        "dBERT":            [(50.37, 3.92), (50.76, 3.91), (48.76, 3.65), (50.02, 3.45), (49.67, 3.59), (49.96, 3.49)],
        "BGE":             [(60.89, 7.85), (59.06, 6.09), (50.69, 3.65), (53.07, 3.95), (51.41, 3.98), (52.02, 3.72)],
        "GTE":        [(64.40, 9.65), (65.48, 8.23), (51.01, 3.83), (55.54, 4.12), (51.43, 4.24), (53.14, 4.31)]
    },
    "MTG-Instrument": {
        "RPC/PR": [(54.31, 8.94), (54.36, 8.26), (50.21, 6.32), (51.93, 7.07), (52.88, 7.70), (50.94, 6.75)],
        "dBERT":            [(49.67, 8.35), (53.08, 8.38), (50.06, 7.21), (52.38, 7.65), (52.66, 8.57), (50.09, 7.55)],
        "BGE":             [(58.90, 12.41), (56.95, 11.35), (48.78, 7.44), (55.63, 9.24), (55.34, 10.98), (53.02, 8.90)],
        "GTE":        [(58.73, 12.62), (61.45, 12.54), (51.37, 7.84), (56.65, 9.33), (55.66, 11.34), (52.72, 8.95)]
    }
}

# Define colors for each embedding method
colors = {
    "RPC/PR": "#c5211c",
    "dBERT": "#e4443f",
    "BGE": "#f59b65",
    "GTE": "#f7ba79"
}
def plot_rectangles_for_task(task_name, task_data):
    # Create a 2x3 grid (6 subplots, one per model)
    fig, axes = plt.subplots(1, 6, figsize=(32, 5))
    axes = axes.flatten()
    for i, model in enumerate(models):
        ax = axes[i]
        # For each model, collect y-limits from all embeddings for proper scaling.
        model_lows = []
        model_highs = []
        # For each embedding method, plot a rectangle
        for j, embed in enumerate(embeddings):
            # Get the two values for this model and embedding
            val1, val2 = task_data[embed][i]
            low = min(val1, val2)
            high = max(val1, val2)
            model_lows.append(low)
            model_highs.append(high)
            # Position on x-axis for the rectangle (j as center; width=0.6)
            rect_x = j - 0.3
            rect_width = 0.3
            rect_height = high - low
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.add_patch(plt.Rectangle((rect_x, low), rect_width, rect_height,
                                       edgecolor=colors[embed], facecolor=colors[embed], alpha=1))
            ax.plot(j-0.15, low, 'o', color='black', markersize=0, label='PR-AUC' if i == 0 else "")
            ax.plot(j-0.15, high, 'o', color='black', markersize=0, label='ROC-AUC' if i == 0 else "")
            # disable x grid
            ax.xaxis.grid(False)
            
            # Annotate the rectangle with its range
            # ax.text(j, (low+high)/2, f"{low:.2f}-{high:.2f}",
            #         ha="center", va="center", fontsize=8, color="black")
        
        # Configure x-axis with embedding labels
        ax.set_xticks(range(len(embeddings)))
        ax.set_xticklabels(embeddings, rotation=45, ha='right', fontsize=18,fontproperties=font_props)
        ax.set_title(model, fontsize=22,fontproperties=font_props)
        # Set y-axis limits with a small margin
        y_min = min(model_lows)
        y_max = max(model_highs)
        y_margin = (y_max - y_min) * 0.1 if (y_max-y_min)!=0 else 1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Overall title and legend for the entire figure
    fig.suptitle(task_name, fontsize=18, fontproperties=font_props)
    # Create custom legend handles from the colors dictionary
    # legend_handles = [plt.Rectangle((0,0),1,1, color=colors[embed], alpha=0.7) for embed in embeddings]
    # fig.legend(legend_handles, embeddings, loc='upper right', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{task_name}-2.pdf", dpi=300)

# Plot for each task separately
for task in tasks:
    plot_rectangles_for_task(task, data[task])