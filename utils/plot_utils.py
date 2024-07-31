import matplotlib.pyplot as plt
from utils import SHORT_SIGNALS
import numpy as np

# avg_percentages is N X (K+1) 2d dataframe, where N is the number of states
# and K is the number of vaccine hesitancy reasons (usually K=15)
def plot_bar_nstates(avg_ps, states, start_day, end_day, idx):
    fig, axes = plt.subplots(avg_ps.shape[0])
    for i in range(avg_ps.shape[0]):
        axes[i].barh(SHORT_SIGNALS[idx], avg_ps.iloc[i][idx])
        axes[i].set_title(f"Vaccine Hesistancy in {states[i]}")
        # axes[i].set_xlabel(f"Average Percentage ({interval}) days")
    fig.tight_layout()
    plt.savefig(f"fig/{states}-from-{start_day}-to-{end_day}-avg_percentages.png")

# avg_percentages is N X (K+1) 2d dataframe, where N is the number of states
# and K is the number of vaccine hesitancy reasons (usually K=15), the 0th column is the state names
def plot_bar_diff_nstates(avg_ps, states, start_day, end_day, idx, vars):
    fig, ax = plt.subplots(1)
    ax.barh(SHORT_SIGNALS[idx], vars[idx])
    ax.set_title(f"Top {len(idx)} disagreement on vaccine hesitancy")
    ax.set_xlabel(f"Variance between {states}")
    fig.tight_layout()
    plt.savefig(f"fig/{states}-from-{start_day}-to-{end_day}-avg_p_var.png")
    return idx
    

    
