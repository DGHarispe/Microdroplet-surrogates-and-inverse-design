import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
file2 = "../../Datasets/S1/models_output_bothG.txt"
df2 = pd.read_csv(file2, sep="\t")

# Extract votes
votes = df2["Max_Votes"].dropna().astype(int)

# Count occurrences
counts = votes.value_counts().sort_index()

print("Vote distribution (counts):")
print(counts)


plt.figure(figsize=(8, 5))

plt.rcParams.update({
    'axes.labelsize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'font.size': 10,
	"text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "text.latex.preamble": (
        r"\usepackage{mathptmx}"
        r"\usepackage{amsmath}"
        r"\usepackage{amssymb}"
        r"\usepackage[mathscr]{euscript}"
    )
})

bars = plt.bar(counts.index, counts.values, edgecolor='black', color="#ffc34aff")

plt.xlabel("Number of Votes")
plt.ylabel("Number of Samples")
plt.yscale("log")
plt.title("Vote Distribution in the Dataset", fontsize=16)

plt.xticks(counts.index)
plt.grid(axis='y', linestyle='--', alpha=0.5)


for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.tight_layout()
plt.show()
