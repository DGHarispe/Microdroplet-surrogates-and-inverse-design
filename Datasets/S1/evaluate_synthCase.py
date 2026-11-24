import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd


input_file = "./synthetic_cases_1.txt"
output_file = "./models_output_bothG.txt"

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, 3)
        )

    def forward(self, x):
        return self.net(x)

def predict_reg(
    params,
    return_prob=False
):
    # ---------------- Prepare input ----------------
    params = np.asarray(params, dtype=np.float32, copy=True)
    
    params[4] = np.log(params[4] + 1e-12)
    params[5] = np.log(params[5] + 1e-12)

    if params.shape != (7,):
        raise ValueError(f"Expected parameter vector of shape (7,), got {params.shape}")

    # Add batch dimension (1, 7)
    X_norm = (params - X_mean_reg) / X_std_reg
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)

    # ---------------- Predict ----------------
    with torch.no_grad():
        logits = model_reg(X_tensor)          # shape (1, 3)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()

    if return_prob:
        return pred_class, probs.squeeze(0).cpu().numpy()
    else:
        return pred_class

def class_decision(pred_classes, pred_probs):
    pred_classes = np.asarray(pred_classes)
    pred_probs = np.asarray(pred_probs)

    n_models, n_cases = pred_classes.shape
    n_classes = pred_probs.shape[-1]

    consensus = np.zeros(n_cases, dtype=int)
    max_votes = np.zeros(n_cases, dtype=int)

    for i in range(n_cases):
        # ---- 1. vote counts ----
        counts = np.bincount(pred_classes[:, i], minlength=n_classes)
        max_count = np.max(counts)

        max_votes[i] = max_count  # store max vote count

        tied_classes = np.where(counts == max_count)[0]

        # ---- 2. no tie ----
        if len(tied_classes) == 1:
            consensus[i] = tied_classes[0]
            continue

        # ---- 3. tie-break via probability accumulation ----
        class_scores = np.zeros(n_classes)

        for m in range(n_models):
            class_scores += pred_probs[m, i]

        consensus[i] = tied_classes[np.argmax(class_scores[tied_classes])]

    return consensus, max_votes

device = None
model_reg = None
X_mean_reg = None
X_std_reg = None

def init_model(folder, index):
    global device
    global model_reg
    global X_mean_reg
    global X_std_reg
    
    device = torch.device("cpu")

    # --- Load models ---
    model_reg = SimpleDNN().to(device)
    model_reg.load_state_dict(torch.load(os.path.join(folder, f"best_model_{index}.pth"), map_location=device))
    model_reg.eval()

    # --- Load normalization ---
    X_mean_reg = np.load(os.path.join(folder, f"X_mean_{index}.npy"))
    X_std_reg = np.load(os.path.join(folder, f"X_std_{index}.npy"))

# -----------------------
base_dirs = ["../../Ensembles/Er1"]
folders = []

for bd in base_dirs:
    if os.path.exists(bd):
        group_folders = sorted([os.path.join(bd, f) for f in os.listdir(bd) if f.isdigit()])
        folders.extend(group_folders)
# -----------------------

#data = np.loadtxt("synthetic_cases_isoD_2v_new.txt", skiprows=1)
data = np.loadtxt(input_file, skiprows=1)

case_ids = data[:, 0]
X = data[:, 1:]   # shape (N, 7)

out_file = output_file

all_outputs = []
all_probs = []

for folder in folders:
    model_id = os.path.basename(folder)
    init_model(folder, model_id)
    outputs = []
    probs_cases = []
    
    for params in X:
        y, probs = predict_reg(params, return_prob=True)
        outputs.append(y)   #outputs per case
        probs_cases.append(probs)
    
    all_outputs.append(np.array(outputs))   #all model outputs
    all_probs.append(np.array(probs_cases))

outputs_array = np.array(all_outputs)   # shape (n_models, n_cases)
outProbs_array = np.array(all_probs)

consensus_classes, max_votes = class_decision(outputs_array, outProbs_array)

variance_per_case = np.var(outputs_array, axis=0)
std_per_case = np.std(outputs_array, axis=0)
mean_per_case = np.mean(outputs_array, axis=0)

output_dict = {
    "Case": case_ids
}

# -----------------------
for folder, outputs in zip(folders, all_outputs):
    group_name = os.path.basename(os.path.dirname(folder)) # e.g. "grupoA"
    model_id = os.path.basename(folder)   # e.g. "1", "2", ...
    col_name = f"NN_{group_name}_{model_id}"
    output_dict[col_name] = outputs
# -----------------------

df_outputs = pd.DataFrame(output_dict)
df_outputs["Case"] = df_outputs["Case"].astype(int)

df_outputs["Mean"] = mean_per_case
df_outputs["Std"] = std_per_case
df_outputs["Variance"] = variance_per_case
df_outputs["Consensus_Class"] = consensus_classes
df_outputs["Max_Votes"] = max_votes

cols_to_round = ["Mean", "Std", "Variance"]

df_outputs[cols_to_round] = df_outputs[cols_to_round].round(3)

df_outputs.to_csv(out_file, index=False, sep="\t")

print(f"Saved results to {out_file}")
