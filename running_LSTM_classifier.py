import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import torch
from training import train
from LSTM_classifier_model import LSTMClassifier, LSTMConfig
from creating_dataset import TwoBandLC
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import random
import os

list_of_dicts = pkl.load(open("all_objects_dicts_prog1.pkl", "rb")) # Load preprocessed dataset
list_of_dicts = [obj for obj in list_of_dicts if "label" in obj] # Filter out objects without labels
labels = np.array([obj["label"] for obj in list_of_dicts], dtype=np.int64) # Extract labels
n_pos = int((labels == 1).sum()) # Count positive samples (TDEs)
n_neg = int((labels == 0).sum()) # Count negative samples (Ia SNe)

pos_weight_value = (n_neg / max(n_pos, 1)) # Compute positive class weight to handle class imbalance
print("n_pos (TDE):", n_pos, "n_neg (Ia):", n_neg, "pos_weight:", pos_weight_value) # Print class distribution info

pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32) # Create tensor for positive class weight


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dataset = TwoBandLC(list_of_dicts, T=300.0, L=300, use_err=True, normalize_per_object=True, up_to_peak=False) # Create dataset instance

# split 80/20

# n_total = len(dataset)
# n_val = max(1, int(0.3 * n_total))
# n_train = n_total - n_val
# train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
# --- extract labels ---

y_all = np.array([dataset[i][1].item() for i in range(len(dataset))])

# --- stratified split ---
splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.3,
    random_state=seed
)

train_idx, val_idx = next(splitter.split(np.zeros(len(y_all)), y_all))

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

print("Train positives:", y_all[train_idx].sum(),
      "Val positives:", y_all[val_idx].sum())

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, drop_last=False)

device = "cuda" if torch.cuda.is_available() else "mps"
model = LSTMClassifier(LSTMConfig())

n_eposchs = 600
lr = 1e-4
model, history = \
train(model, train_loader, val_loader, device, pos_weight=pos_weight, epochs=n_eposchs, lr=lr, weight_decay=lr/10)

# folder_name = "output_models/up_to_peak_lc/seed{}".format(seed)
# if not os.path.exists(folder_name):
#     os.makedirs(folder_name)
# with open(folder_name + "/model_evals.pkl", "wb") as f:
#     pkl.dump(
#         {"epoch": out_epoch,
#           "tp": out_tp,
#           "fp": out_fp,
#           "tn": out_tn,
#           "fn": out_fn,
#           "acc": out_acc,
#           "bal_acc": out_bal_acc,
#           "train_loss": train_losses,
#           "val_loss": val_losses},
#         f
#     )

# with open(folder_name + "/data_loaders.pkl", "wb") as f:
#     pkl.dump(
#         {
#           "train": train_loader,
#           "val": val_loader},
#         f
#     )

# torch.save(model.state_dict(), folder_name + "/model.pth")