import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random
import numpy as np
from dataclasses import dataclass, field

@dataclass
class TrainingHistory:
    epochs:      np.ndarray = field(default_factory=lambda: np.array([]))
    train_loss:  np.ndarray = field(default_factory=lambda: np.array([]))
    val_loss:    np.ndarray = field(default_factory=lambda: np.array([]))
    acc:         np.ndarray = field(default_factory=lambda: np.array([]))
    bal_acc:     np.ndarray = field(default_factory=lambda: np.array([]))
    tp:          np.ndarray = field(default_factory=lambda: np.array([]))
    fp:          np.ndarray = field(default_factory=lambda: np.array([]))
    tn:          np.ndarray = field(default_factory=lambda: np.array([]))
    fn:          np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def tpr(self):
        return self.tp / np.maximum(self.tp + self.fn, 1)

    @property
    def tnr(self):
        return self.tn / np.maximum(self.tn + self.fp, 1)

@torch.no_grad()
def evaluate(model, loader, device, pos_weight=None, threshold=0.5):
    model.eval() # set eval mode
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
        ) # handling imbalance in the loss
    
    total_loss = 0.0 # sum of losses
    total = 0 # total samples
    correct = 0 # correct predictions
    total_uw = 0.0
    # confusion matrix components
    tp = fp = tn = fn = 0

    for X, y in loader:
        X = X.to(device) # move to device
        y = y.to(device) # move to device

        logit = model(X) # forward pass
        loss = loss_fn(logit, y) # compute loss

        loss_uw = F.binary_cross_entropy_with_logits(logit, y, reduction="mean")

        prob = torch.sigmoid(logit) # convert logits to probabilities
        pred = (prob >= threshold).float() # threshold to get binary predictions

        total_loss += loss.item() * X.size(0) # accumulate loss
        total_uw += loss_uw.item() * X.size(0)
        correct += (pred == y).sum().item() # accumulate correct predictions
        total += X.size(0) # accumulate total samples

        # update confusion matrix
        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()

    acc = correct / max(total, 1) # compute accuracy
    avg = total_loss / max(total, 1)
    avg_uw = total_uw / max(total, 1)
    return avg, avg_uw, acc, (tp, fp, tn, fn)

def train(model, train_loader, val_loader, device, pos_weight=None, epochs=30, lr=1e-3, weight_decay=1e-4):
    model.to(device) # move model to device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) # optimizer
    # warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=50)
    # cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - 50, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[50])

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None) # handling imbalance in the loss

    best_val_metric = -1.0 # best validation metric (balanced accuracy)
    best_state = None # best model state

    buf = {k: [] for k in ("train_loss", "val_loss", "acc", "bal_acc", "tp", "fp", "tn", "fn")}

    for epoch in range(1, epochs + 1):
        model.train() # set train mode
        running = 0.0 # running loss
        n = 0 # number of samples

        for X, y in train_loader:
            X = X.to(device) # move to device
            y = y.to(device) # move to device

            opt.zero_grad() # zero gradients, this is needed before backward()
            logit = model(X) # forward pass
            loss = loss_fn(logit, y) # compute loss
            loss.backward() # backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping to prevent exploding gradients, 1.0 to be safe
            opt.step() # optimizer step

            running += loss.item() * X.size(0) # accumulate loss
            n += X.size(0) # accumulate number of samples

        train_loss = running / max(n, 1) # average training loss

        # For imbalance, accuracy can be misleading; we'll print confusion matrix too
        val_loss, val_logloss, val_acc, (tp, fp, tn, fn) = evaluate(
                                                        model, 
                                                        val_loader, 
                                                        device, 
                                                        pos_weight=pos_weight, 
                                                        threshold=0.5
                                                        )
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (tpr + tnr)

        # scheduler.step()
        
        # A better single-number target than accuracy: TPR + TNR (balanced accuracy)
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (tpr + tnr)
        buf["train_loss"].append(train_loss)
        buf["val_loss"].append(val_loss)
        buf["acc"].append(val_acc)
        buf["bal_acc"].append(bal_acc)
        buf["tp"].append(tp); buf["fp"].append(fp)
        buf["tn"].append(tn); buf["fn"].append(fn)
        
        if epoch % 10 == 0 or epoch == 1:# or epoch == epochs:
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                f"| acc={val_acc:.3f} | bal_acc={bal_acc:.3f} | TP={tp} FP={fp} TN={tn} FN={fn} | val_logloss={val_logloss}"
            )
        # save best model based on balanced accuracy
        if bal_acc > best_val_metric:
            best_val_metric = bal_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # load best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    history = TrainingHistory(
        epochs=np.arange(1, epochs + 1),
        **{k: np.array(v) for k, v in buf.items()},
    )
    return model, history