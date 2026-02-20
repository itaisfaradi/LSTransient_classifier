import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field

@dataclass
class TrainingHistory:
    """Per-epoch training metrics recorded during a training run.

    Attributes:
        epochs:     Epoch indices (1-based).
        train_loss: Weighted BCE loss on the training set.
        val_loss:   Weighted BCE loss on the validation set.
        acc:        Raw accuracy on the validation set.
        bal_acc:    Balanced accuracy = 0.5 * (TPR + TNR).
        tp, fp, tn, fn: Confusion matrix counts on the validation set.

    Properties:
        tpr: True positive rate (sensitivity) per epoch.
        tnr: True negative rate (specificity) per epoch.
    """

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
    """Evaluate the model on a data loader without updating weights.

    Args:
        model:      The model to evaluate.
        loader:     DataLoader yielding (X, y) batches.
        device:     Device to run evaluation on.
        pos_weight: Optional class-imbalance weight for the positive class.
        threshold:  Decision threshold for converting probabilities to labels.

    Returns:
        avg_loss:   Mean weighted BCE loss over all samples.
        avg_loss_uw: Mean unweighted BCE loss over all samples.
        acc:        Fraction of correctly classified samples.
        (tp, fp, tn, fn): Confusion matrix counts.
    """

    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
        )
    total_loss = 0.0
    total = 0
    correct = 0
    total_uw = 0.0
    tp = fp = tn = fn = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logit = model(X)
        loss = loss_fn(logit, y)

        loss_uw = F.binary_cross_entropy_with_logits(logit, y, reduction="mean")

        prob = torch.sigmoid(logit)
        pred = (prob >= threshold).float()

        total_loss += loss.item() * X.size(0)
        total_uw += loss_uw.item() * X.size(0)
        correct += (pred == y).sum().item()
        total += X.size(0)

        tp += ((pred == 1) & (y == 1)).sum().item()
        fp += ((pred == 1) & (y == 0)).sum().item()
        tn += ((pred == 0) & (y == 0)).sum().item()
        fn += ((pred == 0) & (y == 1)).sum().item()

    acc = correct / max(total, 1)
    avg = total_loss / max(total, 1)
    avg_uw = total_uw / max(total, 1)
    return avg, avg_uw, acc, (tp, fp, tn, fn)

def train(model, train_loader, val_loader, device, pos_weight=None, epochs=30, lr=1e-3, weight_decay=1e-4, verbose=True):
    """Train the model and return the best checkpoint by balanced accuracy.

    Uses AdamW with gradient clipping. After each epoch, evaluates on the
    validation set and retains the state dict with the highest balanced accuracy.

    Args:
        model:        The model to train (moved to device in-place).
        train_loader: DataLoader for training data.
        val_loader:   DataLoader for validation data.
        device:       Device to run training on.
        pos_weight:   Optional class-imbalance weight for the positive class.
        epochs:       Number of training epochs.
        lr:           Learning rate for AdamW.
        weight_decay: L2 regularisation for AdamW.
        verbose: If True, print metrics every 10 epochs (default True).

    Returns:
        model:   Model loaded with the best-performing state dict.
        history: TrainingHistory containing per-epoch metrics.
    """

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
        )

    best_val_metric = float("inf")
    best_state = None

    buf = {k: [] for k in ("train_loss", "val_loss", "acc", "bal_acc", "tp", "fp", "tn", "fn")}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            logit = model(X)
            loss = loss_fn(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item() * X.size(0)
            n += X.size(0)

        train_loss = running / max(n, 1)

        val_loss, val_loss_uw, val_acc, (tp, fp, tn, fn) = evaluate(
                                                        model, 
                                                        val_loader, 
                                                        device, 
                                                        pos_weight=pos_weight, 
                                                        threshold=0.5
                                                        )
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (tpr + tnr)

        buf["train_loss"].append(train_loss)
        buf["val_loss"].append(val_loss)
        buf["acc"].append(val_acc)
        buf["bal_acc"].append(bal_acc)
        buf["tp"].append(tp); buf["fp"].append(fp)
        buf["tn"].append(tn); buf["fn"].append(fn)
        
        if verbose:
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                    f"| acc={val_acc:.3f} | bal_acc={bal_acc:.3f} | TP={tp} FP={fp} TN={tn} FN={fn} " 
                    f"| val_logloss={val_loss_uw:.3f}"
                )
        # save best model based on balanced accuracy
        if val_loss < best_val_metric:
            best_val_metric = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    # load best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    history = TrainingHistory(
        epochs=np.arange(1, epochs + 1),
        **{k: np.array(v) for k, v in buf.items()},
    )
    return model, history