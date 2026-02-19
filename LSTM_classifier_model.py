from dataclasses import dataclass
import torch.nn as nn

@dataclass
class LSTMConfig:
    input_dim: int = 6
    hidden_dim: int = 64
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.2

class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for irregularly sampled time-series.

    Expects input of shape (B, L, F) where the last two features are binary
    coverage masks for the g and r photometric bands. Uses masked mean pooling
    over valid time steps before classification.

    Output is a single logit per sample (use BCEWithLogitsLoss for training).
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        self.lstm = nn.LSTM(
            input_size = config.hidden_dim,
            hidden_size = config.hidden_dim,
            num_layers = config.num_layers,
            batch_first = True, # input/output tensors have shape (batch, seq, feature)
            bidirectional = config.bidirectional,
            dropout = config.dropout if config.num_layers > 1 else 0.0
        )

        out_dim = config.hidden_dim * (2 if config.bidirectional else 1)

        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(out_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, F) — batch, sequence length, features.
                x[..., -2] and x[..., -1] must be binary g/r band coverage masks.

        Returns:
            logit: Tensor of shape (B,) — one raw logit per sample.
        """
        
        # combined mask: if either band has coverage at that time step
        g_mask = x[..., -2] # (B, L)
        r_mask = x[..., -1] # (B, L)
        time_mask = ((g_mask + r_mask) > 0).float()  # (B, L)

        h = self.input_proj(x)        # (B, L, H)
        h, _ = self.lstm(h)           # (B, L, H*dir)

        # masked mean pooling
        w = time_mask.unsqueeze(-1)   # (B, L, 1)
        h_sum = (h * w).sum(dim=1)    # (B, H*dir)
        denom = w.sum(dim=1).clamp(min=1.0)  # (B, 1)
        h_pool = h_sum / denom # (B, H*dir)

        logit = self.head(h_pool).squeeze(-1)  # (B,)
        return logit