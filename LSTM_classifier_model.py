from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class LSTMConfig:
    input_dim: int = 6
    hidden_dim: int = 64
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.2

class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super().__init__() # initialize parent class
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim) # project input features to hidden_dim

        # define LSTM layer
        self.lstm = nn.LSTM(
            input_size = config.hidden_dim, # hidden size after input projection
            hidden_size = config.hidden_dim, # hidden size of LSTM
            num_layers = config.num_layers, # number of LSTM layers
            batch_first = True, # input/output tensors have shape (batch, seq, feature)
            bidirectional = config.bidirectional, # use bidirectional LSTM
            dropout = config.dropout if config.num_layers > 1 else 0.0 # apply dropout only if multiple layers
        )

        out_dim = config.hidden_dim * (2 if config.bidirectional else 1) # adjust output dim for bidirectionality

        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim), # fully connected layer
            # nn.LayerNorm(out_dim), # layer normalization
            nn.ReLU(), # activation
            nn.Dropout(config.dropout), # dropout for regularization
            nn.Linear(out_dim, 1)   # single logit for binary classification
        )

    def forward(self, x):
        # """
        # x: (B, L, F)
        # We'll use masks from the last two features: g_mask, r_mask
        # to create a combined time mask.
        # """
        
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