import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepDTA(nn.Module):
    def __init__(self, drug_vocab_size, target_vocab_size, hidden_dim=128):
        super(DeepDTA, self).__init__()
        
        # Drug Branch
        self.drug_embed = nn.Embedding(drug_vocab_size, 128)
        self.drug_cnn = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=6),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=8),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Target Branch
        self.target_embed = nn.Embedding(target_vocab_size, 128)
        self.target_cnn = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=12),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # FC Layers
        self.fc = nn.Sequential(
            nn.Linear(96 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, drug_seq, target_seq):
        # Shape: [batch, len]
        
        # Drug pipeline
        d = self.drug_embed(drug_seq) # [batch, len, 128]
        d = d.permute(0, 2, 1) # [batch, 128, len] for Conv1d
        d = self.drug_cnn(d) # [batch, 96, 1]
        d = d.squeeze(-1) # [batch, 96]
        
        # Target pipeline
        t = self.target_embed(target_seq)
        t = t.permute(0, 2, 1)
        t = self.target_cnn(t)
        t = t.squeeze(-1)
        
        # Concat
        x = torch.cat((d, t), dim=1)
        
        # FC
        out = self.fc(x)
        return out
