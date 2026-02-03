import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from model import DeepDTA
from preprocess import drug_encoder, target_encoder
import os

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
MAX_DRUG_LEN = 100
MAX_TARGET_LEN = 1200

# Try to import TDC, fallback if fails
try:
    from tdc.multi_pred import DTI
    TDC_AVAILABLE = True
except ImportError:
    TDC_AVAILABLE = False
    print("TDC library not fully available (missing dependencies). Switching to SYNTHETIC data mode.")
except Exception:
    TDC_AVAILABLE = False
    print("TDC library import failed. Switching to SYNTHETIC data mode.")

class DTIDataset(Dataset):
    def __init__(self, df):
        self.drugs = df['Drug'].values
        self.targets = df['Target'].values
        self.y = df['Y'].values

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, idx):
        drug_seq = drug_encoder.encode(self.drugs[idx], max_len=MAX_DRUG_LEN)
        target_seq = target_encoder.encode(self.targets[idx], max_len=MAX_TARGET_LEN)
        label = float(self.y[idx])
        return torch.LongTensor(drug_seq), torch.LongTensor(target_seq), torch.tensor(label, dtype=torch.float32)

def train():
    global TDC_AVAILABLE
    if TDC_AVAILABLE:
        print("Loading BindingDB Data via TDC...")
        try:
            data = DTI(name = 'BindingDB_Kd')
            # Convert to log scale (pKd)
            data.convert_to_log(form = 'binding')
            
            split = data.get_split(method = 'random', seed = 42, frac = [0.8, 0.1, 0.1])
            train_df = split['train']
            valid_df = split['valid']
        except Exception as e:
            print(f"Error loading TDC data: {e}. Switching to SYNTHETIC data.")
            TDC_AVAILABLE = False
    
    if not TDC_AVAILABLE:
        # Synthetic Data Generation
        print("Generating synthetic data for demonstration...")
        drugs = ["CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=C(N5)C6=CC=CC=C6"] * 100
        targets = ["MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAG"] * 100
        y = np.random.rand(100) * 10 # Random pKd 0-10
        train_df = pd.DataFrame({'Drug': drugs, 'Target': targets, 'Y': y})
        valid_df = pd.DataFrame({'Drug': drugs[:20], 'Target': targets[:20], 'Y': y[:20]})

    print(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")
    
    train_dataset = DTIDataset(train_df)
    valid_dataset = DTIDataset(valid_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = DeepDTA(drug_vocab_size=drug_encoder.vocab_size, 
                    target_vocab_size=target_encoder.vocab_size).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for b_drug, b_target, b_y in train_loader:
            b_drug, b_target, b_y = b_drug.to(device), b_target.to(device), b_y.to(device)
            
            optimizer.zero_grad()
            output = model(b_drug, b_target)
            loss = criterion(output.squeeze(), b_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_drug, b_target, b_y in valid_loader:
                b_drug, b_target, b_y = b_drug.to(device), b_target.to(device), b_y.to(device)
                output = model(b_drug, b_target)
                loss = criterion(output.squeeze(), b_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            # Save to specific directory
            save_path = os.path.join(os.path.dirname(__file__), '../data/model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    if not os.path.exists('../data'):
        # Just in case run from src
        try:
             os.makedirs('../data')
        except:
             pass
    if not os.path.exists('data'):
         try:
             os.makedirs('data')
         except:
             pass
             
    # Ensure data dir exists relative to this script
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train()
