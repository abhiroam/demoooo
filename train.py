import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score, f1_score

# ✅ Load the final unified architecture
from person23_graph_encoder import UnifiedGraphEncoder
from person4_transformer import TransformerSSL
from Person5 import SeizureClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXUP = True
MIXUP_PROB = 0.5
LR_ENC = 8e-5
LR_TX  = 1e-4
LR_CLS = 3e-4

# ── DIRS ──
DATA_DIR = "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2/graph_output"
SSL_DIR  = "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2/transformer_output"
SAVE_DIR = "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. DATA LOADER & PATIENT SPLIT ──
def build_patient_split(data_dir, val_split=0.2):
    """✅ FIX: Groups data by Patient ID to prevent Data Leakage."""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith("_nodes.npy")])
    patient_dict = {}
    
    print("Indexing dataset for patient-level split...")
    for f in files:
        base = f.replace("_nodes.npy", "")
        patient_id = base.split("_")[0]  # Extract 'aaaaaaac'
        
        np_ = os.path.join(data_dir, f"{base}_nodes.npy")
        ap_ = os.path.join(data_dir, f"{base}_adj.npy")
        lp_ = os.path.join(data_dir, f"{base}_labels.npy")
        
        try:
            n_rows = np.load(lp_, mmap_mode='r').shape[0]
            if patient_id not in patient_dict:
                patient_dict[patient_id] = []
            for i in range(n_rows):
                patient_dict[patient_id].append((np_, ap_, lp_, i))
        except Exception:
            continue

    patients = list(patient_dict.keys())
    np.random.shuffle(patients)
    split_idx = int(len(patients) * (1 - val_split))
    
    train_patients = patients[:split_idx]
    val_patients   = patients[split_idx:]
    
    train_list = [item for p in train_patients for item in patient_dict[p]]
    val_list   = [item for p in val_patients for item in patient_dict[p]]
    
    print(f"Train Patients: {len(train_patients)} | Val Patients: {len(val_patients)}")
    print(f"Train Samples: {len(train_list)} | Val Samples: {len(val_list)}")
    
    return train_list, val_list

class EEGDataset(Dataset):
    def __init__(self, data_list):
        self.items = data_list

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        np_, ap, lp, i = self.items[idx]
        
        nodes_mm  = np.load(np_, mmap_mode='r')
        adj_mm    = np.load(ap,  mmap_mode='r')
        labels_mm = np.load(lp,  mmap_mode='r')
        
        mask_path = np_.replace("_nodes.npy", "_mask.npy")
        try:
            mask_mm = np.load(mask_path, mmap_mode='r')
            ch_mask = torch.tensor(np.array(mask_mm), dtype=torch.float32)
        except FileNotFoundError:
            ch_mask = torch.ones(16, dtype=torch.float32)
        
        return (
            torch.tensor(np.array(nodes_mm[i]), dtype=torch.float32),
            torch.tensor(np.array(adj_mm[i]),   dtype=torch.float32),
            ch_mask,
            torch.tensor(int(labels_mm[i]), dtype=torch.long)
        )

# ── 2. OPTIMIZER ──
def make_optimizer(p23, p4, p5):
    enc_params = [p for p in p23.parameters() if p.requires_grad]
    tx_params  = [p for p in p4.parameters() if p.requires_grad]
    cls_params = [p for p in p5.parameters() if p.requires_grad]
    
    return torch.optim.AdamW([
        {'params': enc_params, 'lr': LR_ENC, 'weight_decay': 1e-4},
        {'params': tx_params,  'lr': LR_TX,  'weight_decay': 1e-4},
        {'params': cls_params, 'lr': LR_CLS, 'weight_decay': 1e-3}
    ])

# ── 3. EMA HELPER ──
class EMAModel:
    def __init__(self, models, decay=0.999):
        self.models = models
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for m in self.models:
            for name, param in m.named_parameters():
                if param.requires_grad:
                    self.shadow[(m, name)] = param.data.clone()

    def update(self, models):
        with torch.no_grad():
            for m in models:
                for name, param in m.named_parameters():
                    if param.requires_grad:
                        new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[(m, name)]
                        self.shadow[(m, name)] = new_avg.clone()

    def apply_shadow(self):
        with torch.no_grad():
            for m in self.models:
                for name, param in m.named_parameters():
                    if param.requires_grad:
                        self.backup[(m, name)] = param.data.clone()
                        param.data = self.shadow[(m, name)].clone()

    def restore(self):
        with torch.no_grad():
            for m in self.models:
                for name, param in m.named_parameters():
                    if param.requires_grad:
                        param.data = self.backup[(m, name)].clone()

# ── 4. EVALUATION FUNCTION (Clinical Metrics) ──
def evaluate(p23, p4, p5, loader):
    p23.eval(); p4.eval(); p5.eval()
    all_preds, all_true = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for nodes, adj, ch_mask, labels in loader:
            nodes, adj, ch_mask, labels = nodes.to(DEVICE), adj.to(DEVICE), ch_mask.to(DEVICE), labels.to(DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out_p23, _, _ = p23(nodes, adj, ch_mask)
                cls, _        = p4(out_p23)
                logits        = p5(cls)
                loss          = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / max(len(loader), 1)
    
    # ✅ FIX: Calculate Preictal specific metrics
    # labels: 0=Interictal, 1=Preictal, 2=Ictal
    pre_recall = recall_score(all_true, all_preds, labels=[1], average='macro', zero_division=0)
    pre_f1     = f1_score(all_true, all_preds, labels=[1], average='macro', zero_division=0)
    
    return avg_loss, pre_recall, pre_f1

# ── 5. MAIN TRAINING LOOP ──
def train(epochs=50, batch_size=64):
    train_list, val_list = build_patient_split(DATA_DIR, val_split=0.2)
    
    train_loader = DataLoader(EEGDataset(train_list), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(EEGDataset(val_list), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    p23 = UnifiedGraphEncoder().to(DEVICE)
    p4  = TransformerSSL().to(DEVICE)
    p5  = SeizureClassifier().to(DEVICE)

    # Load SSL weights
    try:
        p23.load_state_dict(torch.load(os.path.join(SSL_DIR, "p23_ssl_pretrained.pth"), map_location=DEVICE))
        print("✅ P23 (GAT) SSL weights loaded successfully!")
    except Exception as e:
        print(f"⚠️ P23 SSL weights not found. Error: {e}")

    try:
        p4.load_state_dict(torch.load(os.path.join(SSL_DIR, "transformer_ssl_pretrained.pth"), map_location=DEVICE))
        print("✅ P4 (Transformer) SSL weights loaded successfully!")
    except Exception as e:
        print(f"⚠️ P4 SSL weights not found. Error: {e}")

    optimizer = make_optimizer(p23, p4, p5)
    scaler = torch.amp.GradScaler()
    ema = EMAModel([p23, p4, p5], decay=0.999)
    criterion = nn.CrossEntropyLoss()

    best_preictal_f1 = -1.0  # ✅ Save based on clinical utility!

    for epoch in range(epochs):
        p23.train(); p4.train(); p5.train()
        epoch_loss = 0.0
        
        for batch_idx, (nodes, adj, ch_mask, labels) in enumerate(train_loader):
            nodes, adj, ch_mask, labels = nodes.to(DEVICE), adj.to(DEVICE), ch_mask.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                if USE_MIXUP and torch.rand(1).item() < MIXUP_PROB:
                    lam = np.random.beta(0.2, 0.2)
                    perm = torch.randperm(nodes.size(0)).to(DEVICE)
                    
                    mixed_nodes = lam * nodes + (1 - lam) * nodes[perm]
                    mixed_adj   = lam * adj + (1 - lam) * adj[perm]
                    mixed_mask  = lam * ch_mask + (1 - lam) * ch_mask[perm]
                    labels_a, labels_b = labels, labels[perm]

                    out_p23, _, _ = p23(mixed_nodes, mixed_adj, mixed_mask)
                    cls, _        = p4(out_p23)
                    logits        = p5(cls)

                    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                else:
                    out_p23, _, _ = p23(nodes, adj, ch_mask)
                    cls, _        = p4(out_p23)
                    logits        = p5(cls)

                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            ema.update([p23, p4, p5])
            
        train_loss = epoch_loss / len(train_loader)
        
        # ✅ EVALUATE USING EMA SHADOW WEIGHTS
        ema.apply_shadow()
        val_loss, pre_recall, pre_f1 = evaluate(p23, p4, p5, val_loader)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Preictal Recall: {pre_recall:.4f} | Preictal F1: {pre_f1:.4f}")
        
        # ✅ Save best model based on Preictal F1 Score, NOT global loss
        if pre_f1 > best_preictal_f1:
            best_preictal_f1 = pre_f1
            torch.save({
                "p23": p23.state_dict(),
                "p4":  p4.state_dict(),
                "p5":  p5.state_dict(),
            }, os.path.join(SAVE_DIR, "model_best.pth"))
            print("  ★ Best clinical model saved!")
            
        ema.restore()

    print("\n✅ Training Complete.")

if __name__ == "__main__":
    train(epochs=50)