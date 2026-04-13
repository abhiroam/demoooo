# ============================================================
# 🧠 TRANSFORMER — MULTI-OBJECTIVE SSL (PERSON 4 FINAL)
# Jointly Pretrains P23 (GAT) and P4 (Transformer) end-to-end
# ============================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from person23_graph_encoder import UnifiedGraphEncoder

# ── DIRS ──
# ✅ FIX: Pointing directly to Graph Output
GRAPH_DIR  = "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2/graph_output"
OUTPUT_DIR = "/content/drive/MyDrive/TUH_EEG_SEIZURE_v2/transformer_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CONSTANTS ──
D_MODEL     = 512
NUM_WINDOWS = 10
SEQ_LEN     = NUM_WINDOWS + 1
NUM_HEADS   = 8
NUM_LAYERS  = 6
FFN_DIM     = 1024
DROPOUT     = 0.1

SSL_EPOCHS  = 30
SSL_LR      = 3e-4
BATCH_SIZE  = 32
GRAD_ACCUM  = 4
NUM_WORKERS = 2

# ============================================================
# DATASET
# ============================================================
class JointSSLEEGDataset(Dataset):
    def __init__(self, graph_dir: str):
        self.samples = []   
        self.by_patient = {}  

        files = sorted(glob(os.path.join(graph_dir, "*_nodes.npy"))) 
        if not files:
            raise FileNotFoundError(f"No files found in {graph_dir}")

        for f in files:
            stem = os.path.basename(f).replace("_nodes.npy", "")
            parts = stem.split("_")
            patient_id = parts[0]
            session_id = parts[1] if len(parts) > 1 else "s000"
            file_id    = parts[2] if len(parts) > 2 else "t000"

            try:
                arr = np.load(f, mmap_mode="r")
                if arr.ndim != 4 or arr.shape[1] != NUM_WINDOWS: continue
                n_rows = arr.shape[0]
            except:
                continue

            for row in range(n_rows):
                entry = (f, row, patient_id, session_id, file_id)
                self.samples.append(entry)
                self.by_patient.setdefault(patient_id, []).append(entry)

        self.patient_ids = list(self.by_patient.keys())

    def __len__(self):
        return len(self.samples)

    def _load_data(self, fpath, row):
        nodes = torch.tensor(np.load(fpath, mmap_mode="r")[row], dtype=torch.float32)
        adj_path = fpath.replace("_nodes.npy", "_adj.npy")
        adj = torch.tensor(np.load(adj_path, mmap_mode="r")[row], dtype=torch.float32)
        
        mask_path = fpath.replace("_nodes.npy", "_mask.npy")
        try:
            mask = torch.tensor(np.load(mask_path, mmap_mode="r"), dtype=torch.float32)
        except FileNotFoundError:
            mask = torch.ones(16, dtype=torch.float32)
            
        return nodes, adj, mask

    def __getitem__(self, idx):
        fpath, row, patient_id, session_id, file_id = self.samples[idx]
        nodes, adj, mask = self._load_data(fpath, row)       

        # 1. Temporal Swap (Applied at the raw sequence level)
        n_shuf, a_shuf = nodes.clone(), adj.clone()
        swap = random.randint(0, NUM_WINDOWS - 2)
        n_shuf[swap], n_shuf[swap+1] = n_shuf[swap+1].clone(), n_shuf[swap].clone()
        a_shuf[swap], a_shuf[swap+1] = a_shuf[swap+1].clone(), a_shuf[swap].clone()

        # 2. Hierarchical Temporal Negatives
        patient_entries = self.by_patient[patient_id]
        neg_entry = None
        
        for attempt in range(15):
            cand = random.choice(patient_entries)
            cand_f, cand_row, cand_sid, cand_tid = cand[0], cand[1], cand[3], cand[4]
            
            if attempt < 6:
                if cand_tid == file_id and abs(cand_row - row) > 100:
                    neg_entry = cand; break
            elif attempt < 11:
                if cand_sid == session_id and cand_tid != file_id:
                    neg_entry = cand; break
            else:
                if cand_sid != session_id:
                    neg_entry = cand; break
                
        if neg_entry is None:
            other_pid = patient_id
            while other_pid == patient_id and len(self.patient_ids) > 1:
                other_pid = random.choice(self.patient_ids)
            neg_entry = random.choice(self.by_patient[other_pid])

        n_neg, a_neg, m_neg = self._load_data(neg_entry[0], neg_entry[1])

        return (nodes, adj, mask), (n_shuf, a_shuf, mask), (n_neg, a_neg, m_neg)


# ============================================================
# MODEL
# ============================================================
class TransformerSSL(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NUM_HEADS,
            dim_feedforward=FFN_DIM, dropout=DROPOUT,
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)

        self.order_head = nn.Sequential(nn.Linear(D_MODEL, 128), nn.GELU(), nn.Linear(128, 2))
        self.pred_head_fwd = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        self.proj_head = nn.Sequential(nn.Linear(D_MODEL, 256), nn.GELU(), nn.Linear(256, 128))

        self.log_sigma_order   = nn.Parameter(torch.zeros(1))
        self.log_sigma_pred    = nn.Parameter(torch.zeros(1))
        self.log_sigma_contrast= nn.Parameter(torch.zeros(1))

    def encode(self, x):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        enc = self.transformer(x)
        return self.norm(enc[:, 0, :]), enc[:, 1:, :]

    def forward(self, x, ssl_mode=False):
        if ssl_mode: raise ValueError("Call model.ssl_loss().")
        return self.encode(x)

    def ssl_loss(self, seq, shuffled, neg_seq, tau=0.07):
        B = seq.shape[0]
        device = seq.device

        cls_orig, tok_orig       = self.encode(seq)
        cls_shuffle, _           = self.encode(shuffled)
        cls_neg, _               = self.encode(neg_seq)

        # 1. Order Prediction
        order_input  = torch.cat([cls_orig, cls_shuffle], dim=0)
        order_labels = torch.cat([torch.ones(B, device=device, dtype=torch.long),
                                  torch.zeros(B, device=device, dtype=torch.long)], dim=0)
        L_order = F.cross_entropy(self.order_head(order_input), order_labels)

        # 2. Future Prediction
        L_pred = F.mse_loss(self.pred_head_fwd(tok_orig[:, :-1, :]), tok_orig[:, 1:, :].detach())

        # 3. Contrastive
        z_orig = F.normalize(self.proj_head(cls_orig), dim=-1)  
        z_neg  = F.normalize(self.proj_head(cls_neg),  dim=-1)  
        cls_orig2, _ = self.encode(seq)
        z_orig2 = F.normalize(self.proj_head(cls_orig2), dim=-1)

        pos_sim = (z_orig * z_orig2).sum(dim=-1, keepdim=True) / tau   
        neg_sim = (z_orig @ z_neg.T) / tau                             
        logits   = torch.cat([pos_sim, neg_sim], dim=-1)   
        L_contrast = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=device))

        # 4. Uncertainty Weights
        s1, s2, s3 = torch.exp(-self.log_sigma_order), torch.exp(-self.log_sigma_pred), torch.exp(-self.log_sigma_contrast)
        L_total = (s1 * L_order + self.log_sigma_order + 
                   s2 * L_pred + self.log_sigma_pred + 
                   s3 * L_contrast + self.log_sigma_contrast)

        return L_total, {"order": L_order.item(), "pred": L_pred.item(), "contrast": L_contrast.item()}


# ============================================================
# PRETRAINING LOOP
# ============================================================
def ssl_pretrain():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    dataset = JointSSLEEGDataset(GRAPH_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=use_amp,
                         persistent_workers=(NUM_WORKERS > 0), drop_last=True)

    # ✅ FIX: Jointly optimize P23 and P4!
    p23 = UnifiedGraphEncoder().to(device)
    p4 = TransformerSSL().to(device)
    optimizer = torch.optim.AdamW(list(p23.parameters()) + list(p4.parameters()), lr=SSL_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SSL_EPOCHS, eta_min=1e-5)

    best_loss  = float("inf")

    for ep in range(SSL_EPOCHS):
        p23.train(); p4.train()
        totals = {"order": 0., "pred": 0., "contrast": 0., "total": 0.}
        n_steps = 0
        current_tau = max(0.07, 0.5 - (0.43 * (ep / 10.0)))
        optimizer.zero_grad()

        for step, (orig, shuf, neg) in enumerate(loader):
            n_o, a_o, m_o = [x.to(device, non_blocking=True) for x in orig]
            n_s, a_s, m_s = [x.to(device, non_blocking=True) for x in shuf]
            n_n, a_n, m_n = [x.to(device, non_blocking=True) for x in neg]

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                # Pass through GAT first
                seq_emb, _, _ = p23(n_o, a_o, m_o)
                shuf_emb, _, _ = p23(n_s, a_s, m_s)
                neg_emb, _, _ = p23(n_n, a_n, m_n)
                
                # Pass through Transformer SSL
                loss, info = p4.ssl_loss(seq_emb, shuf_emb, neg_emb, tau=current_tau)

            loss = loss / GRAD_ACCUM
            if use_amp: scaler.scale(loss).backward()
            else: loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                if use_amp: scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(p23.parameters()) + list(p4.parameters()), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            totals["total"] += loss.item() * GRAD_ACCUM
            n_steps += 1

        scheduler.step()
        avg = totals["total"] / max(n_steps, 1)
        if avg < best_loss:
            best_loss = avg
            torch.save(p4.state_dict(), os.path.join(OUTPUT_DIR, "transformer_ssl_pretrained.pth"))
            # Save the pre-trained P23 encoder too!
            torch.save(p23.state_dict(), os.path.join(OUTPUT_DIR, "p23_ssl_pretrained.pth"))
            print(f"Epoch {ep+1:2d} | loss={avg:.4f} | tau={current_tau:.3f}  ★ best")

if __name__ == "__main__":
    ssl_pretrain()