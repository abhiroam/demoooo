import numpy as np
import torch
import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from person23_graph_encoder import UnifiedGraphEncoder
from person4_transformer import TransformerSSL
from Person5 import SeizureClassifier
from analysis import run_analysis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "graph_output")

# ── 1. Load data ──
files = os.listdir(DATA_DIR)
base = sorted([f for f in files if f.endswith("_nodes.npy")])[0].replace("_nodes.npy","")

nodes  = np.load(f"{DATA_DIR}/{base}_nodes.npy")
adj    = np.load(f"{DATA_DIR}/{base}_adj.npy")
labels = np.load(f"{DATA_DIR}/{base}_labels.npy")

# ✅ Safely load mask
try:
    ch_mask = np.load(f"{DATA_DIR}/{base}_mask.npy")
except FileNotFoundError:
    ch_mask = np.ones(16, dtype=np.float32)

nodes   = torch.tensor(nodes, dtype=torch.float32).to(DEVICE)
adj     = torch.tensor(adj, dtype=torch.float32).to(DEVICE)
ch_mask = torch.tensor(ch_mask, dtype=torch.float32).to(DEVICE)
labels  = torch.tensor(labels, dtype=torch.long).to(DEVICE)

# ── 2. Initialize Models ──
p23 = UnifiedGraphEncoder().to(DEVICE)
p4  = TransformerSSL().to(DEVICE)
p5  = SeizureClassifier().to(DEVICE)

# Note: Load your trained weights here...
p23.eval(); p4.eval(); p5.eval()

# ── 3. Forward Pass ──
with torch.no_grad():
    # Pipeline execution
    out_p23, spatial_nodes, band_weights = p23(nodes, adj, ch_mask)
    cls, tokens = p4(out_p23)
    logits = p5(cls)
    preds = logits.argmax(dim=1)

true = labels.cpu().numpy()
preds_np = preds.cpu().numpy()

# ── 4. Metrics & FAR Calculation ──
# Each sequence steps forward by 3.0 seconds
STEP_SECONDS = 3.0
total_hours = (len(true) * STEP_SECONDS) / 3600

# alert = preictal OR seizure prediction
alerts = (preds_np == 1) | (preds_np == 2)

# ✅ Clinically meaningful future window (e.g., 5 mins = 300 sec / 3 sec step = 100 windows)
future_window = 100 
future_seizure = np.zeros_like(true, dtype=bool)

for i in range(len(true)):
    future_seizure[i] = np.any(true[i:i+future_window] == 2)

false_alarms = (alerts & (~future_seizure)).sum()
far_per_hour = false_alarms / (total_hours + 1e-8)

print(f"False Alarm Rate per hour: {far_per_hour:.2f}")

# ── 5. Clinical Dashboard Prep ──
analysis_results = run_analysis(
    p5=p5, 
    cls=cls, 
    spatial_nodes=spatial_nodes, 
    band_weights=band_weights, 
    labels=labels
)
print("Pipeline successful. Analysis ready for frontend.")