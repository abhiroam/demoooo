# ============================================================
# 🧠 PERSON 2+3: UNIFIED GRAPH ENCODER (FINAL PRODUCTION)
# End-to-end MLP Projector + Topology-Preserving Multi-Head GAT
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants to prevent silent breakage if upstream features change
EEG_FEATURE_DIM = 9   # 5 bands + 1 entropy + 3 hjorth
ID_FEATURE_DIM = 16   # One-hot channel identity
NUM_BANDS = 5         # Delta, Theta, Alpha, Beta, Gamma


class FeatureProjector(nn.Module):
    """
    Replaces the flawed CNN. Processes the EEG features respecting their semantic grouping.
    """
    def __init__(self, in_features=EEG_FEATURE_DIM, hidden=32, out_features=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        # x shape: [B, T, N, in_features]
        return self.mlp(x)


class MultiHeadGATLayer(nn.Module):
    """
    Strictly masked, true multi-head Graph Attention Layer.
    """
    def __init__(self, in_features, out_features, num_heads, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"

        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # True per-head attention scorer
        self.a = nn.Parameter(torch.empty(size=(1, 1, 1, 1, num_heads, self.head_dim * 2)))
        nn.init.xavier_uniform_(self.a)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # Safe residual connection
        self.res = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, h, adj, ch_mask=None):
        B, T, N, _ = h.shape
        
        # 1. Linear Projection
        Wh = self.W(h).view(B, T, N, self.num_heads, self.head_dim)
        
        # 2. Compute Attention Logits
        Wh_i = Wh.unsqueeze(3).expand(B, T, N, N, self.num_heads, self.head_dim)
        Wh_j = Wh.unsqueeze(2).expand(B, T, N, N, self.num_heads, self.head_dim)
        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)  # [B, T, N, N, num_heads, head_dim * 2]
        
        # Element-wise multiply with per-head weights, sum over feature dim
        e = (Wh_concat * self.a).sum(dim=-1)  # [B, T, N, N, num_heads]

        # 3. LeakyReLU FIRST
        e = self.leaky_relu(e)

        # 4. Build Strict Mask
        mask = (adj == 0).unsqueeze(-1)  # [B, T, N, N, 1]

        # Safely handle both [16] and [B, 16] ch_mask shapes dynamically
        if ch_mask is not None:
            if ch_mask.dim() == 1:
                ch_mask = ch_mask.unsqueeze(0).expand(B, -1)
            
            missing_i = (ch_mask == 0).view(B, 1, N, 1, 1)
            missing_j = (ch_mask == 0).view(B, 1, 1, N, 1)
            mask = mask | missing_i | missing_j

        # 5. Mask SECOND
        e = e.masked_fill(mask, float('-inf'))
        
        # 6. Softmax THIRD
        attention = F.softmax(e, dim=3)
        
        # Safety catch: If a node has all neighbors masked, softmax(-inf) returns NaN.
        # This converts those NaNs to 0.0 to prevent gradient collapse.
        attention = torch.nan_to_num(attention, nan=0.0) 
        attention = self.dropout(attention)

        # 7. Aggregate
        h_prime = torch.einsum('btijh,btjhd->btihd', attention, Wh)
        h_prime = h_prime.reshape(B, T, N, self.out_features)
        
        # 8. Residual + Activation
        return F.elu(h_prime + self.res(h))


class UnifiedGraphEncoder(nn.Module):
    def __init__(self, proj_dim=64, gat_hidden=128, out_dim=512, num_nodes=16):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Replace CNN
        self.projector = FeatureProjector(in_features=EEG_FEATURE_DIM, out_features=proj_dim)
        
        # Scale identity features so they don't overpower LayerNorm'd EEG features
        self.id_embedding = nn.Linear(ID_FEATURE_DIM, ID_FEATURE_DIM, bias=False)
        nn.init.eye_(self.id_embedding.weight) 
        
        gat_in_dim = proj_dim + ID_FEATURE_DIM 
        
        # Learnable frequency band weights
        self.band_weights = nn.Parameter(torch.ones(NUM_BANDS))
        
        self.gat1 = MultiHeadGATLayer(in_features=gat_in_dim, out_features=gat_hidden, num_heads=4)
        self.gat2 = MultiHeadGATLayer(in_features=gat_hidden, out_features=gat_hidden, num_heads=4)
        
        # Topology-Preserving Flatten: 16 nodes * 128 dims = 2048
        flat_dim = num_nodes * gat_hidden
        
        # Project exactly to Transformer expectation
        self.topology_proj = nn.Sequential(
            nn.Linear(flat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(0.3) 
        )

    def forward(self, nodes, adj, ch_mask=None):
        """
        nodes:   [B, T, 16, 25]
        adj:     [B, T, 5, 16, 16]
        ch_mask: [B, 16] or [16]
        """
        B, T, N, _ = nodes.shape
        
        # ── 1. Safely split features using constants ──
        eeg_feats = nodes[..., :EEG_FEATURE_DIM]
        id_feats  = nodes[..., EEG_FEATURE_DIM : EEG_FEATURE_DIM + ID_FEATURE_DIM]
        
        # ── 2. Project EEG features & Embed Identity ──
        proj_feats = self.projector(eeg_feats)          # [B, T, N, 64]
        embedded_id = self.id_embedding(id_feats)       # [B, T, N, 16]
        
        gat_in = torch.cat([proj_feats, embedded_id], dim=-1)  # [B, T, N, 80]
        
        # ── 3. Band-Aware Adjacency ──
        # Softmax ensures weights sum to 1. Reshape for broadcasting.
        band_w = F.softmax(self.band_weights, dim=0).view(1, 1, NUM_BANDS, 1, 1)
        adj_weighted = (adj * band_w).sum(dim=2)  # [B, T, N, N]
        
        # ── 4. Graph Attention ──
        x = self.gat1(gat_in, adj_weighted, ch_mask)
        x = self.gat2(x, adj_weighted, ch_mask)  # [B, T, N, 128]
        
        # ── 5. Topology-Preserving Flatten ──
        x_flat = x.view(B, T, N * x.shape[-1])  # [B, T, 2048]
        
        # ── 6. Output to Transformer ──
        out = self.topology_proj(x_flat)  # [B, T, 512]
        
        return out, x, self.band_weights


def get_compiled_encoder():
    model = UnifiedGraphEncoder()
    try:
        model = torch.compile(model)
        print("⚡ torch.compile() enabled")
    except Exception as e:
        print(f"⚠️ torch.compile() failed. Using standard eager mode. Error: {e}")
    return model