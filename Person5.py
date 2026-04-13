# ============================================================
# 🧠 SEIZURE CLASSIFIER & CLINICAL DECODER (PERSON 5 FINAL)
# Outputs: Risk, Uncertainty, Trajectory, Bands, Channels
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeizureClassifier(nn.Module):
    def __init__(self, in_dim=512, spatial_dim=128, num_channels=16):
        super().__init__()
        
        self.num_channels = num_channels

        # ── Main Feature Projection (With Uncertainty Dropout) ──
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # ── Main Risk Classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 0: Interictal, 1: Preictal, 2: Ictal
        )

        # ── Clinical Decoders ──
        
        # 1. Trajectory Generator (Optional UI Visualization)
        self.traj_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Sigmoid() 
        )

        # 2. True Spatial Attention (Scores the 16 physical channels)
        # Evaluates the actual graph nodes before they are flattened by the Transformer
        self.spatial_attention = nn.Sequential(
            nn.Linear(spatial_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, cls):
        x = self.feature_proj(cls)
        return self.classifier(x)

    def predict_proba(self, cls):
        return torch.softmax(self.forward(cls), dim=-1)

    def predict_full(self, cls, spatial_nodes=None, band_weights=None, runs=20):
        """
        cls: [B, 512] (from Person 4 Transformer)
        spatial_nodes: [B, T, 16, 128] (passed directly from Person 2+3 Unified Encoder)
        band_weights: [B, 5] or [5] (passed directly from Person 2+3 Unified Encoder)
        """
        
        self.eval()

        # ✅ Enable ONLY dropout layers for Monte Carlo sampling (Confidence Intervals)
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        # ── 1. Confidence Intervals via MC Dropout ──
        with torch.no_grad():
            mc_probs = torch.stack([self.predict_proba(cls) for _ in range(runs)])

        # Get Seizure class (index 2) probabilities
        mean_risk   = mc_probs.mean(dim=0)[:, 2]
        lower_bound = torch.quantile(mc_probs[:, :, 2], 0.05, dim=0)
        upper_bound = torch.quantile(mc_probs[:, :, 2], 0.95, dim=0)

        # ✅ Restore eval mode for safety so downstream evaluation is deterministic
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

        # ── 2. Interpretability Metrics for Frontend ──
        with torch.no_grad():
            features = self.feature_proj(cls)
            trajectory = self.traj_head(features)
            
            # Ground-Truth Band Contributions (Passed directly from Person 3 GAT)
            bands = band_weights if band_weights is not None else None
            
            # Ground-Truth Spatial Localization
            drivers, channel_scores = None, None
            if spatial_nodes is not None:
                # Pool across time to get overall channel state: [B, 16, 128]
                pooled_spatial = spatial_nodes.mean(dim=1)
                
                # Apply attention to score channels: [B, 16, 1] -> [B, 16]
                attn_logits = self.spatial_attention(pooled_spatial).squeeze(-1)
                channel_scores = torch.softmax(attn_logits, dim=-1)
                
                # Extract top 5 physical electrodes for the UI
                drivers = torch.topk(channel_scores, 5, dim=1).indices.tolist()

        return {
            "risk_score":          mean_risk,
            "confidence_interval": (lower_bound, upper_bound),
            "trajectory":          trajectory,
            "driver_channels":     drivers,
            "band_contributions":  bands,
            "channel_scores":      channel_scores
        }