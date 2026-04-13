import numpy as np
import torch

def run_analysis(p5, cls, spatial_nodes, band_weights, labels=None):
    # ✅ Pass true spatial and frequency signals, drop the fused tokens
    output = p5.predict_full(cls, spatial_nodes=spatial_nodes, band_weights=band_weights)

    risk_score   = output["risk_score"]
    lower, upper = output["confidence_interval"]
    trajectory   = output["trajectory"]
    drivers      = output["driver_channels"]
    bands        = output["band_contributions"]
    chan_scores  = output["channel_scores"]

    risk_np   = risk_score.detach().cpu().numpy().flatten()
    lower_np  = lower.detach().cpu().numpy().flatten()
    upper_np  = upper.detach().cpu().numpy().flatten()
    traj_np   = trajectory.detach().cpu().numpy()
    
    # Safely detach clinically grounded metrics
    bands_np  = bands.detach().cpu().numpy() if bands is not None else None
    chan_np   = chan_scores.detach().cpu().numpy() if chan_scores is not None else None

    result = {
        # 🔹 Risk graph (MC Dropout backed)
        "risk_plot": {
            "x": list(range(len(risk_np))),
            "y": risk_np.tolist(),
            "lower": lower_np.tolist(),
            "upper": upper_np.tolist(),
            "gt": labels.cpu().numpy().tolist() if labels is not None else None
        },

        # 🔹 Trajectory graph
        "trajectory": {
            "x": list(range(traj_np.shape[1])),
            "y": traj_np.mean(axis=0).tolist(),
            "std": traj_np.std(axis=0).tolist()
        },
        
        # 🔹 Clinically Grounded Attributions (True Localization)
        "clinical_drivers": {
            "top_channels": drivers,
            "channel_scores": chan_np.tolist() if chan_np is not None else None,
            "band_weights": bands_np.tolist() if bands_np is not None else None
        }
    }
    return result