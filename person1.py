# ============================================================
# 🚀 ULTRA FAST PIPELINE (FULL DATA, NO LIMIT)
# ============================================================

import os
import time
import warnings
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import mne
import antropy as ant
from scipy.signal import welch as scipy_welch

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    "data_root": BASE_DIR,   # 🔥 IMPORTANT
    "output_dir": os.path.join(BASE_DIR, "graph_output"),
    "sfreq": 256,
    "epoch_duration": 4.0,
    "epoch_overlap": 0.25,
    "preictal_sec": 300,
    "min_gap_sec": 60,
    "freq_bands": {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40),
    },
    "seq_len": 10,
    "n_jobs": 2   # 🔥 MAX CPU -> changed to 2 to prevent MemoryError
}

NUM_CHANNELS = 16

# =========================================================
# BASIC FUNCTIONS (UNCHANGED)
# =========================================================
def parse_annotations(path):
    seizures = []
    if path is None:
        return seizures
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                try:
                    seizures.append((float(parts[1]), float(parts[2])))
                except:
                    pass
    return seizures

def get_label(t, seizures):
    for s, e in seizures:
        if s <= t <= e:
            return 2
        if s - CONFIG["preictal_sec"] <= t < s:
            return 1
        if e < t < e + CONFIG["min_gap_sec"]:
            return None
    return 0

def create_epochs(raw, seizures):
    data = raw.get_data()
    times = raw.times
    sfreq = raw.info["sfreq"]

    win = int(CONFIG["epoch_duration"] * sfreq)
    step = int(win * (1 - CONFIG["epoch_overlap"]))

    epochs, labels = [], []
    for i in range(0, data.shape[1] - win + 1, step):
        seg = data[:, i:i+win]
        t = times[i + win//2]
        lbl = get_label(t, seizures)
        epochs.append(seg)
        labels.append(lbl)

    return epochs, labels

def compute_node_features(epoch):
    freqs, psd = scipy_welch(epoch, CONFIG["sfreq"], nperseg=512)

    feats = []
    for low, high in CONFIG["freq_bands"].values():
        idx = (freqs >= low) & (freqs <= high)
        feats.append(psd[:, idx].mean(axis=1))

    entropy = np.array([ant.sample_entropy(ch) for ch in epoch])
    feats.append(entropy)

    feats = np.array(feats).T
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)

    return feats.astype(np.float32)

def compute_adjacency(epoch):
    return np.eye(NUM_CHANNELS, dtype=np.float32)

# =========================================================
# PARALLEL EPOCH PROCESSING
# =========================================================
def process_epoch(epoch):
    return compute_node_features(epoch), compute_adjacency(epoch)

def build_sequences(epochs, labels):
    seq_len = CONFIG["seq_len"]

    with ProcessPoolExecutor(max_workers=CONFIG["n_jobs"]) as pool:
        results = list(pool.map(process_epoch, epochs))

    nodes_all, adj_all, seq_labels = [], [], []

    for i in range(len(epochs) - seq_len + 1):
        seq_lbls = labels[i:i+seq_len]

        if any(l is None for l in seq_lbls):
            continue

        nodes_all.append([results[i+j][0] for j in range(seq_len)])
        adj_all.append([results[i+j][1] for j in range(seq_len)])
        seq_labels.append(seq_lbls[(seq_len-1)//2])

    if len(nodes_all) == 0:
        return None, "empty"

    return (
        np.array(nodes_all),
        np.array(adj_all),
        np.array(seq_labels)
    ), None

# =========================================================
# FILE PROCESSING
# =========================================================
def process_one_file(edf_path, ann_path):
    start = time.time()
    name = Path(edf_path).stem

    try:
        seizures = parse_annotations(ann_path)
        if len(seizures) == 0:
            return "skip_no_seizure"

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        epochs, labels = create_epochs(raw, seizures)
        seq_data, err = build_sequences(epochs, labels)

        if seq_data is None:
            return err

        nodes, adj, seq_labels = seq_data

        # BALANCING
        normal = np.where(seq_labels == 0)[0]
        pre = np.where(seq_labels == 1)[0]
        ictal = np.where(seq_labels == 2)[0]

        keep = list(pre) + list(ictal)

        if len(normal) > 0:
            np.random.seed(42)
            keep.extend(np.random.choice(normal, int(0.5 * len(normal)), replace=False))

        keep = np.array(sorted(keep))

        nodes = nodes[keep]
        adj = adj[keep]
        seq_labels = seq_labels[keep]

        out = CONFIG["output_dir"]
        np.save(f"{out}/{name}_nodes.npy", nodes)
        np.save(f"{out}/{name}_adj.npy", adj)
        np.save(f"{out}/{name}_labels.npy", seq_labels)

        print(f"⏱️ {name} → {time.time()-start:.2f}s")
        return "done"

    except Exception as e:
        print("ERROR:", e)
        return "error"

# =========================================================
# RUN (SKIP + FAST)
# =========================================================
def run():
    total_start = time.time()
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    edfs = list(Path(CONFIG["data_root"]).rglob("*.edf"))

    # SKIP LOGIC
    done_set = set(
        f.stem.replace("_nodes", "")
        for f in Path(CONFIG["output_dir"]).glob("*_nodes.npy")
    )

    print(f"✅ Already processed: {len(done_set)}\n")

    done = skip = err = 0

    for i, edf in enumerate(edfs, 1):
        name = Path(edf).stem
        print(f"[{i}/{len(edfs)}] {name}")

        if name in done_set:
            skip += 1
            print("⏭️ skip")
            continue

        ann = edf.with_suffix(".csv_bi")
        res = process_one_file(str(edf), str(ann) if ann.exists() else None)

        if res == "done":
            done += 1
            done_set.add(name)
        elif res == "skip_no_seizure":
            skip += 1
        else:
            err += 1

        print(f"📊 done:{done} skip:{skip} err:{err}")

    print("\n🚀 TOTAL TIME:", time.time() - total_start)

if __name__ == "__main__":
    run()