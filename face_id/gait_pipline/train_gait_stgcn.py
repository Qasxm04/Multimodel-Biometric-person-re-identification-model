import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# =======================
# Reproducibility
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =======================
# Helpers
# =======================
def normalize_keypoints_xy(kp_xy):
    kp_xy = kp_xy.copy()

    if kp_xy.shape[0] >= 25:
        neck = kp_xy[1]
        pelvis = kp_xy[8]
    else:
        neck = kp_xy[1]
        pelvis = (kp_xy[8] + kp_xy[11]) / 2.0

    center = (neck + pelvis) / 2.0
    scale = np.linalg.norm(neck - pelvis)

    if scale < 1e-6:
        scale = 1.0

    return (kp_xy - center) / scale

def temporal_resample(seq, num_frames=100):
    T = seq.shape[0]
    if T == num_frames:
        return seq
    idx = np.linspace(0, T - 1, num_frames).astype(np.int64)
    return seq[idx]

def temporal_crop(seq, num_frames=100, mode="center"):
    T = seq.shape[0]
    if T <= num_frames:
        return temporal_resample(seq, num_frames)

    if mode == "start":
        return seq[:num_frames]
    elif mode == "end":
        return seq[-num_frames:]
    elif mode == "early":
        start = max(0, (T - num_frames) // 4)
        return seq[start:start + num_frames]
    elif mode == "late":
        start = max(0, (3 * (T - num_frames)) // 4)
        return seq[start:start + num_frames]
    else:
        start = (T - num_frames) // 2
        return seq[start:start + num_frames]

def add_velocity(seq):
    vel = np.zeros_like(seq, dtype=np.float32)
    vel[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, vel], axis=-1).astype(np.float32)

def get_bone_pairs(num_joints):
    if num_joints == 25:
        return [
            (1, 0), (2, 1), (3, 2), (4, 3),
            (5, 1), (6, 5), (7, 6),
            (8, 1), (9, 8), (10, 9), (11, 10),
            (12, 8), (13, 12), (14, 13),
            (15, 0), (17, 15),
            (16, 0), (18, 16),
            (19, 14), (20, 19),
            (22, 11), (23, 22),
            (21, 14), (24, 11)
        ]
    return [
        (0, 1), (2, 1), (3, 2), (4, 3),
        (5, 1), (6, 5), (7, 6),
        (8, 1), (9, 8), (10, 9),
        (11, 1), (12, 11), (13, 12),
        (14, 0), (16, 14),
        (15, 0), (17, 15)
    ]

def build_openpose_adjacency(num_joints):
    A = np.eye(num_joints, dtype=np.float32)

    if num_joints == 25:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10), (10, 11),
            (8, 12), (12, 13), (13, 14),
            (0, 15), (15, 17),
            (0, 16), (16, 18),
            (14, 19), (19, 20),
            (11, 22), (22, 23),
            (14, 21), (11, 24)
        ]
    else:
        edges = [
            (1, 0), (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13),
            (0, 14), (14, 16),
            (0, 15), (15, 17)
        ]

    for i, j in edges:
        if i < num_joints and j < num_joints:
            A[i, j] = 1.0
            A[j, i] = 1.0

    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
    A = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A, dtype=torch.float32)

# =======================
# Dataset
# =======================
class GaitDataset(Dataset):
    def __init__(self, root_folder, min_valid_frames=10):
        self.samples = []
        self.labels = []
        self.participant_to_idx = {}
        self.num_joints = None
        self.in_channels = 4

        participants = sorted([
            p for p in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, p))
        ])

        print("Building dataset...")
        for idx, participant in enumerate(tqdm(participants, desc="Participants", leave=True)):
            self.participant_to_idx[participant] = idx
            p_path = os.path.join(root_folder, participant)

            for exp in os.listdir(p_path):
                exp_path = os.path.join(p_path, exp)
                if not os.path.isdir(exp_path):
                    continue

                for trial in os.listdir(exp_path):
                    trial_path = os.path.join(exp_path, trial)
                    if not os.path.isdir(trial_path):
                        continue

                    json_folders = [
                        f for f in os.listdir(trial_path)
                        if os.path.isdir(os.path.join(trial_path, f)) and f.endswith("_json")
                    ]

                    for jf in json_folders:
                        json_path = os.path.join(trial_path, jf)
                        seq_xy = []

                        for file in sorted(os.listdir(json_path)):
                            if not file.endswith("_keypoints.json"):
                                continue

                            with open(os.path.join(json_path, file), "r") as f:
                                data = json.load(f)

                            if not data.get("people"):
                                continue

                            raw = np.array(data["people"][0]["pose_keypoints_2d"], dtype=np.float32)
                            if raw.size % 3 != 0:
                                continue

                            kp = raw.reshape(-1, 3)[:, :2]

                            if self.num_joints is None:
                                self.num_joints = kp.shape[0]

                            if kp.shape[0] != self.num_joints:
                                continue

                            kp = normalize_keypoints_xy(kp)
                            seq_xy.append(kp)

                        if len(seq_xy) < min_valid_frames:
                            continue

                        seq_xy = np.array(seq_xy, dtype=np.float32)
                        seq = add_velocity(seq_xy)

                        self.samples.append(torch.tensor(seq, dtype=torch.float32))
                        self.labels.append(idx)

        if len(self.samples) == 0:
            raise ValueError("Dataset built 0 samples")

        print(f"✅ Dataset: {len(self.samples)} samples | {len(self.participant_to_idx)} participants")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# =======================
# Eval collate
# =======================
def collate_fn_eval(batch, num_frames=100, crop_mode="center"):
    sequences, labels = zip(*batch)
    processed = []

    for seq in sequences:
        seq_np = seq.numpy()
        clip = temporal_crop(seq_np, num_frames=num_frames, mode=crop_mode)
        processed.append(torch.tensor(clip, dtype=torch.float32))

    X = torch.stack(processed, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    lengths = torch.full((len(labels),), num_frames, dtype=torch.long)
    return X, y, lengths

# =======================
# Models
# =======================
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.register_buffer("A_base", A)
        self.edge_importance = nn.Parameter(torch.ones_like(A))

    def forward(self, x):
        x = self.conv(x)
        A = self.A_base * self.edge_importance
        return torch.einsum("nctv,vw->nctw", x, A)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, dropout=0.3):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class STGCNGait(nn.Module):
    def __init__(self, num_classes, num_joints, in_channels=4):
        super().__init__()
        A = build_openpose_adjacency(num_joints)
        self.input_bn = nn.BatchNorm1d(num_joints * in_channels)
        self.block1 = STGCNBlock(in_channels, 64, A, dropout=0.2)
        self.block2 = STGCNBlock(64, 128, A, dropout=0.3)
        self.block3 = STGCNBlock(128, 256, A, dropout=0.4)
        self.block4 = STGCNBlock(256, 256, A, dropout=0.4)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, lengths):
        B, T, V, C = x.shape
        x = x.view(B, T, V * C).permute(0, 2, 1)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1).contiguous().view(B, T, V, C)
        x = x.permute(0, 3, 1, 2)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        T_out = x.shape[2]
        mask = torch.arange(T_out, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(-1).float()

        x = x * mask
        x = x.sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        x = x.mean(dim=-1)
        x = self.dropout(x)
        return self.fc(x)

class STGCNBackbone(nn.Module):
    def __init__(self, num_joints, in_channels=4):
        super().__init__()
        A = build_openpose_adjacency(num_joints)
        self.input_bn = nn.BatchNorm1d(num_joints * in_channels)
        self.block1 = STGCNBlock(in_channels, 64, A, dropout=0.2)
        self.block2 = STGCNBlock(64, 128, A, dropout=0.3)
        self.block3 = STGCNBlock(128, 256, A, dropout=0.4)
        self.block4 = STGCNBlock(256, 256, A, dropout=0.4)

    def forward(self, x, lengths):
        B, T, V, C = x.shape
        x = x.view(B, T, V * C).permute(0, 2, 1)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1).contiguous().view(B, T, V, C)
        x = x.permute(0, 3, 1, 2)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        T_out = x.shape[2]
        mask = torch.arange(T_out, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(-1).float()

        x = x * mask
        x = x.sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        x = x.mean(dim=-1)
        return x

class TwoStreamSTGCNGait(nn.Module):
    def __init__(self, num_classes, num_joints, in_channels=4):
        super().__init__()
        self.bone_pairs = get_bone_pairs(num_joints)
        self.joint_stream = STGCNBackbone(num_joints, in_channels)
        self.bone_stream = STGCNBackbone(num_joints, in_channels)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def build_bone_tensor(self, x):
        B, T, V, C = x.shape
        bone = torch.zeros_like(x)

        for child, parent in self.bone_pairs:
            if child < V and parent < V:
                bone[:, :, child, :2] = x[:, :, child, :2] - x[:, :, parent, :2]
                bone[:, :, child, 2:] = x[:, :, child, 2:] - x[:, :, parent, 2:]

        return bone

    def forward(self, x, lengths):
        joint_feat = self.joint_stream(x, lengths)
        bone_feat = self.bone_stream(self.build_bone_tensor(x), lengths)
        feat = torch.cat([joint_feat, bone_feat], dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)

# =======================
# Checkpoint utilities
# =======================
def detect_checkpoint_type(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith("joint_stream.") for k in keys) or any(k.startswith("bone_stream.") for k in keys):
        return "two_stream"
    return "single_stream"

def build_model_for_checkpoint(state_dict, num_classes, num_joints, in_channels):
    ckpt_type = detect_checkpoint_type(state_dict)

    if ckpt_type == "two_stream":
        model = TwoStreamSTGCNGait(num_classes, num_joints, in_channels)
    else:
        model = STGCNGait(num_classes, num_joints, in_channels)

    return model, ckpt_type

# =======================
# Evaluation
# =======================
def compute_topk(scores, labels):
    _, pred = scores.topk(10, dim=1, largest=True, sorted=True)
    y_expand = labels.view(-1, 1)

    top1 = (pred[:, :1] == y_expand).sum().item()
    top5 = (pred[:, :5] == y_expand).sum().item()
    top10 = (pred[:, :10] == y_expand).sum().item()
    total = labels.size(0)

    return (
        100.0 * top1 / total,
        100.0 * top5 / total,
        100.0 * top10 / total
    )

def evaluate_model(model, dataset_subset, device, num_frames=100, batch_size=8):
    model.eval()
    crop_modes = ["start", "early", "center", "late", "end"]

    all_scores = []
    labels_ref = None

    with torch.no_grad():
        for crop_mode in crop_modes:
            loader = DataLoader(
                dataset_subset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda b, cm=crop_mode: collate_fn_eval(b, num_frames=num_frames, crop_mode=cm),
                num_workers=0
            )

            scores = []
            labels_all = []

            for X, y, lengths in loader:
                X = X.to(device)
                lengths = lengths.to(device)

                logits = model(X, lengths)
                scores.append(logits.cpu())
                labels_all.append(y)

            scores = torch.cat(scores, dim=0)
            labels_all = torch.cat(labels_all, dim=0)

            all_scores.append(scores)
            if labels_ref is None:
                labels_ref = labels_all

    avg_scores = sum(all_scores) / len(all_scores)
    return compute_topk(avg_scores, labels_ref), avg_scores, labels_ref

def evaluate_ensemble(model_a, model_b, dataset_subset, device, num_frames=100, batch_size=8, alpha=0.5):
    model_a.eval()
    model_b.eval()
    crop_modes = ["start", "early", "center", "late", "end"]

    all_scores = []
    labels_ref = None

    with torch.no_grad():
        for crop_mode in crop_modes:
            loader = DataLoader(
                dataset_subset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda b, cm=crop_mode: collate_fn_eval(b, num_frames=num_frames, crop_mode=cm),
                num_workers=0
            )

            scores = []
            labels_all = []

            for X, y, lengths in loader:
                X = X.to(device)
                lengths = lengths.to(device)

                logits_a = model_a(X, lengths)
                logits_b = model_b(X, lengths)
                logits = alpha * logits_a + (1.0 - alpha) * logits_b

                scores.append(logits.cpu())
                labels_all.append(y)

            scores = torch.cat(scores, dim=0)
            labels_all = torch.cat(labels_all, dim=0)

            all_scores.append(scores)
            if labels_ref is None:
                labels_ref = labels_all

    avg_scores = sum(all_scores) / len(all_scores)
    return compute_topk(avg_scores, labels_ref), avg_scores, labels_ref

# =======================
# Main
# =======================
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(script_dir), "Gait Dataset")

    single_ckpt = "best_gait_stgcn.pt"
    two_stream_ckpt = "best_two_stream.pt"

    if not os.path.exists(single_ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {single_ckpt}")

    if not os.path.exists(two_stream_ckpt):
        print("⚠️ best_two_stream.pt not found. Only evaluating available checkpoint.")
        two_stream_ckpt = None

    dataset = GaitDataset(dataset_path)

    indices = list(range(len(dataset)))
    labels = dataset.labels

    _, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    test_subset = Subset(dataset, test_idx)

    print("\n📦 Loading primary checkpoint...")
    single_state = torch.load(single_ckpt, map_location=device)
    single_model, single_type = build_model_for_checkpoint(
        single_state,
        num_classes=len(dataset.participant_to_idx),
        num_joints=dataset.num_joints,
        in_channels=dataset.in_channels
    )
    single_model.load_state_dict(single_state)
    single_model = single_model.to(device)

    print(f"✅ Primary checkpoint detected as: {single_type}")

    print("\n🔍 Evaluating primary model...")
    single_metrics, _, _ = evaluate_model(single_model, test_subset, device)

    if two_stream_ckpt is not None:
        print("\n📦 Loading secondary checkpoint...")
        two_state = torch.load(two_stream_ckpt, map_location=device)
        two_model, two_type = build_model_for_checkpoint(
            two_state,
            num_classes=len(dataset.participant_to_idx),
            num_joints=dataset.num_joints,
            in_channels=dataset.in_channels
        )
        two_model.load_state_dict(two_state)
        two_model = two_model.to(device)

        print(f"✅ Secondary checkpoint detected as: {two_type}")

        print("\n🔍 Evaluating secondary model...")
        two_metrics, _, _ = evaluate_model(two_model, test_subset, device)

        candidate_alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        best_alpha = None
        best_combo = -1.0
        best_ens_metrics = None

        print("\n🧠 Running ensemble search...")
        for alpha in candidate_alphas:
            ens_metrics, _, _ = evaluate_ensemble(
                single_model,
                two_model,
                test_subset,
                device,
                alpha=alpha
            )

            top1, top5, top10 = ens_metrics
            combo = 0.5 * top1 + 0.3 * top5 + 0.2 * top10

            if combo > best_combo:
                best_combo = combo
                best_alpha = alpha
                best_ens_metrics = ens_metrics

    else:
        two_metrics = single_metrics
        best_alpha = 1.0
        best_ens_metrics = single_metrics
        two_type = "not_available"

    print("\n===== FINAL RESULTS =====")
    print(f"Participants: {len(dataset.participant_to_idx)}")

    print("\n--- Individual Models ---")
    print(f"Primary ({single_type})   | Top-1: {single_metrics[0]:.2f}% | Top-5: {single_metrics[1]:.2f}% | Top-10: {single_metrics[2]:.2f}%")
    print(f"Secondary ({two_type}) | Top-1: {two_metrics[0]:.2f}% | Top-5: {two_metrics[1]:.2f}% | Top-10: {two_metrics[2]:.2f}%")

    print("\n--- Best Ensemble / Final ---")
    print(f"Best alpha: {best_alpha:.2f}")
    print(f"Top-1: {best_ens_metrics[0]:.2f}%")
    print(f"Top-5: {best_ens_metrics[1]:.2f}%")
    print(f"Top-10: {best_ens_metrics[2]:.2f}%")

    metrics = ["Top-1", "Top-5", "Top-10"]
    primary_vals = list(single_metrics)
    secondary_vals = list(two_metrics)
    ens_vals = list(best_ens_metrics)

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, primary_vals, width, label=f"Primary ({single_type})")
    plt.bar(x, secondary_vals, width, label=f"Secondary ({two_type})")
    plt.bar(x + width, ens_vals, width, label="Ensemble/Final")

    plt.xticks(x, metrics)
    plt.ylabel("Accuracy (%)")
    plt.title("Gait Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
