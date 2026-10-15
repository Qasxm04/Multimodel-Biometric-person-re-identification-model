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
    num_joints = kp_xy.shape[0]

    if num_joints >= 25:
        neck = kp_xy[1]
        pelvis = kp_xy[8]
    else:
        neck = kp_xy[1]
        pelvis = (kp_xy[8] + kp_xy[11]) / 2.0

    center = (neck + pelvis) / 2.0
    scale = np.linalg.norm(neck - pelvis)

    if scale < 1e-6:
        min_xy = kp_xy.min(axis=0)
        max_xy = kp_xy.max(axis=0)
        scale = np.linalg.norm(max_xy - min_xy)

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
    elif mode == "random":
        start = random.randint(0, T - num_frames)
        return seq[start:start + num_frames]
    else:
        start = (T - num_frames) // 2
        return seq[start:start + num_frames]

def add_velocity_channels(seq_xy):
    vel = np.zeros_like(seq_xy, dtype=np.float32)
    vel[1:] = seq_xy[1:] - seq_xy[:-1]
    return np.concatenate([seq_xy, vel], axis=-1).astype(np.float32)  # [T,V,4]

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
            participant_path = os.path.join(root_folder, participant)

            experiments = sorted([
                exp for exp in os.listdir(participant_path)
                if os.path.isdir(os.path.join(participant_path, exp))
            ])

            for experiment in experiments:
                experiment_path = os.path.join(participant_path, experiment)

                trials = sorted([
                    t for t in os.listdir(experiment_path)
                    if os.path.isdir(os.path.join(experiment_path, t))
                ])

                for trial in trials:
                    trial_path = os.path.join(experiment_path, trial)

                    json_folders = [
                        f for f in os.listdir(trial_path)
                        if os.path.isdir(os.path.join(trial_path, f)) and f.endswith("_json")
                    ]

                    for json_folder in json_folders:
                        json_path = os.path.join(trial_path, json_folder)
                        json_files = sorted([
                            f for f in os.listdir(json_path)
                            if f.endswith("_keypoints.json")
                        ])

                        seq_xy = []

                        for jf in json_files:
                            with open(os.path.join(json_path, jf), "r") as file:
                                data = json.load(file)

                            if not data.get("people"):
                                continue

                            raw = np.array(data["people"][0]["pose_keypoints_2d"], dtype=np.float32)
                            if raw.size % 3 != 0:
                                continue

                            keypoints = raw.reshape(-1, 3)
                            kp_xy = keypoints[:, :2]

                            if self.num_joints is None:
                                self.num_joints = kp_xy.shape[0]
                                print(f"Detected {self.num_joints} joints per frame")

                            if kp_xy.shape[0] != self.num_joints:
                                continue

                            kp_xy = normalize_keypoints_xy(kp_xy)
                            seq_xy.append(kp_xy)

                        if len(seq_xy) < min_valid_frames:
                            continue

                        seq_xy = np.array(seq_xy, dtype=np.float32)
                        seq = add_velocity_channels(seq_xy)  # [T,V,4]

                        self.samples.append(torch.from_numpy(seq))
                        self.labels.append(idx)

        if self.num_joints is None:
            raise ValueError("Could not detect any valid keypoint frames.")
        if len(self.samples) == 0:
            raise ValueError("Dataset built 0 samples.")

        print(f"Dataset built: {len(self.samples)} samples, {len(self.participant_to_idx)} participants")
        print(f"Using {self.num_joints} joints and {self.in_channels} channels (x, y, vx, vy)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# =======================
# Collate
# =======================
def collate_fn_train(batch, num_frames=100):
    sequences, labels = zip(*batch)
    processed = []

    for seq in sequences:
        seq_np = seq.numpy()
        clip = temporal_crop(seq_np, num_frames=num_frames, mode="random")
        processed.append(torch.from_numpy(clip))

    X = torch.stack(processed, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    lengths = torch.full((len(labels),), num_frames, dtype=torch.long)
    return X, lengths, y

def collate_fn_eval(batch, num_frames=100, crop_mode="center"):
    sequences, labels = zip(*batch)
    processed = []

    for seq in sequences:
        seq_np = seq.numpy()
        clip = temporal_crop(seq_np, num_frames=num_frames, mode=crop_mode)
        processed.append(torch.from_numpy(clip))

    X = torch.stack(processed, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    lengths = torch.full((len(labels),), num_frames, dtype=torch.long)
    return X, lengths, y

# =======================
# Graph Conv
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
        x = torch.einsum("nctv,vw->nctw", x, A)
        return x

# =======================
# ST-GCN Block
# =======================
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

# =======================
# Model
# =======================
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

        x = x.view(B, T, V * C)
        x = x.permute(0, 2, 1)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1).contiguous().view(B, T, V, C)

        x = x.permute(0, 3, 1, 2)  # [B,C,T,V]

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
        x = self.fc(x)
        return x

# =======================
# Train / Eval
# =======================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, lengths, y in loader:
        X = X.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(X, lengths)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_with_multicrop(model, dataset_subset, device, num_frames=100, batch_size=8):
    model.eval()

    crop_modes = ["start", "center", "end"]
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

            for X, lengths, y in loader:
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
    _, pred = avg_scores.topk(10, dim=1, largest=True, sorted=True)

    y_expand = labels_ref.view(-1, 1)
    top1 = (pred[:, :1] == y_expand).sum().item()
    top5 = (pred[:, :5] == y_expand).sum().item()
    top10 = (pred[:, :10] == y_expand).sum().item()
    total = labels_ref.size(0)

    return 100.0 * top1 / total, 100.0 * top5 / total, 100.0 * top10 / total

# =======================
# Main
# =======================
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(os.path.dirname(script_dir), "Gait Dataset")
    dataset = GaitDataset(dataset_path, min_valid_frames=10)

    indices = list(range(len(dataset)))
    labels = dataset.labels

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate_fn_train(b, num_frames=100),
        num_workers=0
    )

    model = STGCNGait(
        num_classes=len(dataset.participant_to_idx),
        num_joints=dataset.num_joints,
        in_channels=dataset.in_channels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    epochs = 30

    best_top1 = 0.0
    best_top5 = 0.0
    best_top10 = 0.0
    best_epoch_top1 = 0
    best_epoch_top5 = 0
    best_epoch_top10 = 0

    print("\nStarting training...\n")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        top1, top5, top10 = evaluate_with_multicrop(
            model,
            test_subset,
            device,
            num_frames=100,
            batch_size=8
        )

        if top1 > best_top1:
            best_top1 = top1
            best_epoch_top1 = epoch + 1
            torch.save(model.state_dict(), "best_gait_stgcn.pt")

        if top5 > best_top5:
            best_top5 = top5
            best_epoch_top5 = epoch + 1

        if top10 > best_top10:
            best_top10 = top10
            best_epoch_top10 = epoch + 1

        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Top-1: {top1:5.2f}% | "
            f"Top-5: {top5:5.2f}% | "
            f"Top-10: {top10:5.2f}% | "
            f"Best Top-1: {best_top1:5.2f}% (Epoch {best_epoch_top1:02d})"
        )

    print("\n===== FINAL ST-GCN GAIT EVALUATION (TEST SET) =====")
    print(f"Participants: {len(dataset.participant_to_idx)}")
    print(f"Best Top-1 Accuracy: {best_top1:.2f}% (Epoch {best_epoch_top1})")
    print(f"Best Top-5 Accuracy: {best_top5:.2f}% (Epoch {best_epoch_top5})")
    print(f"Best Top-10 Accuracy: {best_top10:.2f}% (Epoch {best_epoch_top10})")
    print("Best model saved as: best_gait_stgcn.pt")

    metrics = ["Top-1", "Top-5", "Top-10"]
    values = [best_top1, best_top5, best_top10]

    plt.figure(figsize=(7, 5))
    plt.bar(metrics, values)
    plt.ylabel("Accuracy (%)")
    plt.title("Best ST-GCN Gait Evaluation Metrics")
    for i, v in enumerate(values):
        plt.text(i, v + 0.8, f"{v:.2f}%", ha="center")
    plt.ylim(0, max(values) + 10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
