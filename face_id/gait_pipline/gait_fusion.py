import os
import ast
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ================== PATHS ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PROGRESS_DIR = os.path.join(PROJECT_DIR, "progress")
FUSION_DIR = os.path.join(PROJECT_DIR, "fusion_protocol")
os.makedirs(PROGRESS_DIR, exist_ok=True)

PAIRED_TRAIN_CSV = os.path.join(FUSION_DIR, "paired_train.csv")
PAIRED_TEST_CSV = os.path.join(FUSION_DIR, "paired_test.csv")

# ================== SETTINGS ==================
TARGET_FRAMES = 60
MIN_FRAMES_REQUIRED = 10
MIN_VALID_JOINTS = 8

NECK_IDX = 1
MIDHIP_IDX = 8
RANKLE_IDX = 11
LANKLE_IDX = 14

# ================== HELPERS ==================
def parse_list_cell(cell):
    if pd.isna(cell):
        return []

    if isinstance(cell, list):
        values = cell
    else:
        text = str(cell).strip()
        if not text:
            return []
        try:
            values = ast.literal_eval(text)
        except Exception:
            values = [x.strip() for x in text.split("||") if x.strip()]

    if not isinstance(values, list):
        values = [values]

    cleaned = []
    for v in values:
        p = str(v).strip().strip('"').strip("'")
        if p:
            cleaned.append(os.path.normpath(p))
    return cleaned


def load_paired_split(csv_path):
    df = pd.read_csv(csv_path)

    if "sample_key" not in df.columns or "participant" not in df.columns:
        raise ValueError(f"{csv_path} must contain sample_key and participant columns")

    if "gait_json_path" not in df.columns:
        raise ValueError(f"{csv_path} must contain gait_json_path column")

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "sample_key": str(row["sample_key"]),
            "participant": str(row["participant"]),
            "gait_paths": parse_list_cell(row["gait_json_path"])
        })
    return rows


# ================== JSON LOADING ==================
def load_openpose_sequence_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        return None

    json_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith("_keypoints.json")
    ])

    if len(json_files) == 0:
        return None

    frames = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        people = data.get("people", [])
        if len(people) == 0:
            continue

        pose = people[0].get("pose_keypoints_2d", None)
        if pose is None or len(pose) % 3 != 0:
            continue

        arr = np.array(pose, dtype=np.float32).reshape(-1, 3)
        xy = arr[:, :2]
        conf = arr[:, 2]
        xy[conf < 0.05] = 0.0
        frames.append(xy)

    if len(frames) < MIN_FRAMES_REQUIRED:
        return None

    return np.stack(frames, axis=0)


# ================== PREPROCESSING ==================
def count_valid_joints_per_frame(seq):
    valid = np.any(seq != 0, axis=2)
    return np.sum(valid, axis=1)


def fill_missing_joints_temporally(seq):
    seq = seq.copy()
    T, V, C = seq.shape

    for v in range(V):
        for c in range(C):
            series = seq[:, v, c]
            valid_idx = np.where(series != 0)[0]
            if len(valid_idx) == 0:
                continue

            for t in range(T):
                if series[t] == 0:
                    nearest = valid_idx[np.argmin(np.abs(valid_idx - t))]
                    series[t] = series[nearest]

            seq[:, v, c] = series

    return seq


def normalize_sequence(seq):
    normed_frames = []

    for frame in seq:
        if frame.shape[0] <= max(NECK_IDX, MIDHIP_IDX):
            continue

        neck = frame[NECK_IDX]
        midhip = frame[MIDHIP_IDX]

        if np.allclose(neck, 0) or np.allclose(midhip, 0):
            continue

        scale = np.linalg.norm(neck - midhip)
        if scale < 1e-6:
            continue

        normed_frames.append((frame - midhip) / scale)

    if len(normed_frames) < MIN_FRAMES_REQUIRED:
        return None

    return np.stack(normed_frames, axis=0)


def resample_sequence(seq, target_frames=60):
    T, V, C = seq.shape
    if T == target_frames:
        return seq

    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_frames)

    out = np.zeros((target_frames, V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(new_idx, old_idx, seq[:, v, c])

    return out


# ================== FEATURE EXTRACTION ==================
def compute_gait_embedding_from_sequence(seq):
    if seq is None:
        return None

    valid_counts = count_valid_joints_per_frame(seq)
    seq = seq[valid_counts >= MIN_VALID_JOINTS]

    if len(seq) < MIN_FRAMES_REQUIRED:
        return None

    seq = fill_missing_joints_temporally(seq)
    seq = normalize_sequence(seq)
    if seq is None:
        return None

    seq = resample_sequence(seq, TARGET_FRAMES)

    vel = np.diff(seq, axis=0, prepend=seq[0:1])
    acc = np.diff(vel, axis=0, prepend=vel[0:1])

    features = [
        seq.mean(axis=0).reshape(-1),
        seq.std(axis=0).reshape(-1),
        vel.mean(axis=0).reshape(-1),
        vel.std(axis=0).reshape(-1),
        acc.mean(axis=0).reshape(-1),
        acc.std(axis=0).reshape(-1),
    ]

    if seq.shape[1] > max(LANKLE_IDX, RANKLE_IDX):
        lankle = seq[:, LANKLE_IDX]
        rankle = seq[:, RANKLE_IDX]
        ankle_dist = np.linalg.norm(lankle - rankle, axis=1)
        features.append(np.array([
            ankle_dist.mean(),
            ankle_dist.std(),
            ankle_dist.max(),
            ankle_dist.min()
        ], dtype=np.float32))

    emb = np.concatenate(features).astype(np.float32)
    emb = normalize(emb.reshape(1, -1))[0]
    return emb


def get_gait_embedding(gait_folder_path):
    seq = load_openpose_sequence_from_folder(gait_folder_path)
    return compute_gait_embedding_from_sequence(seq)


def aggregate_score_vectors(score_vectors, mode="mean"):
    arr = np.array(score_vectors, dtype=np.float32)
    if mode == "max":
        return np.max(arr, axis=0)
    return np.mean(arr, axis=0)


def save_fusion_outputs(results, all_identities):
    detailed_rows = []
    for i, r in enumerate(results):
        row = {
            "sample_index": i,
            "sample_key": r["sample_key"],
            "video_path": "MULTI_GAIT_SAMPLE",
            "true_person": r["true_person"],
            "predicted_person": r["predicted_person"],
            "pred_top1_score": r["top1_score"]
        }

        for k in range(10):
            row[f"rank_{k+1}_name"] = r["top10_names"][k] if k < len(r["top10_names"]) else ""
            row[f"rank_{k+1}_score"] = r["top10_scores"][k] if k < len(r["top10_scores"]) else ""

        detailed_rows.append(row)

    gait_fusion_path = os.path.join(PROJECT_DIR, "gait_fusion_results_sample_level.csv")
    pd.DataFrame(detailed_rows).to_csv(gait_fusion_path, index=False)

    matrix_rows = []
    for i, r in enumerate(results):
        row = {
            "sample_index": i,
            "sample_key": r["sample_key"],
            "video_path": "MULTI_GAIT_SAMPLE",
            "true_person": r["true_person"],
            "predicted_person": r["predicted_person"]
        }

        for person in all_identities:
            row[f"score_{person}"] = r["score_vector"].get(person, np.nan)

        matrix_rows.append(row)

    gait_score_matrix_path = os.path.join(PROJECT_DIR, "gait_score_matrix_sample_level.csv")
    pd.DataFrame(matrix_rows).to_csv(gait_score_matrix_path, index=False)

    print(f"✅ Saved fusion-ready file: {gait_fusion_path}")
    print(f"✅ Saved full score matrix: {gait_score_matrix_path}")


# ================== MAIN ==================
if __name__ == "__main__":
    print("Loading paired protocol...")
    paired_train = load_paired_split(PAIRED_TRAIN_CSV)
    paired_test = load_paired_split(PAIRED_TEST_CSV)

    print(f"Paired train samples: {len(paired_train)}")
    print(f"Paired test samples: {len(paired_test)}")

    print("\nBuilding gait gallery embeddings from paired train split...\n")
    gallery_embeddings = {}
    gallery_cache_dir = os.path.join(PROGRESS_DIR, "paired_gait_gallery_cache_sample_level")
    os.makedirs(gallery_cache_dir, exist_ok=True)

    for row_idx, row in enumerate(paired_train, start=1):
        participant = row["participant"]
        sample_key = row["sample_key"]
        gait_paths = row["gait_paths"]

        if participant not in gallery_embeddings:
            gallery_embeddings[participant] = []

        if row_idx <= 10 or row_idx % 100 == 0:
            print(f"[Gallery {row_idx}/{len(paired_train)}] {sample_key} | gait folders: {len(gait_paths)}")

        for gait_path in gait_paths:
            if not os.path.exists(gait_path):
                continue

            safe_name = os.path.basename(gait_path).replace("\\", "_").replace("/", "_")
            cache_name = f"{sample_key}__{safe_name}.npy"
            cache_path = os.path.join(gallery_cache_dir, cache_name)

            emb = None
            if os.path.exists(cache_path):
                try:
                    emb = np.load(cache_path, allow_pickle=True)
                except Exception:
                    emb = None

            if emb is None:
                emb = get_gait_embedding(gait_path)
                if emb is not None:
                    np.save(cache_path, emb)

            if emb is not None:
                gallery_embeddings[participant].append(emb)

    valid_identities = sorted([p for p, embs in gallery_embeddings.items() if len(embs) > 0])

    if len(valid_identities) == 0:
        raise RuntimeError("No valid gait gallery identities were built.")

    print(f"\nValid gallery identities: {len(valid_identities)}")

    print("\nRunning sample-level paired gait validation...\n")
    results = []

    for row in tqdm(paired_test, desc="Processing paired gait samples"):
        true_person = row["participant"]
        sample_key = row["sample_key"]
        gait_paths = row["gait_paths"]

        sample_score_vectors = []

        for gait_path in gait_paths:
            if not os.path.exists(gait_path):
                continue

            test_emb = get_gait_embedding(gait_path)
            if test_emb is None:
                continue

            similarity_scores = {}
            for gallery_person, gallery_list in gallery_embeddings.items():
                if len(gallery_list) == 0:
                    continue

                best_similarity = max(
                    cosine_similarity(test_emb.reshape(1, -1), g_emb.reshape(1, -1))[0][0]
                    for g_emb in gallery_list
                )
                similarity_scores[gallery_person] = float(best_similarity)

            if len(similarity_scores) == 0:
                continue

            sample_score_vectors.append([similarity_scores[p] for p in valid_identities])

        if len(sample_score_vectors) == 0:
            continue

        fused_vector = aggregate_score_vectors(sample_score_vectors, mode="mean")
        score_vector = {p: float(s) for p, s in zip(valid_identities, fused_vector)}

        sorted_matches = sorted(score_vector.items(), key=lambda x: x[1], reverse=True)
        top1 = sorted_matches[0][0]
        top1_score = sorted_matches[0][1]
        top5 = [x[0] for x in sorted_matches[:5]]
        top10 = [x[0] for x in sorted_matches[:10]]
        top10_scores = [float(x[1]) for x in sorted_matches[:10]]

        results.append({
            "sample_key": sample_key,
            "true_person": true_person,
            "video_path": "MULTI_GAIT_SAMPLE",
            "predicted_person": top1,
            "top1_score": top1_score,
            "top5": top5,
            "top10": top10,
            "top10_names": top10,
            "top10_scores": top10_scores,
            "score_vector": score_vector
        })

    if len(results) == 0:
        raise RuntimeError("No paired gait sample-level results were produced.")

    df = pd.DataFrame([{
        "sample_key": r["sample_key"],
        "true_person": r["true_person"],
        "predicted_person": r["predicted_person"]
    } for r in results])

    df.to_csv(os.path.join(PROJECT_DIR, "gait_validation_results_paired_sample_level.csv"), index=False)

    y_true = df["true_person"]
    y_pred = df["predicted_person"]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(PROJECT_DIR, "classification_report_gait_paired_sample_level.csv")
    )

    labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(PROJECT_DIR, "confusion_matrix_gait_paired_sample_level.csv")
    )

    total_tests = len(results)
    top1_acc = 100.0 * sum(r["true_person"] == r["predicted_person"] for r in results) / total_tests
    top5_acc = 100.0 * sum(r["true_person"] in r["top5"] for r in results) / total_tests
    top10_acc = 100.0 * sum(r["true_person"] in r["top10"] for r in results) / total_tests

    print("\n===== FINAL SAMPLE-LEVEL PAIRED GAIT RESULTS =====")
    print(f"Total paired gait samples: {total_tests}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Top-10 Accuracy: {top10_acc:.2f}%")

    save_fusion_outputs(results, valid_identities)
    print("All paired gait sample-level evaluation files saved.")