import os
import ast
import cv2
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm

# ================== MODELS ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mtcnn = MTCNN(
    image_size=160,
    margin=10,
    keep_all=True,
    thresholds=[0.6, 0.7, 0.7],
    device=device
)

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ================== PATHS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROGRESS_DIR = os.path.join(BASE_DIR, "progress")
FUSION_DIR = os.path.join(BASE_DIR, "fusion_protocol")
os.makedirs(PROGRESS_DIR, exist_ok=True)

PAIRED_TRAIN_CSV = os.path.join(FUSION_DIR, "paired_train.csv")
PAIRED_TEST_CSV = os.path.join(FUSION_DIR, "paired_test.csv")

# ================== SETTINGS ==================
FRAME_SKIP = 6
MIN_FRAMES_REQUIRED = 3
RESIZE_SCALE = 1.0
MAX_FRAMES_PER_VIDEO = 60
SCORE_AGGREGATION = "mean"   # "mean" or "max"

# ================== FACE FUNCTIONS ==================
def extract_face_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = mtcnn(rgb)

    if faces is not None:
        if len(faces.shape) == 4:
            return faces[0]
        return faces
    return None


def get_embedding(face_tensor):
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()
        return normalize(emb)[0]


def get_video_embedding(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    embeddings = []
    frame_count = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        processed_frames += 1
        if processed_frames > MAX_FRAMES_PER_VIDEO:
            break

        if RESIZE_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)

        face = extract_face_from_frame(frame)
        if face is not None:
            emb = get_embedding(face)
            embeddings.append(emb)

    cap.release()

    if len(embeddings) < MIN_FRAMES_REQUIRED:
        return None

    return np.median(np.stack(embeddings, axis=0), axis=0)


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
        if not p:
            continue
        p = os.path.normpath(p)
        lower = p.lower()
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv")) and "_skeleton" not in lower:
            cleaned.append(p)

    return cleaned


def load_paired_split(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing paired split file: {csv_path}")

    df = pd.read_csv(csv_path)

    if "sample_key" not in df.columns or "participant" not in df.columns:
        raise ValueError(f"{csv_path} must contain sample_key and participant columns")

    if "face_video_paths" in df.columns:
        face_col = "face_video_paths"
    elif "face_video_path" in df.columns:
        face_col = "face_video_path"
    else:
        raise ValueError(f"{csv_path} must contain face_video_path or face_video_paths")

    rows = []
    for _, row in df.iterrows():
        face_paths = parse_list_cell(row[face_col])
        rows.append({
            "sample_key": str(row["sample_key"]),
            "participant": str(row["participant"]),
            "face_video_paths": face_paths
        })

    return rows


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
            "video_path": "MULTI_FACE_SAMPLE",
            "true_person": r["true_person"],
            "predicted_person": r["predicted_person"],
            "pred_top1_score": r["top1_score"]
        }

        for k in range(10):
            row[f"rank_{k+1}_name"] = r["top10_names"][k] if k < len(r["top10_names"]) else ""
            row[f"rank_{k+1}_score"] = r["top10_scores"][k] if k < len(r["top10_scores"]) else ""

        detailed_rows.append(row)

    face_fusion_path = os.path.join(BASE_DIR, "face_fusion_results_sample_level.csv")
    pd.DataFrame(detailed_rows).to_csv(face_fusion_path, index=False)

    matrix_rows = []
    for i, r in enumerate(results):
        row = {
            "sample_index": i,
            "sample_key": r["sample_key"],
            "video_path": "MULTI_FACE_SAMPLE",
            "true_person": r["true_person"],
            "predicted_person": r["predicted_person"]
        }

        for person in all_identities:
            row[f"score_{person}"] = r["score_vector"].get(person, np.nan)

        matrix_rows.append(row)

    face_score_matrix_path = os.path.join(BASE_DIR, "face_score_matrix_sample_level.csv")
    pd.DataFrame(matrix_rows).to_csv(face_score_matrix_path, index=False)

    print(f"✅ Saved fusion-ready file: {face_fusion_path}")
    print(f"✅ Saved full score matrix: {face_score_matrix_path}")


# ================== MAIN ==================
if __name__ == "__main__":
    print("Loading paired protocol...")
    paired_train = load_paired_split(PAIRED_TRAIN_CSV)
    paired_test = load_paired_split(PAIRED_TEST_CSV)

    print(f"Paired train samples: {len(paired_train)}")
    print(f"Paired test samples: {len(paired_test)}")

    # -------- Build gallery --------
    print("\nBuilding gallery embeddings from paired train split...\n")
    gallery_embeddings = {}
    gallery_cache_dir = os.path.join(PROGRESS_DIR, "paired_gallery_cache_sample_level")
    os.makedirs(gallery_cache_dir, exist_ok=True)

    total_gallery_videos = sum(len(r["face_video_paths"]) for r in paired_train)
    built_count = 0
    cache_count = 0
    fail_count = 0

    pbar = tqdm(total=total_gallery_videos, desc="Gallery videos")

    for row in paired_train:
        participant = row["participant"]
        sample_key = row["sample_key"]
        face_paths = row["face_video_paths"]

        if participant not in gallery_embeddings:
            gallery_embeddings[participant] = []

        for face_video_path in face_paths:
            pbar.update(1)

            if not os.path.exists(face_video_path):
                fail_count += 1
                continue

            video_stem = os.path.splitext(os.path.basename(face_video_path))[0]
            cache_name = f"{sample_key}__{video_stem}.npy"
            cache_path = os.path.join(gallery_cache_dir, cache_name)

            emb = None
            if os.path.exists(cache_path):
                try:
                    emb = np.load(cache_path, allow_pickle=True)
                    cache_count += 1
                except Exception:
                    emb = None

            if emb is None:
                emb = get_video_embedding(face_video_path)
                if emb is not None:
                    np.save(cache_path, emb)
                    built_count += 1
                else:
                    fail_count += 1

            if emb is not None:
                gallery_embeddings[participant].append(emb)

            pbar.set_postfix({
                "built": built_count,
                "cache": cache_count,
                "fail": fail_count
            })

    pbar.close()

    valid_identities = sorted([p for p, embs in gallery_embeddings.items() if len(embs) > 0])

    if len(valid_identities) == 0:
        raise RuntimeError("No valid gallery identities were built.")

    print(f"\nValid gallery identities: {len(valid_identities)}")

    # -------- Sample-level test evaluation --------
    results = []

    print("\nRunning sample-level paired face validation...\n")
    for row in tqdm(paired_test, desc="Processing paired face samples"):
        true_person = row["participant"]
        sample_key = row["sample_key"]
        face_paths = row["face_video_paths"]

        sample_score_vectors = []

        for face_video_path in face_paths:
            if not os.path.exists(face_video_path):
                continue

            test_emb = get_video_embedding(face_video_path)
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

        fused_vector = aggregate_score_vectors(sample_score_vectors, mode=SCORE_AGGREGATION)
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
            "video_path": "MULTI_FACE_SAMPLE",
            "predicted_person": top1,
            "top1_score": top1_score,
            "top5": top5,
            "top10": top10,
            "top10_names": top10,
            "top10_scores": top10_scores,
            "score_vector": score_vector
        })

    if len(results) == 0:
        raise RuntimeError("No paired face sample-level results were produced.")

    # -------- Metrics --------
    df = pd.DataFrame([{
        "sample_key": r["sample_key"],
        "true_person": r["true_person"],
        "predicted_person": r["predicted_person"]
    } for r in results])

    df.to_csv(os.path.join(BASE_DIR, "face_validation_results_paired_sample_level.csv"), index=False)

    y_true = df["true_person"]
    y_pred = df["predicted_person"]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(BASE_DIR, "classification_report_face_paired_sample_level.csv")
    )

    labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(BASE_DIR, "confusion_matrix_face_paired_sample_level.csv")
    )

    total_tests = len(results)
    top1_acc = 100.0 * sum(r["true_person"] == r["predicted_person"] for r in results) / total_tests
    top5_acc = 100.0 * sum(r["true_person"] in r["top5"] for r in results) / total_tests
    top10_acc = 100.0 * sum(r["true_person"] in r["top10"] for r in results) / total_tests

    print("\n===== FINAL SAMPLE-LEVEL PAIRED FACE RESULTS =====")
    print(f"Total paired face samples: {total_tests}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Top-10 Accuracy: {top10_acc:.2f}%")

    save_fusion_outputs(results, valid_identities)
    print("All paired face sample-level evaluation files saved.")