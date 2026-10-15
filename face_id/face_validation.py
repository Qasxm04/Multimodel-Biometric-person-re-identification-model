import os
import random
import cv2
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================== MODELS ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 Improved detection robustness
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True,                      # detect multiple faces
    thresholds=[0.5, 0.6, 0.6],         # lower thresholds for small/blurred faces
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ================== PATH ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Gait Dataset")
PROGRESS_DIR = os.path.join(BASE_DIR, "progress")
os.makedirs(PROGRESS_DIR, exist_ok=True)

# ================== SETTINGS ==================
FRAME_SKIP = 3              # 🔥 better coverage (was 10)
MIN_FRAMES_REQUIRED = 5     # 🔥 minimum valid frames per video
RESIZE_SCALE = 1.3          # 🔥 enlarge frame to detect small faces

# ================== FACE FUNCTIONS ==================
def extract_face_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = mtcnn(rgb)

    if faces is not None:
        # If multiple faces detected, take the first one
        if len(faces.shape) == 4:
            return faces[0]
        return faces
    return None


def get_embedding(face_tensor):
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0)).cpu().numpy()
        emb = normalize(emb)
        return emb


def get_video_embedding(video_path):
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # 🔥 Resize frame to improve small face detection
        frame = cv2.resize(frame, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE)

        face = extract_face_from_frame(frame)
        if face is not None:
            emb = get_embedding(face)
            embeddings.append(emb)

    cap.release()

    # 🔥 Require minimum number of valid face frames
    if len(embeddings) < MIN_FRAMES_REQUIRED:
        return None

    # 🔥 Median pooling (more robust than mean)
    return np.median(embeddings, axis=0)


# ================== THREADING HELPER ==================
def process_test_video(video_tuple):
    person, video_path = video_tuple
    emb = get_video_embedding(video_path)
    return person, video_path, emb


# ================== MAIN ==================
if __name__ == "__main__":
    random.seed(42)

    print("Scanning dataset...")
    person_videos = {}
    VIDEO_EXTENSIONS = (".mp4", ".avi")

    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                full_path = os.path.join(root, file)
                parts = root.split(os.sep)
                participant_folder = [p for p in parts if p.startswith("Participant_")]
                if participant_folder:
                    person_id = participant_folder[0]
                    person_videos.setdefault(person_id, []).append(full_path)

    print(f"Found {len(person_videos)} identities total\n")

    gallery_embeddings = {}
    test_set = []

    print("Building gallery embeddings...\n")

    for person, videos in person_videos.items():

        if len(videos) < 30:
            print(f"{person} skipped (only {len(videos)} videos)")
            continue

        random.shuffle(videos)
        train_videos = videos[:24]
        test_videos = videos[24:30]

        gallery_file = os.path.join(PROGRESS_DIR, f"{person}_gallery.npy")

        if os.path.exists(gallery_file):
            gallery_embeddings[person] = list(np.load(gallery_file, allow_pickle=True))
            print(f"Loaded saved gallery embeddings for {person}")
        else:
            gallery_embeddings[person] = []

        for vid in train_videos[len(gallery_embeddings[person]):]:
            emb = get_video_embedding(vid)
            if emb is not None:
                gallery_embeddings[person].append(emb)
                np.save(gallery_file, np.array(gallery_embeddings[person], dtype=object))

        test_set.extend([(person, v) for v in test_videos])

        print(f"{person}: {len(gallery_embeddings[person])} gallery embeddings | 6 test videos")

    # ================== VALIDATION ==================
    print("\nRunning validation (multi-threaded, resumable)...\n")

    results_file = os.path.join(PROGRESS_DIR, "validation_results.npy")

    if os.path.exists(results_file):
        results = list(np.load(results_file, allow_pickle=True))
        processed_videos = {r['video_path'] for r in results}
        print(f"Loaded {len(results)} previously processed test results")
    else:
        results = []
        processed_videos = set()

    top1_correct = sum(1 for r in results if r["true_person"] == r["predicted_person"])
    top5_correct = sum(1 for r in results if r["true_person"] in r["top5"])
    top10_correct = sum(1 for r in results if r["true_person"] in r["top10"])

    remaining_test_set = [t for t in test_set if t[1] not in processed_videos]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_test_video, tup): tup for tup in remaining_test_set}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Test Videos"):

            true_person, video_path, test_emb = future.result()

            if test_emb is None:
                continue

            similarity_scores = {}

            for gallery_person, gallery_list in gallery_embeddings.items():
                if len(gallery_list) == 0:
                    continue

                best_similarity = max([
                    cosine_similarity(test_emb, g_emb)[0][0]
                    for g_emb in gallery_list
                ])

                similarity_scores[gallery_person] = best_similarity

            sorted_matches = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

            top1 = sorted_matches[0][0]
            top5 = [x[0] for x in sorted_matches[:5]]
            top10 = [x[0] for x in sorted_matches[:10]]

            if true_person == top1:
                top1_correct += 1
            if true_person in top5:
                top5_correct += 1
            if true_person in top10:
                top10_correct += 1

            result_entry = {
                "true_person": true_person,
                "video_path": video_path,
                "predicted_person": top1,
                "top5": top5,
                "top10": top10
            }

            results.append(result_entry)

            np.save(results_file, np.array(results, dtype=object))

            print(f"{os.path.basename(video_path)} → True: {true_person}, Top1: {top1}")

    # ================== METRICS ==================
    df = pd.DataFrame([{
        "true_person": r["true_person"],
        "predicted_person": r["predicted_person"]
    } for r in results])

    df.to_csv("face_validation_results_64.csv", index=False)

    y_true = df["true_person"]
    y_pred = df["predicted_person"]

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report_64.csv")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=sorted(y_true.unique()), columns=sorted(y_true.unique()))
    cm_df.to_csv("confusion_matrix_64.csv")

    total_tests = len(results)

    top1_acc = (top1_correct / total_tests) * 100
    top5_acc = (top5_correct / total_tests) * 100
    top10_acc = (top10_correct / total_tests) * 100

    print("\nValidation complete.")
    print(f"Total Tests: {total_tests}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Top-10 Accuracy: {top10_acc:.2f}%")
    print("All evaluation files saved.")
