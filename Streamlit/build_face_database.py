import os
import cv2
import torch
import numpy as np
import pickle
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image

# ========================================
# DATASET PATH
# ========================================
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
dataset_path = BASE_DIR / "face_id" / "Gait Dataset"

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {dataset_path}. See README.md for the expected layout."
    )

# ========================================
# DEVICE
# ========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# ========================================
# MODELS
# ========================================
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=60,
    thresholds=[0.7, 0.8, 0.9],
    factor=0.709,
    post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ========================================
# DATABASE
# ========================================
database = []

# ========================================
# FACE EXTRACTION
# ========================================
def extract_face(frame):
    """
    Input frame is from OpenCV -> BGR
    Output face is also BGR
    """
    boxes, probs = mtcnn.detect(frame)
    if boxes is None or probs is None:
        return None

    best_face_idx = np.argmax(probs)
    if probs[best_face_idx] < 0.95:
        return None

    x1, y1, x2, y2 = boxes[best_face_idx].astype(int)
    h, w, _ = frame.shape

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    face = frame[y1:y2, x1:x2]
    if face is None or face.size == 0:
        return None

    face = cv2.resize(face, (160, 160))
    return face

# ========================================
# EMBEDDING
# ========================================
def get_embedding(face_rgb):
    """
    Input must be RGB NumPy array
    """
    pil_face = Image.fromarray(face_rgb)
    face_tensor = transforms.ToTensor()(pil_face).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(face_tensor)

    embedding = embedding.cpu().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

# ========================================
# SELECT BEST FACE FROM VIDEO
# ========================================
def select_best_face(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None, None

    step = max(1, total_frames // 6)
    best_face_bgr = None
    best_score = 0

    for i in range(6):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            continue

        face_bgr = extract_face(frame)
        if face_bgr is None:
            continue

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if score > best_score:
            best_score = score
            best_face_bgr = face_bgr

    cap.release()

    if best_face_bgr is None:
        return None, None

    # Convert ONCE from BGR -> RGB
    face_rgb = cv2.cvtColor(best_face_bgr, cv2.COLOR_BGR2RGB)

    embedding = get_embedding(face_rgb)
    return face_rgb, embedding

# ========================================
# BUILD DATABASE
# ========================================
print("\nBuilding face database...\n")
video_count = 0
face_count = 0

for participant in os.listdir(dataset_path):
    participant_path = os.path.join(dataset_path, participant)

    if not os.path.isdir(participant_path):
        continue

    print(f"Processing participant: {participant}")

    for root, dirs, files in os.walk(participant_path):
        for file in files:
            if not (file.lower().endswith(".mp4") or file.lower().endswith(".avi")):
                continue

            video_path = os.path.join(root, file)
            video_count += 1

            face_rgb, embedding = select_best_face(video_path)

            if face_rgb is not None and embedding is not None:
                database.append({
                    "participant": participant,
                    "embedding": embedding,
                    "face": face_rgb,          # stored as TRUE RGB
                    "source_video": file
                })
                face_count += 1

                print(
                    f"  Added face from {file} | "
                    f"Mean R,G,B: "
                    f"{face_rgb[:, :, 0].mean():.1f}, "
                    f"{face_rgb[:, :, 1].mean():.1f}, "
                    f"{face_rgb[:, :, 2].mean():.1f}"
                )

    print(f"Finished participant {participant}, total faces so far: {face_count}")

print("\nVideos scanned:", video_count)
print("Faces extracted:", face_count)
print("Database entries:", len(database))

# ========================================
# SAVE DATABASE
# ========================================
with (APP_DIR / "face_database.pkl").open("wb") as f:
    pickle.dump(database, f)

print("\nDatabase saved as face_database.pkl ✅ Stored as true RGB faces")
