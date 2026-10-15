import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torch

# Load models
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    return face

def get_embedding(face_tensor):
    return resnet(face_tensor.unsqueeze(0)).detach().numpy()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    embeddings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face = extract_face_from_frame(frame)
        if face is not None:
            emb = get_embedding(face)
            embeddings.append(emb)

    cap.release()
    return embeddings

def get_video_embedding(video_path):
    embeddings = process_video(video_path)

    if len(embeddings) == 0:
        return None

    # Average embedding across frames
    return np.mean(embeddings, axis=0)

def distance_to_percentage(distance):
    return max(0, 100 * (1 - distance))

# ---------------- MAIN ----------------
if __name__ == "__main__":

    # 🔹 Probe video (changed to P11)
    probe_video = "P11_E3T1_Front.mp4"

    # 🔹 Gallery of 5 participants
    participants = {
        "P1":  "P1_E3T1_Front.mp4",
        "P2":  "P2_E3T1_Front.mp4",
        "P11": "P11_E3T1_Front.mp4",
        "P13": "P13_E3T1_Front.mp4",
        "P22": "P22_E3T1_Front.mp4",
    }

    print("Processing probe video...")
    probe_embedding = get_video_embedding(probe_video)

    if probe_embedding is None:
        print("No face detected in probe video.")
        exit()

    results = []

    print("\nComparing probe against 5 participants:\n")

    for name, video in participants.items():
        candidate_embedding = get_video_embedding(video)

        if candidate_embedding is None:
            print(f"{name}: No face detected")
            continue

        distance = np.linalg.norm(probe_embedding - candidate_embedding)
        similarity = distance_to_percentage(distance)

        results.append((name, similarity))
        print(f"{name} → Similarity: {similarity:.2f}%")

    # 🔹 Rank participants (highest similarity first)
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n===== FINAL IDENTIFICATION RESULT =====")
    best_match = results[0]

    print(f"Best Match: {best_match[0]}")
    print(f"Similarity: {best_match[1]:.2f}%")

    if best_match[1] > 70:
        print("DECISION: MATCH FOUND")
    else:
        print("DECISION: NO CONFIDENT MATCH")

