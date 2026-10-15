import streamlit as st
import cv2
import torch
import numpy as np
import pickle
import os
import tempfile
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image, ImageEnhance
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# STREAMLIT PAGE SETTINGS
# =============================
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 Face Recognition with Confidence Analysis")

# =============================
# DEVICE
# =============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write("Using device:", device)

# =============================
# MODELS
# =============================
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

# =============================
# LOAD DATABASE (TRUE RGB FACES)
# =============================
APP_DIR = Path(__file__).resolve().parent
DATABASE_PATH = APP_DIR / "face_database.pkl"

if not DATABASE_PATH.exists():
    st.error(
        "Face database not found. Run `python Streamlit/build_face_database.py` "
        "from the repository root, then restart this app."
    )
    st.stop()

with DATABASE_PATH.open("rb") as f:
    database = pickle.load(f)
st.write(f"Database loaded: {len(database)} face samples")

# =============================
# FACE EXTRACTION
# =============================
def extract_face(frame):
    """
    Input frame from OpenCV is BGR
    Output face is also BGR
    """
    boxes, probs = mtcnn.detect(frame)
    if boxes is None or probs is None:
        return None

    best = np.argmax(probs)
    if probs[best] < 0.95:
        return None

    x1, y1, x2, y2 = boxes[best].astype(int)
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

# =============================
# ENHANCEMENT
# =============================
def enhance_image(face):
    """
    Accepts RGB NumPy array or PIL.Image
    Returns enhanced RGB PIL Image
    """
    if isinstance(face, np.ndarray):
        img = Image.fromarray(face)
    elif isinstance(face, Image.Image):
        img = face.copy()
    else:
        raise ValueError("enhance_image expects np.ndarray or PIL.Image")

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp_factor = 2.0 if sharpness > 100 else min(4.0, 1000 / max(sharpness, 1e-6))
    img = ImageEnhance.Sharpness(img).enhance(sharp_factor)

    mean_color = np.mean(np.array(img) / 255.0)
    color_factor = 1.4 if mean_color > 0.4 else min(2.0, 0.8 / max(mean_color, 1e-6))
    img = ImageEnhance.Color(img).enhance(color_factor)

    return img

# =============================
# EMBEDDING
# =============================
def get_embedding(face_rgb):
    """
    Input must be RGB image
    """
    face_np = np.array(face_rgb)
    tensor = transforms.ToTensor()(face_np).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(tensor)

    embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)
    return embedding.cpu().numpy()[0]

# =============================
# SELECT BEST VIDEO FRAME
# =============================
def select_best_face(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return None, None

    step = max(1, total // 6)
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

    if best_face_bgr is not None:
        face_rgb = cv2.cvtColor(best_face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = enhance_image(face_rgb)
        return face_rgb, get_embedding(face_rgb)

    return None, None

# =============================
# FACE MATCHING
# =============================
def match_face(embedding):
    participant_best = {}

    for entry in database:
        participant = entry["participant"]
        dist = np.linalg.norm(embedding - entry["embedding"])

        if participant not in participant_best or dist < participant_best[participant][0]:
            participant_best[participant] = (
                dist,
                entry["face"],   # already RGB
                entry.get("source_video", "")
            )

    results = []
    for p, (d, f, v) in participant_best.items():
        results.append((p, d, f, v))

    results.sort(key=lambda x: x[1])
    max_dist = max(r[1] for r in results) + 1e-6

    results = [(p, 1 - d / max_dist, enhance_image(f), v) for p, d, f, v in results]
    return results[:3]

# =============================
# SELECT DIFFERENT VIDEO FACE
# =============================
def get_different_video_face(participant, current_video):
    for entry in database:
        if entry["participant"] == participant and entry.get("source_video", "") != current_video:
            return entry["face"]  # already RGB
    return None

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi"]
)

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)  # BGR
        face_bgr = extract_face(frame)

        if face_bgr is None:
            st.error("No face detected")
            st.stop()

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = enhance_image(face_rgb)
        embedding = get_embedding(face)
        source_video = "uploaded_image"

    else:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.read())
            temp = temp_file.name

        try:
            face, embedding = select_best_face(temp)
        finally:
            os.unlink(temp)

        if face is None:
            st.error("No face detected in video")
            st.stop()

        source_video = uploaded_file.name

    # =============================
    # MATCHING
    # =============================
    top3 = match_face(embedding)
    predicted_name = top3[0][0]
    predicted_conf = top3[0][1] * 100

    predicted_face = get_different_video_face(predicted_name, top3[0][3])
    if predicted_face is None:
        predicted_face = top3[0][2]
    else:
        predicted_face = enhance_image(predicted_face)

    # =============================
    # METRIC DISPLAY
    # =============================
    st.metric(
        label="Predicted Identity",
        value=predicted_name,
        delta=f"{predicted_conf:.2f}% confidence"
    )

    # =============================
    # IMAGE COMPARISON
    # =============================
    cols = st.columns(2)
    cols[0].image(face, caption="Uploaded Face", width=300)
    cols[1].image(predicted_face, caption="Reference Face (Different Video)", width=300)

    # =============================
    # TOP 3 GRAPH
    # =============================
    st.subheader("Top 3 Closest Participants")
    df = pd.DataFrame({
        "Participant": [p for p, c, f, v in top3],
        "Confidence": [c for p, c, f, v in top3]
    }).set_index("Participant")
    st.bar_chart(df)

    # =============================
    # PIE CHART CONFIDENCE
    # =============================
    st.subheader("Overall Prediction Confidence")
    match_conf = predicted_conf
    other_conf = 100 - match_conf

    fig, ax = plt.subplots()
    ax.pie(
        [match_conf, other_conf],
        labels=["Predicted Match", "Other Participants"],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis("equal")
    st.pyplot(fig)
