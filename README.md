# Multimodal Biometric Person Re-identification

An end-to-end research implementation for identifying people from **face** and **gait** signals. The project evaluates each modality independently, then combines their similarity scores to improve person re-identification across video recordings.

> Portfolio note: the repository contains the complete source pipeline. Raw participant videos, biometric embeddings, generated databases, and trained checkpoints are excluded to protect participant data and keep the repository lightweight.

## What this project demonstrates

- Face detection and 512-dimensional face embeddings with MTCNN and FaceNet (`InceptionResnetV1`, VGGFace2 weights)
- Skeleton-sequence gait modelling with a spatial-temporal graph convolutional network (ST-GCN)
- Gallery/probe evaluation, cosine-similarity scoring, ranking, and classification metrics
- Score-level fusion of face and gait predictions
- A Streamlit interface for uploading an image or video and inspecting the top identity matches
- Reproducible data boundaries: sensitive and generated artifacts are separated from source code

## System overview

```text
                         ┌────────────────────┐
Input recording ────────►│ Face extraction    │──► FaceNet embedding ──┐
                         └────────────────────┘                         │
                                                                        ├─► Score normalization
                         ┌────────────────────┐                         │   and weighted fusion
Pose sequence ──────────►│ ST-GCN gait model │──► Gait embedding ─────┘          │
                                                                                  ▼
                                                                       Ranked identity match
```

## Repository structure

```text
.
├── face_id/
│   ├── face_validation.py               # Face evaluation pipeline
│   ├── face_fusion.py                   # Face score aggregation
│   ├── bridge_face_to_gait_and_fuse.py  # Multimodal score fusion
│   ├── paired_index.py                  # Reproducible gallery/probe pairs
│   └── gait_pipline/
│       ├── train_gait_stgcn.py          # ST-GCN training and evaluation
│       ├── gait_validation.py           # Gait validation pipeline
│       ├── gait_fusion.py               # Gait score aggregation
│       └── build_gait_database.py       # Gait feature database builder
├── Streamlit/
│   ├── App.py                           # Interactive face-matching demo
│   └── build_face_database.py           # Local face database builder
├── DATASET.md                            # Private dataset setup and safety notes
└── requirements.txt
```

The historical folder name `gait_pipline` is retained so existing experiment paths remain compatible.

## Quick start

Python 3.10 or 3.11 is recommended. A CUDA-capable GPU speeds up training and inference, but the face pipeline automatically falls back to CPU.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pretrained FaceNet weights are downloaded by `facenet-pytorch` on first use.

## Data preparation

The scripts expect authorized local data in:

```text
face_id/Gait Dataset/
```

See [DATASET.md](DATASET.md) for the expected layout and privacy guidance. Generate the paired gallery/probe protocol with:

```bash
python face_id/paired_index.py
```

## Run the pipelines

### Face evaluation

```bash
python face_id/face_validation.py
python face_id/face_fusion.py
```

### Gait evaluation

```bash
python face_id/gait_pipline/train_gait_stgcn.py
python face_id/gait_pipline/gait_validation.py
python face_id/gait_pipline/gait_fusion.py
```

### Multimodal fusion

Run the face and gait evaluations first so their score files exist, then run:

```bash
python face_id/bridge_face_to_gait_and_fuse.py
```

The fusion script aligns the face and gait trials, normalizes their scores, applies weighted score fusion, and produces identification metrics and visualizations.

## Streamlit demo

Build the local face database from the authorized dataset, then launch the app:

```bash
python Streamlit/build_face_database.py
streamlit run Streamlit/App.py
```

The generated `Streamlit/face_database.pkl` remains local and is excluded from version control.

## Outputs

Depending on the pipeline, generated artifacts include:

- Per-trial similarity scores and ranked predictions
- Overall and per-class evaluation metrics
- Confusion matrices and fusion comparison plots
- Trained gait checkpoints and cached embeddings

These outputs are reproducible and ignored by Git. Model checkpoints can be distributed separately as GitHub Release assets when their training data and licensing permit it.

## Responsible use

Biometric identification can affect privacy and civil rights. This repository is intended for authorized research and portfolio review—not covert identification, surveillance, or consequential automated decision-making. Evaluate accuracy across relevant demographic and environmental conditions, obtain informed consent, minimize retention, and comply with applicable law and institutional policy.

## Current limitations

- The project depends on a private, consented dataset and therefore does not include one-command sample results.
- Accuracy depends on video quality, pose coverage, and the match between training and deployment conditions.
- The Streamlit demo currently presents the face-recognition path; gait and multimodal fusion are evaluated through the research scripts.
- Thresholds and fusion weights should be calibrated on a validation split before deployment in a new environment.

## Technology stack

Python · PyTorch · facenet-pytorch · OpenCV · NumPy · pandas · scikit-learn · Streamlit · Matplotlib
