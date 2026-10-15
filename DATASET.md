# Dataset guide

The source dataset is intentionally not committed. Biometric recordings are sensitive, and video collections are typically too large for a normal Git repository.

Place authorized local data under `face_id/Gait Dataset/`. The scripts infer participant identity and recording metadata from the existing project naming convention (for example, `P11_E3T1_Front.mp4`). Review `face_id/paired_index.py` before adapting a dataset with different names.

```text
face_id/
└── Gait Dataset/
    ├── P1/
    │   └── ... authorized recordings and pose data ...
    ├── P2/
    │   └── ...
    └── ...
```

Only use recordings for which you have permission. Do not commit raw participant media, derived embeddings, face databases, or identifying evaluation exports. The repository's `.gitignore` excludes these artifacts by default.

After placing the data:

1. Run `python face_id/paired_index.py` to generate the paired evaluation protocol.
2. Run the face and gait pipelines described in the main README.
3. Run `python Streamlit/build_face_database.py` if you want to use the demo.
