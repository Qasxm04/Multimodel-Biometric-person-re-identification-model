import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ================== PATHS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_INPUT = os.path.join(BASE_DIR, "face_score_matrix.csv")
GAIT_INPUT = os.path.join(BASE_DIR, "gait_score_matrix_sample_level.csv")

FACE_COLLAPSED_OUTPUT = os.path.join(BASE_DIR, "face_score_matrix_sample_level_from_existing.csv")
FACE_COLLAPSED_RESULTS_OUTPUT = os.path.join(BASE_DIR, "face_fusion_results_sample_level_from_existing.csv")

FUSED_SCORE_OUTPUT = os.path.join(BASE_DIR, "fused_score_matrix_sample_level.csv")
FUSED_RESULTS_OUTPUT = os.path.join(BASE_DIR, "fusion_results_sample_level.csv")
FUSED_REPORT_OUTPUT = os.path.join(BASE_DIR, "classification_report_fused_sample_level.csv")
FUSED_CM_OUTPUT = os.path.join(BASE_DIR, "confusion_matrix_fused_sample_level.csv")
BAR_CHART_OUTPUT = os.path.join(BASE_DIR, "fusion_accuracy_bar_chart.png")

# ================== SETTINGS ==================
FACE_WEIGHT = 0.7
GAIT_WEIGHT = 0.3

# ================== HELPERS ==================
def get_score_columns(df):
    return [c for c in df.columns if c.startswith("score_")]


def topk_names_scores(row_scores, k=10):
    sorted_scores = row_scores.sort_values(ascending=False)
    top_names = [c.replace("score_", "") for c in sorted_scores.index[:k]]
    top_scores = [float(v) for v in sorted_scores.iloc[:k].values]
    return top_names, top_scores


def minmax_row_normalize(df_scores):
    arr = df_scores.to_numpy(dtype=np.float32)
    row_min = arr.min(axis=1, keepdims=True)
    row_max = arr.max(axis=1, keepdims=True)
    denom = row_max - row_min
    denom[denom == 0] = 1.0
    norm = (arr - row_min) / denom
    return pd.DataFrame(norm, columns=df_scores.columns, index=df_scores.index)


def save_accuracy_bar_chart(top1_acc, top5_acc, top10_acc, output_path):
    labels = ["Top-1", "Top-5", "Top-10"]
    values = [top1_acc, top5_acc, top10_acc]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    plt.title("Score-Level Fusion Accuracy", pad=20)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.3,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# ================== LOAD ==================
if not os.path.exists(FACE_INPUT):
    raise FileNotFoundError(f"Missing face file: {FACE_INPUT}")

if not os.path.exists(GAIT_INPUT):
    raise FileNotFoundError(f"Missing gait file: {GAIT_INPUT}")

face_df = pd.read_csv(FACE_INPUT)
gait_df = pd.read_csv(GAIT_INPUT)

print("===== INPUT SUMMARY =====")
print(f"Face rows: {len(face_df)}")
print(f"Gait rows: {len(gait_df)}")

face_score_cols = get_score_columns(face_df)
gait_score_cols = get_score_columns(gait_df)

shared_score_cols = sorted(set(face_score_cols).intersection(set(gait_score_cols)))
if len(shared_score_cols) == 0:
    raise RuntimeError("No shared score columns found between face and gait.")

print(f"Shared score columns: {len(shared_score_cols)}")

# ================== COLLAPSE FACE TO SAMPLE LEVEL ==================
collapsed_rows = []
collapsed_result_rows = []

face_df[shared_score_cols] = face_df[shared_score_cols].apply(pd.to_numeric, errors="coerce")

for i, (sample_key, group) in enumerate(face_df.groupby("sample_key", sort=True)):
    true_people = group["true_person"].dropna().astype(str).unique()
    if len(true_people) == 0:
        continue

    true_person = true_people[0]
    mean_scores = group[shared_score_cols].mean(axis=0, skipna=True).fillna(0.0)

    predicted_person = mean_scores.idxmax().replace("score_", "")
    pred_top1_score = float(mean_scores.max())

    collapsed_row = {
        "sample_index": i,
        "sample_key": sample_key,
        "video_path": "MULTI_FACE_SAMPLE_FROM_EXISTING",
        "true_person": true_person,
        "predicted_person": predicted_person,
    }

    for col in shared_score_cols:
        collapsed_row[col] = float(mean_scores[col])

    collapsed_rows.append(collapsed_row)

    top10_names, top10_scores = topk_names_scores(mean_scores, k=10)

    result_row = {
        "sample_index": i,
        "sample_key": sample_key,
        "video_path": "MULTI_FACE_SAMPLE_FROM_EXISTING",
        "true_person": true_person,
        "predicted_person": predicted_person,
        "pred_top1_score": pred_top1_score,
    }

    for k in range(10):
        result_row[f"rank_{k+1}_name"] = top10_names[k] if k < len(top10_names) else ""
        result_row[f"rank_{k+1}_score"] = top10_scores[k] if k < len(top10_scores) else ""

    collapsed_result_rows.append(result_row)

face_sample_df = pd.DataFrame(collapsed_rows)
face_sample_results_df = pd.DataFrame(collapsed_result_rows)

face_sample_df.to_csv(FACE_COLLAPSED_OUTPUT, index=False)
face_sample_results_df.to_csv(FACE_COLLAPSED_RESULTS_OUTPUT, index=False)

print("\n===== FACE COLLAPSE SUMMARY =====")
print(f"Original face rows: {len(face_df)}")
print(f"Collapsed face sample rows: {len(face_sample_df)}")

# ================== ALIGN TO GAIT ==================
face_keys = set(face_sample_df["sample_key"])
gait_keys = set(gait_df["sample_key"])

shared_keys = sorted(face_keys.intersection(gait_keys))
face_only_keys = sorted(face_keys - gait_keys)
gait_only_keys = sorted(gait_keys - face_keys)

print("\n===== ALIGNMENT SUMMARY =====")
print(f"Collapsed face unique keys: {len(face_keys)}")
print(f"Gait unique keys: {len(gait_keys)}")
print(f"Shared keys used for fusion: {len(shared_keys)}")
print(f"Face-only keys: {len(face_only_keys)}")
print(f"Gait-only keys: {len(gait_only_keys)}")

if len(gait_only_keys) > 0:
    print("Gait-only keys skipped:")
    for k in gait_only_keys:
        print(f"  - {k}")

if len(shared_keys) == 0:
    raise RuntimeError("No shared sample_key values found after collapsing face.")

face_aligned = face_sample_df.set_index("sample_key").loc[shared_keys].reset_index()
gait_aligned = gait_df.set_index("sample_key").loc[shared_keys].reset_index()

gait_aligned = gait_aligned.set_index("sample_key").loc[face_aligned["sample_key"]].reset_index()

if not (face_aligned["sample_key"].values == gait_aligned["sample_key"].values).all():
    raise RuntimeError("Sample keys are not aligned after reindexing.")

if not (face_aligned["true_person"].values == gait_aligned["true_person"].values).all():
    mismatch = np.sum(face_aligned["true_person"].values != gait_aligned["true_person"].values)
    raise RuntimeError(f"True person mismatch after alignment in {mismatch} rows.")

# ================== NORMALIZE + FUSE ==================
face_scores = face_aligned[shared_score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
gait_scores = gait_aligned[shared_score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

face_scores_norm = minmax_row_normalize(face_scores)
gait_scores_norm = minmax_row_normalize(gait_scores)

fused_scores = FACE_WEIGHT * face_scores_norm + GAIT_WEIGHT * gait_scores_norm

# ================== BUILD FUSED SCORE MATRIX ==================
fused_score_df = pd.DataFrame()
fused_score_df["sample_index"] = np.arange(len(face_aligned))
fused_score_df["sample_key"] = face_aligned["sample_key"]
fused_score_df["video_path"] = "MULTI_MODAL_SAMPLE"
fused_score_df["true_person"] = face_aligned["true_person"]
fused_score_df["predicted_person_face"] = face_aligned["predicted_person"]
fused_score_df["predicted_person_gait"] = gait_aligned["predicted_person"]

for col in shared_score_cols:
    fused_score_df[col] = fused_scores[col].values

# ================== BUILD FUSION RESULTS ==================
fusion_rows = []
top1_correct = 0
top5_correct = 0
top10_correct = 0

for i in range(len(fused_score_df)):
    sample_key = fused_score_df.loc[i, "sample_key"]
    true_person = fused_score_df.loc[i, "true_person"]

    row_scores = fused_scores.iloc[i]
    top10_names, top10_scores = topk_names_scores(row_scores, k=10)

    predicted_person = top10_names[0]
    pred_top1_score = top10_scores[0]
    top5_names = top10_names[:5]

    if true_person == predicted_person:
        top1_correct += 1
    if true_person in top5_names:
        top5_correct += 1
    if true_person in top10_names:
        top10_correct += 1

    row = {
        "sample_index": i,
        "sample_key": sample_key,
        "video_path": "MULTI_MODAL_SAMPLE",
        "true_person": true_person,
        "predicted_person": predicted_person,
        "pred_top1_score": pred_top1_score,
        "predicted_person_face": fused_score_df.loc[i, "predicted_person_face"],
        "predicted_person_gait": fused_score_df.loc[i, "predicted_person_gait"],
    }

    for k in range(10):
        row[f"rank_{k+1}_name"] = top10_names[k] if k < len(top10_names) else ""
        row[f"rank_{k+1}_score"] = top10_scores[k] if k < len(top10_scores) else ""

    fusion_rows.append(row)

fusion_results_df = pd.DataFrame(fusion_rows)

# ================== SAVE ==================
fused_score_df.to_csv(FUSED_SCORE_OUTPUT, index=False)
fusion_results_df.to_csv(FUSED_RESULTS_OUTPUT, index=False)

y_true = fusion_results_df["true_person"]
y_pred = fusion_results_df["predicted_person"]

report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
pd.DataFrame(report).transpose().to_csv(FUSED_REPORT_OUTPUT)

labels = sorted(set(y_true.unique()).union(set(y_pred.unique())))
cm = confusion_matrix(y_true, y_pred, labels=labels)
pd.DataFrame(cm, index=labels, columns=labels).to_csv(FUSED_CM_OUTPUT)

total = len(fusion_results_df)
top1_acc = 100.0 * top1_correct / total if total > 0 else 0.0
top5_acc = 100.0 * top5_correct / total if total > 0 else 0.0
top10_acc = 100.0 * top10_correct / total if total > 0 else 0.0

print("\n===== FINAL FUSION RESULTS =====")
print(f"Fused samples: {total}")
print(f"Face weight: {FACE_WEIGHT}")
print(f"Gait weight: {GAIT_WEIGHT}")
print(f"Top-1 Accuracy: {top1_acc:.2f}%")
print(f"Top-5 Accuracy: {top5_acc:.2f}%")
print(f"Top-10 Accuracy: {top10_acc:.2f}%")

save_accuracy_bar_chart(top1_acc, top5_acc, top10_acc, BAR_CHART_OUTPUT)

print("\nSaved files:")
print(f"✅ {FACE_COLLAPSED_OUTPUT}")
print(f"✅ {FACE_COLLAPSED_RESULTS_OUTPUT}")
print(f"✅ {FUSED_SCORE_OUTPUT}")
print(f"✅ {FUSED_RESULTS_OUTPUT}")
print(f"✅ {FUSED_REPORT_OUTPUT}")
print(f"✅ {FUSED_CM_OUTPUT}")
print(f"✅ {BAR_CHART_OUTPUT}")