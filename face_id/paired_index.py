import os
import re
import random
import pandas as pd

# ================== PATH ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Gait Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "fusion_protocol")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
TEST_RATIO = 0.2

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# ================== HELPERS ==================
def parse_participant_experiment_trial_from_path(path):
    parts = os.path.normpath(path).split(os.sep)

    participant = next((p for p in parts if p.startswith("Participant_")), None)
    experiment = next((p for p in parts if p.startswith("Experiment_")), None)

    trial = None
    if experiment:
        exp_idx = parts.index(experiment)
        if exp_idx + 1 < len(parts):
            candidate = parts[exp_idx + 1]
            if os.path.splitext(candidate)[1] == "":
                trial = candidate

    return participant, experiment, trial


def extract_view(filename):
    name = filename.lower()

    for v in ["front", "rear", "left", "right", "side", "45", "90"]:
        if v in name:
            return v

    return "unknown"


# ================== FACE SCAN ==================
def scan_face(dataset_path):
    rows = []

    for root, _, files in os.walk(dataset_path):
        participant, experiment, trial = parse_participant_experiment_trial_from_path(root)

        if not participant or not experiment or not trial:
            continue

        for file in files:
            if not file.lower().endswith(VIDEO_EXTENSIONS):
                continue

            full_path = os.path.join(root, file)

            rows.append({
                "participant": participant,
                "experiment": experiment,
                "trial": trial,
                "view": extract_view(file),
                "face_video_path": full_path,
                "sample_key": f"{participant}__{experiment}__{trial}"
            })

    return pd.DataFrame(rows)


# ================== GAIT SCAN (FAST) ==================
def scan_gait(dataset_path):
    rows = []

    participants = [
        p for p in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, p)) and p.startswith("Participant_")
    ]

    for participant in participants:
        p_path = os.path.join(dataset_path, participant)

        for experiment in os.listdir(p_path):
            e_path = os.path.join(p_path, experiment)
            if not os.path.isdir(e_path):
                continue

            for trial in os.listdir(e_path):
                t_path = os.path.join(e_path, trial)
                if not os.path.isdir(t_path):
                    continue

                json_folders = [
                    f for f in os.listdir(t_path)
                    if os.path.isdir(os.path.join(t_path, f)) and f.endswith("_json")
                ]

                for jf in json_folders:
                    json_path = os.path.join(t_path, jf)

                    json_files = [
                        f for f in os.listdir(json_path)
                        if f.endswith("_keypoints.json")
                    ]

                    # 🔥 FAST: no JSON reading
                    if len(json_files) == 0:
                        continue

                    rows.append({
                        "participant": participant,
                        "experiment": experiment,
                        "trial": trial,
                        "gait_json_path": json_path,
                        "sample_key": f"{participant}__{experiment}__{trial}"
                    })

    return pd.DataFrame(rows)


# ================== BUILD PAIRED ==================
def build_pairs(face_df, gait_df):
    face_group = face_df.groupby("sample_key").agg({
        "face_video_path": lambda x: list(x),
        "view": lambda x: list(x)
    }).reset_index()

    gait_group = gait_df.groupby("sample_key").agg({
        "gait_json_path": lambda x: list(x)
    }).reset_index()

    merged = pd.merge(face_group, gait_group, on="sample_key")

    merged["participant"] = merged["sample_key"].apply(lambda x: x.split("__")[0])
    merged["experiment"] = merged["sample_key"].apply(lambda x: x.split("__")[1])
    merged["trial"] = merged["sample_key"].apply(lambda x: x.split("__")[2])

    return merged


# ================== SPLIT ==================
def split_data(df):
    random.seed(RANDOM_SEED)

    train_rows = []
    test_rows = []

    for participant, group in df.groupby("participant"):
        rows = group.to_dict("records")
        random.shuffle(rows)

        n = len(rows)
        n_test = max(1, int(n * TEST_RATIO))

        test = rows[:n_test]
        train = rows[n_test:]

        if len(train) == 0:
            train.append(test.pop())

        train_rows.extend(train)
        test_rows.extend(test)

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


# ================== MAIN ==================
def main():
    print("Scanning face videos...")
    face_df = scan_face(DATASET_PATH)
    print(f"Found {len(face_df)} face videos")

    print("Scanning gait sequences...")
    gait_df = scan_gait(DATASET_PATH)
    print(f"Found {len(gait_df)} gait sequences")

    paired_df = build_pairs(face_df, gait_df)
    print(f"Found {len(paired_df)} paired samples")

    train_df, test_df = split_data(paired_df)

    # ================== SAVE ==================
    paired_df.to_csv(os.path.join(OUTPUT_DIR, "paired_full.csv"), index=False)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "paired_train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "paired_test.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "train_keys.txt"), "w") as f:
        for k in train_df["sample_key"]:
            f.write(k + "\n")

    with open(os.path.join(OUTPUT_DIR, "test_keys.txt"), "w") as f:
        for k in test_df["sample_key"]:
            f.write(k + "\n")

    print("\n===== DONE =====")
    print(f"Participants: {paired_df['participant'].nunique()}")
    print(f"Total samples: {len(paired_df)}")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")


if __name__ == "__main__":
    main()