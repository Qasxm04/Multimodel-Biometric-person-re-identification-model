import json
import numpy as np

def load_keypoints(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)

    people = data.get("people", [])

    if len(people) == 0:
        return None

    keypoints = people[0]["pose_keypoints_2d"]
    keypoints = np.array(keypoints).reshape(-1,3)

    return keypoints[:,:2]


def extract_gait_features(sequence):

    hip_distances = []
    knee_distances = []
    ankle_distances = []

    for frame in sequence:

        left_hip = frame[11]
        right_hip = frame[8]

        left_knee = frame[12]
        right_knee = frame[9]

        left_ankle = frame[13]
        right_ankle = frame[10]

        hip_distances.append(np.linalg.norm(left_hip - right_hip))
        knee_distances.append(np.linalg.norm(left_knee - right_knee))
        ankle_distances.append(np.linalg.norm(left_ankle - right_ankle))

    features = np.array([
        np.mean(hip_distances),
        np.std(hip_distances),

        np.mean(knee_distances),
        np.std(knee_distances),

        np.mean(ankle_distances),
        np.std(ankle_distances)
    ])

    return features