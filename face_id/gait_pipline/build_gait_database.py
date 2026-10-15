import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
root_folder = os.path.join(os.path.dirname(SCRIPT_DIR), "Gait Dataset")

participants = sorted([p for p in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, p))])

for participant in participants:
    participant_path = os.path.join(root_folder, participant)
    print(f"Processing Participant: {participant}")
    
    experiments = sorted([exp for exp in os.listdir(participant_path) if os.path.isdir(os.path.join(participant_path, exp))])
    
    for experiment in experiments:
        experiment_path = os.path.join(participant_path, experiment)
        trials = sorted([t for t in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, t))])
        
        for trial in trials:
            trial_path = os.path.join(experiment_path, trial)
            # Only consider folders ending with _json
            json_folders = [f for f in os.listdir(trial_path) 
                            if os.path.isdir(os.path.join(trial_path, f)) and f.endswith("_json")]
            
            total_frames = 0
            empty_frames = 0
            
            for json_folder in json_folders:
                json_path = os.path.join(trial_path, json_folder)
                json_files = sorted([f for f in os.listdir(json_path) if f.endswith("_keypoints.json")])
                
                for jf in json_files:
                    total_frames += 1
                    json_file_path = os.path.join(json_path, jf)
                    with open(json_file_path, 'r') as file:
                        data = json.load(file)
                        if not data.get("people"):
                            empty_frames += 1
            
            print(f"  Trial: {experiment}/{trial}")
            print(f"    Total frames: {total_frames}")
            print(f"    Empty frames: {empty_frames}")
