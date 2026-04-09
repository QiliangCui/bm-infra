import json
import sys
import os

def parse_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract the task results
    task_results = data['results'].get('mmmu_pro', {})
    
    # We want the 'mmmu_pro_acc' metric (ignoring stderr)
    acc = task_results.get('mmmu_pro_acc,none', 0.0)
    
    # Print the specific JSON structure the infra expects
    print(json.dumps({"mmmu_pro_acc": acc}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_results.py <path_to_json>")
        sys.exit(1)
    parse_results(sys.argv[1])