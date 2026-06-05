from pathlib import Path
import csv


def append_experiment_result(csv_path, result):
    
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    result = dict(result)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)