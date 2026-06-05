from pathlib import Path
import csv


EXPERIMENT_COLUMNS = [
    "method",
    "dataset",
    "n_train",
    "n_attack",
    "n_epochs",
    "batch_size",
    "lr",
    "base_dim",
    "repr_dim",
    "proj_dim",
    "k_steps",
    "num_t_samples",
    "classifier",
    "target_byte",
    "device",
    "train_start_time",
    "train_end_time",
    "train_time_sec",
    "train_time_ms",
    "final_rank",
    "min_rank",
    "rank0_trace",
    "figure_path",
    "checkpoint_path",
]


def append_experiment_result(csv_path, result):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {col: "" for col in EXPERIMENT_COLUMNS}

    for key, value in result.items():
        if key in row:
            row[key] = value

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=EXPERIMENT_COLUMNS,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)