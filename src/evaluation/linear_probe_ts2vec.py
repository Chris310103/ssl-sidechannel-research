from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    project_root = Path(__file__).resolve().parents[2]

    repr_path = project_root / "outputs" / "ts2vec" / "ascad_ts2vec_repr_2048.npy"
    label_path = project_root / "outputs" / "ts2vec" / "ascad_ts2vec_labels_2048.npy"

    X = np.load(repr_path)
    y = np.load(label_path)

    print("Representation shape:", X.shape)
    print("Labels shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) < len(y) else None,
    )

    clf = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
    )

    print("Training linear probe...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Linear probe accuracy:", acc)
    print("Random baseline:", 1 / 256)


if __name__ == "__main__":
    main()