import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.utils.key_rank import (
    leakage_labels,
    metadata_plaintext,
)


def split_nonprofiling_attack_data(
    X_attack,
    metadata_attack,
    n_ssl_train: int,
    n_knn_train: int,
    n_knn_eval: int,
    n_neighbors: int = 3,
):
    n_required = max(n_ssl_train, n_knn_train + n_knn_eval)

    if X_attack.shape[0] < n_required:
        raise ValueError(
            f"Requested max(n_ssl_train, n_knn_train + n_knn_eval) = {n_required}, "
            f"but only {X_attack.shape[0]} attack traces are available"
        )

    X_ssl_train = X_attack[:n_ssl_train]
    X_knn_train = X_attack[:n_knn_train]
    metadata_knn_train = metadata_attack[:n_knn_train]
    X_knn_eval = X_attack[n_knn_train : n_knn_train + n_knn_eval]
    metadata_knn_eval = metadata_attack[n_knn_train : n_knn_train + n_knn_eval]

    if X_knn_train.shape[0] < n_neighbors:
        raise ValueError(
            f"n_neighbors={n_neighbors} cannot exceed n_knn_train={X_knn_train.shape[0]}"
        )

    return (
        X_ssl_train,
        X_knn_train,
        metadata_knn_train,
        X_knn_eval,
        metadata_knn_eval,
    )


def compute_knn_candidate_accuracies(
    repr_train,
    metadata_train,
    repr_eval,
    metadata_eval,
    target_byte: int,
    n_neighbors: int = 3,
    leakage_model: str = "HW",
    weights: str = "distance",
):
    plaintext_train = metadata_plaintext(
        metadata_train,
        target_byte=target_byte,
    )
    plaintext_eval = metadata_plaintext(
        metadata_eval,
        target_byte=target_byte,
    )

    accuracies = np.zeros(256, dtype=np.float64)

    for key_guess in tqdm(range(256), desc="Training candidate KNNs"):
        y_train = leakage_labels(
            plaintext_train,
            key_guess=key_guess,
            leakage_model=leakage_model,
        )
        y_eval = leakage_labels(
            plaintext_eval,
            key_guess=key_guess,
            leakage_model=leakage_model,
        )

        classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
        )
        classifier.fit(
            repr_train,
            y_train,
        )

        predictions = classifier.predict(repr_eval)
        accuracies[key_guess] = float(np.mean(predictions == y_eval))

    return accuracies


def rank_key_candidates(candidate_scores, true_key):
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)

    if candidate_scores.shape != (256,):
        raise ValueError(
            f"candidate_scores must have shape (256,), got {candidate_scores.shape}"
        )

    ranked_keys = np.argsort(candidate_scores)[::-1]
    true_key_rank = int(np.where(ranked_keys == true_key)[0][0])

    return ranked_keys, true_key_rank
