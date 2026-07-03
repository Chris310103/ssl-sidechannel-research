from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from src.datasets.ascad_loader import load_ascad_split
from src.evaluation.rank_eval import (
    expand_proba_to_256,
    compute_rank_curve,
    plot_rank_curve,
)
from src.utils.get_device import get_device
from src.utils.experiment_logger import append_experiment_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CPCRefEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=320):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=10, stride=2, padding=4),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):

        x = x.transpose(1, 2)
        z = self.net(x)
        z = z.transpose(1, 2)
        return z


class CPCRefModel(nn.Module):
    def __init__(self, repr_dim=320, context_dim=320, prediction_steps=12):
        super().__init__()

        self.encoder = CPCRefEncoder(
            input_channels=1,
            hidden_dim=repr_dim,
        )

        self.gru = nn.GRU(
            input_size=repr_dim,
            hidden_size=context_dim,
            num_layers=1,
            batch_first=True,
        )

        self.predictor = nn.Linear(
            context_dim,
            repr_dim * prediction_steps,
            bias=False,
        )

        self.repr_dim = repr_dim
        self.context_dim = context_dim
        self.prediction_steps = prediction_steps

    def forward(self, x):
        z = self.encoder(x)
        c, _ = self.gru(z)
        pred = self.predictor(c)
        return z, c, pred

    def encode(self, x, mode="context_mean"):
        z, c, _ = self.forward(x)

        if mode == "context_mean":
            return c.mean(dim=1)

        if mode == "latent_mean":
            return z.mean(dim=1)

        if mode == "context_last":
            return c[:, -1, :]

        if mode == "concat":
            return torch.cat(
                [
                    c.mean(dim=1),
                    c[:, -1, :],
                    z.mean(dim=1),
                ],
                dim=1,
            )
        
        raise ValueError(f"Unknown encode mode: {mode}")

def cpc_ref_loss(
    z,
    pred,
    prediction_steps=12,
    negative_samples=10,
):

    batch_size, seq_len, repr_dim = z.shape
    device = z.device

    if seq_len <= prediction_steps + 1:
        raise ValueError(
            f"Sequence length {seq_len} too short for prediction_steps={prediction_steps}"
        )

    total_loss = 0.0
    total_acc = 0.0
    loss_count = 0

    z_pool = z.reshape(-1, repr_dim)

    for k in range(1, prediction_steps + 1):

        pred_k = pred[:, :-k, (k - 1) * repr_dim : k * repr_dim]

        target_k = z[:, k:, :]

        pred_k = pred_k.reshape(-1, repr_dim)      
        target_k = target_k.reshape(-1, repr_dim)  
        n_pos = pred_k.size(0)

        
        pos_score = torch.sum(pred_k * target_k, dim=1, keepdim=True)

        neg_indices = torch.randint(
            low=0,
            high=z_pool.size(0),
            size=(n_pos, negative_samples),
            device=device,
        )

        neg_z = z_pool[neg_indices]  

        neg_score = torch.bmm(
            neg_z,
            pred_k.unsqueeze(2),
        ).squeeze(2)

        logits = torch.cat([pos_score, neg_score], dim=1)

        labels = torch.zeros(n_pos, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()

        total_loss = total_loss + loss
        total_acc = total_acc + acc.item()
        loss_count += 1

    return total_loss / loss_count, total_acc / loss_count


def train_cpc_ref(
    X_train,
    device,
    repr_dim=320,
    context_dim=320,
    prediction_steps=12,
    negative_samples=10,
    n_epochs=10,
    batch_size=64,
    lr=2e-4,
):
    model = CPCRefModel(
        repr_dim=repr_dim,
        context_dim=context_dim,
        prediction_steps=prediction_steps,
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train).float())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_log = []
    acc_log = []

    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            z, c, pred = model(batch_x)

            if epoch == 0 and num_batches == 0:
                print("z shape:", z.shape)
                print("c shape:", c.shape)
                print("pred shape:", pred.shape)


            loss, acc = cpc_ref_loss(
                z=z,
                pred=pred,
                prediction_steps=prediction_steps,
                negative_samples=negative_samples,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)

        loss_log.append(avg_loss)
        acc_log.append(avg_acc)

        print(f"Epoch #{epoch}: loss={avg_loss:.6f}, cpc_acc={avg_acc:.4f}")

    return model, loss_log, acc_log


def encode_representations(model, X, device, batch_size=256):
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    reps = []

    model.eval()

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            h = model.encode(batch_x)
            reps.append(h.cpu().numpy())

    return np.concatenate(reps, axis=0)


def main():
    ascad_path = PROJECT_ROOT / "data" / "raw" / "ascad" / "ASCAD.h5"

    figure_dir = PROJECT_ROOT / "outputs" / "figures" / "cpc_ref"
    repr_dir = PROJECT_ROOT / "outputs" / "representations" / "cpc_ref"
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints" / "cpc_ref"

    figure_dir.mkdir(parents=True, exist_ok=True)
    repr_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_train = 50000
    n_attack = 10000

    n_epochs = 30
    batch_size = 64
    lr = 2e-4

    repr_dim = 320
    context_dim = 320
    prediction_steps = 12
    negative_samples = 10

    target_byte = 2

    print("Loading ASCAD profiling traces...")
    X_profiling, y_profiling = load_ascad_split(
        h5_path=ascad_path,
        split="profiling",
        add_channel=True,
        normalize=None,
        load_metadata=False,
    )

    print("Loading ASCAD attack traces with metadata...")
    X_attack, y_attack, metadata_attack = load_ascad_split(
        h5_path=ascad_path,
        split="attack",
        add_channel=True,
        normalize=None,
        load_metadata=True,
    )

    X_train = X_profiling[:n_train]
    y_train = y_profiling[:n_train]

    X_attack_small = X_attack[:n_attack]
    metadata_attack_small = metadata_attack[:n_attack]

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_attack shape:", X_attack_small.shape)
    print("metadata_attack shape:", metadata_attack_small.shape)

    device = get_device(prefer_mps=False)
    print("Using device:", device)

    print("Training CPC...")

    train_start_time = time.time()

    model, loss_log, acc_log = train_cpc_ref(
    X_train=X_train,
    device=device,
    repr_dim=repr_dim,
    context_dim=context_dim,
    prediction_steps=prediction_steps,
    negative_samples=negative_samples,
    n_epochs=n_epochs,
    batch_size=batch_size,
    lr=lr,
)
    train_end_time = time.time()
    train_time_sec = train_end_time - train_start_time
    train_time_ms = 1000 * train_time_sec

    print("CPC loss log:", loss_log)
    print(f"Training start time: {train_start_time}")
    print(f"Training end time: {train_end_time}")
    print(f"Training time: {train_time_sec:.2f} sec")
    print(f"Training time: {train_time_ms:.2f} ms")

    checkpoint_path = checkpoint_dir / "cpc_ref_encoder.pt"

    torch.save(model.state_dict(), checkpoint_path)
    print("Saved checkpoint to:", checkpoint_path)

    print("Encoding profiling representations...")
    repr_train = encode_representations(
        model=model,
        X=X_train,
        device=device,
        batch_size=256,
    )

    print("Encoding attack representations...")
    repr_attack = encode_representations(
        model=model,
        X=X_attack_small,
        device=device,
        batch_size=256,
    )

    print("repr_train shape:", repr_train.shape)
    print("repr_attack shape:", repr_attack.shape)

    np.save(repr_dir / "repr_train.npy", repr_train)
    np.save(repr_dir / "repr_attack.npy", repr_attack)
    np.save(repr_dir / "y_train.npy", y_train)

    print("Training linear classifier on CPC representations...")
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    clf.fit(repr_train, y_train)

    print("Predicting attack probabilities...")
    attack_probas_seen = clf.predict_proba(repr_attack)
    attack_probas = expand_proba_to_256(
        attack_probas_seen,
        classes=clf.classes_,
    )

    print("attack_probas shape:", attack_probas.shape)

    print("Computing key rank curve...")
    ranks = compute_rank_curve(
        probas=attack_probas,
        metadata=metadata_attack_small,
        target_byte=target_byte,
        max_traces=n_attack,
        use_log=True,
    )

    print("Final rank:", ranks[-1])
    print("Minimum rank:", ranks.min())

    rank0_indices = np.where(ranks == 0)[0]
    rank0_trace = int(rank0_indices[0] + 1) if len(rank0_indices) > 0 else -1

    print("Rank-0 trace:", rank0_trace)

    rank_path = figure_dir / "cpc_ref_linear_probe_rank.png"
    ranks_path = repr_dir / "cpc_ref_linear_probe_ranks.npy"

    plot_rank_curve(
        ranks,
        save_path=rank_path,
        title="CPC-ref + Linear Probe Key Rank"
    )

    np.save(ranks_path, ranks)

    print("Saved rank curve to:", rank_path)
    print("Saved ranks to:", ranks_path)

    summary_path = PROJECT_ROOT / "outputs" / "logs" / "experiment_summary.csv"

    append_experiment_result(
        summary_path,
        {
            "method": "CPC-ref",
            "dataset": "ASCAD.h5",
            "n_train": n_train,
            "n_attack": n_attack,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "repr_dim": repr_dim,
            "prediction_steps": prediction_steps,
            "negative_samples": negative_samples,
            "context_dim": context_dim,
            "classifier": "LogisticRegression",
            "target_byte": target_byte,
            "device": device,
            "train_start_time": train_start_time,
            "train_end_time": train_end_time,
            "train_time_sec": round(train_time_sec, 2),
            "train_time_ms": round(train_time_ms, 2),
            "final_rank": int(ranks[-1]),
            "min_rank": int(ranks.min()),
            "rank0_trace": rank0_trace,
            "figure_path": str(rank_path.relative_to(PROJECT_ROOT)),
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        },
    )

    print("Saved experiment summary to:", summary_path)


if __name__ == "__main__":
    main()