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

def mae_reconstruction_loss(pred, target, mask, norm_pix_loss=True):
    """
    pred:   [B, num_patches, patch_size]
    target: [B, num_patches, patch_size]
    mask:   [B, num_patches], 1 means masked patch, 0 means visible patch
    """
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - mean) / torch.sqrt(var + 1e-6)

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [B, num_patches]

    mask = mask.float()
    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)

    return loss

class MAE1D(nn.Module):

    def __init__(
        self,
        trace_len=700,
        patch_size=10,
        embed_dim=320,
        depth=4,
        num_heads=8,
        mlp_ratio=4,
        mask_ratio=0.5,
        dropout=0.0,
    ):
        super().__init__()

        if trace_len % patch_size != 0:
            raise ValueError(
                f"trace_len={trace_len} must be divisible by patch_size={patch_size}"
            )

        self.trace_len = trace_len
        self.patch_size = patch_size
        self.n_patches = trace_len // patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_size),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def patchify(self, x):
        # x: [B, L, 1]
        x = x[:, : self.trace_len, 0]
        patches = x.reshape(x.size(0), self.n_patches, self.patch_size)
        return patches

    def make_mask(self, batch_size, device):
        num_mask = max(1, int(round(self.mask_ratio * self.n_patches)))

        noise = torch.rand(batch_size, self.n_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)

        mask = torch.zeros(
            batch_size,
            self.n_patches,
            dtype=torch.bool,
            device=device,
        )
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        return mask

    def forward(self, x):
        patches = self.patchify(x)  # [B, N, P]
        batch_size = patches.size(0)

        tokens = self.patch_embed(patches)  # [B, N, D]
        pos = self.pos_embed.expand(batch_size, -1, -1)

        mask = self.make_mask(batch_size, x.device)

        mask_tokens = self.mask_token.expand(batch_size, self.n_patches, -1)
        input_tokens = torch.where(
            mask.unsqueeze(-1),
            mask_tokens + pos,
            tokens + pos,
        )

        h = self.encoder(input_tokens)
        recon = self.decoder(h)  # [B, N, P]

        loss = mae_reconstruction_loss(
                pred=recon,
                target=patches,
                mask=mask,
                norm_pix_loss=True,
            )
        
        return loss, recon, mask

    def encode(self, x):
        patches = self.patchify(x)
        tokens = self.patch_embed(patches) + self.pos_embed
        h = self.encoder(tokens)
        return h.mean(dim=1)
    
def train_mae(
    X_train,
    device,
    trace_len=700,
    patch_size=10,
    embed_dim=320,
    depth=4,
    num_heads=8,
    mask_ratio=0.5,
    n_epochs=30,
    batch_size=128,
    lr=1e-4,
):
    model = MAE1D(
        trace_len=trace_len,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mask_ratio=mask_ratio,
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train).float())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    loss_log = []
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        num_batches = 0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device)

            loss, _, _ = model(batch_x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        loss_log.append(avg_loss)
        print(f"Epoch #{epoch}: mae_loss={avg_loss:.6f}")

    return model, loss_log


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

    n_train = 50000
    n_attack = 10000

    n_epochs = 100
    batch_size = 128
    lr = 1e-4

    trace_len = 700
    patch_size = 5
    embed_dim = 320
    depth = 4
    num_heads = 8
    mask_ratio = 0.30
    target_byte = 2
    normalize_mode = None

    run_name = (
        f"mae_patch{patch_size}_mask{int(mask_ratio * 100)}"
        f"_dim{embed_dim}_ep{n_epochs}"
    )

    figure_dir = PROJECT_ROOT / "outputs" / "figures" / run_name
    repr_dir = PROJECT_ROOT / "outputs" / "representations" / run_name
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints" / run_name

    figure_dir.mkdir(parents=True, exist_ok=True)
    repr_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ASCAD profiling traces...")
    X_profiling, y_profiling = load_ascad_split(
        h5_path=ascad_path,
        split="profiling",
        add_channel=True,
        normalize=normalize_mode,
        load_metadata=False,
    )

    print("Loading ASCAD attack traces with metadata...")
    X_attack, y_attack, metadata_attack = load_ascad_split(
        h5_path=ascad_path,
        split="attack",
        add_channel=True,
        normalize=normalize_mode,
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

    print("Training MAE-1D...")
    train_start_time = time.time()

    model, loss_log = train_mae(
        X_train=X_train,
        device=device,
        trace_len=trace_len,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mask_ratio=mask_ratio,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
    )

    train_end_time = time.time()
    train_time_sec = train_end_time - train_start_time
    train_time_ms = 1000 * train_time_sec

    print("MAE loss log:", loss_log)
    print(f"Training start time: {train_start_time}")
    print(f"Training end time: {train_end_time}")
    print(f"Training time: {train_time_sec:.2f} sec")
    print(f"Training time: {train_time_ms:.2f} ms")

    checkpoint_path = checkpoint_dir / f"{run_name}_encoder.pt"
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

    print("Training linear classifier on MAE representations...")
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
    )
    clf.fit(repr_train, y_train)

    train_acc = float(clf.score(repr_train, y_train))
    print("Linear probe train accuracy:", train_acc)

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

    rank_path = figure_dir / f"{run_name}_linear_probe_rank.png"
    ranks_path = repr_dir / f"{run_name}_linear_probe_ranks.npy"

    plot_rank_curve(
        ranks,
        save_path=rank_path,
        title="MAE-1D + Linear Probe Key Rank",
    )

    np.save(ranks_path, ranks)

    print("Saved rank curve to:", rank_path)
    print("Saved ranks to:", ranks_path)

    summary_path = PROJECT_ROOT / "outputs" / "logs" / "experiment_summary.csv"

    append_experiment_result(
        summary_path,
        {
            "method": "MAE-1D",
            "run_name": run_name,
            "dataset": "ASCAD.h5",
            "n_train": n_train,
            "n_attack": n_attack,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "trace_len": trace_len,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mask_ratio": mask_ratio,
            "normalize": normalize_mode,
            "classifier": "LogisticRegression",
            "linear_probe_train_acc": round(train_acc, 6),
            "target_byte": target_byte,
            "device": str(device),
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