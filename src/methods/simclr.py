import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

import src.utils.data_loader as data_loader 
import src.models.model_zoo as model_zoo
import src.methods.knn_model as knn_model
import src.utils.trace_transforms as trace_transforms 
import src.utils.supporting_functions as supporting_func 

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SimCLRModel(nn.Module):
    def __init__(self, backbone_name, pool_mode, input_length, projector_hidden_dim, proj_dim):
        super().__init__()

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode

        self.encoder = build_backbone(model_name=backbone_name, input_channels=1, input_length=input_length)

        self.repr_dim = self.encoder.get_output_dim(pool=pool_mode)

        self.projector = nn.Sequential(
            nn.Linear(self.repr_dim, projector_hidden_dim),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder.encode(x, pool=self.pool_mode)
        z = self.projector(h)
        z = F.normalize(z, dim=1)

        return h, z

    def encode(self, x):
        return self.encoder.encode(x, pool=self.pool_mode)


def nt_xent_loss(z1, z2, temperature=0.2):
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have the same shape, received {tuple(z1.shape)} and {tuple(z2.shape)}")

    if temperature <=0:
        raise ValueError("Temperature valude need to be greater than 0, current value is: ", temperature)

    batch_size = z1.shape[0]
    if batch_size < 2:
        raise ValueError("NT-Xent loss requires batch_size >= 2")

    z = torch.cat([z1, z2], dim=0)

    similarity = torch.matmul(z, z.T) / temperature

    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)

    similarity = similarity.masked_fill(self_mask, -1e9)

    labels = torch.arange(2 * batch_size, device=z.device)

    labels = (labels + batch_size) % (2 * batch_size)

    return F.cross_entropy(similarity, labels)


@print_run_time
def train_simclr(params, X_train, device, trace_window):
    """ training function of simclr """
    seed = params['seed']
    epochs = params['epochs']
    temperature = params['temperature']
    max_shift = params['max_shift']
    noise_std = params['noise_std']
    denoise_kernel_size = params['denoise_kernel_size']
    view_augmentation = params['view_augmentation']
    augmentations = params['augmentation_family']
    augmentation_probability = params['augmentation_probability']
    start, end = trace_window[0], trace_window[1]
    input_length = end - start

    run_name = (
        f"simclr_{params['backbone_name']}_window{start}-{end}_{params['pool_mode']}_proj{params['proj_dim']}"
        f"_tau{temperature}_n{len(X_train)}_ep{epochs}_seed{seed}_knn{params['knn_neighbors']}"
    )

    if start < 0 or end <= start or end > X_train.shape[1]:
        raise ValueError("Invalid trace window")
    if params["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    if params["batch_size"] < 2 or len(X_train) < params["batch_size"]:
        raise ValueError("SimCLR requires at least one full batch of size >= 2")

    X_train = ensure_trace_matrix(X_train)

    # Seed here, not only in the caller, so the seed in run_name is the one actually used
    # for weight init, data shuffling and augmentation on every entry point.
    set_seed(seed)

    model = SimCLRModel(
        backbone_name=params['backbone_name'],
        pool_mode=params['pool_mode'],
        projector_hidden_dim=params['projector_hidden_dim'],
        proj_dim=params['proj_dim'],
        input_length=input_length
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(X_train).float())

    data_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    backbone_params = sum(param.numel() for param in model.encoder.parameters() if param.requires_grad)
    print("Shared backbone trainable parameters:", backbone_params)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Full SimCLR trainable parameters:", trainable_params)

    model.train()
    loss_log = []
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_index, (batch_x,) in enumerate(data_loader):
            batch_x = batch_x.to(device)

            x1 = make_trace_view(
                batch_x,
                trace_window=trace_window,
                augmentation=view_augmentation,
                augmentation_family=augmentations,
                augmentation_probability=augmentation_probability,
                max_shift=max_shift,
                noise_std=noise_std,
                denoise_kernel_size=denoise_kernel_size,
            )

            x2 = make_trace_view(
                batch_x,
                trace_window=trace_window,
                augmentation=view_augmentation,
                augmentation_family=augmentations,
                augmentation_probability=augmentation_probability,
                max_shift=max_shift,
                noise_std=noise_std,
                denoise_kernel_size=denoise_kernel_size,
            )

            h1, z1 = model(x1)
            _, z2 = model(x2)

            # if statement here is pure for data checking purpose and it is the code in it never used for training.
            if epoch == 0 and batch_index == 0:
                model_input = prepare_model_input(batch_x, trace_window=trace_window)

                temporal_features = model.encoder.forward_features(model_input)

                print("Batch input shape:", batch_x.shape)
                print("Model input shape:", model_input.shape)

                print("Temporal feature shape:", temporal_features.shape)

                print("Backbone representation shape:", h1.shape)
                print("Projection shape:", z1.shape)

            loss = nt_xent_loss(z1=z1, z2=z2, temperature=temperature)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        loss_log.append(avg_loss)

        print(f"Epoch = #{epoch}, simclr_loss = {avg_loss:.6f}")

    return model, loss_log, run_name


def encode_representations(model, X, device, trace_window, batch_size=256):
    X = ensure_trace_matrix(X)

    dataset = TensorDataset(torch.from_numpy(X).float())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    representations = []

    model.eval()

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            batch_x = prepare_model_input(batch_x, trace_window=trace_window)

            h = model.encode(batch_x)

            representations.append(h.cpu().numpy())

    repr = np.concatenate(representations, axis=0)
    return repr


def print_dataset_details(trace_window, input_length, X_train, y_train, plaintext, key):
    print("Trace window:", trace_window)
    print("Window size:", input_length)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("plaintext shape:", plaintext.shape)
    print("key shape:", key.shape)


def main(opts):
    """Run the SimCLR training and evaluation pipeline standalone"""
    # Imported here, not at the top: ssl_factory imports this module at load time.
    from src.methods.ssl_factory import get_ssl_method_preset

    device = get_device(prefer_mps=False)
    print("[LOG] -- Using device: ", device)   # type: ignore
    common_config_path = os.path.join(PROJECT_ROOT, "src", "configs", "ssl_shared_params.json")
    # Same merge as ssl_factory.train_ssl_model: model preset first, shared training params on top.
    params = {**get_ssl_method_preset("simclr"), **load_json_file(common_config_path)}
    encode_batch_size = params["encode_batch_size"]

    n_train = opts.n_train
    trace_window, input_length = get_trace_window(opts.trace_window)

    output_dir = opts.output
    os.makedirs(output_dir, exist_ok=True)

    # Load full traces. train_simclr and encode_representations apply trace_window
    # exactly once, after view augmentation, so the loader must not cut it as well.
    print("Loading the profiling traces for model training...")
    X_train, y_train, plaintext_train, key = load_dataset(opts.input_train, which_one="train", attack_window=None, method="norm", trace_num=n_train)
    print_dataset_details(trace_window, input_length, X_train, y_train, plaintext_train, key)

    print("Loading the attack traces for KNN evaluation...")
    X_test, _, plaintext_test, _ = load_dataset(opts.input_test, which_one="test", attack_window=None, method="norm")
    print("X_test shape:", X_test.shape)

    print("Start Training SimCLR...")
    model, loss_log, run_name = train_simclr(params, X_train, device, trace_window)
    print("SimCLR loss log: ", loss_log)

    # Same payload layout as train.py, so ssl_factory.load_ssl_encoder can reload it.
    checkpoint_path = os.path.join(output_dir, f"{run_name}_encoder.pt")
    checkpoint_payload = {
        "method": "simclr",
        "params": params,
        "model_state_dict": model.state_dict(),
        "input_length": input_length,
        "trace_window": opts.trace_window,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    print("Saved checkpoint to:", checkpoint_path)

    print("Encoding KNN-train and KNN-eval representations...")
    repr_knn_train = encode_representations(model=model, X=X_train, device=device, trace_window=trace_window, batch_size=encode_batch_size)
    repr_knn_eval = encode_representations(model=model, X=X_test, device=device, trace_window=trace_window, batch_size=encode_batch_size)
    print("repr_knn_train shape:", repr_knn_train.shape)
    print("repr_knn_eval shape:", repr_knn_eval.shape)

    expected_repr_dim = model.repr_dim
    for repr_name, representations in (("repr_knn_train", repr_knn_train), ("repr_knn_eval", repr_knn_eval)):
        if representations.shape[1] != expected_repr_dim:
            raise ValueError(f"Unexpected {repr_name} dimension: expected {expected_repr_dim}, " f"received {representations.shape[1]}")

    print("Now, Training 256 candidate KNNs and computing candidate accuracies...")  # type: ignore
    knn_models = train_256_knn(repr_knn_train, plaintext_train, opts.target_byte, opts.leakage_model,
                               n_neighbors=params["knn_neighbors"], weights=params["knn_weights"])
    acc_list = test_256_knn(knn_models, repr_knn_eval, plaintext_test, opts.target_byte, opts.leakage_model)

    real_key = int(key[opts.target_byte])
    rank = int((acc_list > acc_list[real_key]).sum())
    print(f"Real key {real_key}: acc={acc_list[real_key]:.4f}, rank={rank}, best guess={int(acc_list.argmax())}")

    # The encoder does not depend on the target byte or leakage model, but the KNN accuracies do.
    eval_tag = f"byte{opts.target_byte}_{opts.leakage_model}"
    acc_path = os.path.join(output_dir, f"{run_name}_{eval_tag}_knn_acc.npy")
    np.save(acc_path, acc_list)
    fig_path = os.path.join(output_dir, f"{run_name}_{eval_tag}_knn_acc.png")
    plot_distinguisher_figure(range(256), acc_list, real_key, "key guess", "accuracy", f"real key {real_key}, rank {rank}", fig_path)
    print("Saved KNN accuracies to:", acc_path)
    print("Saved KNN accuracy figure to:", fig_path)

    print("SSL model and KNN model training is done!")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-itr", "--input_train", help="Path to input dataset")
    parser.add_argument("-ite", "--input_test", help="Path to input dataset")
    parser.add_argument("-o", "--output", default=os.path.join(PROJECT_ROOT, "outputs", "simclr"), help="path to output folder")
    parser.add_argument("-tw", "--trace_window", type=str, default="0_700", help="")
    parser.add_argument("-tb", "--target_byte", type=int, default=2)
    parser.add_argument("-lm", "--leakage_model", choices=("ID", "HW"), default="HW")
    parser.add_argument("--n_train", type=int, default=0, help="num of traces used for training")

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parse_args(sys.argv[1:])
    opts.input_train = "/home/carl/work_dir/ssl-sidechannel-research/data/npz_data/ASCAD/ATM_AES_v1_fixed_key/ASCAD_train.npz"
    opts.input_test = "/home/carl/work_dir/ssl-sidechannel-research/data/npz_data/ASCAD/ATM_AES_v1_fixed_key/ASCAD_test.npz"
    opts.output = "/home/carl/work_dir/ssl-sidechannel-research/outputs/ASCAD"
    opts.trace_window = "0_700"
    opts.target_byte = 2
    opts.leakage_model = "HW"
    opts.n_train = 0
    main(opts)
