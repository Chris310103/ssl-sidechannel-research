import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.data_loader as data_loader 
import src.models.model_zoo as model_zoo
import src.methods.knn_model as knn_model
import src.utils.trace_transforms as trace_transforms 
import src.utils.supporting_functions as supporting_func
import src.methods.losses as loss_func


# --------------------------------------------------------------------------------
# 1. Data augmentation: turn one image into two different "views"
# --------------------------------------------------------------------------------
class DataTransform():
    def __init__(self):
        pass

    def transform(self):
        pass


class TwoViews():
    """
    Applies the same random augumentation pipeline twice to one image
    Because the pipeline is random, the two ouputs differ (e.g., different crop,
    colors, flip, ...), but they still show the same underlying object.
    """
    def __init__(self):
        self.transformer = DataTransform()

    def __call__(self, trace):
        view1 = self.transformer(trace)
        view2 = self.transformer(trace)
        return view1, view2


# --------------------------------------------------------
# 2. The model: encoder f(.) + projection head g(.)
# --------------------------------------------------------
def make_encoder():
    pass


class SimCLRModel(model_zoo.SCAModel):
    def __init__(self, backbone_name, pool_mode, input_length, projection_dim, emb_size):
        super().__init__()

        self.backbone_name = backbone_name
        self.pool_mode = pool_mode
        self.emb_size = emb_size

        self.encoder = model_zoo.build_model(model_name=backbone_name, input_length=input_length, emb_size=emb_size)

        # self.repr_dim = self.encoder.get_output_dim(pool=pool_mode)
        self.feature_dim = emb_size   # is this correct?

        self.projector = nn.Sequential(         # f: g: h -> z (only used for the loss)
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


# ------------------------------------------------------------------------
# 3. Stage 1: self-supervised pretraining (labels are never used)
# ------------------------------------------------------------------------
def simclr_pretrain(params, model, data_loader, optimizer, scheduler):
    """pretrain of the simclr model."""
    epochs = params['epochs']
    temperature = params['temperature']

    loss_log = []
    device = supporting_func.get_device()

    model.train()
    for epoch in range(1, epochs+1):
        total_loss, count = 0.0, 0

        for batch_index, (x1, x2) in enumerate(data_loader):
            x1, x2 = x1.to(device), x2.to(device)

            h1, z1 = model(x1)
            h2, z2 = model(x2)
            loss = loss_func.nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x1.size(0)
            count += x1.size(0)

            # if statement here is pure for data checking purpose and it is the code in it never used for training.
            if epoch == 0 and batch_index == 0:
                temporal_features = model.encoder.forward_features(x1)

                print("Batch input shape:", x1.shape)
                print("Model input shape:", model.inp_shape)

                print("Temporal feature shape:", temporal_features.shape)

                print("Backbone representation shape:", h1.shape)
                print("Projection shape:", z1.shape)

        avg_loss = total_loss / max(count, 1)
        loss_log.append(avg_loss)

        scheduler.step()
        print(f"Pretrain epoch = #{epoch:3d}, simclr_loss = {avg_loss:.4f}")

    return loss_log


@supporting_func.print_run_time
def run_training(params, data_loader, ):
    """ training function of simclr """
    seed = params['seed']
    epochs = params['epochs']
    lr = params['lr']
    temperature = params['temperature']
    max_shift = params['max_shift']
    noise_std = params['noise_std']
    denoise_kernel_size = params['denoise_kernel_size']
    view_augmentation = params['view_augmentation']
    augmentations = params['augmentation_family']
    augmentation_probability = params['augmentation_probability']
    start, end = trace_window[0], trace_window[1]
    input_length = end - start

    device = supporting_func.get_device()

    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    run_name = (
        f"simclr_{params['backbone_name']}_window{start}-{end}_{params['pool_mode']}_proj{params['proj_dim']}"
        f"_tau{temperature}_ep{epochs}_seed{seed}_knn{params['knn_neighbors']}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    backbone_params = sum(param.numel() for param in model.encoder.parameters() if param.requires_grad)
    print("Shared backbone trainable parameters:", backbone_params)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Full SimCLR trainable parameters:", trainable_params)

    loss_log = simclr_pretrain(params, model, data_loader, optimizer, scheduler)

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

    print("all done!")
    return model, loss_log


def run_testing():
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

    # Throw away the projection head; keep only the encoder
    simclr_eval(model.encoder, eval_train_loader, eval_test_loader, device)
    

@torch.no_grad()
def extract_features(encoder, data_loader, device):
    """extract features from the data."""

    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            x = encoder(x).cpu()
            feats.append(x)
            labels.append(y)

    feats = torch.cat(feats)
    labels = torch.cat(labels)
    return feats, labels


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

    device = supporting_func.get_device(prefer_mps=False)
    print("[LOG] -- Using device: ", device)   # type: ignore
    common_config_path = os.path.join(supporting_func.get_project_root(), "src", "configs", "ssl_shared_params.json")
    # Same merge as ssl_factory.train_ssl_model: model preset first, shared training params on top.
    common_params = supporting_func.load_json_file(common_config_path)
    params = {**get_ssl_method_preset("simclr"), **common_params}
    encode_batch_size = params["encode_batch_size"]

    n_train = opts.n_train
    trace_window, input_length = supporting_func.get_trace_window(opts.trace_window)

    output_dir = opts.output
    os.makedirs(output_dir, exist_ok=True)

    target_byte = params['target_byte']
    attack_window = params['attack_window']
    leakage_model = opts.leakage_model
    random_shift = params['random_shift']

    # Load full traces. train_simclr and encode_representations apply trace_window
    # exactly once, after view augmentation, so the loader must not cut it as well.
    print("Loading the profiling traces for model training...")
    train_data_dict = data_loader.load_dataset_base(opts.train_input)
    train_loader = data_loader.build_loader(train_data_dict, target_byte, attack_window, leakage_model, random_shift=0)

    print("Loading the attack traces for KNN evaluation...")
    test_data_dict = data_loader.load_dataset_base(opts.test_input)
    test_loader = data_loader.build_loader(test_data_dict, target_byte, attack_window, leakage_model, random_shift)

    print("Start Training SimCLR...")
    model, loss_log = run_training(params, X_train, device, trace_window)
    print("SimCLR loss log: ", loss_log)

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
