#!/usr/bin/env python3
"""
PyTorch port of ASCAD_train_models.py (originally TensorFlow / Keras). Using this to validate all
the supporting functions and util functions are working correctly in this project.
"""

import os
import sys
import pdb
import ast
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.supporting_functions as supporting_functions
import src.models.model_zoo as model_zoo
import src.utils.data_loader as data_loader
import src.utils.key_rank as key_rank


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_one_epoch(model, data_loader, criterion, optimizer, device):
    """train the model for one epoch, and evaluate on the validation set if provided. Returns the loss and accuracy."""
    is_train = True if optimizer else False
    model.train(is_train)

    loss_sum = 0.0
    correct = 0
    seen = 0
    n_batches = len(data_loader)

    with torch.set_grad_enabled(is_train):     # no_grad() when evaluating
        for batch_index, (x, y, _) in enumerate(data_loader):
            x = x.float()
            x = x.to(device=device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            y = y.to(device=device)

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                 optimizer.zero_grad()
                 loss.backward()
                 optimizer.step()

            bs = x.shape[0]
            loss_sum += loss.item() * bs
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += bs

            tmp_acc_ave = correct / seen
            tmp_loss_ave = loss_sum / seen
            if (batch_index + 1) % 100 == 0 or (batch_index + 1) == n_batches:
                print(f"batch: {batch_index+1:4d}/{batch_index} | loss: {tmp_loss_ave:.4f} | acc: {tmp_acc_ave:.4f}")

        acc_ave = correct / seen
        loss_ave = loss_sum / seen

    return loss_ave, acc_ave


def train_model(model, data_dict, params, save_file_name, verbose=1):
    """PyTorch equivalent of the Keras train_model().
    Mirrors the Keras callbacks: the model is checkpointed after every epoch,
    and early_stopping!=0 adds patience-10 early stopping on val_loss with
    best-weight restoration. validation_split uses Keras semantics: the LAST
    fraction of the data, taken before shuffling.
    """
    device = supporting_functions.get_device()
    save_dir = os.path.dirname(save_file_name)

    early_stopping = params['early_stop']
    split_ratio = params['split_ratio']
    target_byte = params['target_byte']
    attack_window = params['attack_window']
    leakage_model = params['leakage_model']
    random_shift = params['random_shift']
    epochs = params['epochs']

    train_dict, val_dict = data_loader.train_val_split(data_dict, split_ratio)

    # build_loader(data_dict, target_byte, attack_window, leakage_model, random_shift=0, trace_num=0)
    train_loader, train_data_params = data_loader.build_loader(train_dict, target_byte, attack_window, leakage_model, random_shift)
    val_loader, val_data_params = data_loader.build_loader(val_dict, target_byte, attack_window, leakage_model, random_shift)

    # Sanity check (replaces the Keras input_layer_shape check)
    if model.input_length is not None and model.input_length != len(train_dict['X_train'][0]):
        raise ValueError("Error: model input shape {} instead of {} is not expected ...".format(model.input_dim, len(train_dict['X_train'][0])))

    model.to(device)
    optimizer = model.configure_optimizer()
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    best_val = math.inf
    best_state = None
    patience = 10
    waited = 0

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("[train...]")
        loss, acc = run_one_epoch(model, train_loader, criterion, optimizer, device)
        history["loss"].append(loss)
        history["accuracy"].append(acc)

        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, None, device)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        line = " - loss: {:.4f} - accuracy: {:.4f}".format(loss, acc)
        line += " - val_loss: {:.4f} - val_accuracy: {:.4f}".format(val_loss, val_acc)
        print(line)

        # ModelCheckpoint: save after every epoch
        model_zoo.save_sca_model(model, save_file_name)

        # EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        if early_stopping and val_loss is not None:
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                waited = 0
            else:
                waited += 1
                if waited >= patience:
                    if verbose:
                        print("Early stopping at epoch {} (restoring best weights)".format(epoch + 1))
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    model_zoo.save_sca_model(model, save_file_name)
                    break

    supporting_functions.plot_loss_curve(history, save_dir)
    return model


# Run training with the data and a CNN model for profiling attack
def run_training(dataset_dir, dataset_name, params):
    # default parameters values, CNN training
    dataset_path = os.path.join(dataset_dir, dataset_name)
    model_save_name = f"cnn_{dataset_name}_100.pt"
    model_save_dir = os.path.join(dataset_dir, 'model_saved')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, model_save_name)

    # load traces and get network type
    attack_window = params["attack_window"]
    model_name = "cnn_best_norm"
    _, input_length = supporting_functions.get_attack_window(attack_window)
    emb_size = params['emb_size']

    ### Training the model
    best_model = model_zoo.build_model(model_name, input_length, emb_size)
    print(best_model)
    data_dict = data_loader.load_dataset_base(dataset_path)
    trained_model = train_model(best_model, data_dict, params, model_save_path)

    return trained_model


@torch.no_grad()
def compute_acc_and_predict_proba(model, test_data_loader, data_num):
    """Function to compute the acc and predict proba with a trained DL model."""
    device = supporting_functions.get_device()

    model.eval()
    model.to(device)

    test_loss, correct = 0.0, 0
    proba_list, textin_list = [], []

    for data, target, textin in test_data_loader:
        data, target = data.to(device), target.to(device)

        s_output = model(data)    # logits

        # Loss: crossentropyloss expect logits
        test_loss += F.cross_entropy(s_output, target, reduction="sum").item()

        pred = s_output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

        # probabilities for ranking curve
        softmax_output = F.softmax(s_output, dim=1)
        proba_list.append(softmax_output.cpu().numpy())
        textin_list.append(textin)

    preds_proba = np.concatenate(proba_list, axis=0)
    textin_mat = np.concatenate(textin_list, axis=0)

    test_loss /= data_num
    acc = correct / data_num

    return acc, preds_proba, textin_mat


# Test a saved model against one of the ASCAD databases Attack traces
def test_model(params, trained_model, dataset_dir, output_dir):
    """The function run model on testing dataset and computing the rank and plot the results."""
    #default parameters values                                                                                                                                                                                                                                                                                                                                                                                                                                           
    dataset_path = os.path.join(dataset_dir, "ASCAD_test.npz")
    trace_num = params['trace_num']
    attack_window = params["attack_window"] 
    target_byte = params["target_byte"]
    leakage_model = params["leakage_model"]
    random_shift = params["random_shift"]
    
    # build_loader(data_dict, target_byte, attack_window, leakage_model, random_shift=0, trace_num=0)
    data_dict = data_loader.load_dataset_base(dataset_path)
    test_data_loader, test_data_params = data_loader.build_loader(data_dict, target_byte, attack_window, leakage_model, random_shift, trace_num)

    # def compute_acc_and_predict_proba(model, test_data_loader, data_num):
    data_num = test_data_params['data_num']
    acc, preds_proba, plaintext = compute_acc_and_predict_proba(trained_model, test_data_loader, data_num)
    print("The testing accuracy is: ", acc)

    key = test_data_params['key']
    key_rank.compute_key_ranks(preds_proba, plaintext, key, target_byte, leakage_model, output_dir)


def main():
    # set params:
    params = {
        "attack_window": "0_700",
        "trace_num": 0,
        "target_byte": 2,
        "epochs": 20,
        "leakage_model": "HW",
        "batch_size": 100,
        'early_stop': False,
        'split_ratio': 0.1,
        'random_shift': -1,
        'emb_size': 0
    }

    params['emb_size'] = 9 if 'HW' == params['leakage_model'] else 256
    # Run training
    dataset_name = "ASCAD_train.npz"
    dataset_dir = "/home/carl/work_dir/ssl-sidechannel-research/data/npz_data/ASCAD/ATM_AES_v1_fixed_key"
    trained_model = run_training(dataset_dir, dataset_name, params)

    # Test model
    output_dir = "/home/carl/work_dir/ssl-sidechannel-research/output/tmp"
    os.makedirs(output_dir, exist_ok=True)
    test_model(params, trained_model, dataset_dir, output_dir)


if __name__ == "__main__":
    main()
