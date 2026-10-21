import os
import time
import pdb
import joblib
import numpy as np
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import src.utils.key_rank as key_rank
from src.utils.data_loader import get_labels, load_dataset
from src.utils.supporting_functions import plot_distinguisher_figure, load_or_save_ml_model


def train_one_knn(x_train, y_train, knn_model_path, leakage_model):
    ''' train and save the knn model  '''
    n_neighbors = 10

    # make sure the data is correct
    print('[LOG] -- using {} leakage model now'.format(leakage_model))
    if 'HW' == leakage_model:
        assert(set(y_train) == set(range(9)))
    elif 'ID' == leakage_model:
        assert(set(y_train) == set(range(256)))
    else:
        raise ValueError("The labels is not correct according to the leakage model")

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,  # n-shot Number of neighbors to use by default for kneighbors queries. n for n-shot learning
                               weights='distance',
                               p=2,  # Power parameter for the Minkowski metric.
                               metric='cosine',  # the distance metric to use for the tree.
                               algorithm='brute'  # Algorithm used to compute the nearest neighbors
                               )
    knn.fit(x_train, y_train)

    acc = accuracy_score(y_train, knn.predict(x_train))
    print('[LOG] -- acc on training data set is: ', acc)

    joblib.dump(knn, knn_model_path)
    print('[LOG] -- kNN model save to path: {}'.format(knn_model_path))
    return knn


def train_256_knn(train_x_feat, plaintext, target_byte, leakage_model="HW", n_neighbors=7, weights="distance", model_root_dir=None):
    """Fit one KNN per key guess on the same features; only the labels change.
    Saving is optional: every saved model holds its own copy of train_x_feat.
    """
    if model_root_dir is not None:                
        os.makedirs(model_root_dir, exist_ok=True)
    
    knn_models = []
    for guess_key in range(256):
        train_y = key_rank.leakage_labels(plaintext[:, target_byte], guess_key, leakage_model)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric="cosine", algorithm="brute").fit(train_x_feat, train_y)
        knn = train_one_knn(train_x_feat, train_y, os.path.join(model_root_dir, leakage_model, f"guess_key_{guess_key}_knn"))
        knn_models.append(knn)

    print('[LOG] -- 256 knn model training are all done!')
    return knn_models


def test_one_knn(x_test, y_test, guess_key, real_key, knn_model_path):
    """ generate ranking raw data """

    # trained knn models path
    knn_model = load_or_save_ml_model(knn_model_path, mode='load')
    print('[LOG] -- making predictions of guessed key {} and its corresopnding k-nn models {}'.format(guess_key, knn_model_path))

    y_preds = knn_model.predict(x_test)
    acc = accuracy_score(y_test, y_preds)
    print('[LOG] -- knn model: {}, guess key is: {}, real key is: {} test acc is: {}'.format(knn_model_path, guess_key, real_key, acc))

    del knn_model
    return acc


def test_256_knn(knn_models, test_x_feat, plaintext, target_byte, leakage_model="HW"):
    """Accuracy of each key guess's KNN on the attack features, as an array of 256."""
    acc_list = np.zeros(len(knn_models))
    for guess_key, knn in enumerate(tqdm(knn_models, desc="Testing 256 KNNs")):
        test_y = key_rank.leakage_labels(plaintext[:, target_byte], guess_key, leakage_model)
        acc_list[guess_key] = accuracy_score(test_y, knn.predict(test_x_feat))

    print('[LOG] -- test 256 knn all done!')
    return acc_list


def test_256_knn_shared_neighbors(knn_models, test_x_feat, train_plaintext, test_plaintext, target_byte, leakage_model="HW"):
    if len(knn_models) != 256:
        raise ValueError("Expected 256 fitted KNN models")
    if len(train_plaintext) != knn_models[0].n_samples_fit_:
        raise ValueError("Training plaintext must match the fitted training rows")
    if len(test_plaintext) != len(test_x_feat):
        raise ValueError("Test plaintext and features must have the same length")

    n_classes = {"HW": 9, "ID": 256}[leakage_model]

    # Every model was fitted on the same features with the same metric.
    distances, neighbor_rows = knn_models[0].kneighbors(test_x_feat)

    if knn_models[0].weights == "uniform":
        vote_weights = np.ones_like(distances)
    elif knn_models[0].weights == "distance":
        zero = distances == 0
        inverse = np.divide(
            1.0, distances,
            out=np.zeros_like(distances),
            where=~zero,
        )
        # Match sklearn: when a neighbor has distance zero,
        # only zero-distance neighbors get votes.
        vote_weights = np.where(
            zero.any(axis=1, keepdims=True),
            zero.astype(float),
            inverse,
        )
    else:
        raise ValueError("This function supports uniform or distance weights")

    row_ids = np.broadcast_to(
        np.arange(len(test_x_feat))[:, None],
        neighbor_rows.shape,
    )
    accuracies = np.empty(256, dtype=float)

    for guess_key in range(256):
        # These are the labels fitted into knn_models[guess_key].
        train_y = key_rank.leakage_labels(
            train_plaintext[:, target_byte], guess_key, leakage_model
        )
        test_y = key_rank.leakage_labels(
            test_plaintext[:, target_byte], guess_key, leakage_model
        )

        votes = np.zeros((len(test_x_feat), n_classes), dtype=float)
        np.add.at(
            votes,
            (row_ids, train_y[neighbor_rows]),
            vote_weights,
        )
        predictions = votes.argmax(axis=1)
        accuracies[guess_key] = np.mean(predictions == test_y)

    return accuracies


def compute_knn_pred_proba(knn_model_path, x_test, plaintext, key, params):                                   
    """ generate ranking raw data for one guessed key and its corresponding k-nn model """                                 
    print('[LOG] -- making predictions of guessed key and its corresopnding k-nn models {}...'.format(knn_model_path))     
    model_dir = params['model_dir']                                                                                                              
    ranking_root = os.path.join(model_dir, 'ranking_curve')

    target_byte = params['target_byte']                                                                                    
    leakage_model = params['leakage_model']                                                                                
    max_trace_num = params['max_trace_num']     # default is 5000                                                          
                                                                                                                           
                                                                                                                           
    if len(x_test) != len(plaintext):                                                                                      
        raise ValueError("Features and plaintext must have the same number of rows")                                       
    max_trace_num = min(params["max_trace_num"], len(x_test))                                                              
    if max_trace_num < 1:                                                                                                  
        raise ValueError("At least one test trace is required")                                                            
                                                                                                                                                                                                                           
    y_test = get_labels(plaintext, key, target_byte, leakage_model)                                                        
    knn_model = load_or_save_ml_model(knn_model_path, mode='load')                                                         
                                                                                                                                                                                                                              
    # trained knn models path                                                                                              
    print('[LOG] -- k-nn models path: ', knn_model_path)                                                                   
    acc = accuracy_score(y_test, knn_model.predict(x_test))                                                                
    print('[LOG] -- knn model: {}, real key is: {} test acc is: {}'.format(knn_model_path, key[target_byte], acc))       

    preds = knn_model.predict_proba(x_test)
    return acc, preds


def compute_knn_ranks(preds_proba, plaintext, key, target_byte, leakage_model, output_dir):                                                                                                                                                                                                                         
    """Function to compute the key guess rank, plot the figure and save the raw data."""
    # now key_rank ranking curve is a class
    min_trace_idx = 0
    max_trace_idx = 10000
    rank_step = 10 

    rank_curve_obj = key_rank.KeyRanking(leakage_model, min_trace_idx, max_trace_idx, target_byte, rank_step, output_dir)
    x, y = rank_curve_obj.ranking_curve(preds_proba, plaintext, key)
    rank_curve_obj.save_rank_data(x, y)           
                                                                                                                                                                                                                                
    print('[LOG] -- all done for model testing!') # type: ignore                                                           
                                                                                                                                                                                                                          