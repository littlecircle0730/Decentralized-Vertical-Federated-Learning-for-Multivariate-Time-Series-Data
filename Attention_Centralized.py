# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import torch
import numpy as np
import argparse
import time
import random
from datetime import datetime

from networks.causal_cnn import CausalCNNEncoder
from datetime import datetime

from NUMO_load_data import load_numo_dataset, smoothData
from Attention_utils import print_and_log, regression_score, regression_score_Attention, LinearRegressor, fit_attention_hyperparameters, remove_features, LSTMModel, CausalCNNEncoderWithAttention, CombinedModelWithGeneralAttention


def list_of_ints(arg):
   return list(map(int, arg.split(',')))
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for NUMO repository datasets'
    )
    parser.add_argument('--Perform_NUMO_experiments', type=bool, default=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', default='NUMO/',
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', default="Save_models/",
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', default="default_hyperparameters.json",
                        help='path of the file of hyperparameters to use ' +
                             'for training; must be a JSON file')

    parser.add_argument('--batching', type=bool, default=True)
    parser.add_argument('--save_pre_average_model', type=bool, default=False)
    parser.add_argument('--save_post_average_model', type=bool, default=False)
    parser.add_argument('--save_post_cluster_model', type=bool, default=False)

    parser.add_argument('--log_filename', type=str, default=f"./Log/log_central_TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    parser.add_argument('--raw_data_filename', type=str, default="NUMO_raw6")
    parser.add_argument('--processed_data_filename', type=str, default="cent2")
    parser.add_argument('--history_filename', type=str, default=f"central_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    parser.add_argument('--num_total_clients', type=int, default=3000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_length', type=int, default=5)
    parser.add_argument('--num_epoch', type=int, default=100)

    parser.add_argument('--in_channel', type=int, default=31)
    parser.add_argument('--remove_fea_index', type=list, default=[]) #time
    
    parser.add_argument('--isCommu', type=bool, default=False) ####all the pass in string will become True
    parser.add_argument('--isDCNN', type=int, default=1)
    parser.add_argument('--isAttention', type=int, default=1)


    return parser.parse_args()

def extract_features_from_model(model):
    weights = model.regressor.linear.weight.data.cpu().numpy().flatten()
    bias = model.regressor.linear.bias.data.cpu().numpy().flatten()
    return np.concatenate((weights, bias))

def pad_collate_fn(batch):
    batch = [item.clone().detach() for item in batch]
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)


if __name__=='__main__':
    args = parse_arguments()

    save_pre_average_model=args.save_pre_average_model
    save_post_average_model=args.save_post_average_model
    save_post_cluster_model=args.save_post_cluster_model

    log_filename = args.log_filename 
    his_filename = args.history_filename
    seq_length = args.seq_length

    cuda = args.cuda
    path=args.path
    save_path = args.save_path
    gpu = args.gpu
    hyper = args.hyper

    # Param
    processed_data_filename = args.processed_data_filename
    history_filename = args.history_filename
    log_filename = args.log_filename
    raw_data_filename = args.raw_data_filename
    learning_rate = args.learning_rate
    num_epoch=args.num_epoch
    seq_length = args.seq_length
    remove_fea_index = args.remove_fea_index
    isCommu = args.isCommu
    isDCNN = args.isDCNN
    isAttention = args.isAttention
    print("Parsed isCommu:", args.isCommu)

    if cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        cuda = False

    print("CUDA check:", torch.cuda.is_available())

    torch.manual_seed(4321)
    np.random.seed(1234)
    
    # train[i] should be feature_set for client i, train_labels[i] should be target/ labels for client i
    train, train_labels, test, test_labels = load_numo_dataset(path, seq_length=seq_length, processed_filename=processed_data_filename, raw_data_filename=raw_data_filename, isCommu=isCommu)

    #smooth data
    train = smoothData(train, [i for i in range(16)], 2)
    test = smoothData(test, [i for i in range(16)], 2)
    # remove log of log_cnt
    for i in range(len(train)):
        train[i][:, 16:20, :] = torch.exp(train[i][:, 16:20, :]) - 1
    for i in range(len(test)):
        test[i][:, 16:20, :] = torch.exp(test[i][:, 16:20, :]) - 1

    # Remove specified features from train and test data
    train= remove_features(train, remove_fea_index)
    test = remove_features(test, remove_fea_index)

    batching= args.batching

    # Param
    num_total_clients = args.num_total_clients
    learning_rate = args.learning_rate
    num_clients = num_total_clients
    batch_size = args.batch_size
    batching= args.batching
    in_channels = args.in_channel

    local_classifier_coef = []
    local_classifier_intercept = []
    output_dim = 3  # output types

    #shuffle
    random.seed(444)
    combined = list(zip(train, train_labels))
    random.shuffle(combined)
    train, train_labels = zip(*combined)
    train = list(train)
    train_labels = list(train_labels)

    # for mem load
    train = train[:num_clients]
    train_labels = train_labels[:num_clients]
    # test = test[:num_clients]
    # test_labels = test_labels[:num_clients]
    print_and_log(log_filename, f"Total Vehicle size: {len(train)}, Current chosen client size: {num_clients}")


    # set 
    depths = [2]
    channels_lists = [20]
    reduced_size_lists = [16]
    out_channels_lists = [20]
    kernel_size_lists = [2]
    learning_rates_lists = [learning_rate]
    batch_size_lists = [batch_size]
    epoch_size_lists = [num_epoch]
    attention_dim = 10


    best_MSE = 1000000
    best_params = {}
    for batch_size in batch_size_lists:
        for num_epochs in epoch_size_lists:
            for depth in depths:
                for channels in channels_lists:
                    for reduced_size in reduced_size_lists:
                        if reduced_size>channels: continue
                        for out_channels in out_channels_lists:
                            for kernel_size in kernel_size_lists:
                                for learning_rate in learning_rates_lists:

                                    log_filename = log_filename
                                    params = {
                                        'batch_size': batch_size,
                                        'num_epochs': num_epochs,
                                        'depth': depth,
                                        'channels': channels,
                                        'reduced_size': reduced_size,
                                        'out_channels': out_channels,
                                        'kernel_size': kernel_size,
                                        'learning_rate': learning_rate,
                                        'isDCNN': isDCNN,
                                        'isAttention': isAttention
                                    }
                                    
                                    if isDCNN:
                                        if isAttention:
                                            query_list = []
                                            key_list = []
                                            value_list = []
                                            encoder = CausalCNNEncoderWithAttention(
                                                in_channels, channels, depth, reduced_size, out_channels, kernel_size, attention_dim, num_attentions=5
                                            )
                                            regressor = LinearRegressor(out_channels, output_dim)
                                            model = CombinedModelWithGeneralAttention(encoder, regressor)


                                    if cuda:
                                        model = model.cuda(gpu)
                                        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

                                    local_models = model

                                    central_train = []
                                    central_train_labels = []

                                    print_and_log(log_filename, f"Curretn Vehicle size: {len(train)}, Current Client size: {num_clients}")

                                    for i in range(num_clients):
                                        vehicle_data = train[i]  # (sample, features, seq)
                                        vehicle_labels = train_labels[i]  # (sample, value)
                                        central_train.append(vehicle_data)
                                        central_train_labels.append(vehicle_labels)

                                    # Concatenate all reshaped data
                                    central_train = torch.cat(central_train, dim=0)
                                    central_train_labels = torch.cat(central_train_labels, dim=0)

                                    print_and_log(log_filename, f"len(central_train) {len(central_train)} len(test) {len(test)}")


                                    # Start training

                                    batch_idx_list = []
                                    initial_weights = None

                                    device_id = gpu

                                    start_time = time.time()

                                    # Train combined model; 
                                    # make sure train_data shape is [batch_size, in_channels, sequence_length]
                                    if isAttention:
                                        local_models, history = fit_attention_hyperparameters(
                                            local_models, 
                                            central_train, 
                                            central_train_labels, 
                                            None, 
                                            None, 
                                            initial_weights, 
                                            cuda, 
                                            gpu=device_id, 
                                            num_epochs=num_epochs, 
                                            learning_rate=learning_rate, 
                                            batch_size=batch_size, 
                                            scheduler=None, 
                                            filename=history_filename, 
                                            client_id=0, 
                                        )

                                    end_time = time.time()
                                    training_time = end_time - start_time

                                    print_and_log(log_filename, f"Current param: { params}")
                                    print_and_log(log_filename, f"Training time: {training_time:.2f} seconds")
                                    
                                    mDevice = torch.device(f'cuda:{gpu}' if cuda else "cpu")
                                    test_data_on_device = [data.to(mDevice) for data in test]
                                    test_labels_on_device = [label.to(mDevice) for label in test_labels]

                                    if isAttention:
                                        test_MSE = regression_score_Attention(local_models, test, test_labels, filename=log_filename, check=True)
                                    else:
                                        test_MSE = regression_score(local_models, test, test_labels, filename=log_filename, check=True)
                                    print_and_log(log_filename, f"Test MSE: {test_MSE}\n")

                                    if test_MSE < best_MSE:
                                        best_MSE = test_MSE
                                        best_params = params


    print_and_log( log_filename, f'Best MSE: {best_MSE}')
    print_and_log( log_filename, f'Best Params: {best_params}')