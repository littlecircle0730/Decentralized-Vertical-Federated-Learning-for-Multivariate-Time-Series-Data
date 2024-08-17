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

import os
import torch
import numpy as np
import argparse
import copy
from networks.causal_cnn import CausalCNNEncoder
from datetime import datetime
import random
import math

import torch.optim as optim

from NUMO_load_data import load_numo_dataset
from NUMO_utils import print_and_log, regression_score, LinearRegressor, CombinedModel, fit_model_hyperparameters, remove_features, LSTMModel

global num_total_clients
global batch_size
global learning_rate

def save_local_model_weights(local_models, round_num, log_filename):
    for i, model in enumerate(local_models):
        filename = f"{log_filename}_round{round_num}_client{i}_weights.pth"
        torch.save(model.state_dict(), filename)

def list_of_ints(arg):
    return list(map(int, arg.strip('[]').split(',')))

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

    parser.add_argument('--cluster_beta', type=float, default=1.5)
    parser.add_argument('--client_partition_beta', type=float, default=20.0)

    parser.add_argument('--save_pre_average_model', type=bool, default=False)
    parser.add_argument('--save_post_average_model', type=bool, default=False)
    parser.add_argument('--save_post_cluster_model', type=bool, default=False)

    parser.add_argument('--batching', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--seq_length', type=int, default=5)

    parser.add_argument('--log_filename', type=str, default=f"./Log/log_weighted_hasCommu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")    
    parser.add_argument('--raw_data_filename', type=str, default="NUMO_raw6")
    parser.add_argument('--processed_data_filename', type=str, default="cent2") #raw6
    
    parser.add_argument('--history_filename', type=str, default=f"weighted{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--num_total_clients', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epoch', type=int, default=3)
    parser.add_argument('--isCommu', type=bool, default=False)
    parser.add_argument('--isDCNN', type=int, default=1)

    parser.add_argument('--in_channel', type=int, default=31)
    parser.add_argument('--remove_fea_index', type=list_of_ints, default=[])

    return parser.parse_args()


def aggregate_model_params(local_models, W_client, device):
    combined_model_params_list_all = [model.state_dict() for model in local_models]
    agg_params = copy.deepcopy(combined_model_params_list_all[0])
    for key in agg_params.keys():
        stacked_params = torch.stack([W_client[k] * combined_model_params_list_all[k][key] for k in range(len(local_models))])
        agg_params[key] = torch.sum(stacked_params, dim=0)
    return agg_params

if __name__=='__main__':
    args = parse_arguments()

    save_pre_average_model=args.save_pre_average_model;
    save_post_average_model=args.save_post_average_model;
    save_post_cluster_model=args.save_post_cluster_model;

    cuda = args.cuda
    path=args.path
    save_path = args.save_path
    gpu = args.gpu

    processed_data_filename = args.processed_data_filename
    history_filename = args.history_filename
    log_filename = args.log_filename
    raw_data_filename = args.raw_data_filename
    learning_rate = args.learning_rate
    num_epoch=args.num_epoch
    seq_length = args.seq_length
    isCommu = args.isCommu
    remove_fea_index = args.remove_fea_index
    isDCNN = args.isDCNN


    if cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        cuda = False

    print("CUDA check:", torch.cuda.is_available())


    torch.manual_seed(4321)
    np.random.seed(1234)

    # Load custom dataset
    # train[i] should be feature_set for client i, train_labels[i] should be target/labels for client i
    train, train_labels, test, test_labels = load_numo_dataset(path, seq_length=seq_length, processed_filename=processed_data_filename, raw_data_filename=raw_data_filename, isCommu=isCommu)

    
    # num_total_clients = len(train)
    num_total_clients = args.num_total_clients
    
    learning_rate = args.learning_rate
    num_clients = num_total_clients
    batch_size = args.batch_size
    batching= args.batching
    num_rounds = args.num_rounds
    in_channels = args.in_channel  # numo features

    local_models = []
    local_classifier_coef = []
    local_classifier_intercept = []

    # set
    depth = 2  # network depth
    channels = 20  # hidden channel
    reduced_size = 16
    out_channels = 20
    kernel_size = 2
    output_dim = 3  # output types


    # For numo
    for i in range(num_clients):
        
        if isDCNN:
            # DCNN
            encoder = CausalCNNEncoder(in_channels, channels, depth, reduced_size, out_channels, kernel_size)
            regressor = LinearRegressor(out_channels, output_dim)
            model = CombinedModel(encoder, regressor)
        else:
            #LSTM
            LSTMEncoder = LSTMModel(in_channels, 80, 2, out_channels, 1)
            regressor = LinearRegressor(out_channels, output_dim)
            model = CombinedModel(LSTMEncoder, regressor)

        local_models.append(model)
        
    # #smooth data
    # train = smoothData(train, [i for i in range(16)], 2)
    # test = smoothData(test, [i for i in range(16)], 2)
    
    # # remove log of log_cnt
    # for i in range(len(train)):
    #     train[i][:, 16:20, :] = torch.exp(train[i][:, 16:20, :]) - 1
    # for i in range(len(test)):
    #     test[i][:, 16:20, :] = torch.exp(test[i][:, 16:20, :]) - 1

    # Remove specified features from train and test data
    train= remove_features(train, remove_fea_index)
    test = remove_features(test, remove_fea_index)
    
    # #shuffle
    random.seed(444)
    combined = list(zip(train, train_labels))
    random.shuffle(combined)
    train, train_labels = zip(*combined)
    train = list(train)
    train_labels = list(train_labels)

    # for mem load
    train = train[:num_clients]
    train_labels = train_labels[:num_clients]

    print_and_log(log_filename, f"len(train): {len(train)}  len(test): {len(test)}")
    
    total_data = sum(train[i].shape[0] for i in range(len(train)))
    W_num_data = [train[i].shape[0]/train[i].shape[0] for i in range(num_clients)]
    W_num_data = np.array(W_num_data)

    local_schedulers = []

    # calculate the loss of each client
    s_loss = np.array([1/(num_clients) for i in range(num_clients)])
    local_loss = np.array([1/(num_clients) for i in range(num_clients)])
    W_client = np.array([1/num_clients for i in range(num_clients)])

    m_t = np.zeros(num_clients)
    v_t = np.zeros(num_clients)

    # Start training
    for t in range(num_rounds):

        for i in range(num_clients):

            local_train = train[i] #shape=[sample, features, seq]
            local_train_labels = train_labels[i] #spahe=[sample, seq]

            if t == 0:
                initial_weights = None
            else:
                initial_weights = agg_weights

            device_id = None
            if cuda:
                device_id = gpu
        
            # train by the updated return model with 'local' data 
            local_models[i], history = fit_model_hyperparameters(local_models[i], local_train, local_train_labels, None, None, initial_weights, cuda, gpu=device_id, num_epochs=num_epoch, learning_rate=learning_rate, batch_size=batch_size, filename=history_filename, client_id=i, scheduler=None)
            # save model
            save_local_model_weights(local_models, t, log_filename)

        #V1
        local_loss = []
        for i in range(num_clients):
            local_loss.append(regression_score(local_models[i], train[i].unsqueeze(0), train_labels[i].unsqueeze(0),  filename=None))
        local_loss = np.array(local_loss)


        # update weight
        W_client = local_loss * W_num_data
        # W_client = W_client + local_loss * W_num_data
        W_client = (W_client) / np.sum(W_client)

        # aggregate the model base on previous loss
        mDevice = torch.device(f'cuda:{gpu}' if cuda else "cpu")
        agg_weights = aggregate_model_params(local_models, W_client, mDevice)

        # update local models
        for i in range(num_clients):
            local_models[i].load_state_dict(agg_weights)

        # test on the new model on local data
        for i in range(num_clients):
            s_loss[i] = regression_score(local_models[i], train[i].unsqueeze(0), train_labels[i].unsqueeze(0),  filename=None)

        print_and_log(log_filename, f"Test MSE Global (All Test) on round {t}: [{regression_score(local_models[0], test, test_labels,  filename=log_filename)}]" )
  
        params = {
            'batch_size': batch_size,
            'num_epochs': num_epoch,
            'depth': depth,
            'channels': channels,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'learning_rate': learning_rate,
            'isDCNN': isDCNN,
            'isCommu': isCommu
        }
        print_and_log( log_filename, f'Params: {params}')

    # # check model size
    # for i in range(num_clients):
    #     torch.save(local_models[i].state_dict(), f"{log_filename}_{i}_Weight_best_model.pth")
    #     model_size = os.path.getsize(f"{log_filename}_{i}_Weight_best_model.pth")
    #     print_and_log( log_filename, f"Model size: {model_size / 1e6} MB" )