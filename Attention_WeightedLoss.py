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
import copy
from networks.causal_cnn import CausalCNNEncoder
from datetime import datetime
import random
import math

import torch.optim as optim

from NUMO_load_data import load_numo_dataset
from Attention_utils import print_and_log, regression_score, regression_score_Attention, LinearRegressor, fit_attention_hyperparameters, remove_features, CausalCNNEncoderWithAttention, CombinedModelWithGeneralAttention


global num_total_clients
global batch_size
global learning_rate


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
    parser.add_argument('--in_channel', type=int, default=31)
    parser.add_argument('--isCommu', type=bool, default=False)
    parser.add_argument('--isDCNN', type=int, default=1)
    parser.add_argument('--isAttention', type=int, default=1)

    return parser.parse_args()


def aggregate_model_params(local_models, W_client, device):
    combined_model_params_list_all = [model.state_dict() for model in local_models]
    agg_params = copy.deepcopy(combined_model_params_list_all[0])
    for key in agg_params.keys():
        stacked_params = torch.stack([W_client[k] * combined_model_params_list_all[k][key].to(device) for k in range(len(local_models))])
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
    isDCNN = args.isDCNN
    isAttention = args.isAttention


    if cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        cuda = False

    print("CUDA check:", torch.cuda.is_available())


    torch.manual_seed(4321)
    np.random.seed(1234)

    # # For reducing learning rate
    # scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.995)

    # Load custom dataset here and specify nb_classes
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

    #set
    depth = 2  # network depth
    channels = 20  # hidden channel
    reduced_size = 16
    out_channels = 20
    kernel_size = 2
    output_dim = 3  # output types
    attention_dim = 5


    # For numo
    for i in range(num_clients):
        
        # DCNN
        encoder = CausalCNNEncoderWithAttention(in_channels, channels, depth, reduced_size, out_channels, kernel_size, attention_dim, num_attentions=5)
        regressor = LinearRegressor(out_channels, output_dim)
        model = CombinedModelWithGeneralAttention(encoder, regressor)

        local_models.append(model)
    
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
    test = test[:num_clients]
    test_labels = test_labels[:num_clients]


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

            device_id = gpu
        
            # train by the updated return model with 'local' data 
            local_models[i], history = fit_attention_hyperparameters(
                local_models[i], 
                local_train, 
                local_train_labels,
                None, 
                None, 
                initial_weights, 
                cuda, 
                gpu=device_id, 
                num_epochs=num_epoch,
                learning_rate=learning_rate, 
                batch_size=batch_size, 
                scheduler=None, 
                filename=history_filename, 
                client_id=0
            )

        #V1
        local_loss = []
        for i in range(num_clients):
            local_loss.append(regression_score(local_models[i], train[i].unsqueeze(0), train_labels[i].unsqueeze(0),  filename=None))
        local_loss = np.array(local_loss)


        # update weight
        W_client = local_loss * W_num_data
        W_client = (W_client) / np.sum(W_client)

        # aggregate the model base on previous loss
        mDevice = torch.device(f'cuda:{t%8}' if cuda else "cpu")
        agg_weights = aggregate_model_params(local_models, W_client, mDevice)

        # update local models
        for i in range(num_clients):
            local_models[i].load_state_dict(agg_weights)

        # test on the new model on local data
        for i in range(num_clients):
            s_loss[i] = regression_score(local_models[i], train[i].unsqueeze(0), train_labels[i].unsqueeze(0),  filename=None)
        

        test_data_on_device = [data.to(mDevice) for data in test]
        test_labels_on_device = [label.to(mDevice) for label in test_labels]

        print_and_log(log_filename, f"------------------------------------------------Test MSE Global (All Test) on round {t}: [{regression_score(local_models[0], test, test_labels,  filename=log_filename)}]" )
  
        params = {
            'batch_size': batch_size,
            'num_epochs': num_epoch,
            'depth': depth,
            'channels': channels,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'learning_rate': learning_rate,
            'isDCNN': isDCNN
        }
        print_and_log( log_filename, f'Params: {params}')  