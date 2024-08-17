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
import time
import math
import random

import torch.optim as optim

from NUMO_load_data import load_numo_dataset
from NUMO_utils import print_and_log, regression_score, LinearRegressor, CombinedModel, fit_model_hyperparameters, remove_features, LSTMModel, IndexedMaxHeap

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

    parser.add_argument('--num_groups', type=int, default=5, help='Number of groups for local aggregation') # number of communities
    parser.add_argument('--bandwidth', type=float, default=10.0, help='Bandwidth in Mbps for model transmission') #Mbps

    return parser.parse_args()

def assign_cpu_specifications(num_clients):
    cpu_specs = []
    for _ in range(num_clients):
        cpu_frequency = round(random.uniform(1.5, 3.5), 2)
        cpu_cores = random.randint(2, 8)
        cpu_specs.append({'frequency': cpu_frequency, 'cores': cpu_cores})
    return cpu_specs

def calculate_time_based_on_cpu(cpu_spec, base_time):
    # base on the "base device"
    base_frequency = 2 #GHz
    # base_cores = 1 
    base_performance = base_frequency
    
    target_performance = cpu_spec['frequency']
    performance_ratio = base_performance / target_performance
    adjusted_time = base_time * performance_ratio
    
    return adjusted_time


def print_group_info(group_assignment, log_filename):
    for i, group in enumerate(group_assignment):
        Ng = len(group)
        print_and_log(log_filename, f"Group {i+1}: {Ng} clients")

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

@measure_time
def aggregate_model_params(local_models, W_client, device):
    combined_model_params_list_all = [model.state_dict() for model in local_models]
    agg_params = copy.deepcopy(combined_model_params_list_all[0])
    for key in agg_params.keys():
        stacked_params = torch.stack([W_client[k] * combined_model_params_list_all[k][key] for k in range(len(local_models))])
        agg_params[key] = torch.sum(stacked_params, dim=0)
    return agg_params

@measure_time
def aggregate_within_group(local_models, group_indices, W_client, device):
    group_model_params = [local_models[i].state_dict() for i in group_indices]
    agg_params = copy.deepcopy(group_model_params[0])
    for key in agg_params.keys():
        stacked_params = torch.stack([W_client[i] * group_model_params[j][key] for j, i in enumerate(group_indices)])
        agg_params[key] = torch.sum(stacked_params, dim=0)
    return agg_params

def random_community_assignment(num_clients, num_community):
    community_sizes = np.random.multinomial(num_clients, np.ones(num_community)/num_community)

    indices = list(range(num_clients))
    random.shuffle(indices)
    community_assignment = []
    current_index = 0
    
    for size in community_sizes:
        community = indices[current_index:current_index + size]
        community_assignment.append(community)
        current_index += size

    return community_assignment

def calculate_aggregation_time_per_model(cpu_frequency_ghz, model_size_kb, base_frequency):
    base_time = model_size_kb / 1024 / base_frequency
    return base_time / (cpu_frequency_ghz / base_frequency)

def calculate_transmission_time(model_size_kb, bandwidth_mbps, num_models):
    # Adjust the transmission time to account for simultaneous transmissions
    effective_bandwidth_mbps = bandwidth_mbps / num_models
    return (model_size_kb / 1024) / effective_bandwidth_mbps

def calculate_total_time(M, B, Ng, Nc, cpu_frequency_ghz):
    if Ng == 1: 
        return 0
    M_bits = M * 1024 * 8 # M should be converted to bits (M * 1024 * 8)
    transmission_time = ((Ng - 1) * M_bits) / (B * 1e6)
    T_agg_one = calculate_aggregation_time_per_model(cpu_frequency_ghz, M, cpu_frequency_ghz)
    aggregation_time = (transmission_time + Ng * T_agg_one) * math.log(Nc, Ng)
    return aggregation_time

def find_last_layer_start_index(N, Ng):
    h = math.ceil(math.log(N * (Ng - 1) + 1) / math.log(Ng))
    prev_layer_total = (Ng**(h - 1) - 1) // (Ng - 1)
    start_index = prev_layer_total
    return start_index

def hierarchical_aggregation(models, W_client, Ng, device, model_size, bandwidth, community_max_heap, cpu_specs):
    if len(models) == 1 or Ng == 1:
        return models[0].state_dict(), 0

    cur_models = copy.deepcopy(models)
    cur_W_client = copy.deepcopy(W_client)
    orig_cpu_specs = copy.deepcopy(cpu_specs)
    total_aggregation_time = 0

    while len(cur_models) > 1:  # Continue until only one model remains
        # new_heap = [None for _ in range(len(cur_models) - len(cur_models) // Ng)]  # Reduce one layer
        new_heap_size = math.ceil(len(cur_models) / Ng)  # Number of parent nodes in the next layer
        new_heap = [None for _ in range(new_heap_size)]
        new_w_client = [0] * len(new_heap)

        for i in range(len(cur_models)):
            # _, _, model_id_in_group = cur_community_max_heap[i]
            model_indices_in_group = [ i*Ng+j+1 for j in range(Ng) if i*Ng+j+1<len(cur_models)]   # Convert range to list of integers

            if i == 0:  # If it's the root node, aggregate with itself
                group = [0] + model_indices_in_group
            else:
                group = model_indices_in_group  # Aggregate only with current model, parent will aggregate with its own group

            if len(group)==0: break

            group_models = [cur_models[idx] for idx in group]
            group_weights = [cur_W_client[idx] for idx in group]

            max_cpu_spec = orig_cpu_specs[i]  # Use faster CPU, do not update orig_cpu_specs

            agg_params, aggregation_time = aggregate_within_group(
                group_models, list(range(len(group_models))),
                [w / sum(group_weights) for w in group_weights], device
            )
            aggregation_time = calculate_time_based_on_cpu(max_cpu_spec, aggregation_time)
            total_aggregation_time += aggregation_time

            new_model = copy.deepcopy(group_models[0])
            new_model.load_state_dict(agg_params)
            new_heap[i] = new_model
            new_w_client[i] = sum(group_weights)

        # Update models and weights for the next layer
        cur_models = [model for model in new_heap if model is not None]
        cur_W_client = [w for w in new_w_client if w != 0]

    final_model_params = cur_models[0].state_dict()  # Final aggregated model
    return final_model_params, total_aggregation_time


def hierarchical_aggregation_ORIG(models, W_client, Ng, device, model_size, bandwidth, cpu_specs):
    """
    Hierarchical aggregation of models in groups of size group_t.
    Args:
    - models: List of models to be aggregated.
    - W_client: Weights for each client.
    - t: Group size for aggregation.
    - device: Device to perform computations on.
    - model_size: Size of each model in bytes.
    - bandwidth: Bandwidth in Mbps.
    Returns:
    - Aggregated model parameters.
    - Total aggregation time including transmission time.
    """

    if len(models)==1 or Ng == 1:
        return models[0].state_dict(), 0

    total_aggregation_time = 0
    transmission_time_per_model = (model_size / bandwidth) / 1024 / 1024 * 8 # to second

    cur_models = copy.deepcopy(models)
    cur_W_client = copy.deepcopy(W_client)
    cur_cpu_specs = copy.deepcopy(cpu_specs)

    while len(cur_models) >= Ng:
        new_models = []
        new_w_client = []
        new_cpu_specs = []
        cur_group_time = 0
        for i in range(0, len(cur_models), Ng):
            groupModels = cur_models[i:i+Ng]
            group_weights = cur_W_client[i:i+Ng]/sum(cur_W_client[i:i+Ng]) #normalization
            agg_params, local_agg_time = aggregate_within_group(groupModels, list(range(len(groupModels))), group_weights, device)

            selected_cpu_spec = random.choice( cur_cpu_specs[i : i+Ng] )
            cur_group_time = calculate_time_based_on_cpu(selected_cpu_spec, local_agg_time)

            new_cpu_specs.append(selected_cpu_spec)
            new_model = copy.deepcopy(cur_models[0])
            new_model.load_state_dict(agg_params)
            new_models.append(new_model)
            new_w_client.append(sum(cur_W_client[i:i+Ng]))

            # total_aggregation_time += cur_group_time + transmission_time_per_model * (len(models) - len(new_models))
            total_aggregation_time += cur_group_time #compare the aggregation step only

        cur_models = new_models[:]
        cur_W_client = new_w_client[:]
        cur_cpu_specs = new_cpu_specs[:]
    
    if len(cur_models) > 1:
        agg_params, local_agg_time = aggregate_within_group(cur_models, list(range(len(cur_models))), cur_W_client/sum(cur_W_client), device)
        final_model = copy.deepcopy(cur_models[0])
        final_model.load_state_dict(agg_params)
        
        selected_cpu_spec = random.choice(cur_cpu_specs)
        local_agg_time = calculate_time_based_on_cpu(selected_cpu_spec, local_agg_time)
        total_aggregation_time += local_agg_time
    else:
        final_model = copy.deepcopy(cur_models[0])
    return final_model.state_dict(), total_aggregation_time

def update_community_weights(local_loss, W_num_data, group_indices):
    group_loss = local_loss[group_indices]
    group_W_num_data = W_num_data[group_indices]
    group_score = group_loss * group_W_num_data
    group_base = np.sum(group_score)
    return group_score, group_base


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

    num_groups = args.num_groups
    B = args.bandwidth

    if cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        cuda = False

    print("CUDA check:", torch.cuda.is_available())


    torch.manual_seed(4321)
    np.random.seed(1234)
    
    # Load custom dataset here and specify nb_classes
    # train[i] should be feature_set for client i, train_labels[i] should be target/labels for client i
    train, train_labels, test, test_labels = load_numo_dataset(path, seq_length=seq_length, processed_filename=processed_data_filename, raw_data_filename=raw_data_filename, isCommu=isCommu)

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

    M = 37  # model size in KB

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

    # CPU specification randomly assign
    cpu_specs = assign_cpu_specifications(num_total_clients)
    for i, spec in enumerate(cpu_specs):
        print(f"Vehicle {i} - CPU Frequency: {spec['frequency']} GHz, Cores: {spec['cores']}")
    average_cpu_frequency = 2.5 # use the average to estimate

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

    m_t = np.zeros(num_clients)
    v_t = np.zeros(num_clients)

    # assign group
    community_assignment = random_community_assignment(num_clients, num_groups)
    print_group_info(community_assignment, log_filename)

    # find best vehicle number for a gorup, local_nv
    best_N_list = [] # best vehicle numbers for hieharchy of each group
    prev_N_list = best_N_list
    for i, community in enumerate(community_assignment):
        local_nv = None
        min_total_time = float('inf')
        Nc = len(community) 
        if Nc<=1:
            local_nv = 1
        else:
            for tempT in range(2, Nc + 1):
                total_time = calculate_total_time(M, B, tempT, Nc, average_cpu_frequency)
                if total_time < min_total_time:
                    min_total_time = total_time
                    local_nv = tempT
        best_N_list.append(local_nv)

    # build the heap for the first time
    community_max_heaps = []
    for i, community in enumerate(community_assignment):
        if prev_N_list and prev_N_list[i]!=best_N_list[i]:
            print("########## CHANGES !!!!!!!!!!!!!!!")
        community_cpu_specs = [cpu_specs[j] for j in community]
        vehicle_ids = community  # community includes vehicle_id
        community_max_heap = IndexedMaxHeap(community_cpu_specs, vehicle_ids, best_N_list[i])
        community_max_heaps.append(community_max_heap)


    for t in range(num_rounds):

        # random leave
        if t >= 79:
            for i, community in enumerate(community_assignment):
                for client_index in community[:]:  # Copy the list to avoid modification issues
                    if random.random() < 0.01:  # 0.01% leave
                        if len(community_assignment[i]) > 1:  # Ensure that the community has more than one client
                            community_assignment[i].remove(client_index)
                            community_assignment.append([client_index])

                            # Remove from heap
                            removed_successfully = community_max_heaps[i].remove(client_index)

                            # Create a new heap for the new community
                            new_heap = IndexedMaxHeap([cpu_specs[client_index]], [client_index], 1)
                            community_max_heaps.append(new_heap)
                            num_groups += 1

            community_max_heaps = [heap for commu, heap in zip(community_assignment, community_max_heaps) if len(commu) > 0]
            community_assignment = [commu for commu in community_assignment if len(commu) > 0]
            print_and_log(log_filename, f"Round {t}: Number of communities: {len(community_assignment)}, num_heap: {len(community_max_heaps)}")


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

        print_and_log(log_filename, "start calculate loss")
        local_loss = []
        for i in range(num_clients):
            local_loss.append(regression_score(local_models[i], train[i].unsqueeze(0), train_labels[i].unsqueeze(0),  filename=None))
        local_loss = np.array(local_loss)


        # update weight

        community_models = []
        base = np.zeros(num_groups)
        W_client = np.zeros(num_clients)
        community_weights = np.zeros(num_groups)

        print_and_log(log_filename, "start count weight")
        for i, community in enumerate(community_assignment):
            cur_score, cur_base = update_community_weights(local_loss, W_num_data, community)
            base[i] = cur_base
            if cur_base > 0:
                W_client[community] = cur_score / cur_base
            else:
                W_client[community] = 0

        # aggregate the model base on previous loss
        mDevice = torch.device(f'cuda:{gpu}' if cuda else "cpu")

        # find best vehicle number for a gorup, local_nv
        prev_N_list = best_N_list
        best_N_list = []
        for i, community in enumerate(community_assignment):
            local_nv = None
            min_total_time = float('inf')
            Nc = len(community) 
            if Nc<=1:
                local_nv = 1
            else:
                # for tempT in range(3, Nc + 1):
                for tempT in range(2, Nc + 1):
                    total_time = calculate_total_time(M, B, tempT, Nc, average_cpu_frequency)
                    if total_time < min_total_time:
                        min_total_time = total_time
                        local_nv = tempT
            best_N_list.append(local_nv)

        ## LOCAL AGG        
        max_local_agg_time = -1
        print_and_log(log_filename, "start local agg")
        for i, community in enumerate(community_assignment):
            
            # Use the precomputed community_max_heap
            community_max_heap = community_max_heaps[i]

            # get community_max_heap last element
            last_heap_indices = [int(entry[2]) for entry in community_max_heap.heap]

            # check & rearrange 
            if all(idx in community for idx in last_heap_indices):
                # rearrange base on last_heap_indices
                community_local_models = copy.deepcopy([local_models[j] for j in last_heap_indices])
                community_W_client = copy.deepcopy([W_client[j] for j in last_heap_indices])
                community_cpu_specs = copy.deepcopy([cpu_specs[j] for j in last_heap_indices])
            else:
                print_and_log(log_filename, f"Error: Some indices in last_heap_indices {last_heap_indices} are not in community {community}")
                continue

            community_agg_params, local_agg_time = hierarchical_aggregation(
                community_local_models, 
                community_W_client, 
                best_N_list[i], 
                mDevice, 
                M, 
                B, 
                community_max_heap, 
                community_cpu_specs
            )

            if local_agg_time > max_local_agg_time:
                max_local_agg_time = local_agg_time

            community_model = copy.deepcopy(local_models[0])
            community_model.load_state_dict(community_agg_params)
            community_models.append(community_model)
            community_weights[i] = base[i] / sum(base)
        print_and_log(log_filename, f"[Selected] Local aggregation time: {max_local_agg_time} seconds")

        ## Global AGG
        print_and_log(log_filename, "start Global Agg")
        agg_weights, global_agg_time = aggregate_model_params(community_models, community_weights, mDevice) # revised

        ## Compare: Random LOCAL AGG        
        max_local_agg_time_orig = -1
        print_and_log(log_filename, "start random agg")
        for i, community in enumerate(community_assignment):
            orig_community_local_models = copy.deepcopy([local_models[j] for j in community])
            orig_community_W_client = copy.deepcopy([W_client[j] for j in community])
            orig_community_cpu_specs = copy.deepcopy([cpu_specs[j] for j in community])

            orig_community_agg_params, orig_local_agg_time = hierarchical_aggregation_ORIG(orig_community_local_models, orig_community_W_client, best_N_list[i], mDevice, M, B, orig_community_cpu_specs)
            
            if orig_local_agg_time > max_local_agg_time_orig:
                max_local_agg_time_orig = orig_local_agg_time

        print_and_log(log_filename, f"[Centralized] Local aggregation time: {max_local_agg_time_orig} seconds")

        print_and_log(log_filename, f"Time Ratio: {max_local_agg_time/max_local_agg_time_orig if max_local_agg_time_orig>0 else 1}")


        # update local models
        for i in range(num_clients):
            local_models[i].load_state_dict(agg_weights)

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
            'isDCNN': isDCNN,
            'isCommu': isCommu
        }
        print_and_log( log_filename, f'Params: {params}')
