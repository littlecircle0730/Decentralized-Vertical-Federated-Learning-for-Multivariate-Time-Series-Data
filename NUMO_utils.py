import torch
import numpy as np
from networks.causal_cnn import CausalCNN, CausalCNNEncoder, SqueezeChannels
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time
import psutil
import json
import os
import losses
import math

def print_and_log(log_filename, *args, **kwargs):
    print(*args, **kwargs)
    with open(log_filename, "a") as log_file:
        print(*args, **kwargs, file=log_file)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # print("####### scores", scores.shape) #[32,1,5]
        attention_weights = self.softmax(scores)
        attended_features = torch.matmul(attention_weights, v)
        return attended_features  # (batch_size, 1, out_channels)
    
# class AttentionLayer(nn.Module):
#     def __init__(self, d_model, k_channels, out_channels):
#         super(AttentionLayer, self).__init__()
#         self.query = nn.Linear(d_model, out_channels)
#         self.key = nn.Linear(k_channels, out_channels)
#         self.value = nn.Linear(k_channels, out_channels)
#         self.attention = ScaledDotProductAttention(out_channels)
#         self.out_channels = out_channels

#     def forward(self, query, key, value):
#         q = self.query(query).unsqueeze(1)  # (batch_size, 1, out_channels)
#         k = self.key(key)  # (batch_size, seq_len, out_channels)
#         v = self.value(value)  # (batch_size, seq_len, out_channels)
        
#         attended_features = self.attention(q, k, v)  # (batch_size, 1, out_channels)
#         return attended_features.squeeze(1)  # (batch_size, out_channels)

class CombinedModelWithGeneralAttention(nn.Module):
    def __init__(self, encoder, regressor, query_dim, key_dim, value_dim):
        super(CombinedModelWithGeneralAttention, self).__init__()
        self.encoder = encoder
        self.regressor = regressor
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

    def forward(self, x, query, key, value):
        x = x.to(next(self.encoder.parameters()).device)
        features = self.encoder(x, query, key, value)
        outputs = self.regressor(features)
        return outputs
    

# class CausalCNNEncoderWithAttention(nn.Module):
#     def __init__(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size, attention_dim):
#         super(CausalCNNEncoderWithAttention, self).__init__()
#         self.causal_cnn = CausalCNN(in_channels, channels, depth, reduced_size, kernel_size)
#         self.reduce_size = nn.AdaptiveMaxPool1d(1)
#         self.squeeze = SqueezeChannels()
#         self.linear = nn.Linear(reduced_size, attention_dim)
#         self.attention_layer = AttentionLayer(attention_dim, in_channels, out_channels)
#         # self.attention_layer = AttentionLayer(attention_dim, attention_dim, out_channels)
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#     def forward(self, x, query, key, value):
#         x = x.to(next(self.causal_cnn.parameters()).device)
#         features = self.causal_cnn(x)
#         features = self.reduce_size(features)
#         features = self.squeeze(features)
#         features = self.linear(features)

#         # print("################### ", features.shape)
#         query = features  # (batch_size, attention_dim)
#         key = x.transpose(1, 2)  # (batch_size, seq_len, features)
#         value = x.transpose(1, 2)  # (batch_size, seq_len, features)

#         attended_features = self.attention_layer(query, key, value) #(batch_size, out_channels)
#         # print("############### attended_features ", attended_features.shape) #[32,20]
#         return attended_features



class CombinedModel(nn.Module):
    def __init__(self, encoder, regressor):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.regressor = regressor
        self.output_dim = regressor.output_dim

    def forward(self, x):
        x = x.to(next(self.encoder.parameters()).device)
        features = self.encoder(x)
        outputs = self.regressor(features)
        return outputs

class CausalCNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )
        self.in_channels = in_channels

    def forward(self, x):
        x = x.to(next(self.network.parameters()).device) 
        return self.network(x)

class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, features):
        outputs = self.linear(features)
        return outputs.view(features.size(0), self.output_dim)
# class LinearRegressor(nn.Module):
#     def __init__(self, input_dim, output_dim=3):
#         super(LinearRegressor, self).__init__()
#         layers = []
#         current_dim = input_dim
#         for hidden_dim in [16, 16]:
#             layers.append(nn.Linear(current_dim, hidden_dim))
#             layers.append(nn.LeakyReLU())
#             current_dim = hidden_dim
#         layers.append(nn.Linear(current_dim, output_dim))
#         self.network = nn.Sequential(*layers)
#         self.output_dim = output_dim 

#     def forward(self, features):
#         outputs = self.network(features)
#         return outputs.view(features.size(0), -1)

def normalize_data_with_scaler(data, scaler):
    reshaped_data = data.reshape(data.shape[0], -1)
    normalized_data = scaler.transform(reshaped_data)
    normalized_data = normalized_data.reshape(data.shape)
    return normalized_data


def fit_model_hyperparameters(model, train_data, train_labels, test_data=None, test_labels=None, initial_weights=None, cuda=False, gpu=0, num_epochs=20, learning_rate=0.003, batching=True, batch_size=16, scheduler=None, filename=None, client_id=0):
    
    initial_gpu_memory = torch.cuda.memory_allocated(gpu)
    history = TrainingHistory(gpu)

    if initial_weights is not None:
        model.load_state_dict(initial_weights)

    model.train()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-3) ####
    criterion = nn.MSELoss()

    device = torch.device(f"cuda:{gpu}" if cuda else "cpu")

    model = model.to(device)
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    # if test_data is not None and test_labels is not None:
    #     test_data = [data.to(device) for data in test_data]
    #     test_labels = [label.to(device) for label in test_labels]

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, train_data.size(0), batch_size):
            num_samples = train_data.shape[0]
            if batch_size > num_samples:
                batch_size = num_samples

            if batching:
                idx_list_batching = np.random.choice(train_data.shape[0], batch_size, replace=False)
                local_train = train_data[idx_list_batching]
                local_train_labels = train_labels[idx_list_batching]
            else:
                local_train = train_data[i:i+batch_size]
                local_train_labels = train_labels[i:i+batch_size]
    
            optimizer.zero_grad()
            batch_data = local_train
            batch_data = batch_data.to(torch.float32)
            batch_labels = local_train_labels.float()
            outputs = model(batch_data)
            
            # loss = criterion(outputs, batch_labels)
            loss = regression_score_calculate(outputs, batch_labels)

            loss.backward()

            optimizer.step()
            
            epoch_loss += loss.item()

        history.on_epoch_end(epoch_loss / (train_data.size(0) / batch_size), epoch, client_id)

    model.cpu()
    train_data = train_data.cpu()
    train_labels = train_labels.cpu()
    torch.cuda.empty_cache()

    history.save(f'{filename}_training_history.json')
    return model, history

# def fit_encoder_hyperparameters(encoder, train_data, train_labels, test_data=None, test_labels=None, initial_weights=None, cuda=False, gpu=0, num_epochs=20, learning_rate=0.001, batching=True, batch_size=16, scheduler=None, filename=None, client_id=0):
#     if initial_weights is not None:
#         encoder.load_state_dict(initial_weights)

#     encoder.train()
#     optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
#     # criterion = nn.MSELoss()
#     compared_length=50
#     nb_random_samples=10
#     negative_penalty=1
#     triplet_loss = losses.triplet_loss.TripletLoss(compared_length, nb_random_samples, negative_penalty)

#     if scheduler:
#         scheduler = scheduler(optimizer)


#     device = torch.device(f"cuda:{gpu}" if cuda else "cpu")

#     encoder = encoder.to(device)
#     train_data = train_data.to(device)
#     train_labels = train_labels.to(device)
#     if test_data is not None and test_labels is not None:
#         test_data = [data.to(device) for data in test_data]
#         test_labels = [label.to(device) for label in test_labels]


#     history = TrainingHistory()

#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         for i in range(0, train_data.size(0), batch_size):
#             num_samples = train_data.shape[0]
#             if batch_size > num_samples:
#                 batch_size = num_samples

#             if batching:
#                 idx_list_batching = np.random.choice(train_data.shape[0], batch_size, replace=False)
#                 local_train = train_data[idx_list_batching]
#                 local_train_labels = train_labels[idx_list_batching]
#             else:
#                 local_train = train_data[i:i+batch_size]
#                 local_train_labels = train_labels[i:i+batch_size]
    
#             optimizer.zero_grad()
#             batch_data = local_train
#             batch_data = batch_data.to(torch.float32)
#             # batch_labels = local_train_labels.float()
#             # outputs = model(batch_data)
            
#             #semi-supervise losses
#             loss = triplet_loss(batch_data, encoder, train_data)
            
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()

#         history.on_epoch_end(epoch_loss / (train_data.size(0) / batch_size), epoch, client_id)

#     history.save(f'{filename}_training_history.json')
#     return encoder, history


# def fit_regressor_hyperparameters(model, train_data, train_labels, test_data=None, test_labels=None, initial_weights=None, cuda=False, gpu=0, num_epochs=20, learning_rate=0.001, batching=True, batch_size=16, scheduler=None, filename=None, client_id=0):
#     if initial_weights is not None:
#         model.load_state_dict(initial_weights)

#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

#     if scheduler:
#         scheduler = scheduler(optimizer)

#     device = torch.device(f"cuda:{gpu}" if cuda else "cpu")

#     model = model.to(device)
#     train_data = train_data.to(device)
#     train_labels = train_labels.to(device)
#     if test_data is not None and test_labels is not None:
#         test_data = [data.to(device) for data in test_data]
#         test_labels = [label.to(device) for label in test_labels]

#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         for i in range(0, train_data.size(0), batch_size):
#             num_samples = train_data.shape[0]
#             if batch_size > num_samples:
#                 batch_size = num_samples

#             if batching:
#                 idx_list_batching = np.random.choice(train_data.shape[0], batch_size, replace=False)
#                 local_train = train_data[idx_list_batching]
#                 local_train_labels = train_labels[idx_list_batching]
#             else:
#                 local_train = train_data[i:i+batch_size]
#                 local_train_labels = train_labels[i:i+batch_size]
    
#             optimizer.zero_grad()
#             batch_data = local_train
#             batch_data = batch_data.to(torch.float32)
#             batch_labels = local_train_labels.float()
#             outputs = model(batch_data)
            
#             # loss = criterion(outputs, batch_labels)
#             loss = regression_score_calculate(outputs, batch_labels)

#             loss.backward()
#             optimizer.step()
#             if scheduler:
#                 scheduler.step()
            
#             epoch_loss += loss.item()

#     return model, None


def print_and_log(filename, message):
    with open(filename, 'a') as f:
        f.write(f"{message}\n")
    print(message)

def regression_score_calculate(outputs, targets):
    """
    Compute the mean squared error for a batch of outputs and targets.
    """
    total_mse = 0.0
    total_samples = 0

    for i in range(outputs.size(0)):
        for j in range(outputs.size(1)):
            label_value = targets[i, j].item()
            if label_value == -1:
                continue  # there is no traffic light, skip this direction
            
            error = (outputs[i, j] - targets[i, j]) ** 2 #MSE
            total_mse += error
            total_samples += 1

    mean_squared_error = total_mse / total_samples if total_samples > 0 else torch.tensor(0.0, requires_grad=True)
    return mean_squared_error

def regression_score(model, test_data, test_labels, filename=None, isSSL=False, encoder=None, check=False):
    """
    Compute the mean squared error of the regression model on the test data.

    Args:
    model (torch.nn.Module): The trained model to evaluate.
    test_data (list of torch.Tensor): List of test data for each vehicle, where each tensor has shape (sample, features, seq_length).
    test_labels (list of torch.Tensor): List of test labels for each vehicle, where each tensor has shape (sample, 3).

    Returns:
    float: The mean squared error of the model on the test data.
    """
    model.eval()
    total_samples = 0
    total_mse = 0.0

    # Initialize MSE lists for different label ranges
    mse_groups = {
        "0-5": {"mse": 0.0, "count": 0},
        "5-10": {"mse": 0.0, "count": 0},
        "10-20": {"mse": 0.0, "count": 0},
        "20+": {"mse": 0.0, "count": 0}
    }
    
    with torch.no_grad():
        for i in range(len(test_data)):  # vehicle_id
            for s in range(test_data[i].shape[0]):  # sample
                inputs = test_data[i][s].unsqueeze(0)  # [1, features, seq_length]

                #For SSL
                if isSSL and encoder!=None:
                    inputs = encoder(inputs)
                
                outputs = model(inputs)  # [1, 3]
                actual_labels = test_labels[i][s].unsqueeze(0)  # [1, 3]

                for j in range(3):
                    label_value = actual_labels[0][j].item()

                    if label_value == -1:
                        continue  # there is no traffic light, skip this direction
                    
                    error = ((outputs[0][j] - actual_labels[0][j]) ** 2).item()

                    total_mse += error
                    total_samples += 1
                    
                    if label_value <= 5:
                        mse_groups["0-5"]["mse"] += error
                        mse_groups["0-5"]["count"] += 1
                    elif 5 < label_value <= 10:
                        mse_groups["5-10"]["mse"] += error
                        mse_groups["5-10"]["count"] += 1
                    elif 10 < label_value <= 20:
                        mse_groups["10-20"]["mse"] += error
                        mse_groups["10-20"]["count"] += 1
                    else:
                        mse_groups["20+"]["mse"] += error
                        mse_groups["20+"]["count"] += 1
                
                # print_and_log(f"./{filename} check result", f"check result: {actual_labels}, {outputs}, MSE: {error}")
    
    # Calculate average MSE for each group
    for key in mse_groups:
        if mse_groups[key]["count"] > 0:
            mse_groups[key]["mse"] /= mse_groups[key]["count"]
        else:
            mse_groups[key]["mse"] = "No samples in this range"

    if filename:
        print_and_log(filename, f"MSE for different ranges: {mse_groups}")
        print_and_log(filename, f"Overall MSE: {total_mse / total_samples}")

    mean_squared_error = total_mse / total_samples
    return mean_squared_error

def regression_score_Attention(model, test_data, test_labels, filename=None, isSSL=False, encoder=None, check=False, query=None, key=None, value=None):
    model.eval()
    total_samples = 0
    total_mse = 0.0

    mse_groups = {
        "0-5": {"mse": 0.0, "count": 0},
        "5-10": {"mse": 0.0, "count": 0},
        "10-20": {"mse": 0.0, "count": 0},
        "20+": {"mse": 0.0, "count": 0}
    }
    
    with torch.no_grad():
        for i in range(len(test_data)):  # vehicle_id
            for s in range(test_data[i].shape[0]):  # sample
                inputs = test_data[i][s].unsqueeze(0)  # [1, features, seq_length]

                if isSSL and encoder != None:
                    inputs = encoder(inputs)

                if query is not None and key is not None and value is not None:
                    local_query = query[i].unsqueeze(0)
                    local_key = key[i].unsqueeze(0)
                    local_value = value[i].unsqueeze(0)
                    outputs = model(inputs, local_query, local_key, local_value)
                else:
                    outputs = model(inputs)

                actual_labels = test_labels[i][s].unsqueeze(0)  # [1, 3]

                for j in range(3):
                    label_value = actual_labels[0][j].item()

                    if label_value == -1:
                        continue

                    error = ((outputs[0][j] - actual_labels[0][j]) ** 2).item()

                    total_mse += error
                    total_samples += 1

                    if label_value <= 5:
                        mse_groups["0-5"]["mse"] += error
                        mse_groups["0-5"]["count"] += 1
                    elif 5 < label_value <= 10:
                        mse_groups["5-10"]["mse"] += error
                        mse_groups["5-10"]["count"] += 1
                    elif 10 < label_value <= 20:
                        mse_groups["10-20"]["mse"] += error
                        mse_groups["10-20"]["count"] += 1
                    else:
                        mse_groups["20+"]["mse"] += error
                        mse_groups["20+"]["count"] += 1

    for key in mse_groups:
        if mse_groups[key]["count"] > 0:
            mse_groups[key]["mse"] /= mse_groups[key]["count"]
        else:
            mse_groups[key]["mse"] = "No samples in this range"

    if filename:
        print_and_log(filename, f"MSE for different ranges: {mse_groups}")
        print_and_log(filename, f"Overall MSE: {total_mse / total_samples}")

    mean_squared_error = total_mse / total_samples
    return mean_squared_error



def adjust_direction(base_direction, target_direction):
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    direction_to_index = {d: i for i, d in enumerate(directions)}
    base_index = direction_to_index[base_direction]
    target_index = direction_to_index[target_direction]
    adjusted_index = (target_index - base_index + len(directions)) % len(directions)
    return directions[adjusted_index]


def map_to_state(triple):
    return triple[0] * 4 + triple[1] * 2 + triple[2]


def remove_features(data, features_to_remove):
    """
    Remove specified features from the data.
    Args:
    data (list of torch.Tensor): The input data.
    features_to_remove (list of int): The indices of the features to remove (0-based).
    Returns:
    list of torch.Tensor: The data with specified features removed.
    """
    filtered_data = []
    for d in data:
        d_np = d.detach().numpy()  # Convert each tensor to np array after moving to CPU
        d_filtered_np = np.delete(d_np, features_to_remove, axis=1)  # Remove specified features
        filtered_data.append(torch.tensor(d_filtered_np, dtype=torch.float32).to(d.device))  # Convert back to tensors and move to original device
    return filtered_data


class TrainingHistory:
    def __init__(self, gpu=None):
        self.times = []
        self.losses = []
        self.cpu_usages = []
        self.gpu_usages = []
        self.memory_usages = []
        self.epochs = []
        self.client_ids = []
        self.test_MSE = []
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())  # Get the current process
        self.gpu = gpu
        self.start_usage = self.process.cpu_times()  # Get initial CPU times
        # if self.gpu is not None:
        #     self.start_GPU = torch.cuda.memory_allocated(self.gpu)
        # else:
        #     self.start_GPU = 0

    def on_epoch_end(self, loss, epoch, client_id, test_score=None):
        current_time = time.time()
        epoch_time = current_time - self.start_time
        self.times.append(epoch_time)

        self.losses.append(loss)
        
        # Calculate the increment of CPU usage for this epoch
        current_cpu_times = self.process.cpu_times()
        current_cpu_usage = (current_cpu_times.user + current_cpu_times.system) - (self.start_usage.user + self.start_usage.system)
        self.cpu_usages.append(current_cpu_usage)  # Record actual CPU usage time
        
        self.memory_usages.append(self.process.memory_info().rss)  # Record RSS memory usage in bytes
        self.epochs.append(epoch)
        self.client_ids.append(client_id)
        if test_score is not None:
            self.test_MSE.append(test_score)

        if self.gpu is not None:
            # current_gpu = torch.cuda.memory_allocated(self.gpu)
            # current_gpu_usage = current_gpu - self.start_GPU
            # self.gpu_usages.append(current_gpu_usage)  # Record GPU memory usage
            # self.start_GPU = current_gpu  # Reset start_GPU to current usage
            self.gpu_usages.append(torch.cuda.memory_allocated(self.gpu))

        self.start_time = current_time  # Reset start time for next epoch
        self.start_usage = current_cpu_times # Reset start time for next epochs

    def to_dict(self):
        return {
            "times": self.times,
            "losses": self.losses,
            "cpu_usages": self.cpu_usages,
            "memory_usages": self.memory_usages,
            "gpu_usages": self.gpu_usages,
            "epochs": self.epochs,
            "client_ids": self.client_ids,
            "test_MSE": self.test_MSE
        }

    @classmethod
    def from_dict(cls, history_dict):
        history = cls()
        history.times = history_dict["times"]
        history.losses = history_dict["losses"]
        history.cpu_usages = history_dict["cpu_usages"]
        history.memory_usages = history_dict["memory_usages"]
        history.gpu_usages = history_dict.get("gpu_usages", [])
        history.epochs = history_dict["epochs"]
        history.client_ids = history_dict["client_ids"]
        history.test_MSE = history_dict.get("test_MSE", [])
        return history

    def save(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            existing_data["times"].extend(self.times)
            existing_data["losses"].extend(self.losses)
            existing_data["cpu_usages"].extend(self.cpu_usages)
            existing_data["memory_usages"].extend(self.memory_usages)
            existing_data["gpu_usages"].extend(self.gpu_usages)
            existing_data["epochs"].extend(self.epochs)
            existing_data["client_ids"].extend(self.client_ids)
            existing_data["test_MSE"].extend(self.test_MSE)
        else:
            existing_data = self.to_dict()

        with open(filepath, 'w') as f:
            json.dump(existing_data, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            history_dict = json.load(f)
        return cls.from_dict(history_dict)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pred_length):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim * pred_length)
        self.output_dim = output_dim
        self.pred_length = pred_length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        x = x.transpose(1, 2) 
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        # lstm_out = self.layer_norm(lstm_out)  # Apply Layer Normalization if needed
        out = self.fc(lstm_out[:, -1, :])
        return out.view(x.size(0), self.pred_length, self.output_dim)