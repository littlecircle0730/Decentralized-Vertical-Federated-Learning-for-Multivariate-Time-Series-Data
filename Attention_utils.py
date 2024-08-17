import torch
import numpy as np
from networks.causal_cnn import CausalCNN, SqueezeChannels
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time
import psutil
import json
import os

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
        attention_weights = self.softmax(scores)
        attended_features = torch.matmul(attention_weights, v)
        return attended_features  # (batch_size, seq_length, out_channels)

class AttentionLayer(nn.Module):
    def __init__(self, d_model, k_channels, out_channels):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(d_model, out_channels)
        self.key = nn.Linear(k_channels, out_channels)
        self.value = nn.Linear(k_channels, out_channels)
        self.attention = ScaledDotProductAttention(out_channels)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = key.size()
        # q = self.query(query)
        q = self.query(query).unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, out_channels)
        k = self.key(key)  # (batch_size, seq_len, out_channels)
        v = self.value(value)  # (batch_size, seq_len, out_channels)
        attended_features = self.attention(q, k, v)  # (batch_size, seq_len, out_channels)
        return attended_features  # (batch_size, seq_len, out_channels)

class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.leaky_relu(x)
        return x

class CausalCNNEncoderWithAttention(nn.Module):
    def __init__(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size, attention_dim, num_attentions):
        super(CausalCNNEncoderWithAttention, self).__init__()
        self.fully_connected_layers = nn.ModuleList([FullyConnectedLayer(6, attention_dim) for _ in range(4)])
        self.res_fc_layer = FullyConnectedLayer(7, attention_dim)
        self.causal_cnns = nn.ModuleList([
            CausalCNN(attention_dim, channels, depth, reduced_size, kernel_size) for _ in range(4)
        ] + [CausalCNN(attention_dim, channels, depth, reduced_size, kernel_size)])
        self.reduce_size = nn.AdaptiveMaxPool1d(1)
        self.squeeze = SqueezeChannels()
        self.linear = nn.Linear(reduced_size * num_attentions, out_channels)

    def calculate_pa_scores(self, speed, acce, cnt):
        # Ensure speed and cnt are tensors
        if not isinstance(speed, torch.Tensor):
            speed = torch.tensor(speed, dtype=torch.float32)
        if not isinstance(cnt, torch.Tensor):
            cnt = torch.tensor(cnt, dtype=torch.float32)
        
        log_cnt = torch.log(cnt + 1)

        pa_scores = torch.sigmoid(torch.abs(acce)) + torch.sigmoid(log_cnt)
        return torch.exp(pa_scores)

    def forward(self, x):
        cnn_features_list = []
        for i in range(4):
            causal_cnn = self.causal_cnns[i]
            cur_indices = [j for j in range(i, 24, 4)]
            cur_feature = x[:, cur_indices, :]  # Direction-specific features
            speed, acce,  cnt = cur_feature[:, 0, :], cur_feature[:, 1, :], cur_feature[:, 4, :] 
            pa_scores = self.calculate_pa_scores(speed, acce, cnt)
            
            # Apply fully connected layer to cur_feature
            fc_layer = self.fully_connected_layers[i]
            cur_feature = cur_feature.permute(0, 2, 1)  # Change shape to (batch_size, seq_length, features)
            cur_feature = fc_layer(cur_feature)  # Apply fully connected layer
            cur_feature = cur_feature.permute(0, 2, 1)  # Change back to (batch_size, features, seq_length)
            
            # Apply attention scores
            pa_scores = pa_scores.unsqueeze(1).expand_as(cur_feature)
            cur_feature = cur_feature * pa_scores
            
            cnn_features = causal_cnn(cur_feature)  # (batch_size, cnn_out_channels, seq_length)
            cnn_features = self.reduce_size(cnn_features)  # (batch_size, cnn_out_channels, 1)
            cnn_features = self.squeeze(cnn_features)  # (batch_size, cnn_out_channels)
            cnn_features_list.append(cnn_features)

        res_features = x[:, 24:, :]  # Remaining features
        res_features = res_features.permute(0, 2, 1)  # Change shape to (batch_size, seq_length, features)
        res_features = self.res_fc_layer(res_features)  # Apply fully connected layer
        res_features = res_features.permute(0, 2, 1)  # Change back to (batch_size, features, seq_length)
        causal_cnn = self.causal_cnns[-1]
        cnn_features = causal_cnn(res_features)  # (batch_size, cnn_out_channels, seq_length)
        cnn_features = self.reduce_size(cnn_features)  # (batch_size, cnn_out_channels, 1)
        cnn_features = self.squeeze(cnn_features)  # (batch_size, cnn_out_channels)
        cnn_features_list.append(cnn_features)

        cnn_features_concat = torch.cat(cnn_features_list, dim=1)  # (batch_size, cnn_out_channels * num_attentions)
        outputs = self.linear(cnn_features_concat)  # (batch_size, out_channels)

        return outputs

class CombinedModelWithGeneralAttention(nn.Module):
    def __init__(self, encoder, regressor):
        super(CombinedModelWithGeneralAttention, self).__init__()
        self.encoder = encoder
        self.regressor = regressor

    def forward(self, x):
        x = x.to(next(self.encoder.parameters()).device)
        features = self.encoder(x)
        outputs = self.regressor(features)
        return outputs

class LinearRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=3):
        super(LinearRegressor, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in [16, 16]:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        outputs = self.network(features)
        return outputs.view(features.size(0), -1)

def normalize_data_with_scaler(data, scaler):
    reshaped_data = data.reshape(data.shape[0], -1)
    normalized_data = scaler.transform(reshaped_data)
    normalized_data = normalized_data.reshape(data.shape)
    return normalized_data

def fit_attention_hyperparameters(model, train_data, train_labels, test_data=None, test_labels=None, initial_weights=None, cuda=False, gpu=0, 
                              num_epochs=20, learning_rate=0.001, batching=True, batch_size=16, scheduler=None, filename=None, client_id=0):
    if initial_weights is not None:
        model.load_state_dict(initial_weights)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = nn.MSELoss()

    if scheduler:
        scheduler = scheduler(optimizer)

    device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
    model = model.to(device)
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    if test_data is not None and test_labels is not None:
        test_data = [data.to(device) for data in test_data]
        test_labels = [label.to(device) for label in test_labels]

    history = TrainingHistory()

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
            
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, batch {i}")
                return model, None

            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()
            
            epoch_loss += loss.item()

        history.on_epoch_end(epoch_loss / (train_data.size(0) / batch_size), epoch, client_id)

    history.save(f'{filename}_training_history.json')
    return model, history

def regression_score(model, test_data, test_labels, filename=None, isSSL=False, encoder=None, check=False):
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

class TrainingHistory:
    def __init__(self):
        self.times = []
        self.losses = []
        self.cpu_usages = []
        self.memory_usages = []
        self.epochs = []
        self.client_ids = []
        self.test_MSE = []
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())  # Get the current process

    def on_epoch_end(self, loss, epoch, client_id, test_score=None):
        current_time = time.time()
        epoch_time = current_time - (self.times[-1] if self.times else self.start_time)
        self.times.append(epoch_time)
        self.losses.append(loss)
        self.cpu_usages.append(self.process.cpu_times().user + self.process.cpu_times().system)  # Record actual CPU usage time
        self.memory_usages.append(self.process.memory_info().rss)  # Record RSS memory usage in bytes
        self.epochs.append(epoch)
        self.client_ids.append(client_id)
        if test_score is not None:
            self.test_MSE.append(test_score)

    def to_dict(self):
        return {
            "times": self.times,
            "losses": self.losses,
            "cpu_usages": self.cpu_usages,
            "memory_usages": self.memory_usages,
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


def regression_score_Attention(model, test_data, test_labels, filename=None, isSSL=False, encoder=None, check=False):
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
        out = self.fc(lstm_out[:, -1, :])
        return out.view(x.size(0), self.pred_length, self.output_dim)