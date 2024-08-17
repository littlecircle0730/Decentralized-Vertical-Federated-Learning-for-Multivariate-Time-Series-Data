import pandas as pd
import numpy as np
import torch
import os
import pickle
import collections
import random
import math
from collections import Counter
from NUMO_utils import adjust_direction, map_to_state
import ast

def smoothData(data, indexSets, window_size=3):
    smoothed_data = []
    for seq in data:
        smoothed_seq = np.copy(seq) #[sample, f, seq]
        for feature_index in indexSets:
            for sample in range(seq.shape[0]): # each sample
                feature_data = seq[sample, feature_index, :]
                smoothed_feature_data = np.convolve(feature_data, np.ones(window_size)/window_size, mode='same')
                smoothed_seq[sample, feature_index, :] = smoothed_feature_data
        smoothed_data.append(torch.tensor(smoothed_seq, dtype=torch.float32))
    
    return smoothed_data

def custom_three_bit_encode(signal_state):
    encoding = {
        -1: [0, 0, 0],  # no info
        0: [-1, -1, -1],  # 000
        1: [-1, -1, 1],  # 001
        2: [-1, 1, -1],  # 010
        3: [-1, 1, 1],  # 011
        4: [1, -1, -1],  # 100
        5: [1, -1, 1],  # 101
        6: [1, 1, -1],  # 110
        7: [1, 1, 1],  # 111
    }
    return encoding[signal_state]


def refactorState(data):
    newstate_data = []

    for i in range(len(data)):  # Iterate over each vehicle data sample
        seq = data[i]
        state_seq = np.copy(seq)  # [samples, features, sequence_length]
        state_seq_new = np.zeros((seq.shape[0], seq.shape[1] + 8, seq.shape[2]))  # [samples, features + 8, sequence_length]

        for sample_idx in range(seq.shape[0]):  # Iterate over each sample
            for time_step in range(seq.shape[2]):  # Iterate over each time step
                # Collect states for four directions
                signal_states = [seq[sample_idx, 20 + d, time_step].item() for d in range(4)]
                # Encode states
                encoded_states = [custom_three_bit_encode(state) for state in signal_states]
                # Flatten the encoded states into a single list
                flat_encoded_states = [bit for encoded_state in encoded_states for bit in encoded_state]
                # Move subsequent features backwards
                state_seq_new[sample_idx, 20+12:, time_step] = state_seq[sample_idx, 20+4:, time_step]
                # Insert one-hot encoded signal states
                state_seq_new[sample_idx, 20:20 + 12, time_step] = flat_encoded_states
        state_seq_new = torch.tensor(state_seq_new, dtype=torch.float32)
        newstate_data.append(state_seq_new)
    
    return newstate_data


def create_sequences(data, labels, seq_length):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - seq_length + 1):
        # sequence = data[i:i + seq_length] # seq of data
        # label = labels[i + seq_length -1] # time of phase switching time

        #Adjust time to hour
        sequence = data[i:i + seq_length]
        label = labels[i + seq_length - 1]
        time_steps = sequence[:, -1] // 3600
        sequence[:, -1] = time_steps

        sequences.append(sequence)
        sequence_labels.append(label)

    if len(sequences) > 0 and len(sequence_labels):
        sequences = np.array(sequences, dtype=np.float32)
        sequence_labels = np.array(sequence_labels, dtype=np.float32).reshape(-1, 3) #(3,) ->(1,3)
    else:
        sequences = np.empty((0, seq_length, data.shape[1]), dtype=np.float32)
        sequence_labels = np.empty((0, 3), dtype=np.float32)

    return sequences, sequence_labels


def split_data_alternating(data, labels, interval=1, seq_length=5):
    # data in one window: T-seq_length ~ T
    # the last feature has to be timestep

    data = pd.DataFrame(data)
    labels = pd.DataFrame(labels)

    minT, maxT = int(data[data.columns[-1]].min()), int(data[data.columns[-1]].max())
    data_list = []
    labels_list = []

    for start_step in range(minT, maxT - seq_length - 1, interval):
        end_step = start_step + seq_length
        mask = (data[data.columns[-1]] >= start_step) & (data[data.columns[-1]] < end_step)

        if mask.sum() >= seq_length:
            temp_data = data.loc[mask]
            temp_labels = labels.loc[mask]
            
            sequences, sequence_labels = create_sequences(temp_data.values, temp_labels.values, seq_length)
            if len(sequences) > 0: 
                data_list.append(sequences)
                labels_list.append(sequence_labels)

    if data_list:
        sequences = np.concatenate(data_list)
        sequence_labels = np.concatenate(labels_list)
    else:
        sequences, sequence_labels = np.empty((0, seq_length, data.shape[1]), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    
    sequences = torch.tensor(sequences, dtype=torch.float32)
    sequence_labels = torch.tensor(sequence_labels, dtype=torch.float32)

    return sequences, sequence_labels

def check_data_proportion(train_list, test_list):
    train_sum = sum(seq.shape[0] for seq in train_list)
    test_sum = sum(seq.shape[0] for seq in test_list)
    proportion = train_sum / (train_sum + test_sum)
    print("Train Test proportion:", proportion)
    # return 0.775 < proportion < 0.825
    return 0.5 <= proportion


def process_group(group, vehicle_ids, isCommu, directions, interval, seq_length, train_index, group_index):
    vehicle_train_sequences = collections.defaultdict(list)
    vehicle_test_sequences = collections.defaultdict(list)
    vehicle_train_labels = collections.defaultdict(list)
    vehicle_test_labels = collections.defaultdict(list)

    for vehicle_id in vehicle_ids:
        if isCommu:
            vehicle_data = group[group['VehicleID'] == vehicle_id]
        else:
            # Do not have neighbors' data
            vehicle_data = group[(group['VehicleID'] == vehicle_id) & (group['Obs_VehicleID'] == vehicle_id)]

        sorted_group = vehicle_data.sort_values(by='Step')
        grouped = sorted_group.groupby('Step')

        features_list = []
        labels_list = []

        for step, sub_group in grouped:
            states = sub_group[sub_group['Obs_VehicleID'] == vehicle_id]['TLS_State'].iloc[0]

            # new label
            labels = sub_group[sub_group['Obs_VehicleID'] == vehicle_id]['timeSwitch'].iloc[0] #labels is phase changing time

            base_direction = sub_group[sub_group['Obs_VehicleID'] == vehicle_id]['Obs_Direction'].iloc[0]
            direction_speed = {d: [] for d in directions}
            direction_acceleration = {d: [] for d in directions}
            direction_count = {d: 0 for d in directions}
            direction_states = {d: None for d in directions}

            for _, row in sub_group.iterrows():
                adj_direction = adjust_direction(base_direction, row['Obs_Direction'])
                direction_speed[adj_direction].append(float(row['Obs_Speed']))
                direction_acceleration[adj_direction].append(float(row['Obs_Acceleration']))
                direction_count[adj_direction] += 1
                if not direction_states[adj_direction]: direction_states[adj_direction] = states

            avg_speeds = [np.mean(direction_speed[d]) if direction_speed[d] else 0 for d in directions]
            avg_accelerations = [np.mean(direction_acceleration[d]) if direction_acceleration[d] else 0 for d in directions]
            var_speed = [np.var(direction_speed[d]) if direction_speed[d] else 0 for d in directions]
            var_accelerations = [np.var(direction_acceleration[d]) if direction_acceleration[d] else 0 for d in directions]
            cnt = [direction_count[d] for d in directions]
            log_cnt = [math.log(1 + c) for c in cnt]

            # new features
            minDur = sub_group[sub_group['Obs_VehicleID'] == vehicle_id]['minDur'].iloc[0]
            maxDur = sub_group[sub_group['Obs_VehicleID'] == vehicle_id]['maxDur'].iloc[0]

            # Clean and convert minDur and maxDur from string to list of floats
            if isinstance(minDur, str):
                minDur = ast.literal_eval(minDur)
            minDur = list(map(float, minDur))
            if isinstance(maxDur, str):
                maxDur = ast.literal_eval(maxDur)
            maxDur = list(map(float, maxDur))

            # Ensure labels are lists of floats
            if isinstance(labels, str):
                labels = ast.literal_eval(labels)
            labels = list(map(float, labels))

            minDur = [float(minDur_i) if minDur_i != -1 else 0.0 for minDur_i in minDur]
            maxDur = [float(maxDur_i) if maxDur_i != -1 else 0.0 for maxDur_i in maxDur]

            direction_states_list = [direction_states[adj_direction] if direction_states[d] else -1 for d in directions]
            # features = avg_speeds + avg_accelerations + var_speed + var_accelerations + log_cnt + direction_states_list + [step]  # 8+8+4+4+(1) noMinMax
            features = avg_speeds + avg_accelerations + var_speed + var_accelerations + log_cnt + direction_states_list + minDur + maxDur + [step]  # 8+8+4+4+[3+3](1) features
            features_list.append(features)
            labels_list.append(labels)

        features_list = np.array(features_list, dtype=np.float32)
        labels_list = np.array(labels_list, dtype=np.float32)

        # shape of seq: (n, seq=3, features=18, torch tensor
        seqs, lbls = split_data_alternating(features_list, labels_list, interval, seq_length)  # the last feature is timestep

        seqs = seqs.transpose(2, 1)

        if seqs.shape[0]>0:
            if group_index in train_index:
                vehicle_train_sequences[vehicle_id].append(seqs)
                vehicle_train_labels[vehicle_id].append(lbls)
            else:
                vehicle_test_sequences[vehicle_id].append(seqs)
                vehicle_test_labels[vehicle_id].append(lbls)

    return vehicle_train_sequences, vehicle_train_labels, vehicle_test_sequences, vehicle_test_labels

def load_and_prepare_data(data, seq_length, interval=1, train_intersections=None, test_intersections=None, isCommu=True):
    """
    Output shape = (n, features, seq)
    """
    directions = ['N', 'E', 'S', 'W']
    
    tls_groups = data.groupby(['TLS_X', 'TLS_Y'])
    print(f"Number of intersections: {len(tls_groups)}")

    # Use provided intersections for consistency
    if train_intersections is None or test_intersections is None:
        # num_train = int(len(tls_groups) * 0.8)
        num_train = int(len(tls_groups) * 0.9)
        # all_indices = list(tls_groups.groups.keys())
        # train_intersections = random.sample(all_indices, num_train)
        all_indices = list(range(len(tls_groups)))
        train_index = random.sample(all_indices, num_train)
        test_index = [i for i in all_indices if i not in train_index]
    
    tryTimes = 0
    
    while tryTimes < 20:
        print("####### Try time", tryTimes)
        vehicle_train_sequences_commu, vehicle_train_labels_commu = collections.defaultdict(list), collections.defaultdict(list)
        vehicle_test_sequences_commu, vehicle_test_labels_commu = collections.defaultdict(list), collections.defaultdict(list)

        vehicle_train_sequences_no_commu, vehicle_train_labels_no_commu = collections.defaultdict(list), collections.defaultdict(list)
        vehicle_test_sequences_no_commu, vehicle_test_labels_no_commu = collections.defaultdict(list), collections.defaultdict(list)

        group_index = 0  # Initialize the group index counter

        for (tls_x, tls_y), group in tls_groups:
            vehicle_ids = group['VehicleID'].unique()
            
            # Process with communication
            train_sequences_commu, train_labels_commu, test_sequences_commu, test_labels_commu = process_group(group, vehicle_ids, True, directions, interval, seq_length, train_index, group_index)
            # Process without communication
            train_sequences_no_commu, train_labels_no_commu, test_sequences_no_commu, test_labels_no_commu = process_group(group, vehicle_ids, False, directions, interval, seq_length, train_index, group_index)
            for vehicle_id in train_sequences_commu.keys():
                vehicle_train_sequences_commu[vehicle_id].extend(train_sequences_commu[vehicle_id])
                vehicle_train_labels_commu[vehicle_id].extend(train_labels_commu[vehicle_id])

            for vehicle_id in test_sequences_commu.keys():
                vehicle_test_sequences_commu[vehicle_id].extend(test_sequences_commu[vehicle_id])
                vehicle_test_labels_commu[vehicle_id].extend(test_labels_commu[vehicle_id])

            for vehicle_id in train_sequences_no_commu.keys():
                vehicle_train_sequences_no_commu[vehicle_id].extend(train_sequences_no_commu[vehicle_id])
                vehicle_train_labels_no_commu[vehicle_id].extend(train_labels_no_commu[vehicle_id])

            for vehicle_id in test_sequences_no_commu.keys():
                vehicle_test_sequences_no_commu[vehicle_id].extend(test_sequences_no_commu[vehicle_id])
                vehicle_test_labels_no_commu[vehicle_id].extend(test_labels_no_commu[vehicle_id])

            group_index += 1

        # Concatenate all sequences and labels for train and test sets
        train_sequences_list_commu = [torch.cat(vehicle_train_sequences_commu[vehicle_id], dim=0) for vehicle_id in vehicle_train_sequences_commu.keys()]
        train_labels_list_commu = [torch.cat(vehicle_train_labels_commu[vehicle_id], dim=0) for vehicle_id in vehicle_train_labels_commu.keys()]
        test_sequences_list_commu = [torch.cat(vehicle_test_sequences_commu[vehicle_id], dim=0) for vehicle_id in vehicle_test_sequences_commu.keys()]
        test_labels_list_commu = [torch.cat(vehicle_test_labels_commu[vehicle_id], dim=0) for vehicle_id in vehicle_test_labels_commu.keys()]

        train_sequences_list_no_commu = [torch.cat(vehicle_train_sequences_no_commu[vehicle_id], dim=0) for vehicle_id in vehicle_train_sequences_no_commu.keys()]
        train_labels_list_no_commu = [torch.cat(vehicle_train_labels_no_commu[vehicle_id], dim=0) for vehicle_id in vehicle_train_labels_no_commu.keys()]
        test_sequences_list_no_commu = [torch.cat(vehicle_test_sequences_no_commu[vehicle_id], dim=0) for vehicle_id in vehicle_test_sequences_no_commu.keys()]
        test_labels_list_no_commu = [torch.cat(vehicle_test_labels_no_commu[vehicle_id], dim=0) for vehicle_id in vehicle_test_labels_no_commu.keys()]

        tryTimes += 1
        if check_data_proportion(train_sequences_list_commu, test_sequences_list_commu) and check_data_proportion(train_sequences_list_no_commu, test_sequences_list_no_commu):
            break
        else:
            print("Train-test split proportion not satisfied, retrying...")

    print(f"Train sequences shape (Commu): {train_sequences_list_commu[-1].shape if train_sequences_list_commu else 'Empty'}")
    print(f"Train labels shape (Commu): {train_labels_list_commu[-1].shape if train_labels_list_commu else 'Empty'}")
    print(f"Test sequences shape (Commu): {test_sequences_list_commu[-1].shape if test_sequences_list_commu else 'Empty'}")
    print(f"Test labels shape (Commu): {test_labels_list_commu[-1].shape if test_labels_list_commu else 'Empty'}")

    print(f"Train sequences shape (No Commu): {train_sequences_list_no_commu[-1].shape if train_sequences_list_no_commu else 'Empty'}")
    print(f"Train labels shape (No Commu): {train_labels_list_no_commu[-1].shape if train_labels_list_no_commu else 'Empty'}")
    print(f"Test sequences shape (No Commu): {test_sequences_list_no_commu[-1].shape if test_sequences_list_no_commu else 'Empty'}")
    print(f"Test labels shape (No Commu): {test_labels_list_no_commu[-1].shape if test_labels_list_no_commu else 'Empty'}")

    return (
        train_sequences_list_commu, train_labels_list_commu, test_sequences_list_commu, test_labels_list_commu,
        train_sequences_list_no_commu, train_labels_list_no_commu, test_sequences_list_no_commu, test_labels_list_no_commu,
        train_intersections, test_intersections
    )


def load_numo_dataset(csv_path, seq_length=5, processed_filename="weighted", raw_data_filename="NUMO_raw6", isCommu=True):
    processed_data_dir = os.path.join(csv_path, "processed_data")
    processed_data_files = {
        f"{processed_filename}_train_commu": os.path.join(processed_data_dir, f"{processed_filename}_train_commu.pkl"),
        f"{processed_filename}_train_labels_commu": os.path.join(processed_data_dir, f"{processed_filename}_train_labels_commu.pkl"),
        f"{processed_filename}_test_commu": os.path.join(processed_data_dir, f"{processed_filename}_test_commu.pkl"),
        f"{processed_filename}_test_labels_commu": os.path.join(processed_data_dir, f"{processed_filename}_test_labels_commu.pkl"),
        f"{processed_filename}_train_no_commu": os.path.join(processed_data_dir, f"{processed_filename}_train_no_commu.pkl"),
        f"{processed_filename}_train_labels_no_commu": os.path.join(processed_data_dir, f"{processed_filename}_train_labels_no_commu.pkl"),
        f"{processed_filename}_test_no_commu": os.path.join(processed_data_dir, f"{processed_filename}_test_no_commu.pkl"),
        f"{processed_filename}_test_labels_no_commu": os.path.join(processed_data_dir, f"{processed_filename}_test_labels_no_commu.pkl"),
        f"{processed_filename}_intersections": os.path.join(processed_data_dir, f"{processed_filename}_intersections.pkl"),
    }

    if all(os.path.isfile(f) for f in processed_data_files.values()):
        print("USE the Previous Processed File!!!!")
        with open(processed_data_files[f"{processed_filename}_train_commu"], 'rb') as f:
            train_sequences_commu = pickle.load(f)
        with open(processed_data_files[f"{processed_filename}_train_labels_commu"], 'rb') as f:
            train_labels_commu = pickle.load(f)
        with open(processed_data_files[f"{processed_filename}_test_commu"], 'rb') as f:
            test_sequences_commu = pickle.load(f)
        with open(processed_data_files[f"{processed_filename}_test_labels_commu"], 'rb') as f:
            test_labels_commu = pickle.load(f)
        with open(processed_data_files[f"{processed_filename}_train_no_commu"], 'rb') as f:
            train_sequences_no_commu = pickle.load(f)
        with open(processed_data_files[f"{processed_filename}_train_labels_no_commu"], 'rb') as f:
            train_labels_no_commu = pickle.load(f)
        with open(processed_data_files[f"{processed_filename}_test_no_commu"], 'rb') as f:
            test_sequences_no_commu = pickle.load(f)
        with open(processed_data_files[f"{processed_filename}_test_labels_no_commu"], 'rb') as f:
            test_labels_no_commu = pickle.load(f)
        with open(processed_data_files[f"{processed_filename}_intersections"], 'rb') as f:
            intersections = pickle.load(f)
        print("Data loaded successfully from processed files.")
        train_intersections, test_intersections = intersections
    else:
        csv_file_path = os.path.join(csv_path, f"{raw_data_filename}.csv")
        if not os.path.isfile(csv_file_path):
            raise FileNotFoundError(f"CSV file not found at path: {csv_file_path}")

        data = pd.read_csv(csv_file_path, engine='python')

        # Drop rows where 'TLS_State' contains 'NA'
        data = data[~data['TLS_State'].apply(lambda x: 'NA' in x)]
        data['TLS_State'] = data['TLS_State'].str.strip("[]").str.split(',')

        label_mapping = {'r': 0, 'g': 1, 'y': 1, 'R': 0, 'G': 1, 'Y': 1}
        data['TLS_State'] = data['TLS_State'].apply(lambda x: [label_mapping.get(light.strip().replace("'", "").strip(), 1) for light in x])
        data['TLS_State'] = data['TLS_State'].apply(map_to_state)

        (
            train_sequences_commu, train_labels_commu, test_sequences_commu, test_labels_commu,
            train_sequences_no_commu, train_labels_no_commu, test_sequences_no_commu, test_labels_no_commu,
            train_intersections, test_intersections
        ) = load_and_prepare_data(data, seq_length, interval=1)

        # Filter sequences and labels
        train_sequences_commu = [seq for seq in train_sequences_commu if seq.numel() != 0]
        train_labels_commu = [label for label in train_labels_commu if label.numel() != 0]
        test_sequences_commu = [seq for seq in test_sequences_commu if seq.numel() != 0]
        test_labels_commu = [label for label in test_labels_commu if label.numel() != 0]

        train_sequences_no_commu = [seq for seq in train_sequences_no_commu if seq.numel() != 0]
        train_labels_no_commu = [label for label in train_labels_no_commu if label.numel() != 0]
        test_sequences_no_commu = [seq for seq in test_sequences_no_commu if seq.numel() != 0]
        test_labels_no_commu = [label for label in test_labels_no_commu if label.numel() != 0]

        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        with open(processed_data_files[f"{processed_filename}_train_commu"], 'wb') as f:
            pickle.dump(train_sequences_commu, f)
        with open(processed_data_files[f"{processed_filename}_train_labels_commu"], 'wb') as f:
            pickle.dump(train_labels_commu, f)
        with open(processed_data_files[f"{processed_filename}_test_commu"], 'wb') as f:
            pickle.dump(test_sequences_commu, f)
        with open(processed_data_files[f"{processed_filename}_test_labels_commu"], 'wb') as f:
            pickle.dump(test_labels_commu, f)
        with open(processed_data_files[f"{processed_filename}_train_no_commu"], 'wb') as f:
            pickle.dump(train_sequences_no_commu, f)
        with open(processed_data_files[f"{processed_filename}_train_labels_no_commu"], 'wb') as f:
            pickle.dump(train_labels_no_commu, f)
        with open(processed_data_files[f"{processed_filename}_test_no_commu"], 'wb') as f:
            pickle.dump(test_sequences_no_commu, f)
        with open(processed_data_files[f"{processed_filename}_test_labels_no_commu"], 'wb') as f:
            pickle.dump(test_labels_no_commu, f)
        with open(processed_data_files[f"{processed_filename}_intersections"], 'wb') as f:
            pickle.dump((train_intersections, test_intersections), f)
        print("Data processed and saved successfully.")

    if isCommu:
        print("Return data WITH Communication")
        return (train_sequences_commu, train_labels_commu, test_sequences_commu, test_labels_commu)
    else:
        print("Return data without Communication")
        return (train_sequences_no_commu, train_labels_no_commu, test_sequences_no_commu, test_labels_no_commu)
    
    
def save_to_csv(data, labels, data_filename, label_filename):
    """
    Save the data and labels to separate CSV files.

    Args:
    data (list of torch.Tensor): List of tensors containing vehicle data with shape (num_sample, num_features, num_timesteps).
    labels (list of torch.Tensor): List of tensors containing vehicle labels with shape (num_sample, num_timesteps).
    data_filename (str): The name of the output CSV file for data.
    labels_filename (str): The name of the output CSV file for labels.
    """
    data_df_list = []
    labels_df_list = []
    print(len(data), len(labels))
    if not len(data) or not len(labels):
        exit()

    for vehicle_data, vehicle_labels in zip(data, labels):
        print(vehicle_data.shape)
        num_samples, num_features, num_timesteps = vehicle_data.shape
        num_labels, num_output_timesteps = vehicle_labels.shape

        # Reshape data to (num_samples * num_timesteps, num_features)
        vehicle_data_2d = vehicle_data.permute(0, 2, 1).reshape(num_samples * num_timesteps, num_features)

        # Create DataFrame for data
        data_df = pd.DataFrame(vehicle_data_2d.numpy(), columns=[f'feature_{i}' for i in range(num_features)])
        data_df_list.append(data_df)

        # Create DataFrame for labels
        labels_df = pd.DataFrame(vehicle_labels.numpy(), columns=[f'label_{i+1}' for i in range(num_output_timesteps)])
        labels_df_list.append(labels_df)

    # Concatenate all DataFrames and save to CSV
    data_result = pd.concat(data_df_list, ignore_index=True)
    labels_result = pd.concat(labels_df_list, ignore_index=True)

    data_result.to_csv(f"NUMO/processed_data/{data_filename}", index=False)
    labels_result.to_csv(f"NUMO/processed_data/{label_filename}", index=False)