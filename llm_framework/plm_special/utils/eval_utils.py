import numpy as np
import torch
import time
import json
import psutil
import GPUtil
from munch import Munch
from torch.utils.data import DataLoader
import pandas as pd
import os
from datetime import datetime
from plm_special.utils.utils import process_batch
from plm_special.data.dataset import ExperienceDataset
import random
import pickle
import logging

col_dict = {
    'queue_type': 0,
    'burst_allowance': 1,
    'drop_probability': 2,
    'current_queue_delay': 3,
    'accumulated_probability': 4,
    'length_in_bytes': 5,
    'packet_length': 6
}
state_columns = list(col_dict.keys())


def convert_exp_pool_to_dataframe(exp_pool, csv_output_path='exp_pool_data.csv'):
    """
    Converts the given experience pool into a pandas DataFrame.
    Optionally saves the DataFrame to a CSV file and the experience pool as a dictionary to a pickle file.

    Args:
        exp_pool (object): The experience pool object containing states, actions, rewards, and dones.
        csv_output_path (str): Path to save the resulting DataFrame as a CSV file (default: 'exp_pool_data.csv').

    Returns:
        pd.DataFrame: The DataFrame representation of the experience pool.
    """
    
    # Step 1: Convert the Experience Pool to a DataFrame    
    # Create state column names based on the length of each state vector
    state_columns = [f'state_{i}' for i in range(len(exp_pool.states[0]))]  # Assuming each state is a 1D array
    state_columns = list(col_dict.keys())

    
    # Flatten the states into individual columns
    expanded_states = np.array([state for state in exp_pool.states])
    
    # Create the DataFrame with expanded states
    df = pd.DataFrame(expanded_states, columns=state_columns)
    
    # Add actions, rewards, and dones as columns to the DataFrame
    df['actions'] = exp_pool.actions
    df['rewards'] = exp_pool.rewards
    df['dones'] = exp_pool.dones

    # Step 2: Save the DataFrame to a CSV file
    # df.to_csv(csv_output_path, index=False)
    # print(f"DataFrame saved successfully to: {csv_output_path}")
    return df


def find_nearest_length(df, cur_index,states, new_action, new_queue_length):

    # Filter the DataFrame based on the queue type and action
    df_qt= df[df['queue_type']== int(states[0][0][col_dict['queue_type']])]
    df_ats= df_qt[df_qt['actions']== int(new_action.item())] 

    df_filtered = df_ats    
    
    # Find the nearest data point based on the queue length in bytes
    column = 'length_in_bytes'

    if column not in df_filtered.columns or df_filtered.empty:
        print(f"Column '{column}' not found or DataFrame is empty.")
        return None

    nearest_idx = (df_filtered[column] - new_queue_length).abs().idxmin()

    logging.debug("Nearest index: %s", nearest_idx)
    logging.debug()
    logging.debug("()()()()()"* 20)
    logging.debug("%s",(df_filtered[column] - new_queue_length).abs())
    logging.debug("df_filtered[column][nearest_idx]: %s",df_filtered[column][nearest_idx])
    logging.debug("new_queue_length",new_queue_length)
    logging.debug("df_filtered[column][nearest_idx] - new_queue_length: %s",df_filtered[column][nearest_idx] - new_queue_length)
    logging.debug("Nearest index: %s", nearest_idx)
    logging.debug("Current index: %s", cur_index)
    logging.debug("()()()()()"* 20)
    logging.debug()

    if nearest_idx >= df.index.max():
        print()
        print("---"* 20)
        print("Nearest index is out of bounds, returning None.", nearest_idx)
        print("len(df_filtered)", len(df_filtered))
        print("Outside df_ats limits")
        print("---"* 20)
        print()

    if pd.isna(nearest_idx):
        return None

    return nearest_idx


def tensor_to_list(tensor):
    # Detach the tensor and then convert it to a NumPy array and then to a list
    return tensor.detach().cpu().numpy().tolist()


def test_step(args, model, loss_fn, raw_batch, target_return):
    states, actions, returns, timesteps = raw_batch

    # Convert states to tensor and ensure correct shape
    states = torch.tensor(states[0], dtype=torch.float32).to(args.device).unsqueeze(0)  # Shape [1, 8]

    # Convert actions, returns, and timesteps to tensors
    actions = torch.tensor(actions, dtype=torch.float32).to(args.device)  # Shape [1, 1]
    returns = torch.tensor(returns, dtype=torch.float32).to(args.device)  # Shape [1, 1]
    timesteps = torch.tensor(timesteps, dtype=torch.int32).to(args.device)  # Shape [1, 1]

    # Create a batch with the correctly formatted tensors
    # Wrap states in a list to avoid TypeError in process_batch
    batch = ([states], [actions], [returns], [timesteps])  # Ensure states is a list

    # Call process_batch
    states, actions, returns, timesteps, labels = process_batch(batch, device=args.device)

    # Predict actions using the model
    # actions_pred1 = model(states, actions, returns, timesteps)
    queue_action = 0
    actions_pred1, queue_action = model.sample(states, target_return, timesteps)    

    # Permute for loss calculation
    actions_pred = actions_pred1.permute(0, 2, 1)
    loss = loss_fn(actions_pred, labels)

    # print("actions_pred1",actions_pred1)
    # print("actions_pred",actions_pred)
    # print("llm-queue_action",queue_action)
    # print("actual-queue_action",labels)

    return loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred


def otest_step(args, model, loss_fn, raw_batch, target_return):
    # Assuming raw_batch is a tuple of numpy arrays or lists
    states, actions, returns, timesteps = raw_batch

    # Convert states to tensor and ensure correct shape
    states = torch.tensor(states[0], dtype=torch.float32).to(args.device).unsqueeze(0)  # Shape [1, 8]

    # Convert actions, returns, and timesteps to tensors
    actions = torch.tensor(actions, dtype=torch.float32).to(args.device)  # Shape [1, 1]
    returns = torch.tensor(returns, dtype=torch.float32).to(args.device)  # Shape [1, 1]
    timesteps = torch.tensor(timesteps, dtype=torch.int32).to(args.device)  # Shape [1, 1]

    # Create a batch with the correctly formatted tensors
    # Wrap states in a list to avoid TypeError in process_batch
    batch = ([states], [actions], [returns], [timesteps])  # Ensure states is a list

    # Call process_batch
    states, actions, returns, timesteps, labels = process_batch(batch, device=args.device)

    # Predict actions using the model
    # actions_pred1 = model(states, actions, returns, timesteps)
    actions_pred1 = model(states, actions, returns, timesteps)

    # Permute for loss calculation
    actions_pred = actions_pred1.permute(0, 2, 1)
    loss = loss_fn(actions_pred, labels)

    queue_action = 0

    # print("actions_pred1",actions_pred1)
    # print("actions_pred",actions_pred)
    # print("actual-queue_action",labels)

    return loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred