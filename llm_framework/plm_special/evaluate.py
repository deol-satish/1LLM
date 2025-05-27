import numpy as np
import torch
import time
import json
import psutil
import GPUtil
from munch import Munch
from torch.utils.data import DataLoader
import pandas as pd
from collections import defaultdict
import os


# List of log files to delete
log_files = ['app.log', 'eval_utils.log']

for file in log_files:
    if os.path.isfile(file):
        os.remove(file)
        print(f"Deleted file: {file}")
    else:
        print(f"File not found: {file}")

def remove_file(file_path):
    """Remove a file if it exists."""
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    else:
        print(f"File not found: {file_path}")

from datetime import datetime
from plm_special.utils.utils import process_batch
from plm_special.data.dataset import ExperienceDataset
from plm_special.utils.eval_utils import tensor_to_list, convert_exp_pool_to_dataframe, find_nearest_length, col_dict, test_step, otest_step
import random
import pickle

import logging
# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG,  # Set level to DEBUG to see all messages
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='app.log',
#     filemode='a'
# )


state_columns = list(col_dict.keys())

def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()

def log_step(step_index, test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred):
    return {
        'step': step_index,
        'test_loss': test_loss.item(),
        'actions_pred1': tensor_to_list(actions_pred1),
        'actions_pred': tensor_to_list(actions_pred),
        'states': tensor_to_list(states),
        'actions': tensor_to_list(actions),
        'returns': tensor_to_list(returns),
        'timestamps': str(time.time()),
        'timesteps': tensor_to_list(timesteps),
        'labels': tensor_to_list(labels)
    }

def evaluate_on_simulated_env(args, model, exp_pool, target_return, loss_fn ,llm_freq=100,process_reward_fn=None, seed=0):
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    # Initialization
    llm_freq = llm_freq
    print("llm_freq", llm_freq)
    logging.debug("llm_freq: %s", llm_freq)
    current_date = datetime.today().strftime('%Y-%m-%d')
    base_path = f'./results_eval/{args.plm_type}/{current_date}'
    os.makedirs(base_path, exist_ok=True)

    df = convert_exp_pool_to_dataframe(exp_pool)
    df_all = df.copy()
    # Filter and reset index for classic queue (queue_type == 0)
    df_classic = df[df['queue_type'] == 0].reset_index(drop=True)
    # Filter and reset index for L4S queue (queue_type == 1)
    df_l4s = df[df['queue_type'] == 1].reset_index(drop=True)


    print("Dataframe shape:", df.shape)
    print("Classic traffic shape:", df_classic.shape)
    print("L4S traffic shape:", df_l4s.shape)

    logging.debug("Dataframe shape: %s", df.shape)
    logging.debug("Classic traffic shape: %s", df_classic.shape)
    logging.debug("L4S traffic shape: %s", df_l4s.shape)


    def run_policy(df_subset, traffic_type, use_model_decision=True):
        logs = defaultdict(list)
        step_index = 0 # records the number of steps taken
        cur_index = 0 # records which datapoint we are currently at
        # step_limit = len(df_subset) - 1
        step_limit = (df_subset.index.max() - 1) * 0.2
        # step_limit = 20
        step_limit = 16000
        logging.debug("Step limit: %s", step_limit)
        logging.debug(f"Processing index: {cur_index}, Traffic Type: {traffic_type}, Use Model Decision: {use_model_decision}")

        while step_index < step_limit:
            logging.debug("---"* 20)
            logging.debug(f"Current Step Index: {step_index},Processing index: {cur_index}, Traffic Type: {traffic_type}, Use Model Decision: {use_model_decision}")
            # row = df_subset.iloc[cur_index]


            row = df_subset.loc[cur_index]
            logging.debug("row: %s",row)

            state = np.array(row[state_columns], dtype=np.float32)
            current_action = row['actions']
            reward=row['rewards']
            done=0
            batch = [state],[current_action],[reward],[done]
            test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred = test_step(args, model, loss_fn, batch,target_return)
            new_action = actions_pred.detach().cpu().numpy().argmax(axis=1).flatten()

            logging.debug("new_action.item(): %s",new_action.item())
            logging.debug("new_action: %s",new_action)
            logging.debug("current_action: %s",current_action)
            logging.debug("type(current_action): %s",type(current_action))
            if step_index % llm_freq == 0:
                logging.debug(f"Step {step_index}: Using LLM decision for action selection")

            if step_index % llm_freq == 0 and use_model_decision and int(new_action.item()) != int(current_action):            
                # Use the model to decide the action    
                # Model-based policy
                logging.debug(f"Using model decision at step {step_index}, current action: {current_action}, new action: {new_action.item()}")
                if new_action.item() == 1 or new_action.item() == 2:
                    logging.debug(f"action 1 or 2:,Current Ep Index: {step_index},Processing index: {cur_index}, Traffic Type: {traffic_type}, Use Model Decision: {use_model_decision} ")
         
                new_queue_length = float(states[0][0][col_dict['length_in_bytes']])
                # CHeck whether the new action is 0 or 2 ( or ENQUEUE or MARKECN )
                if new_action.item() == 0 or new_action.item() == 2 :
                    new_queue_length = (float(states[0][0][col_dict['length_in_bytes']]) + float(states[0][0][col_dict['packet_length']]))
                    new_iloc = find_nearest_length(df_subset, cur_index, states, new_action, new_queue_length)
                    if new_iloc is None or new_iloc >= len(df_subset):
                        logging.debug("new_iloc is None or out of bounds, breaking loop")
                        break
                    cur_index = new_iloc
                    cur_index += 1
                else:
                    new_queue_length = float(states[0][0][col_dict['length_in_bytes']])
                    new_iloc = find_nearest_length(df_subset, cur_index, states, new_action, new_queue_length)
                    if new_iloc is None or new_iloc >= len(df_subset):
                        logging.debug("new_iloc is None or out of bounds, breaking loop")
                        break
                    cur_index = new_iloc
                    cur_index += 1

                model.reset_dq()
            else:
                # Sequential policy
                cur_index += 1

            logs['steps'].append(log_step(
                step_index, test_loss, states, actions, returns, timesteps, labels, actions_pred1, actions_pred
            ))
            step_index += 1

        return logs
    


    def setup_new_logging(log_filename):
        """Resets logging handlers and configures a new log file."""
        root_logger = logging.getLogger()
        # Remove all existing handlers
        for handler in root_logger.handlers[:]:
            handler.close()  # Close the handler to release resources
            root_logger.removeHandler(handler)
        
        # Configure logging for the new file
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_filename,
            filemode='a'
        )

    # --- Start of your script's logic ---

    # Initial cleanup of debug log files
    # Ensure args.plm_type, args.plm_size, and llm_freq are defined
    # Example: if not all(hasattr(args, attr) for attr in ['plm_type', 'plm_size']) or 'llm_freq' not in globals():
    #    raise ValueError("args.plm_type, args.plm_size, or llm_freq is not defined.")

    remove_file(f'{args.plm_type}_{args.plm_size}_{llm_freq}_all_traffic_llm_eval_logs_debug.log')
    remove_file(f'{args.plm_type}_{args.plm_size}_{llm_freq}_all_traffic_original_eval_logs_debug.log')
    remove_file(f'{args.plm_type}_{args.plm_size}_{llm_freq}_classic_traffic_llm_eval_logs_debug.log')
    remove_file(f'{args.plm_type}_{args.plm_size}_{llm_freq}_classic_traffic_original_eval_logs_debug.log')
    remove_file(f'{args.plm_type}_{args.plm_size}_{llm_freq}_l4s_traffic_llm_eval_logs_debug.log')
    remove_file(f'{args.plm_type}_{args.plm_size}_{llm_freq}_l4s_traffic_original_eval_logs_debug.log')

    # Part 0: Consider both L4S and classic traffic with LLM Policy
    debug_log_filename_part0 = f'{args.plm_type}_{args.plm_size}_{llm_freq}_all_traffic_llm_eval_logs_debug.log'
    setup_new_logging(debug_log_filename_part0)

    all_traffic_llm_logs = run_policy(df_all, traffic_type='all_traffic', use_model_decision=True)
    all_traffic_llm_filename = f'{args.plm_type}_{args.plm_size}_{llm_freq}_all_traffic_llm_eval_logs.json'
    all_traffic_llm_file_path = os.path.join(base_path, all_traffic_llm_filename)

    with open(all_traffic_llm_file_path, 'w') as f:
        json.dump(all_traffic_llm_logs, f, indent=4)
    logging.debug(f"Saved all_traffic_llm logs to {all_traffic_llm_file_path}")

    # Part 1: Consider both L4S and classic traffic with Original Policy
    debug_log_filename_part1 = f'{args.plm_type}_{args.plm_size}_{llm_freq}_all_traffic_original_eval_logs_debug.log'
    setup_new_logging(debug_log_filename_part1)

    all_traffic_original_logs = run_policy(df_all, traffic_type='all_traffic', use_model_decision=False)
    all_traffic_original_filename = f'{args.plm_type}_{args.plm_size}_{llm_freq}_all_traffic_original_eval_logs.json'
    all_traffic_original_file_path = os.path.join(base_path, all_traffic_original_filename)

    with open(all_traffic_original_file_path, 'w') as f:
        json.dump(all_traffic_original_logs, f, indent=4)
    logging.debug(f"Saved all_traffic_original logs to {all_traffic_original_file_path}")

    # Part 2: Classic traffic with LLM policy
    debug_log_filename_part2 = f'{args.plm_type}_{args.plm_size}_{llm_freq}_classic_traffic_llm_eval_logs_debug.log'
    setup_new_logging(debug_log_filename_part2)

    classic_llm_logs = run_policy(df_classic, traffic_type='classic_llm', use_model_decision=True)
    classic_llm_filename = f'{args.plm_type}_{args.plm_size}_{llm_freq}_classic_traffic_llm_eval_logs.json'
    classic_llm_file_path = os.path.join(base_path, classic_llm_filename)

    with open(classic_llm_file_path, 'w') as f:
        json.dump(classic_llm_logs, f, indent=4)
    logging.debug(f"Saved classic_traffic_llm logs to {classic_llm_file_path}")

    # Part 3: Classic traffic without LLM (original, sequential)
    debug_log_filename_part3 = f'{args.plm_type}_{args.plm_size}_{llm_freq}_classic_traffic_original_eval_logs_debug.log'
    setup_new_logging(debug_log_filename_part3)

    classic_original_logs = run_policy(df_classic, traffic_type='classic_original', use_model_decision=False)
    classic_original_filename = f'{args.plm_type}_{args.plm_size}_{llm_freq}_classic_traffic_original_eval_logs.json'
    classic_original_file_path = os.path.join(base_path, classic_original_filename)

    with open(classic_original_file_path, 'w') as f:
        json.dump(classic_original_logs, f, indent=4)
    logging.debug(f"Saved classic_traffic_original logs to {classic_original_file_path}")

    # Part 4: L4S traffic with LLM policy
    debug_log_filename_part4 = f'{args.plm_type}_{args.plm_size}_{llm_freq}_l4s_traffic_llm_eval_logs_debug.log'
    setup_new_logging(debug_log_filename_part4)

    l4s_llm_logs = run_policy(df_l4s, traffic_type='l4s_llm', use_model_decision=True)
    l4s_llm_filename = f'{args.plm_type}_{args.plm_size}_{llm_freq}_l4s_traffic_llm_eval_logs.json'
    l4s_llm_file_path = os.path.join(base_path, l4s_llm_filename)

    with open(l4s_llm_file_path, 'w') as f:
        json.dump(l4s_llm_logs, f, indent=4)
    logging.debug(f"Saved l4s_traffic_llm logs to {l4s_llm_file_path}")

    # Part 5: L4S traffic without LLM (original, sequential)
    debug_log_filename_part5 = f'{args.plm_type}_{args.plm_size}_{llm_freq}_l4s_traffic_original_eval_logs_debug.log'
    setup_new_logging(debug_log_filename_part5)

    l4s_original_logs = run_policy(df_l4s, traffic_type='l4s_original', use_model_decision=False)
    l4s_original_filename = f'{args.plm_type}_{args.plm_size}_{llm_freq}_l4s_traffic_original_eval_logs.json'
    l4s_original_file_path = os.path.join(base_path, l4s_original_filename)

    with open(l4s_original_file_path, 'w') as f:
        json.dump(l4s_original_logs, f, indent=4)
    logging.debug(f"Saved l4s_traffic_original logs to {l4s_original_file_path}")

