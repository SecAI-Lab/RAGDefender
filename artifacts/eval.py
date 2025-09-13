import os
import json
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def calculate_mean_std_confidence(values):
    n = len(values)
    if n == 0:
        return None, None, None

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std_deviation = math.sqrt(variance)
    # 95% Confidence Interval
    confidence_interval = 1.96 * (std_deviation / math.sqrt(n))

    return mean, std_deviation, confidence_interval

def extract_metrics_from_log(log_file):
    asr_mean = None
    acc_mean = None
    negatives = []
    predicted = []
    
    inside_target_block = False

    # Read the log file and extract the ASR Mean, F1 mean, and negative/predicted values
    with open(log_file, 'r') as f:
        for line in f:
            if "ASR Mean:" in line:
                asr_mean = float(line.split(":")[1].strip())
            if "Accuracy Mean:" in line:
                acc_mean = float(line.split(":")[1].strip())

            # Start looking after "Golden Passage"
            if "Golden Passage:" in line:
                inside_target_block = True
                continue

            # Capture the values between Golden Passage and Output
            if inside_target_block and "Output:" not in line:
                values = line.strip().split()
                if len(values) == 2 and values[0].isdigit() and values[1].isdigit():
                    negative = int(values[0])  # First value is negative
                    predicted_value = int(values[1])  # Second value is predicted
                    negatives.append(negative)
                    predicted.append(predicted_value)

            # Stop parsing after "Output:"
            if "Output:" in line:
                inside_target_block = False
    
    return asr_mean, acc_mean, negatives, predicted

def evaluate_dataset(eval_dataset, eval_model_code, top_k, adv_per_query, adv_q_list, query_results_dir_format, model_names, idx_list):
    results_table = []
    M = 10
    repeat_times = 10
    attack_method = 'LM_targeted'
    score_function = 'dot'

    for model_name in model_names:
        for additional_adv_per_query in adv_q_list:
            asr_values = []
            acc_values = []

            for idx in idx_list:
                query_results_dir = query_results_dir_format.format(idx=idx)

                log_name2 = f"{eval_dataset}-{eval_model_code}-{model_name}-Top{top_k}--M{M}x{repeat_times}-adv-{attack_method}-{score_function}-{adv_per_query}-{top_k}_{additional_adv_per_query}"
                # log_file = f"logs/{query_results_dir2}/{log_name2}.txt"
                log_file = f"logs/{query_results_dir}/{log_name2}.txt"
                
                if os.path.exists(log_file):
                    asr_mean, acc_mean, negatives, predicted = extract_metrics_from_log(log_file)
                    if asr_mean is not None:
                        asr_values.append(asr_mean)
                    if acc_mean is not None:
                        acc_values.append(acc_mean)

            asr_mean, asr_std, asr_ci = calculate_mean_std_confidence(asr_values)

            results_table.append({
                'Model': model_name,
                'perterb_ratio (%)': f"{(additional_adv_per_query+adv_per_query)*100/top_k:.2f}%",
                'Accuracy (%)': f"{acc_mean:.2f}" if acc_mean is not None else 'N/A',
                'ASR (%)': f"{asr_mean:.2f}" if asr_mean is not None else 'N/A',
            })

    return pd.DataFrame(results_table)

def get_datasets_config(method):
    """Get dataset configuration based on the selected method"""
    
    if method == 'PoisonedRAG':
        return [
            {
                'eval_dataset': 'nq',
                'top_k': 5,
                'adv_per_query': 5,
                'adv_q_list': [0, 5, 15, 25],
                'query_results_dir_format': 'main_logs_PoisonedRAG_{idx}'
            },
            {
                'eval_dataset': 'hotpotqa',
                'top_k': 2,
                'adv_per_query': 2,
                'adv_q_list': [0, 2, 6, 10],
                'query_results_dir_format': 'main_logs_PoisonedRAG_{idx}'
            },
            {
                'eval_dataset': 'msmarco',
                'top_k': 2,
                'adv_per_query': 2,
                'adv_q_list': [0, 2, 6, 10],
                'query_results_dir_format': 'main_logs_PoisonedRAG_{idx}'
            }
        ]
    elif method == 'Blind':
        return [
            {
                'eval_dataset': 'nq',
                'top_k': 5,
                'adv_per_query': 5,
                'adv_q_list': [0],
                'query_results_dir_format': 'main_logs_blind_{idx}'
            },
            {
                'eval_dataset': 'hotpotqa',
                'top_k': 2,
                'adv_per_query': 2,
                'adv_q_list': [0],
                'query_results_dir_format': 'main_logs_blind_{idx}'
            },
            {
                'eval_dataset': 'msmarco',
                'top_k': 2,
                'adv_per_query': 2,
                'adv_q_list': [0],
                'query_results_dir_format': 'main_logs_blind_{idx}'
            }
        ]
    elif method == 'GARAG':
        return [
            {
                'eval_dataset': 'nq',
                'top_k': 5,
                'adv_per_query': 5,
                'adv_q_list': [0],
                'query_results_dir_format': 'main_logs_GARAG_{idx}'
            },
            {
                'eval_dataset': 'hotpotqa',
                'top_k': 2,
                'adv_per_query': 2,
                'adv_q_list': [0],
                'query_results_dir_format': 'main_logs_GARAG_{idx}'
            },
            {
                'eval_dataset': 'msmarco',
                'top_k': 2,
                'adv_per_query': 2,
                'adv_q_list': [0],
                'query_results_dir_format': 'main_logs_GARAG_{idx}'
            }
        ]
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'PoisonedRAG', 'Blind', or 'GARAG'")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate RAG models with different methods')
    parser.add_argument('--method', type=str, required=True, 
                       choices=['PoisonedRAG', 'Blind', 'GARAG'],
                       help='Evaluation method to use (PoisonedRAG, Blind, or GARAG)')
    
    args = parser.parse_args()
    
    # List of idx values to generate query_results_dir
    idx_list = [12] 
    
    # Model configurations
    eval_model_code_full = ['contriever', 'dpr', 'ance']
    eval_model_code_limited = ['contriever']
    
    # Select eval_model_code based on method
    if args.method == 'PoisonedRAG':
        eval_model_code = eval_model_code_full
    else:  # Blind or GARAG
        eval_model_code = eval_model_code_limited

    # Get dataset configuration based on method
    datasets_config = get_datasets_config(args.method)
    
    model_names = ['llama7b', 'vicuna7b']
    
    print(f"Running evaluation with method: {args.method}")
    print(f"Using eval_model_code: {eval_model_code}")
    
    for config in datasets_config:
        print(f"\nProcessing for Dataset: {config['eval_dataset']}")
        
        for eval_model in eval_model_code:
            print(f"\nProcessing for eval_model = {eval_model}")
            eval_dataset = config['eval_dataset']
            top_k = config['top_k']
            adv_per_query = config['adv_per_query']
            adv_q_list = config['adv_q_list']
            query_results_dir_format = config['query_results_dir_format']

            # Evaluate the dataset and get the table as a DataFrame
            df = evaluate_dataset(eval_dataset, eval_model, top_k, adv_per_query, adv_q_list, query_results_dir_format, model_names, idx_list)

            # Print the DataFrame for the current dataset
            print(f"\nResults for dataset: {eval_dataset} using method: {args.method}")
            print(df)

if __name__ == '__main__':
    main()