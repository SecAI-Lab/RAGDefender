# run for {str({str(test_params['method'])})} and blind

import os
import subprocess

def run(test_params):
    log_file, log_name = get_log_name(test_params)

    cmd = f"python3 -u main_abl.py \
        --eval_model_code {test_params['eval_model_code']} \
        --eval_dataset {test_params['eval_dataset']} \
        --split {test_params['split']} \
        --query_results_dir {test_params['query_results_dir']} \
        --model_name {test_params['model_name']} \
        --top_k {test_params['top_k']} \
        --use_truth {test_params['use_truth']} \
        --gpu_id {test_params['gpu_id']} \
        --attack_method {test_params['attack_method']} \
        --adv_per_query {test_params['adv_per_query']} \
        --additional_adv_per_query {test_params['additional_adv_per_query']} \
        --score_function {test_params['score_function']} \
        --repeat_times {test_params['repeat_times']} \
        --M {test_params['M']} \
        --seed {test_params['seed']} \
        --name {log_name} \
        --method {test_params['method']} \
        > {log_file}"

    # Run the command and wait for it to complete
    subprocess.run(cmd, shell=True, check=True)

def get_log_name(test_params):
    # Generate a log file name
    #os.makedirs(f"logs/main_logs_v5_{str(test_params['seed'])}", exist_ok=True)
    os.makedirs(f"logs/main_logs_{str(test_params['method'])}_{str(test_params['seed'])}", exist_ok=True)

    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    
    if test_params['attack_method'] is not None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"

    if test_params['note'] is not None:
        log_name = test_params['note']
    
    return f"logs/main_logs_{str(test_params['method'])}_{str(test_params['seed'])}/{log_name}_{test_params['additional_adv_per_query']}.txt", log_name

# Test parameters
test_params = {
    # beir_info
    'eval_model_code': "contriever", # "ance"
    'eval_dataset': "nq",
    'split': "test",
    #'query_results_dir': 'main',
    'query_results_dir': 'main_GARAG',

    # LLM setting
    'model_name': 'llama7b', 
    'use_truth': False,
    'top_k': 5,
    'gpu_id': 0,

    # attack
    'attack_method': 'LM_targeted',
    'adv_per_query': 5,
    #'adv_per_query': 2,
    'additional_adv_per_query': 0,
    'score_function': 'dot', #cos_sim
    'repeat_times': 10,
    'M': 10,
    'seed': 12,

    'method': 'GARAG',

    'note': None
}

for eval_code in ["contriever"]:
    #nq
    for seed in [12]:
        for model in ['llama7b', 'vicuna7b']:
        # for model in ['vicuna13b']:            
            test_params['eval_model_code'] = eval_code
            test_params['eval_dataset'] = "nq"
            test_params['split'] = "test"
            test_params['adv_per_query'] = 5
            test_params['top_k'] = 5     
            test_params['model_name'] = model       
            test_params['additional_adv_per_query'] = 0
            test_params['seed'] = seed
            run(test_params)
    #hotpotqa
    for seed in [12]:
        for model in ['llama7b', 'vicuna7b']:
            test_params['eval_model_code'] = eval_code
            test_params['eval_dataset'] = "hotpotqa"
            test_params['split'] = "test"
            test_params['adv_per_query'] = 2
            test_params['top_k'] = 2
            test_params['model_name'] = model
            test_params['additional_adv_per_query'] = 0
            test_params['seed'] = seed
            run(test_params)
    #msmarco
    for seed in [12]:
        for model in ['llama7b', 'vicuna7b']:
            test_params['eval_model_code'] = eval_code
            test_params['eval_dataset'] = "msmarco"
            test_params['split'] = "train"
            test_params['adv_per_query'] = 2
            test_params['top_k'] = 2
            test_params['model_name'] = model
            test_params['additional_adv_per_query'] = 0
            test_params['seed'] = seed
            run(test_params)