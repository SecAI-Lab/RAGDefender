import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import json
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from src.models import create_model
from src.utils import load_beir_datasets_md, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, f1_score, top_similar_pairs
from src.attack import Attacker
from src.prompts import wrap_prompt, wrap_prompt_llama
import torch
import sys
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import math
from sklearn.cluster import KMeans, AgglomerativeClustering
import sklearn.feature_extraction.text as text

def find_num_adv(text_list, s_model):
    embeddings = s_model.encode(text_list, convert_to_tensor=True)
    cos_sim_matrix = util.cos_sim(embeddings, embeddings)
    avg = torch.mean(cos_sim_matrix, dim=0)
    median = torch.median(cos_sim_matrix, dim=0)
    avg_avg = avg.mean()
    avg_median = median.values.median()
    above_avg = [1 if score > avg_avg else 0 for score in avg]
    above_median = [1 if score > (avg_median+avg_avg)/2 else 0 for score in median.values]
    final = [1 if above_avg[i] == 1 or above_median[i] == 1 else 0 for i in range(len(above_avg))]
    result = sum(final) if sum(final) > 0 and avg_avg < avg_median else len(text_list) - sum(final)

    # Clean up tensors to free memory
    del embeddings, cos_sim_matrix, avg, median
    torch.cuda.empty_cache()
    return result

def find_num_adv_tfidf(text_list):
    stop_words = list(text.ENGLISH_STOP_WORDS)
    tfidf = text.TfidfVectorizer(stop_words=stop_words)
    X = tfidf.fit_transform(text_list)
    all_data = tfidf.get_feature_names_out()
    dense = X.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=all_data)
    dict_tfidf = df.T.sum(axis=1)
    dict_tfidf = dict_tfidf.sort_values(ascending=False)
    top_3 = dict_tfidf[:5]
    indices = []
    for word in top_3.index:
        indices.append([1 if word in sentence else 0 for sentence in text_list])
    final = [1 if sum([index[i] for index in indices]) > math.floor(len(indices)/2) else 0 for i in range(len(text_list))]
    return sum(final)

def find_num_adv_agg(text_list, s_model):
    embeddings = s_model.encode(text_list, convert_to_tensor=True)
    model = AgglomerativeClustering(n_clusters=2)
    model.fit(embeddings.cpu().detach().numpy())
    labels = model.labels_
    labels = list(labels)
    num_labels = sum(labels)
    num_tfidf = find_num_adv_tfidf(text_list)
    return min(num_labels, len(text_list)-num_labels) if num_labels > 0 and num_tfidf <= int(len(text_list)/2) else max(num_labels, len(text_list)-num_labels)

def parse_args():
    parser = argparse.ArgumentParser(description='test')
    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')
    parser.add_argument("--query_results_dir", type=str, default='main')

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='palm2')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_truth', type=str, default='False')
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--attack_method', type=str, default='LM_targeted')
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--additional_adv_per_query', type=int, default=0, help='The number of additional adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=10, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=10, help='one of our parameters, the number of target queries')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")

    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda'
    setup_seeds(args.seed)
    if args.model_config_path == None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'

    # load target queries and answers
    if args.eval_dataset == 'msmarco':
        corpus, queries, qrels = load_beir_datasets_md('msmarco', 'train')
        incorrect_answers = load_json(f'results/target_queries/{args.eval_dataset}.json')
    else:
        corpus, queries, qrels = load_beir_datasets_md(args.eval_dataset, args.split)
        incorrect_answers = load_json(f'results/target_queries/{args.eval_dataset}.json')
    A_N = args.additional_adv_per_query + args.adv_per_query

    if args.use_truth == 'True':
        args.attack_method = None

    if args.attack_method not in [None, 'None']:
        # Load retrieval models
        model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
        model.eval()
        model.to(device)
        c_model.eval()
        c_model.to(device) 
        attacker = Attacker(args,
                            model=model,
                            c_model=c_model,
                            tokenizer=tokenizer,
                            get_emb=get_emb) 
    
    llm = create_model(args.model_config_path)

    # Clear memory after model loading
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    all_results = []
    asr_list = []
    accuracy_list = []  # New list for accuracy calculation
    ret_list = []

    s_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', trust_remote_code=True)
    
    for iter in range(args.repeat_times):
        print(f'######################## Iter: {iter+1}/{args.repeat_times} #######################')

        target_queries_idx = range(iter * args.M, iter * args.M + args.M)
        target_queries = [incorrect_answers[idx]['question'] for idx in target_queries_idx]
        
        if args.attack_method not in [None, 'None']:
            for i in target_queries_idx:
                target_queries[i - iter * args.M] = {'query': target_queries[i - iter * args.M], 'id': incorrect_answers[i]['id']}
                
            adv_text_groups = attacker.get_attack(target_queries)
            adv_text_list = sum(adv_text_groups, [])
            
            adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
            adv_input = {key: value.cuda() for key, value in adv_input.items()}
            
            with torch.no_grad():
                adv_embs = get_emb(c_model, adv_input)

            # Clear tensors to free RAM
            del adv_input
            gc.collect()
            torch.cuda.empty_cache()

        asr_cnt = 0
        accuracy_cnt = 0  # Counter for correct answers
        ret_sublist = []
        
        iter_results = []
        for i in target_queries_idx:
            iter_idx = i - iter * args.M

            print(f'############# Target Question: {iter_idx+1}/{args.M} #############')
            question = incorrect_answers[i]['question']
            print(f'Question: {question}\n') 
            
            gt_ids = list(qrels[incorrect_answers[i]['id']].keys())
            
            # Find positive passages
            pos_ids = []
            if args.eval_dataset == 'nq':
                title = [corpus[id]["title"] for id in gt_ids if id in corpus.keys()][0]
                for id in corpus.keys():
                    if corpus[id]["title"] == title:
                        pos_ids.append(id)
                pos_ans = [corpus[id]["text"] for id in pos_ids]
            ground_truth = [corpus[id]["text"] for id in gt_ids]
            incco_ans = incorrect_answers[i]['incorrect answer']
            correct_ans = incorrect_answers[i]['correct answer']  # Get correct answer for accuracy check
            
            adv_text_groups = [adv_text_list[i*args.adv_per_query:(i+1)*args.adv_per_query] for i in range(args.M)]

            if args.use_truth == 'True':
                if 'llama' in args.model_name:
                    query_prompt = wrap_prompt_llama(question, ground_truth, 4)
                else:
                    query_prompt = wrap_prompt(question, ground_truth, 4)
                response = llm.query(query_prompt)
                print(f"Output: {response}\n\n")
                
                # Check accuracy for truth mode
                if clean_str(correct_ans) in clean_str(response):
                    accuracy_cnt += 1
                    
                iter_results.append(
                    {
                        "question": question,
                        "input_prompt": query_prompt,
                        "output": response,
                    }
                )  

            else:  # topk mode
                topk_results = []
                adv_text_now = adv_text_groups[iter_idx]
                adv_text_copy = adv_text_now.copy()
                adv_path = f'poisoned_corpus/{args.eval_dataset}/gpt4/{args.eval_dataset}_{iter}_{iter_idx}.txt'
                with open(adv_path, 'r') as f:
                    passage = f.readlines()
                    for p in passage:
                        if (p != '' and p != '\n'):
                            adv_text_now.append(question + '.' + p)
                            adv_text_copy.append(p)
                adv_text_now = adv_text_now[:A_N]
                
                if args.eval_dataset == 'nq':
                    adv_text_now = adv_text_now + pos_ans
                    print(len(pos_ans), len(adv_text_now)-len(pos_ans))
                else:
                    adv_text_now = adv_text_now + ground_truth
                    print(len(ground_truth), len(adv_text_now)-len(ground_truth))
                    
                if args.eval_dataset == 'nq' or args.eval_dataset == 'msmarco':
                    adv_text_num = find_num_adv_agg(adv_text_now, s_model)
                else:
                    adv_text_num = find_num_adv(adv_text_now, s_model)
                print(adv_text_num)
                
                gen_num = max(1,int(adv_text_num*(adv_text_num-1)/2))
                adv_pairs = top_similar_pairs(adv_text_now, s_model, gen_num)
                
                pair_cnt = Counter()
                for x, y, sim in adv_pairs:
                    pair_cnt[x] += math.copysign(sim * sim, sim)
                    pair_cnt[y] += math.copysign(sim * sim, sim)
                sorted_pairs = sorted(pair_cnt.items(), key=lambda item: item[1], reverse=True)[:adv_text_num]
                top_5_indices = [idx for idx, _ in sorted_pairs]
                
                top_5_texts = [clean_str(adv_text_now[i]) for i in range(len(adv_text_now)) if i in top_5_indices]
                adv_text_now_res = [adv_text_now[i] for i in range(len(adv_text_now)) if i not in top_5_indices]
                adv_text_now_res2 = [clean_str(adv_text_now[i]) for i in range(len(adv_text_now)) if i not in top_5_indices]
                
                if len(adv_text_now) == 0:
                    print("Alert!")
                    adv_text_num = len(adv_text_now) - len(pos_ans)
                    gen_num = max(1, int(adv_text_num * (adv_text_num - 1) / 2))
                    adv_pairs = top_similar_pairs(adv_text_now, s_model, gen_num)
                    pair_cnt = Counter()
                    for x, y, sim in adv_pairs:
                        pair_cnt[x] += math.copysign(sim * sim, sim)
                        pair_cnt[y] += math.copysign(sim * sim, sim)
                    sorted_pairs = sorted(pair_cnt.items(), key=lambda item: item[1], reverse=True)[:adv_text_num]
                    top_5_indices = [idx for idx, _ in sorted_pairs]
                    adv_text_now_res = [adv_text_now[i] for i in range(len(adv_text_now)) if i not in top_5_indices]
                
                adv_input = tokenizer(adv_text_now_res2, padding=True, truncation=True, return_tensors="pt")
                adv_input = {key: value.cuda() for key, value in adv_input.items()}
                with torch.no_grad():
                    adv_embs = get_emb(c_model, adv_input)

                # Clear tensors
                del adv_input
                gc.collect()
                torch.cuda.empty_cache()

                if args.attack_method not in [None, 'None']: 
                    query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
                    query_input = {key: value.cuda() for key, value in query_input.items()}
                    with torch.no_grad():
                        query_emb = get_emb(model, query_input)

                    # Clear query tensors
                    del query_input
                    gc.collect()
                    torch.cuda.empty_cache()

                    for j in range(len(adv_text_now_res)):
                        adv_emb = adv_embs[j, :].unsqueeze(0) 
                        if args.score_function == 'dot':
                            adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                        elif args.score_function == 'cos_sim':
                            adv_sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                        topk_results.append({'score': adv_sim, 'context': adv_text_now_res[j]})
                    
                    topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)
                    topk_contents = []
                    for j in range(min(args.top_k, len(topk_results))):
                        topk_contents.append(topk_results[j]["context"])
                    adv_text_set = set(adv_text_copy)

                    cnt_from_adv = sum([i in adv_text_set for i in topk_contents])
                    ret_sublist.append(cnt_from_adv)
                    
                if 'llama' in args.model_name:
                    query_prompt = wrap_prompt_llama(question, topk_contents, 4)
                else:
                    query_prompt = wrap_prompt(question, topk_contents, 4)

                response = llm.query(query_prompt)
                print(f'Output: {response}\n\n')
                
                injected_adv = [i for i in topk_contents if i in adv_text_set]
                iter_results.append(
                    {
                        "id": incorrect_answers[i]['id'],
                        "question": question,
                        "injected_adv": injected_adv,
                        "input_prompt": query_prompt,
                        "output_poison": response,
                        "incorrect_answer": incco_ans,
                        "answer": correct_ans
                    }
                )

                # Check if incorrect answer is in response (ASR)
                if clean_str(incco_ans) in clean_str(response) or clean_str(response) in clean_str(incco_ans):
                    asr_cnt += 1
                    
                # Check if correct answer is in response (Accuracy)
                if clean_str(correct_ans) in clean_str(response) or clean_str(response) in clean_str(correct_ans):
                    accuracy_cnt += 1
                    
                topk_results = []

        asr_list.append(asr_cnt)
        accuracy_list.append(accuracy_cnt)  # Add accuracy count for this iteration
        ret_list.append(ret_sublist)

        all_results.append({f'iter_{iter}': iter_results})
        save_results(all_results, args.query_results_dir, args.name+str(A_N))
        print(f'Saving iter results to results/query_results/{args.query_results_dir}/{args.name+str(A_N)}' + '.json')

        # Clear memory after each iteration
        if 'adv_embs' in locals():
            del adv_embs
        if 'query_emb' in locals():
            del query_emb
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate metrics
    asr = np.array(asr_list) / args.M
    asr_mean = round(np.mean(asr), 2)
    
    accuracy = np.array(accuracy_list) / args.M
    accuracy_mean = round(np.mean(accuracy), 2)
    
    ret_precision_array = np.array(ret_list) / args.top_k
    ret_precision_mean = round(np.mean(ret_precision_array), 2)
    ret_recall_array = np.array(ret_list) / A_N
    ret_recall_mean = round(np.mean(ret_recall_array), 2)
    ret_f1_array = f1_score(ret_precision_array, ret_recall_array)
    ret_f1_mean = round(np.mean(ret_f1_array), 2)
  
    print(f"ASR: {asr}")
    print(f"ASR Mean: {asr_mean}\n")
    
    print(f"Accuracy: {accuracy}")
    print(f"Accuracy Mean: {accuracy_mean}\n")

    print(f"Ret: {ret_list}")
    print(f"Precision mean: {ret_precision_mean}")
    print(f"Recall mean: {ret_recall_mean}")
    print(f"F1 mean: {ret_f1_mean}\n")

    print(f"Ending...")

if __name__ == '__main__':
    main()