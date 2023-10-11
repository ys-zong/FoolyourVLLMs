import transformers
import torch
import argparse
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from torch.nn.functional import softmax
import random
import math
import itertools
from transformers import AutoTokenizer, AutoModelForCausalLM


choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def move_answers_to_position(df, position):
    # Create a new DataFrame to store the modified questions, choices, and labels
    new_df = df.copy()
    position_idx = ord(position) - ord('A')
    
    for idx in range(len(df)):
        # Extract the question, choices, and label
        question = df.iloc[idx, 0]
        k = df.shape[1] - 2  # Minus 2 for the question and answer columns
        choices = [df.iloc[idx, i + 1] for i in range(k)]
        original_label = df.iloc[idx, k + 1]
        
        # Find the index of the original label
        original_label_idx = ord(original_label) - ord("A")
        
        # Move the answer to the new desired position
        choices[position_idx], choices[original_label_idx] = choices[original_label_idx], choices[position_idx]
        
        # Update the choices and label in the new DataFrame
        for i in range(k):
            new_df.iat[idx, i + 1] = choices[i]
        
        new_df.iat[idx, k + 1] = position  # Update the label to the new position

    return new_df

def generate_permutation_indices(n_choices, k_remaining, original_label_idx):
    # Generate combinations for remaining choices (keeping the correct answer)
    combinations = list(itertools.combinations(set(range(n_choices)) - {original_label_idx}, k_remaining - 1))
    
    # Generate all permutations for each combination along with the correct answer (at original_label_idx)
    all_permutations = []
    for comb in combinations:
        for perm in itertools.permutations([original_label_idx] + list(comb)):
            all_permutations.append(list(perm))
    
    return all_permutations

def choice_reduce_permute(df, idx, perm_i, n_reduced):
    k = df.shape[1] - 2  # Number of choices (excluding question and answer columns)
    original_choices = [df.iloc[idx, i+1] for i in range(k)]
    original_label = df.iloc[idx, k + 1]  # Original answer (e.g., "A", "B", etc.)
    
    # Identify the index of the correct answer
    original_label_idx = ord(original_label) - ord('A')
    
    # Generate permutation indices
    all_perms = generate_permutation_indices(k, n_reduced, original_label_idx)
    
    if perm_i >= len(all_perms):
        return "Invalid perm_i"
    
    # Choose the permutation based on the perm_i
    chosen_permutation = all_perms[perm_i]  # Removed the problematic line
    
    # Generate the new choices and label
    new_choices = [original_choices[i] for i in chosen_permutation]
    new_label_idx = np.where(np.array(chosen_permutation) == original_label_idx)[0][0]
    new_label = chr(ord('A') + new_label_idx)
    
    return new_choices, new_label

def format_example(args, df, idx, n_reduced=None, include_answer=True, permute_pos=None, perm_i=None):
    prompt = df.iloc[idx, 0]  # The question
    
    if args.permutation_attack:
        choices, label = permute_choices_and_answer(df, idx, perm_i)
    elif args.reduce_attack:
        choices, label = choice_reduce_permute(df, idx, perm_i, n_reduced)
    else:
        if n_reduced is not None:
            choices, label = reduce_choices_and_answer(df, idx, n_reduced, permute_pos=permute_pos)
        else:
            k = df.shape[1] - 2  # Minus 2 for the question and answer columns
            choices = [df.iloc[idx, i+1] for i in range(k)]
            label = df.iloc[idx, k + 1]

    choice_labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # All possible choice labels

    # Include the new set of choices
    for j, choice in enumerate(choices):
        prompt += "\n{}. {}".format(choice_labels[j], choice)
        
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(label)
        return prompt
    else:
        return prompt, label

def permute_choices_and_answer(df, idx, perm_i):
    num_choices = df.shape[1] - 2
    choice_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:num_choices]
    perm_list = list(itertools.permutations(choice_labels))
    
    # Extract the original choices and label from the DataFrame row
    original_choices = [df.iloc[idx, i+1] for i in range(num_choices)]
    original_label = df.iloc[idx, num_choices + 1]  # Original answer (e.g., "A", "B", etc.)
    
    # Generate all possible permutations of choices
    perm_list = list(itertools.permutations(range(num_choices)))
    if perm_i >= len(perm_list):
        return "Invalid perm_i"
    
    # Extract the perm_i-th permutation
    permutation = perm_list[perm_i]
    
    # Apply the permutation to the original choices
    new_choices = [original_choices[i] for i in permutation]
    
    # Find the new position of the original answer label in the permuted list
    original_label_idx = ord(original_label) - ord("A")
    new_label_idx = np.where(np.array(permutation) == original_label_idx)[0][0]
    new_label = chr(ord("A") + new_label_idx)
    
    return new_choices, new_label

def reduce_choices_and_answer(df, idx, n_reduced, permute_pos=None):
    # Extract the original choices and label from the DataFrame row
    k = df.shape[1] - 2  # Minus 2 for the question and answer columns
    original_choices = [df.iloc[idx, i+1] for i in range(k)]
    original_label = df.iloc[idx, k + 1]  # Original answer (e.g., "A", "B", etc.)
    
    # Identify the index of the original answer in the choices
    original_label_idx = ord(original_label) - ord("A")
    
    # Choose a random subset of choices, while keeping the original answer
    other_choices = [i for i in range(k) if i != original_label_idx]
    random_subset = np.random.choice(other_choices, n_reduced - 1, replace=False).tolist()
    
    # If permute_pos is not None, make sure the answer remains at that position
    if permute_pos is not None:
        permute_pos_idx = ord(permute_pos) - ord("A")
        if permute_pos_idx in random_subset:
            random_subset.remove(permute_pos_idx)
    
    # Add the original answer index and sort
    random_subset.append(original_label_idx)
    random_subset.sort()

    # Create a new list of choices and a new label
    new_choices = [original_choices[i] for i in random_subset]
    
    if permute_pos is not None:
        new_choices[permute_pos_idx], new_choices[original_label_idx] = new_choices[original_label_idx], new_choices[permute_pos_idx]
        new_label_idx = permute_pos_idx
    else:
        new_label_idx = random_subset.index(original_label_idx)
        
    new_label = chr(ord("A") + new_label_idx)

    return new_choices, new_label

def gen_prompt(args, train_df, subject, k=-1, n_reduced=None):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(subject)
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(args, train_df, i, n_reduced=n_reduced, include_answer=True)
    return prompt

def full_search_eval(args, subject, dev_df, test_df, model, tokenizer, n_reduced=0, permute_pos=None):
    cors = []
    num_choices = test_df.shape[1] - 2
    answers = choices[:num_choices]
    device = model.device
    all_preds, all_gt = [], []
    for i in range(test_df.shape[0]):
        corr = 1
        if args.reduce_attack:
            total_perms = len(generate_permutation_indices(num_choices, n_reduced, original_label_idx=ord(test_df.iloc[i, test_df.shape[1]-1]) - ord('A')))
        elif args.permutation_attack:
            total_perms = math.factorial(num_choices)
        right_wrong = []
        for perm_i in range(total_perms):
            k = args.ntrain
            prompt_end, new_label = format_example(args, test_df, i, n_reduced=n_reduced, 
                                                   include_answer=False, permute_pos=permute_pos, perm_i=perm_i)
            train_prompt = gen_prompt(args, dev_df, subject, k, n_reduced=n_reduced)
            prompt = train_prompt + prompt_end
            
            label = new_label

            model_inputs = tokenizer(prompt, return_tensors='pt')
            input_ids = model_inputs.input_ids
            if not args.load_in_8bit:
                input_ids = input_ids.to(device)
            with torch.no_grad():
                output = model(input_ids)
            logits = output.logits
            token_probs = softmax(logits[0, -1, :], dim=-1)

            lprobs = []
            for ans in answers:
                ans_id = tokenizer(ans, add_special_tokens=False, return_tensors="pt").input_ids[0].item()
                lprobs.append(token_probs[ans_id].item())
                
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
            
            all_preds.append(pred)
            all_gt.append(label)

            cor = pred == label
            right_wrong.append(cor)
            if cor == False:
                corr = 0
                break
        cors.append(corr)
    
    acc = np.mean(cors)
    cors = np.array(cors)
    
    print("Average accuracy {:.2f} - {}".format(acc*100, subject))
    
    return cors, acc

def eval(args, subject, dev_df, test_df, model, tokenizer, n_reduced=0, permute_pos=None):
    cors = []
    num_choices = test_df.shape[1] - 2

    answers = choices[:num_choices]
    device = model.device
    all_preds, all_gt = [], []
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end, new_label = format_example(args, test_df, i, n_reduced=n_reduced, include_answer=False, permute_pos=permute_pos)
        train_prompt = gen_prompt(args, dev_df, subject, k, n_reduced=n_reduced)
        prompt = train_prompt + prompt_end
        
        label = new_label

        model_inputs = tokenizer(
            prompt,
            return_tensors='pt',
        )
        input_ids = model_inputs.input_ids
        if not args.load_in_8bit:
            input_ids = input_ids.to(device)
        with torch.no_grad():
            output = model(input_ids)
        logits = output.logits
        token_probs = softmax(logits[0, -1, :], dim=-1)

        lprobs = []
        for ans in answers:
            ans_id = tokenizer(ans, add_special_tokens=False, return_tensors="pt").input_ids[0].item()
            lprobs.append(token_probs[ans_id].item())
            
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        all_preds.append(pred)
        all_gt.append(label)

        cor = pred == label
        cors.append(cor)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.2f} - {}".format(acc*100, subject))

    return cors, acc


def load_model(args, engine):
    if engine == 'llama2-7b':
        tokenizer = transformers.LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', load_in_8bit=args.load_in_8bit)
        model = transformers.LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', load_in_8bit=args.load_in_8bit)
    elif engine == 'llama2-13b':
        tokenizer = transformers.LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf', load_in_8bit=args.load_in_8bit)
        model = transformers.LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", load_in_8bit=args.load_in_8bit)
    elif engine == 'llama2-70b':
        tokenizer = transformers.LlamaTokenizer.from_pretrained('meta-llama/Llama-2-70b-hf', load_in_8bit=True)
        model = transformers.LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", load_in_8bit=True)
    elif engine == 'llama2-7b-chat':
        tokenizer = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", load_in_8bit=args.load_in_8bit)
        model = AutoModelForCausalLM.from_pretrained("daryl149/llama-2-7b-chat-hf", load_in_8bit=args.load_in_8bit)
    elif engine == 'vicuna7b':
        tokenizer = transformers.AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5', load_in_8bit=args.load_in_8bit)
        model = transformers.AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", load_in_8bit=args.load_in_8bit)
    elif engine == 'vicuna13b':
        tokenizer = transformers.AutoTokenizer.from_pretrained('lmsys/vicuna-13b-v1.5', load_in_8bit=args.load_in_8bit)
        model = transformers.AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5", load_in_8bit=args.load_in_8bit)
    elif engine == 'wizard-7b':
        tokenizer = transformers.AutoTokenizer.from_pretrained('WizardLM/WizardLM-7B-V1.0', load_in_8bit=args.load_in_8bit)
        model = transformers.AutoModelForCausalLM.from_pretrained("WizardLM/WizardLM-7B-V1.0", load_in_8bit=args.load_in_8bit)
    elif engine == 'wizard-13b':
        tokenizer = transformers.AutoTokenizer.from_pretrained('WizardLM/WizardLM-13B-V1.1', load_in_8bit=args.load_in_8bit)
        model = transformers.AutoModelForCausalLM.from_pretrained("WizardLM/WizardLM-13B-V1.1", load_in_8bit=args.load_in_8bit)
    elif engine == 'internlm-20b':
        tokenizer = transformers.AutoTokenizer.from_pretrained('internlm/internlm-20b', load_in_8bit=args.load_in_8bit, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("internlm/internlm-20b", load_in_8bit=args.load_in_8bit, trust_remote_code=True)
    elif engine == 'falcon-7b':
        tokenizer = transformers.AutoTokenizer.from_pretrained('tiiuae/falcon-7b', load_in_8bit=args.load_in_8bit, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", load_in_8bit=args.load_in_8bit, trust_remote_code=True)
    elif engine == 'mpt-7b':
        tokenizer = transformers.AutoTokenizer.from_pretrained("mosaicml/mpt-7b", load_in_8bit=args.load_in_8bit, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b", load_in_8bit=args.load_in_8bit, trust_remote_code=True)
    return model, tokenizer

def main(args):
    engines = args.engine
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    print(subjects)
    print(args)

    for engine in engines:
        all_cors = []
        all_accs = []
        print("=====================================")
        print("Engine: {}".format(engine))
        print("=====================================")
        model, tokenizer = load_model(args, engine)
        if not args.load_in_8bit:
            model.cuda().half()
        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
            if args.use_subset:
                test_df = test_df[:100]
            
            if args.permutation_attack or args.reduce_attack:
                cors, acc = full_search_eval(args, subject, dev_df, test_df, model, tokenizer, n_reduced=args.n_reduced)
                all_cors.append(cors)

            elif args.position_permute:
                tmp_acc = {i: 0 for i in ['A', 'B', 'C', 'D']}
                for i in ['A', 'B', 'C', 'D']:
                    new_df = move_answers_to_position(test_df, i)
                    cors, acc = eval(args, subject, dev_df, new_df, model, tokenizer, n_reduced=args.n_reduced, permute_pos=i)
                    tmp_acc[i] = acc
                all_accs.append(tmp_acc)
            else:
                cors, acc = eval(args, subject, dev_df, test_df, model, tokenizer, n_reduced=args.n_reduced)
                all_cors.append(cors)
                all_accs.append(acc)
            
        if not args.position_permute:
            weighted_acc = np.mean(np.concatenate(all_cors))
            print("Average accuracy: {:.2f}".format(weighted_acc*100))
        elif args.permutation_attack:
            weighted_acc = np.mean(np.concatenate(all_cors))
            print("Worst permutation accuracy: {:.2f}".format(weighted_acc*100))
        else:
            weighted_acc = {i: np.mean([d[i] for d in all_accs]) for i in ['A', 'B', 'C', 'D']}
            print("Average accuracy")
            for key, value in weighted_acc.items():
                print(f"Length {key} - {value * 100:.2f}")
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        print("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--data_dir", "-d", type=str, default="/home/co-zong1/rds/rds-bmai-cdt-zong-hiFUWVB9vu0/multimodal_data/MMLU")
    
    parser.add_argument("--engine", "-e", choices=["llama2-7b", "llama2-13b", "llama2-70b", "llama2-7b-chat", "vicuna7b", "vicuna13b",
                                                   "wizard-7b", "wizard-13b", "internlm-20b", "falcon-7b", "mpt-7b"],
                        default=["llama2-7b", "llama2-13b", "vicuna7b", "vicuna13b","wizard-7b", "wizard-13b", "internlm-20b", "falcon-7b", "mpt-7b"], nargs="+")
    
    parser.add_argument("--n_reduced", "-n", type=int, default=None)
    parser.add_argument("--use_subset", "-u", action='store_true', default=False)

    parser.add_argument("--permutation_attack", "-es", action='store_true', default=False)
    parser.add_argument("--position_permute", "-r", action='store_true', default=False)
    parser.add_argument("--reduce_attack", "-ra", action='store_true', default=False)

    parser.add_argument("--load_in_8bit", "-8bit", action='store_true', default=False)

    args = parser.parse_args()
    main(args)
