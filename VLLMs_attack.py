import transformers
import torch
import argparse
from PIL import Image
import os
import numpy as np
import pandas as pd
import time
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
import random
from collections import defaultdict
import math
import itertools
from open_flamingo import create_model_and_transforms
from otter_ai import OtterForConditionalGeneration
from llava.model.builder import load_pretrained_model as load_llava_model
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

import sys
sys.path.append('../mPLUG-Owl')
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor


choices = ["A", "B", "C", "D", "E"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers).\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def compute_avg_acc(all_accs):
    # Group dicts by their length (number of choices)
    grouped_accs = defaultdict(list)
    for acc_dict in all_accs:
        grouped_accs[len(acc_dict)].append(acc_dict)

    # Compute the average accuracy for each choice in each group
    avg_accs = {}
    for length, acc_list in grouped_accs.items():
        avg_dict = defaultdict(float)
        num_dicts = len(acc_list)
        for acc_dict in acc_list:
            for choice, acc in acc_dict.items():
                avg_dict[choice] += acc
        # Calculate average
        for choice in avg_dict:
            avg_dict[choice] /= num_dicts
        avg_accs[length] = dict(avg_dict)

    return avg_accs

def reduce_choices_and_answer(df, idx, n):
    # Extract original choices and label based on the DataFrame structure
    choices_columns = [col for col in df.columns if col.startswith("choice")]
    original_choices = [df.loc[idx, col] for col in choices_columns]
    original_label = df.loc[idx, "label"]  # Original answer (e.g., "A", "B", etc.)

    # Identify the index of the original answer in the choices
    original_label_idx = ord(original_label) - ord("A")
    
    # Choose a random subset of choices, while keeping the original answer
    other_choices = [i for i in range(len(choices_columns)) if i != original_label_idx]
    random_subset = np.random.choice(other_choices, n - 1, replace=False).tolist()
    random_subset.append(original_label_idx)
    random_subset.sort()
    
    # Create new list of choices and new label
    new_choices = [original_choices[i] for i in random_subset]
    new_label_idx = random_subset.index(original_label_idx)
    new_label = chr(ord("A") + new_label_idx)
    
    return new_choices, new_label

def permute_choices_and_answer(df, idx, perm_i):
    row = df.iloc[idx]
    num_choices = sum(col.startswith("choice") for col in df.columns)
    
    # Extract the original choices and label from the DataFrame row
    original_choices = [value for key, value in row.items() if key.startswith('choice')]
    original_label = row['label']  # Original answer (e.g., "A", "B", etc.)
    
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

def generate_permutation_indices(n_choices, k_remaining, original_label_idx):
    combinations = [list(comb) for comb in itertools.combinations(set(range(n_choices)) - {original_label_idx}, k_remaining - 1)]
    all_permutations = []
    for comb in combinations:
        for perm in itertools.permutations([original_label_idx] + comb):
            all_permutations.append(list(perm))
    return all_permutations

def choice_reduce_permute(df, idx, perm_i, n_reduced):
    # Extract existing choices for the given row index
    row = df.iloc[idx]
    label = row['label']
    choices = [value for key, value in row.items() if key.startswith('choice')]
    k = len(choices)

    # Identify the index of the original answer in the choices
    original_label_idx = ord(label) - ord('A')

    # Generate permutation indices for reduced choices
    combinations = list(itertools.combinations(set(range(k)) - {original_label_idx}, n_reduced - 1))
    all_perms = []
    for comb in combinations:
        for perm in itertools.permutations([original_label_idx] + list(comb)):
            all_perms.append(list(perm))

    if perm_i >= len(all_perms):
        return "Invalid perm_i value"

    chosen_permutation = all_perms[perm_i]

    # Create new choices and label based on the permutation
    new_choices = [choices[i] for i in chosen_permutation]
    new_label_idx = np.where(np.array(chosen_permutation) == original_label_idx)[0][0]
    new_label = chr(ord('A') + new_label_idx)

    return new_choices, new_label

def move_answers_to_position(df, position):
    # Create a new DataFrame to store the modified questions, choices, and labels
    new_df = df.copy()
    position_idx = ord(position) - ord('A')
    
    for idx in range(len(df)):
        # Extract the question, choices, and label
        row = df.iloc[idx]
        question = row['question']
        label_col_name = 'label'
        
        choices = [(key, value) for key, value in row.items() if key.startswith('choice')]
        original_label = row[label_col_name]
        
        # Find the index of the original label
        original_label_idx = ord(original_label) - ord("A")
        
        # Move the answer to the new desired position
        choices[position_idx], choices[original_label_idx] = choices[original_label_idx], choices[position_idx]
        
        # Update the choices and label in the new DataFrame
        for i, (key, _) in enumerate(choices):
            new_df.at[idx, key] = choices[i][1]
        
        new_df.at[idx, label_col_name] = position  # Update the label to the new position

    return new_df

def format_example(args, df, idx, n_reduced=None, include_answer=True, permute_pos=None, perm_i=None):
    row = df.iloc[idx]
    prompt = "Question: " + row['question']  # The question
    
    if args.permutation_attack:
        choices, label = permute_choices_and_answer(df, idx, perm_i)
    elif args.reduce_attack:
        choices, label = choice_reduce_permute(df, idx, perm_i, n_reduced)
    else:
        if n_reduced is not None:
            choices, label = reduce_choices_and_answer(df, idx, n_reduced)
        else:
            choices = [value for key, value in row.items() if key.startswith('choice')]
            label = row['label']

    choice_labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # All possible choice labels

    # Include the new set of choices
    prompt += "\nOptions:"
    for j, choice in enumerate(choices):
        prompt += "\n{}. {}".format(choice_labels[j], choice)
        
    prompt += "\nAnswer:"
    prefix = "The following are multiple choice questions (with answers).\n\n"
    
    prompt = prefix + prompt
    if include_answer:
        prompt += " {}".format(label)
        return prompt
    else:
        return prompt, label

def full_search_eval(args, subject, dev_df, test_df, model, processor, tokenizer, n_reduced=0, permute_pos=None, engine=None):
    cors = []
    num_choices = sum(col.startswith("choice") for col in test_df.columns)
    answers = choices[:num_choices]
    if 'instructblip' in engine or 'blip2' in engine:
        device = model.device
    all_preds, all_gt = [], []
    for i in tqdm(range(test_df.shape[0])):
        try:
            corr = 1
            if args.reduce_attack:
                total_perms = len(generate_permutation_indices(num_choices, n_reduced, ord(test_df.iloc[i]['label']) - ord('A')))
            elif args.permutation_attack:
                total_perms = math.factorial(num_choices)
            for perm_i in range(total_perms):
                
                image = Image.open(os.path.join(args.data_dir, test_df.iloc[i]['image_path'])).convert("RGB")
                prompt, label = format_example(args, test_df, i, n_reduced=n_reduced,
                                                    include_answer=False, permute_pos=permute_pos, perm_i=perm_i)
                
                if 'instructblip' in engine or 'blip2' in engine:
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                    inputs = {k: v.half() if v.dtype == torch.float else v for k, v in inputs.items()}
                    with torch.no_grad():
                        output = model(**inputs)
                elif engine == 'openflamingo':
                    vision_x = [processor(image).unsqueeze(0)]
                    vision_x = torch.cat(vision_x, dim=0)
                    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
                    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
                    lang_x = tokenizer([f"<image>{prompt}"], return_tensors="pt",)
                    with torch.no_grad():
                        output = model(
                            vision_x=vision_x.cuda().half(),
                            lang_x=lang_x["input_ids"].cuda(),
                            attention_mask=lang_x["attention_mask"].cuda(),
                        )
                elif 'otter' in engine:
                    vision_x = processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
                    tokenizer.padding_side = "left"
                    model_dtype = next(model.parameters()).dtype
                    lang_x = tokenizer([f"<image>{prompt}"], return_tensors="pt",)
                    # Convert tensors to the model's data type
                    vision_x = vision_x.to(dtype=model_dtype)
                    with torch.no_grad():
                        output = model(
                            vision_x=vision_x.cuda().half(),
                            lang_x=lang_x["input_ids"].cuda(),
                            attention_mask=lang_x["attention_mask"].cuda(),
                        )
                elif 'llava' in engine:
                    image_tensor = processor.preprocess([image], return_tensors='pt')['pixel_values'].half().cuda()
                    conv_mode = 'llava_llama_2'
                    conv = conv_templates[conv_mode].copy()
                    # first message
                    if model.config.mm_use_im_start_end:
                        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
                    else:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                    conv.append_message(conv.roles[0], inp)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    
                    with torch.no_grad():
                        output = model(input_ids, images=image_tensor,)
                elif 'mplug-owl' in engine:
                    prompt = [f"<image>{prompt}"]
                    inputs = processor(text=prompt, images=[image], return_tensors='pt')
                    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        output = model(**inputs, num_images=torch.tensor([1]))
                else:
                    raise NotImplementedError
            
                logits = output.logits
                token_probs = softmax(logits[0, -1, :], dim=-1)

                lprobs = []
                for ans in answers:
                    ans_id = tokenizer(ans, add_special_tokens=False, return_tensors="pt").input_ids[0].item()
                    lprobs.append(token_probs[ans_id].item())
                    
                pred = {0: "A", 1: "B", 2: "C", 3: "D", 4: 'E'}[np.argmax(lprobs)]
                
                all_preds.append(pred)
                all_gt.append(label)

                cor = pred == label
                if cor == False:
                    corr = 0
                    break
            cors.append(corr)
        except:
            print("Error on {}".format(i))
            cors.append(0)
            all_gt.append(label)
            all_preds.append('A')
    
    acc = np.mean(cors)
    cors = np.array(cors)
    print("Average accuracy {:.2f} - {}".format(acc*100, subject))
    
    return cors, acc

def eval(args, subject, dev_df, test_df, model, processor, tokenizer, permute_pos=None, engine=None):
    cors = []
    all_preds, all_gt = [], []
    num_choices = sum(col.startswith("choice") for col in test_df.columns)
    
    answers = choices[:num_choices]
    if 'instructblip' in engine or 'blip2' in engine:
        device = model.device
    for i in tqdm(range(test_df.shape[0]), desc="Generating Answers"):
        try:
            image = Image.open(os.path.join(args.data_dir, test_df.iloc[i]['image_path'])).convert("RGB")
            prompt, label = format_example(args, test_df, i, n_reduced=args.n_reduced, include_answer=False, permute_pos=permute_pos)
            if 'instructblip' in engine or 'blip2' in engine:
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                inputs = {k: v.half() if v.dtype == torch.float else v for k, v in inputs.items()}
                with torch.no_grad():
                    output = model(**inputs)
            elif engine == 'openflamingo':
                vision_x = [processor(image).unsqueeze(0)]
                vision_x = torch.cat(vision_x, dim=0)
                vision_x = vision_x.unsqueeze(1).unsqueeze(0)
                tokenizer.padding_side = "left" # For generation padding tokens should be on the left
                lang_x = tokenizer([f"<image>{prompt}"], return_tensors="pt",)
                with torch.no_grad():
                    output = model(
                        vision_x=vision_x.cuda().half(),
                        lang_x=lang_x["input_ids"].cuda(),
                        attention_mask=lang_x["attention_mask"].cuda(),
                    )
            elif 'otter' in engine:
                vision_x = processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
                tokenizer.padding_side = "left"
                model_dtype = next(model.parameters()).dtype
                lang_x = tokenizer([f"<image>{prompt}"], return_tensors="pt",)
                # Convert tensors to the model's data type
                vision_x = vision_x.to(dtype=model_dtype)
                with torch.no_grad():
                    output = model(
                        vision_x=vision_x.cuda().half(),
                        lang_x=lang_x["input_ids"].cuda(),
                        attention_mask=lang_x["attention_mask"].cuda(),
                    )
            elif 'llava' in engine:
                image_tensor = processor.preprocess([image], return_tensors='pt')['pixel_values'].half().cuda()
                conv_mode = 'llava_llama_2'
                conv = conv_templates[conv_mode].copy()
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                
                with torch.no_grad():
                    output = model(input_ids, images=image_tensor,)
            elif 'mplug-owl' in engine:
                prompt = [f"<image>{prompt}"]
                inputs = processor(text=prompt, images=[image], return_tensors='pt')
                inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = model(**inputs, num_images=torch.tensor([1]))
            else:
                raise NotImplementedError

            logits = output.logits
            token_probs = softmax(logits[0, -1, :], dim=-1)

            lprobs = []
            for ans in answers:
                ans_id = tokenizer(ans, add_special_tokens=False, return_tensors="pt").input_ids[0].item()
                lprobs.append(token_probs[ans_id].item())
            
            pred = {0: "A", 1: "B", 2: "C", 3: "D", 4: 'E'}[np.argmax(lprobs)]
            all_preds.append(pred)
            all_gt.append(label)

            cor = pred == label
            cors.append(cor)
        except:
            print("Error on {}".format(i))
            cors.append(0)
            all_gt.append(label)
            all_preds.append('A')

    acc = np.mean(cors)
    cors = np.array(cors)
    print("Average accuracy {:.2f} - {}".format(acc*100, subject))
    
    return cors, acc

def load_model(args, engine):
    if engine == 'instructblip7b':
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/instructblip-vicuna-7b")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    if engine == 'instructblip13b':
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b")
        tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/instructblip-vicuna-13b")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
    elif engine == 'blip2':
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
        tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/blip2-opt-6.7b")
        model = Blip2ForConditionalGeneration.from_pretrained( "Salesforce/blip2-opt-6.7b")
    elif engine == 'openflamingo':
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
            #cache_dir=os.environ["TRANSFORMERS_CACHE"]
        )
        processor = image_processor
    elif engine == 'otter-llama':
        model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-9B-LA-InContext", device_map="auto")
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif engine == 'otter-mpt':
        model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-MPT7B", device_map="auto")
        tokenizer = model.text_tokenizer
        image_processor = transformers.CLIPImageProcessor()
        processor = image_processor
    elif engine == 'llava7b':
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='PahaII/MM-LLaMA-2-7B-ft', model_base=None, model_name='llava', load_8bit=args.load_in_8bit)
        processor = image_processor
    elif engine == 'llava13b':
        tokenizer, model, image_processor, context_len = load_llava_model(model_path='liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3', model_base=None, model_name='llava', load_8bit=args.load_in_8bit)
        processor = image_processor
    elif engine == 'mplug-owl':
        pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
        model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16,
        )
        image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_ckpt)
        processor = MplugOwlProcessor(image_processor, tokenizer)
    elif engine == 'mplug-owl-pt':
        pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b-pt'
        model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16,
        )
        image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_ckpt)
        processor = MplugOwlProcessor(image_processor, tokenizer)
    return model, processor, tokenizer


def main(args):
    engines = args.engine
    
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, 'meta_data')) if "_test.csv" in f])

    print(subjects)
    print(args)

    for engine in engines:
        print("=====================================")
        print("Engine: {}".format(engine))
        print("=====================================")
        try:
            model, processor, tokenizer = load_model(args, engine)
        except:
            print("Failed to load model for {}".format(engine))
            continue
        model.cuda().half()

        all_cors = []
        all_accs = []
        acc_per_num = {2: [], 3: [], 4: [], 5: []}
        for subject in subjects:
            
            dev_df = pd.read_csv(os.path.join(args.data_dir, "meta_data", subject + "_test.csv"))[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "meta_data", subject + "_test.csv"))
            num_choices = sum(col.startswith("choice") for col in test_df.columns)
            if args.permutation_attack or args.reduce_attack:
                cors, acc = full_search_eval(args, subject, dev_df, test_df, model, processor, tokenizer, n_reduced=args.n_reduced, engine=engine)
                all_cors.append(cors)
                acc_per_num[num_choices].append(acc)
            elif args.position_permute:
                curr_choices = choices[:num_choices]
                tmp_acc = {i: 0 for i in curr_choices}
                for i in curr_choices:
                    new_df = move_answers_to_position(test_df, i)
                    cors, acc = eval(args, subject, dev_df, new_df, model, processor, tokenizer, permute_pos=i, engine=engine)
                    tmp_acc[i] = acc
                all_accs.append(tmp_acc)
                all_cors.append(cors)
            else:
                cors, acc = eval(args, subject, dev_df, test_df, model, processor, tokenizer, engine=engine)
                all_cors.append(cors)
                acc_per_num[num_choices].append(acc)
                all_accs.append(acc)
            
        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.2f}".format(weighted_acc*100))
        if not args.position_permute:
            acc_per_length = {key: np.mean(acc_list) if acc_list else 0 for key, acc_list in acc_per_num.items()}
            print("Average accuracy per number of choices")
            for key, value in acc_per_length.items():
                print(f"Length {key} - {value * 100:.2f}")

        elif args.permutation_attack:
            acc_per_length = {key: np.mean(acc_list) if acc_list else 0 for key, acc_list in acc_per_num.items()}
            print("Average accuracy per number of choices")
            for key, value in acc_per_length.items():
                print(f"Length {key} - {value * 100:.2f}")

        del model
        del processor
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--data_dir", "-d", type=str, default="data/ScienceQA")
    
    parser.add_argument("--engine", "-e", choices=["blip2", "instructblip7b", "instructblip13b", "openflamingo", "otter-llama", "otter-mpt", 
                                                   "llava7b", "llava13b", 'mplug-owl-pt', "mplug-owl"],
                        default=["instructblip7b"], nargs="+")
    
    parser.add_argument("--n_reduced", "-n", type=int, default=None)

    parser.add_argument("--permutation_attack", "-es", action='store_true', default=False)
    parser.add_argument("--position_permute", "-r", action='store_true', default=False)
    parser.add_argument("--reduce_attack", "-ra", action='store_true', default=False)

    parser.add_argument("--load_in_8bit", "-8bit", action='store_true', default=False)

    args = parser.parse_args()
    main(args)