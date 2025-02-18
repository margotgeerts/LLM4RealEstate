import pandas as pd
import os
import numpy as np
import time
import json
import argparse
from utils import *
import joblib
from transformers import AutoModelForCausalLM, AutoTokenizer 
import vllm
from vllm import LLM, SamplingParams
from typing import List


parser = argparse.ArgumentParser(description='Gather output from chatbots')
parser.add_argument(
                    "--dataset",
                    action="store",
                    type=str,
                    default='KC',
                    choices=['KC', 'beijing', 'barcelona'],
                    help="Give the dataset you want to use",
)

parser.add_argument(
                    "--model",
                    action="store",
                    type=str,
                    default='neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8',
                    choices=['neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8'],
                    help="give the model you want to use"
)

parser.add_argument(
                    "--examples",
                    action="store",
                    type=int,
                    default=0,
                    choices=[0, 3, 10],
                    help="give the number of examples you want to use"
)

parser.add_argument(
                    "--example_selection",
                    action="store",
                    type=str,
                    default='geo',
                    choices=['geo', 'hedonic', 'mixed'],
                    help="give the type of example selection you want to use")

parser.add_argument(
                    "--context",
                    action="store",
                    type=bool,
                    default=False,
                    choices=[False, True],
                    help="give True if you want to use context, False otherwise"
)

parser.add_argument(
                    "--output_file",
                    action="store",
                    type=str,
                    help="Give the full path of where you want to save the output file",
)

parser.add_argument(
                    "--n_gpu",
                    action="store",
                    type=int,
                    default=None,
                    help="Give the number of gpus you want to use",
)


def gather_answers(idx, prompt, answers, model='neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8', n_examples=0, context=False, example_selection='geo', n_gpu=1, output_file=None):

    temperature = 0
    seed = 0
    
    
    row = prompt.df.loc[idx]
    print(answers.shape)


    messages = []
    if len(idx) == 1:
        messages = [[{"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.get_price_prompt(row, context=context, n_examples=n_examples, example_selection=example_selection)},
                {"role": "assistant", "content": str(answers.loc[idx, 'predicted_price'])},
                {"role": "user", "content": "Please provide a price interval with 90% coverage around your estimated price for this property in the format 'min_price - max_price'."}]]
    else:
        messages = [[{"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.get_price_prompt(r, context=context, n_examples=n_examples, example_selection=example_selection)},
                {"role": "assistant", "content": str(answers.loc[i, 'predicted_price'])},
                {"role": "user", "content": "Please provide a price interval with 90% coverage around your estimated price for this property in the format 'min_price - max_price'."}]  for i, r in row.iterrows()]
    # print(messages)

    tokenizer = AutoTokenizer.from_pretrained(model)
    input_list_tok = [tokenizer.apply_chat_template(user_input, tokenize=False,add_special_tokens=False, add_generation_prompt=True) for user_input in messages]

    cpu_offload_gb = 210 if n_gpu == 1 else 100
    model = LLM(model=model, cpu_offload_gb=cpu_offload_gb, download_dir=cache_dir)
    # Model generation
    sampling_params = SamplingParams(temperature=temperature, max_tokens=1024, seed=seed)
    
    # split up the input list in chunks of 50
    chunk_size = 50
    for i in range(0, len(input_list_tok), chunk_size):
        start_id = i
        end_id = min(i+chunk_size, len(input_list_tok))
        print(f'Processing chunk {start_id} to {end_id}')

        output_list = model.generate(input_list_tok[start_id:end_id], sampling_params=sampling_params,use_tqdm=True)
        output_list = [output.outputs[0].text.strip() for output in output_list]
        mins, maxs = zip(*[get_interval_from_response(output) for output in output_list])
        print(len(mins))
        answers.loc[idx[start_id:end_id], 'min_price'] = mins
        answers.loc[idx[start_id:end_id], 'max_price'] = maxs
        if output_file is not None:
            print(f'Saving to {output_file}')
            answers.to_csv(output_file)
        
    return answers



if __name__ == '__main__':

    args = parser.parse_args()


    indices = np.loadtxt(f'config/{args.dataset}_test_indices.txt', dtype=int)
    

    if args.output_file is not None and os.path.exists(args.output_file):
        results = pd.read_csv(args.output_file, index_col=0)
        
    else:
        results = pd.DataFrame(index=indices)

    if 'price' not in results.columns and 'log_price' in results.columns:
        results['price'] = np.exp(results['log_price']).round()
    
    
    prompt = Prompt(args.dataset)

    out = gather_answers(indices, prompt, results, args.model, args.examples, args.context, args.example_selection, args.n_gpu, args.output_file)
   

    out.to_csv(args.output_file)