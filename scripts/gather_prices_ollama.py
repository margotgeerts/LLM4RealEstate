from ollama import Client
import ollama
import pandas as pd
import os
import numpy as np
import time
import json
import argparse
from utils import *
import joblib


client = Client(host='http://localhost:11434')

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
                    default='llama3.2:3b-instruct-fp16',
                    choices=['llama3.2:3b-instruct-fp16', 'llama3.2','deepseek-r1:7b'],
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
                    "--n_jobs",
                    action="store",
                    type=int,
                    default=None,
                    help="Give the number of jobs you want to run in parallel",
)

def gather_answers(idx, prompt, model='llama3.2', n_examples=0, context=False, example_selection='geo'):

    temperature = 0
    seed = 0
    num_ctx = 16384
    
    
    row = prompt.df.loc[idx]

    messages = [{"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.get_price_prompt(row, context=context, n_examples=n_examples, example_selection=example_selection)}]
    print(messages)


    response = ollama.chat(messages=messages, model=model,
     options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx},
     stream=False)

    print(response['message']['content'])

    price = None
    price = get_price_from_response(response['message']['content'], currency=prompt.currency)

        
    return price



if __name__ == '__main__':

    args = parser.parse_args()

    if args.output_file is not None and os.path.exists(args.output_file):
        results = pd.read_csv(args.output_file, index_col=0)
    else:
        results = pd.DataFrame()

    if 'price' not in results.columns and 'log_price' in results.columns:
        results['price'] = np.exp(results['log_price']).round()
    
    col = 'predicted_price'

    indices = np.loadtxt(f'config/{args.dataset}_test_indices.txt', dtype=int)
    indices_to_complete = [idx for idx in indices if idx not in results.index]

    prompt = Prompt(args.dataset)

    for idx in indices_to_complete:
        price = gather_answers(idx, prompt, args.model, args.examples, args.context, args.example_selection)
        results.loc[idx, col] = price

        results.to_csv(args.output_file)

        print(f"Index {idx} done")
