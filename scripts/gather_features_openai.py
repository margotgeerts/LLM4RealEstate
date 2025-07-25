import argparse
from openai import OpenAI
import os
import pandas as pd
import numpy as np
import time
from utils import * 
import json

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
                    default='gpt-4o-mini',
                    choices=['gpt-4o-mini'],
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
                    help="give the type of example selection you want to use"
)

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
                    '--key',
                    action='store',
                    type=str,
                    help='give the API key'
)

parser.add_argument(
                    '--n_chunks',
                    action='store',
                    type=int,
                    default=2,
                    help='give the number of chunks to split the batch into'
)

def get_content_from_response(response, request_id):
    response = list(filter(lambda x: x['custom_id'] == request_id, response))[0]
    return response["response"]["body"]["choices"][0]["message"]["content"]

def gather_answers(batch_idx, prompt, answers, model='gpt-4o-mini', n_examples=0, context=False, example_selection='geo', n_chunks=2):

    temperature = 0
    seed = 0
    

    batch_requests = []
    print(answers.shape)
    print(answers.min_price.notnull().sum())

    for idx in batch_idx:
        row = prompt.df.loc[idx]

        messages = [{"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.get_price_prompt(row, context=context, n_examples=n_examples, example_selection=example_selection)},
                {"role": "assistant", "content": str(answers.loc[idx, 'predicted_price'])},
                {"role": "user",
                        "content": "Please provide a price interval with 90% coverage around your estimated price for this property in the format 'min_price - max_price'."},
                {"role": "assistant", "content": f"{str(answers.loc[idx, 'min_price'])} - {str(answers.loc[idx, 'max_price'])}"},
                {"role": "user",
                        "content": "Please provide the top 5 features that you deemed most important for your previous predictions. Answer with a comma-separated list of features, using the feature names: {features}.".format(features=', '.join(['coordinates', 'address']+prompt.features['hedonic']+['transaction_date']))}]
        batch_requests.append(
            {"custom_id": f"request3-{idx}", "method": "POST", "url": "/v1/chat/completions", 
            "body": {"model":model, "messages": messages, "max_tokens": 1000, "temperature": temperature, "seed": seed}})
    
    if n_chunks > 1:
        responses = []
        for i in range(n_chunks):
            split_idx = len(batch_requests)//n_chunks
            if i == n_chunks-1:
                batch_requests_split = batch_requests[i*split_idx:]
                chunk_idx = batch_idx[i*split_idx:]
            else:
                batch_requests_split = batch_requests[i*split_idx:(i+1)*split_idx]
                chunk_idx = batch_idx[i*split_idx:(i+1)*split_idx]
            
            # Save the requests to a file
            with open(f'tmp/requests3{i+1}.jsonl', 'w') as f:
                # save each request as a separate line
                for request in batch_requests_split:
                    json.dump(request, f)
                    f.write('\n')
            
            # create a batch input file
            batch_input_file = client.files.create(
                file=open(f"tmp/requests3{i+1}.jsonl", "rb"),
                purpose="batch"
                )

            batch_input_file_id = batch_input_file.id

            batch = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                "description": f"request3"
                }
            )

            # Wait for the batch to complete
            while True:
                time.sleep(10)
                batch = client.batches.retrieve(batch.id)
                if batch.status == "completed":
                    break
                elif batch.status == "failed":
                    time.sleep(60)
                    batch = client.batches.create(
                        input_file_id=batch_input_file_id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={
                        "description": f"request3"
                        }
                    )
                    print(f"Batch failed, resubmitted with id: {batch.id}")
            

            # Get the results
            batch_output_file_id = batch.output_file_id
            r = client.files.content(batch_output_file_id).text
            r = r.split('\n')
            r = [json.loads(response) for response in r if response != '']
            features = [get_content_from_response(r, f"request3-{idx}") for idx in chunk_idx]
            answers.loc[chunk_idx, 'features'] = features
            answers.to_csv('tmp/answers.csv')
            responses += r



    else:
        # Save the requests to a file
        with open('tmp/requests3.jsonl', 'w') as f:
            # save each request as a separate line
            for request in batch_requests:
                json.dump(request, f)
                f.write('\n')
        
        batch_input_file2 = client.files.create(
            file=open("tmp/requests3.jsonl", "rb"),
            purpose="batch"
            )
        
        batch_input_file_id2 = batch_input_file2.id

        batch = client.batches.create(
            input_file_id=batch_input_file_id2,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "request3"
            }
        )

        # Wait for the batch to complete
        while True:
            time.sleep(10)
            batch = client.batches.retrieve(batch.id)
            if batch.status == "completed":
                break
            elif batch.status == "failed":
                time.sleep(60)
                batch = client.batches.create(
                    input_file_id=batch_input_file_id2,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                    "description": "request3"
                    }
                )
                print(f"Batch failed, resubmitted with id: {batch.id}")
        
        # Get the results
        batch_output_file_id2 = batch.output_file_id
        print(batch_output_file_id2)
        responses = client.files.content(batch_output_file_id2).text
        responses = responses.split('\n')
        responses = [json.loads(response) for response in responses if response != '']
        features = [get_content_from_response(responses, f"request3-{idx}") for idx in batch_idx]
        answers.loc[batch_idx, 'features'] = features
        answers.to_csv('tmp/answers.csv')

    # write responses to a file
    with open('tmp/responses.jsonl', 'w') as f:
        # save each request as a separate line
        for response in responses:
            json.dump(response, f)
            f.write('\n')

        
    return answers

if __name__ == '__main__':

    args = parser.parse_args()

    os.environ["OpenAI_API_KEY"]=args.key

    client = OpenAI(api_key=os.environ.get("OpenAI_API_KEY"),)

    if args.output_file is not None and os.path.exists(args.output_file):
        results = pd.read_csv(args.output_file, index_col=0)
        # drop the rows where all values are missing
        # results = results.dropna(how='all')
    else:
        results = pd.DataFrame()

    if 'price' not in results.columns and 'log_price' in results.columns:
        results['price'] = np.exp(results['log_price']).round()
    
    col = 'predicted_price'

    indices = np.loadtxt(f'config/{args.dataset}_test_indices.txt', dtype=int)

    batch_idx = [idx for idx in indices if (idx not in results.index) or ('features' not in results.columns) or (results.loc[idx].isna().sum()>0)]
    print(f"Batch indices: {batch_idx}")

    prompt = Prompt(args.dataset)
    if prompt.df.loc[batch_idx[0]].address is None:
        print("No address")
        raise Exception("No address")
    else:
        print(prompt.df.loc[batch_idx[0]].address)

    df = gather_answers(batch_idx, prompt=prompt, model=args.model, answers=results,
    n_examples=args.examples, context=args.context, example_selection=args.example_selection,
    n_chunks=args.n_chunks)

    df.to_csv(args.output_file)




    






    
