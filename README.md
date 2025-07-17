# On the Performance of LLMs for Real Estate Appraisal </br><sub><sub>*Margot Geerts, Manon Reusens, Bart Baesens, Seppe vanden Broucke, Jochen De Weerdt* [[*2025*]]([link-to-paper](https://arxiv.org/abs/2506.11812))</sub></sub>
This study explores the potential of Large Language Models (LLMs) in real estate appraisal, leveraging prompt optimization with in-context learning (ICL). This project evaluates various prompting strategies, and investigates LLMs' ability to provide a viable alternative to traditional machine learning (ML) models, in terms of prediction accuracy, uncertainty estimation and interpretability.

## Repository structure
This repository is organised as follows:
```bash
├── config/
│    ├── barcelona_test_indices.txt: the test indices for the Barcelona dataset
│    ├── beijing_test_indices.txt: the test indices for the Beijing dataset
│    ├── KC_test_indices.txt: the test indices for the King County dataset
│    ├── context.json: the market reports for the three datasets in different time periods
│    ├── features.json: the features used in the three datasets
│    ├── prompt_template.json: the prompt templates
│    └── task_definition.json: the task definition prompts
│
├── data/
│    ├── raw/
│    │    
│    └── processed/
│            ├── barcelona.csv: the Barcelona dataset
│            ├── beijing.csv: the Beijing dataset
│            └── KC.csv: the King County dataset
└── scripts/
        ├── baselines_hedonic.py: script to run the hedonic lightgbm baseline
        ├── baselines_knn.py: script to run the k-nearest neighbours baseline
        ├── baselines.py: script to run the lightgbm and k-nearest neighbours baselines
        ├── gather_features_ollama.py: script to gather feature rankings from Ollama
        ├── gather_features_openai.py: script to gather feature rankings from OpenAI
        ├── gather_features_vllm.py: script to gather feature rankings using VLLM
        ├── gather_intervals_ollama.py: script to gather prediction intervals from Ollama
        ├── gather_intervals_openai.py: script to gather prediction intervals from OpenAI
        ├── gather_intervals_vllm.py: script to gather prediction intervals using VLLM
        ├── gather_prices_ollama.py: script to gather price predictions from Ollama
        ├── gather_prices_openai.py: script to gather price predictions from OpenAI
        ├── gather_prices_vllm.py: script to gather price predictions using VLLM
        └── utils.py: utility functions for creating prompts and processing responses

```

## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

## Reproducing the results
To reproduce the baseline results, taking the barcelona dataset as an example, run the following commands:
```bash
python scripts/baselines.py --dataset barcelona
python scripts/baselines_hedonic.py --dataset barcelona
python scripts/baselines_knn.py --dataset barcelona
```
Arguments:
- `--dataset`: the dataset to use, options are `barcelona`, `beijing`, and `KC`.


To reproduce the results with Ollama, first install Ollama according to the instructions in the [Ollama website](https://ollama.com), then run the following command:
```bash
python scripts/gather_prices_ollama.py --dataset barcelona --model llama3.2 --output_file results/barcelona_ollama.csv 
python scripts/gather_intervals_ollama.py --dataset barcelona --model llama3.2 --output_file results/barcelona_ollama.csv
python scripts/gather_features_ollama.py --dataset barcelona --model llama3.2 --output_file results/barcelona_ollama.csv
```
Arguments:
- `--dataset`: the dataset to use, options are `barcelona`, `beijing`, and `KC`.
- `--model`: the Ollama model to use, options are `llama3.2` and `deepseek-r1:7b`.
- `--output_file`: the output file to save the results to.
- `--examples`: the number of examples to use for the prompt strategy, options are `0`, `3`, `10`.
- `--example_selection`: the prompt strategy to use, options are `geo`, `hedonic`, `mixed`.
- `--context`: whether to use context or not, options are `False`, `True`.


e.g. using the `10 ex. mixed` prompt strategy:
```bash
python scripts/gather_prices_ollama.py --dataset barcelona --model llama3.2 --output_file results/barcelona_ollama_prices_prompt_strategy.csv --examples 10 --example_selection mixed
```

To reproduce the results with OpenAI, make sure you have an API key and run the following command with the same arguments as above:
```bash
python scripts/gather_prices_openai.py --dataset barcelona --model gpt-4o-mini --output_file results/barcelona_openai_prices.csv --key <API_KEY>
```

## Citing

Please cite our paper and/or code as follows:
```bibtex
@article{Geerts2025LLMsRealEstate,
  title={On the Performance of LLMs for Real Estate Appraisal},
  author={Margot Geerts and Manon Reusens and Bart Baesens and Seppe {vanden Broucke} and Jochen {De Weerdt}},
  journal={arXiv preprint arXiv:2506.11812},
  year={2025}
}
