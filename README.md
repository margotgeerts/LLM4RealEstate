# *On the Performance of LLMs for Real Estate Appraisal* 🏡📊
LLM4RealEstate explores the potential of Large Language Models (LLMs) in real estate appraisal, leveraging prompt optimization with in-context learning (ICL). This project evaluates various prompting strategies, and investigates LLMs' ability to provide a viable alternative to traditional machine learning (ML) models, in terms of prediction accuracy, uncertainty estimation and interpretability.

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

