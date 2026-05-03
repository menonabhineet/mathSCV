import json
import os
import re
import pandas as pd

def extract_gsm8k_answer(answer_str):
    """Extracts the final numeric answer from GSM8K's '####' format."""
    try:
        return str(answer_str).split('####')[1].strip()
    except IndexError:
        return str(answer_str).strip()

def load_gsm8k(filepath="data/raw/gsm8k/main/test-00000-of-00001.parquet"):
    """Loads and standardizes the GSM8K parquet dataset."""
    df = pd.read_parquet(filepath)
    standardized_data = []
    
    for idx, row in df.iterrows():
        standardized_data.append({
            "id": f"gsm8k_{idx}",
            "source": "GSM8K",
            "problem": str(row['question']),
            "ground_truth": extract_gsm8k_answer(row['answer'])
        })
    print(f"Loaded {len(standardized_data)} problems from GSM8K.")
    return standardized_data

def load_svamp(filepath="data/raw/SVAMP.csv"):
    """Loads and standardizes the SVAMP csv dataset."""
    df = pd.read_csv(filepath)
    standardized_data = []
    
    for idx, row in df.iterrows():
        # SVAMP splits the problem into 'Body' and 'Question'
        body = str(row.get('Body', ''))
        question = str(row.get('Question', row.get('question', '')))
        
        # If it's split, combine them. Otherwise, just use the question.
        full_problem = f"{body} {question}".strip() if body and body != 'nan' else question
        
        standardized_data.append({
            "id": f"svamp_{idx}",
            "source": "SVAMP",
            "problem": full_problem,
            "ground_truth": str(row.get('Answer', row.get('answer', ''))).strip()
        })
    print(f"Loaded {len(standardized_data)} problems from SVAMP.")
    return standardized_data

def load_asdiv(filepath="data/raw/nlu_asdiv.csv"):
    """Loads and standardizes the ASDiv csv dataset."""
    df = pd.read_csv(filepath)
    standardized_data = []
    
    for idx, row in df.iterrows():
        # ASDiv column naming can vary depending on the specific release
        body = str(row.get('Body', ''))
        question = str(row.get('Question', row.get('question', '')))
        # Extracts only the first sequence of numbers/decimals
        raw_answer = str(row.get('Answer', row.get('answer', '')))
        match = re.search(r'-?\d+\.?\d*', raw_answer)
        clean_answer = match.group(0) if match else raw_answer.strip()
        full_problem = f"{body} {question}".strip() if body and body != 'nan' else question
        
        standardized_data.append({
            "id": f"asdiv_{idx}",
            "source": "ASDiv",
            "problem": full_problem,
            "ground_truth": clean_answer
        })
    print(f"Loaded {len(standardized_data)} problems from ASDiv.")
    return standardized_data

def create_tiny_sample_set(output_path="data/sample/tiny_math.json"):
    """Creates a local sample dataset for API testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_data = [
        {
            "id": "gsm8k_sample_1",
            "source": "GSM8K",
            "problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "ground_truth": "72"
        }
    ]
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=4)
    return sample_data