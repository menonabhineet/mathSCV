import json
from collections import Counter

def analyze_dataset(filepath, dataset_name):
    print(f"\n{'='*50}")
    print(f"MINING JSON FOR: {dataset_name}")
    print(f"{'='*50}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)

    # 1. Error Categorization Analysis
    error_types = []
    for r in results:
        if not r['results']['SCV_Pipeline']['correct']:
            error_types.append(r['analysis'].get('error_category', 'Unknown'))
            
    error_counts = Counter(error_types)
    print("\n--- ERROR CATEGORIZATION (SCV Failures) ---")
    for err, count in error_counts.items():
        print(f"{err}: {count}")

    # 2. Token Efficiency Analysis
    b1_tokens_est = sum(r['metrics']['total_tokens'] for r in results) / len(results) / 4 # Rough fraction for just B1
    scv_tokens_est = sum(r['metrics']['total_tokens'] for r in results) / len(results)
    print("\n--- TOKEN EFFICIENCY ---")
    print(f"Avg Tokens (Full Pipeline): {scv_tokens_est:.0f}")

    # 3. Find the Golden Examples
    print("\n--- QUALITATIVE EXAMPLES TO MINE ---")
    
    improvements = [r['id'] for r in results if r['analysis']['improvement_from_B1']]
    print(f"Improvements (SCV fixed B1): {improvements}")
    
    regressions = [r['id'] for r in results if r['analysis']['regression_from_B1']]
    print(f"Regressions (SCV broke B1): {regressions}")

def main():
    analyze_dataset("data/results/svamp_deepseek_eval.json", "SVAMP")
    analyze_dataset("data/results/gsm8k_deepseek_eval.json", "GSM8K")
    analyze_dataset("data/results/asdiv_deepseek_eval.json", "ASDiv")

if __name__ == "__main__":
    main()