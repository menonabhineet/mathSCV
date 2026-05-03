import os
import json
from src.api_harness import OpenRouterHarness
from src.data_loader import load_gsm8k
from src.data_loader import load_svamp 
from src.data_loader import load_asdiv 
from src.unified_evaluator import run_unified_evaluation

def save_checkpoint(results_list, output_path):
    """Saves the current evaluation results to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_list, f, indent=4)

def main():
    print("--- Phase 4: Unified Evaluation Engine ---")
    
    harness = OpenRouterHarness()
    
    print("Loading dataset...")
    #full_dataset = load_gsm8k()
    #full_dataset = load_svamp()
    full_dataset = load_asdiv()
    
    # Running a robust 250-question sample
    test_subset = full_dataset[:250] 
    
    output_file = "data/results/ASDiv_deepseek_eval.json"
    results = []
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        print(f"Resuming from checkpoint. {len(results)} problems already evaluated.")
    
    processed_ids = {res['id'] for res in results}
    
    for i, problem_data in enumerate(test_subset):
        if problem_data['id'] in processed_ids:
            continue
            
        print(f"\nEvaluating Problem {i+1}/{len(test_subset)} (ID: {problem_data['id']})")
        print(f"Question: {problem_data['problem'][:80]}...")
        
        try:
            # Run the unified pipeline
            eval_data = run_unified_evaluation(harness, problem_data, k=3)
            results.append(eval_data)
            
            # Save checkpoint immediately
            save_checkpoint(results, output_file)
            
            # Extract stats for the console printout
            b1_corr = eval_data['results']['B1_SingleAgent']['correct']
            b3_corr = eval_data['results']['B3_SelfRefine']['correct']
            scv_corr = eval_data['results']['SCV_Pipeline']['correct']
            tokens = eval_data['metrics']['total_tokens']
            time_sec = eval_data['metrics']['time_seconds']
            
            print(f"B1 Correct: {b1_corr} | B3 Correct: {b3_corr} | SCV Correct: {scv_corr}")
            print(f"Cost: {tokens} tokens | Speed: {time_sec}s")
            
            if eval_data['analysis']['improvement_from_B1']: 
                print(">> SUCCESS: SCV fixed a wrong answer! (Improvement)")
            if eval_data['analysis']['regression_from_B1']: 
                print(f">> FAIL: SCV broke a right answer. Error Type: [{eval_data['analysis']['error_category']}]")
            
        except Exception as e:
            print(f"Error evaluating problem {problem_data['id']}: {e}")
            print("Stopping evaluation loop.")
            break
            
    # Calculate final stats for this run
    if results:
        total = len(results)
        b1_acc = sum(1 for r in results if r['results']['B1_SingleAgent']['correct']) / total * 100
        b2_acc = sum(1 for r in results if r['results']['B2_SelfConsistency']['correct']) / total * 100
        b3_acc = sum(1 for r in results if r['results']['B3_SelfRefine']['correct']) / total * 100
        b4_acc = sum(1 for r in results if r['results']['B4_BasicDebate']['correct']) / total * 100
        scv_acc = sum(1 for r in results if r['results']['SCV_Pipeline']['correct']) / total * 100
        
        improvements = sum(1 for r in results if r['analysis']['improvement_from_B1'])
        regressions = sum(1 for r in results if r['analysis']['regression_from_B1'])
        
        avg_tokens = sum(r['metrics']['total_tokens'] for r in results) / total
        
        print("\n=== FINAL DEEPSEEK ASDIV METRICS ===")
        print(f"Total Evaluated: {total}")
        print(f"B1 (Single Agent):    {b1_acc:.1f}%")
        print(f"B2 (Self-Consistency):{b2_acc:.1f}%")
        print(f"B3 (Self-Refine):     {b3_acc:.1f}%")
        print(f"B4 (Basic Debate):    {b4_acc:.1f}%")
        print(f"SCV Pipeline:         {scv_acc:.1f}%")
        print("-" * 30)
        print(f"Improvements vs B1: {improvements}")
        print(f"Regressions vs B1:  {regressions}")
        print(f"Average Tokens/Problem: {avg_tokens:.0f}")
        print("==============================")

if __name__ == "__main__":
    main()