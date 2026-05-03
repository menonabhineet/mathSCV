import time
import json
import os
from collections import Counter
from src.prompts import (
    get_single_agent_cot_prompt, get_critique_prompt, 
    get_revision_prompt, get_verifier_prompt, get_error_categorization_prompt
)
from src.extractors import extract_xml_answer

def check_correctness(extracted, truth):
    if extracted is None: return False
    try:
        return float(extracted) == float(truth)
    except (ValueError, TypeError):
        return str(extracted).strip().lower() == str(truth).strip().lower()

def run_unified_evaluation(harness, problem_data, model=None, k=3):
    start_time = time.time()
    total_tokens = 0
    
    problem_text = problem_data['problem']
    ground_truth = str(problem_data['ground_truth']).strip()
    
    sys_solve, user_solve = get_single_agent_cot_prompt()
    sys_crit_adv, user_crit_adv = get_critique_prompt(is_self_reflection=False)
    sys_crit_self, user_crit_self = get_critique_prompt(is_self_reflection=True)
    sys_verify, user_verify = get_verifier_prompt()
    sys_revise, user_revise = get_revision_prompt()
    
    candidates = []
    adv_critiques = []
    extracted_candidates = []
    
    # 1. Generate K Candidates and Adversarial Critiques (Used for B1, B2, B4, and SCV)
    for i in range(k):
        sol, t = harness.generate_response_with_tokens(sys_solve, user_solve.format(problem=problem_text), model=model, temperature=0.8)
        total_tokens += t
        candidates.append(sol)
        extracted_candidates.append(extract_xml_answer(sol))
        
        crit, t = harness.generate_response_with_tokens(sys_crit_adv, user_crit_adv.format(problem=problem_text, solution=sol), model=model, temperature=0.3)
        total_tokens += t
        adv_critiques.append(crit)

    # Baseline 1: Single-Agent
    b1_ans = extracted_candidates[0]
    b1_correct = check_correctness(b1_ans, ground_truth)

    # Baseline 2: Self-Consistency
    valid_answers = [ans for ans in extracted_candidates if ans is not None]
    b2_ans = Counter(valid_answers).most_common(1)[0][0] if valid_answers else None
    b2_correct = check_correctness(b2_ans, ground_truth)

    # Baseline 3: Self-Refine (Self-Critique -> Revise on Candidate 1)
    self_crit, t = harness.generate_response_with_tokens(
        sys_crit_self, user_crit_self.format(problem=problem_text, solution=candidates[0]), model=model, temperature=0.3
    )
    total_tokens += t
    
    b3_rev, t = harness.generate_response_with_tokens(
        sys_revise, user_revise.format(problem=problem_text, solution=candidates[0], critique=self_crit), model=model
    )
    total_tokens += t
    b3_ans = extract_xml_answer(b3_rev)
    b3_correct = check_correctness(b3_ans, ground_truth)

    # Baseline 4: Basic Debate (Adversarial Critique -> Revise on Candidate 1)
    b4_rev, t = harness.generate_response_with_tokens(
        sys_revise, user_revise.format(problem=problem_text, solution=candidates[0], critique=adv_critiques[0]), model=model
    )
    total_tokens += t
    b4_ans = extract_xml_answer(b4_rev)
    b4_correct = check_correctness(b4_ans, ground_truth)

    # SCV Pipeline: Verifier looks at all K candidates and adversarial critiques
    debate_history = "".join([f"--- Candidate {i+1} ---\nSolution:\n{candidates[i]}\n\nCritique:\n{adv_critiques[i]}\n\n" for i in range(k)])
    ver_rat, t = harness.generate_response_with_tokens(
        sys_verify, user_verify.format(problem=problem_text, debate_history=debate_history), model=model, temperature=0.1
    )
    total_tokens += t
    scv_ans = extract_xml_answer(ver_rat)
    scv_correct = check_correctness(scv_ans, ground_truth)

    # Error Categorization
    error_category = "N/A"
    if not scv_correct:
        sys_err, user_err = get_error_categorization_prompt()
        err_cat, t = harness.generate_response_with_tokens(
            sys_err, 
            user_err.format(problem=problem_text, ground_truth=ground_truth, wrong_answer=scv_ans, rationale=ver_rat), 
            model=model, temperature=0.1
        )
        total_tokens += t
        error_category = err_cat.strip()

    end_time = time.time()

    return {
        "id": problem_data.get('id', 'unknown'),
        "problem": problem_text,
        "ground_truth": ground_truth,
        "metrics": {
            "time_seconds": round(end_time - start_time, 2),
            "total_tokens": total_tokens
        },
        "results": {
            "B1_SingleAgent": {"answer": b1_ans, "correct": b1_correct},
            "B2_SelfConsistency": {"answer": b2_ans, "correct": b2_correct},
            "B3_SelfRefine": {"answer": b3_ans, "correct": b3_correct},
            "B4_BasicDebate": {"answer": b4_ans, "correct": b4_correct},
            "SCV_Pipeline": {"answer": scv_ans, "correct": scv_correct, "rationale": ver_rat}
        },
        "analysis": {
            "improvement_from_B1": (not b1_correct) and scv_correct,
            "regression_from_B1": b1_correct and (not scv_correct),
            "error_category": error_category
        }
    }