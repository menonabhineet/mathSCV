def get_single_agent_cot_prompt():
    system_prompt = (
        "You are an expert mathematical reasoning assistant. "
        "Your task is to solve the given math word problem."
    )
    
    # We explicitly instruct the model on HOW to format the output
    user_template = (
        "Solve the following math problem step-by-step. "
        "Show all your reasoning. "
        "Once you have your final answer, you MUST enclose the final numerical value "
        "inside <answer> and </answer> tags. "
        "For example: <answer>42</answer>.\n\n"
        "Problem: {problem}"
    )
    return system_prompt, user_template

def get_critique_prompt(is_self_reflection=False):
    """Generates prompts for the critique phase. Changes tone based on the baseline type."""
    if is_self_reflection:
        system_prompt = (
            "You are a meticulous mathematician reviewing your own previous work."
        )
        user_template = (
            "Problem: {problem}\n\n"
            "Your previous solution:\n{solution}\n\n"
            "Review your solution step-by-step. Identify any logical gaps, misinterpretations "
            "of the problem, or arithmetic errors. Do not provide a new final answer yet, "
            "just output your critique and analysis."
        )
    else:
        system_prompt = (
            "You are a strict, adversarial math critic. Your goal is to find flaws in proposed solutions."
        )
        user_template = (
            "Problem: {problem}\n\n"
            "Proposed solution:\n{solution}\n\n"
            "Critique the proposed solution. Actively hunt for arithmetic slips, brittle reasoning, "
            "or failure to read the prompt correctly. Be direct and point out specific errors. "
            "If the solution is absolutely flawless, concede that it is correct. Do not provide a new final answer."
        )
    return system_prompt, user_template

def get_revision_prompt():
    """Generates the prompt for the Solver to fix its answer based on a critique."""
    system_prompt = (
        "You are an expert mathematical assistant tasked with fixing flawed solutions based on feedback."
    )
    user_template = (
        "Problem: {problem}\n\n"
        "Original Solution:\n{solution}\n\n"
        "Critique:\n{critique}\n\n"
        "Based on the critique, provide a completely revised step-by-step solution. "
        "Even if the critique says the original was correct, you must output the full reasoning again. "
        "You MUST enclose your final numerical value inside <answer> and </answer> tags. "
        "For example: <answer>42</answer>."
    )
    return system_prompt, user_template

def get_verifier_prompt():
    """Generates the prompt for the final Verifier agent."""
    system_prompt = (
        "You are the final Verifier in a mathematical reasoning system. "
        "Your job is to review a math problem, look at several proposed solutions and their critiques, "
        "recompute the key quantities yourself, and determine the single correct answer."
    )
    
    # We pass the K candidates and critiques as a formatted string
    user_template = (
        "Problem: {problem}\n\n"
        "Here are the proposed solutions and their critiques:\n"
        "{debate_history}\n\n"
        "Task:\n"
        "1. Recompute the math step-by-step to verify the logic.\n"
        "2. Select the best solution, or provide your own if all are flawed.\n"
        "3. You MUST enclose the final numerical value of your chosen answer inside <answer> and </answer> tags. "
        "For example: <answer>42</answer>."
    )
    return system_prompt, user_template

def get_error_categorization_prompt():
    system_prompt = (
        "You are an AI auditor. Your job is to classify mathematical reasoning failures into "
        "one of four strict categories: [Arithmetic Error], [Reading Comprehension], "
        "[Logic/Reasoning Flaw], or [Other]."
    )
    user_template = (
        "Problem: {problem}\n"
        "Ground Truth Answer: {ground_truth}\n"
        "Model's Incorrect Answer: {wrong_answer}\n"
        "Model's Rationale:\n{rationale}\n\n"
        "Analyze the rationale and output ONLY the category name in brackets, e.g., [Arithmetic Error]."
    )
    return system_prompt, user_template