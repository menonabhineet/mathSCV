# MathSCV: Multi-Agent Debate Tutor for Math

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![DeepSeek-V4-Flash](https://img.shields.io/badge/Model-DeepSeek--V4--Flash-orange)
![License](https://img.shields.io/badge/License-MIT-green)


## Overview
Large Language Models (LLMs) frequently struggle with math word problems due to brittle multi-step reasoning and cascading arithmetic slips. **MathSCV** explores whether a structured multi-agent pipeline composed of a **Solver**, an adversarial **Critic**, and a final **Verifier** (SCV) can correct reasoning flaws better than single-agent baselines.

Using `DeepSeek-V4-Flash`, we evaluated the SCV architecture against four established baselines across the **GSM8K**, **SVAMP**, and **ASDiv** datasets. 

### Key Findings
1. **The "Ceiling Effect":** The base model achieved >92% zero-shot accuracy across all single-agent baselines, starving the multi-agent pipeline of errors to fix.
2. **Degeneration-of-Thought:** Introducing adversarial Critics actively *degraded* performance on GSM8K and ASDiv. The Critic frequently hallucinated false constraints, bullying the Verifier into abandoning correct arithmetic.
3. **Self-Consistency is King:** Simple Self-Consistency (majority vote over K=3 paths) tied or outperformed the complex multi-agent debate across all datasets while consuming half the API tokens.

## Architecture

The project implements a token-efficient **Unified Evaluation Engine**. Rather than running 5 separate experiments, it generates $K=3$ initial Chain-of-Thought candidate solutions and recycles them across the baseline evaluations to drastically reduce API costs.

* **Baseline 1:** Single-Agent CoT
* **Baseline 2:** Self-Consistency Voting ($K=3$)
* **Baseline 3:** Self-Refine (Self-Reflection loop)
* **Baseline 4:** Basic Debate (Adversarial Critique $\rightarrow$ Revision)
* **SCV Pipeline:** $K=3$ Solvers $\rightarrow$ $K=3$ Adversarial Critics $\rightarrow$ 1 Verifier



## Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/menonabhineet/mathSCV.git](https://github.com/menonabhineet/mathSCV.git)
cd mathSCV
```

**2. Create a virtual environment and install dependencies**
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

**3. Configure API Keys**
```
DEEPSEEK_API_KEY=your_deepseek_key_here
```

## Running the Pipeline
**1. Run the Evaluation Engine**

To run the evaluation pipeline on the datasets, execute the Phase 4 script. You can modify the script to swap between load_svamp(), load_gsm8k(), or load_asdiv().
```
python run_phase4_evaluation.py
```
Results will automatically checkpoint and save to data/results/.

**2. Run the Error Analysis**

To extract qualitative insights, token efficiency metrics, and error categorizations (e.g., Reading Comprehension vs. Arithmetic Error), run:
```
python final_analysis.py
```

##  License
This project is licensed under the MIT License - see the LICENSE file for details.
