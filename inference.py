#!/usr/bin/env python3
"""
Production Inference Script for Adaptive Financial Decision Environment.

Uses OpenAI-compatible API to run an intelligent agent through complex
financial reasoning scenarios with sophisticated reward shaping.
"""

import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from env.environment import ExpenseLeakDetectionEnv
from env.models import Action, ActionType, LeakType, UncertaintyType
from env.grader import ExpenseGrader


# Constants
MAX_RETRIES = 3
DEFAULT_TASK = "easy"


def create_client() -> OpenAI:
    """Create OpenAI client with environment variables."""
    api_key = os.getenv("HF_TOKEN")
    base_url = os.getenv("API_BASE_URL")
    
    if not api_key:
        raise ValueError("HF_TOKEN environment variable not set")
    if not base_url:
        raise ValueError("API_BASE_URL environment variable not set")
    
    return OpenAI(api_key=api_key, base_url=base_url)


def format_observation(obs) -> str:
    """Format observation as structured text for the LLM."""
    lines = [
        "═══════════════════════════════════════════",
        "           CURRENT STATE",
        "═══════════════════════════════════════════",
        f"Budget: ${obs.budget:.2f}",
        f"Step: {obs.step_count}/{obs.max_steps}",
        f"Leaks flagged: {obs.identified_leaks_count}",
        f"Suggestions made: {obs.suggestions_count}",
        f"Clarifications used: {obs.clarifications_count}",
        f"Patterns discovered: {obs.patterns_discovered_count}",
        f"Efficiency: {obs.efficiency_score:.1%}",
        "",
        "── USER PROFILE ──",
    ]
    
    profile = obs.user_profile
    lines.append(f"Income: {profile.get('income_level', 'unknown')} | Lifestyle: {profile.get('lifestyle', 'unknown')}")
    if profile.get('financial_goals'):
        lines.append(f"Goals: {', '.join(profile['financial_goals'])}")
    lines.append(f"Debt: {profile.get('debt_level', 'unknown')} | Dependents: {profile.get('dependents', 0)}")
    
    lines.append("")
    lines.append("── EXPENSES ──")
    
    by_category = {}
    for exp in obs.expenses:
        cat = exp.get('category', 'other')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(exp)
    
    for category, expenses in sorted(by_category.items()):
        lines.append(f"\n[{category.upper()}]")
        for exp in expenses:
            recurring = " (recurring)" if exp.get("is_recurring") else ""
            usage = ""
            if exp.get("usage_score") is not None:
                usage = f" [usage: {exp['usage_score']*100:.0f}%]"
            
            flags = []
            if exp.get("has_uncertainty"):
                flags.append("uncertain")
            if exp.get("requires_clarification"):
                flags.append("needs-info")
            if exp.get("conflicting_signals"):
                flags.append("conflicting")
            if exp.get("alternatives_exist"):
                flags.append("alternatives")
            
            flag_str = f" <{', '.join(flags)}>" if flags else ""
            
            lines.append(f"  [{exp['id']}] {exp['description']} - ${exp['amount']:.2f}{recurring}{usage}{flag_str}")
            
            if exp.get("related_expenses"):
                lines.append(f"       -> Related: {', '.join(exp['related_expenses'])}")
    
    if obs.last_action_result:
        lines.append("")
        lines.append("── LAST ACTION RESULT ──")
        lines.append(obs.last_action_result)
    
    if obs.last_analysis_insight:
        lines.append(f"\nINSIGHT: {obs.last_analysis_insight}")
    
    if obs.uncertainties_remaining > 0:
        lines.append(f"\n{obs.uncertainties_remaining} expense(s) have unresolved uncertainties")
    
    if obs.redundant_actions_count > 0:
        lines.append(f"Warning: {obs.redundant_actions_count} redundant action(s) taken")
    
    return "\n".join(lines)


def get_system_prompt() -> str:
    """Get system prompt for financial reasoning agent."""
    return """You are an expert financial analysis agent. Identify expense leaks with MAXIMUM ACCURACY and MINIMAL STEPS.

LEAK TYPES:
- unused_subscription: Services not being used
- overpriced_service: Paying too much for usage level
- duplicate_expense: Multiple services for same need
- hidden_fee: Avoidable bank/service fees
- lifestyle_creep: Convenience spending adding up
- impulse_purchase: Unplanned discretionary buys
- unnecessary_upgrade: Premium when basic suffices
- forgotten_recurring: Subscriptions user forgot about
- aggregated_small: Small daily costs that add up
- misaligned_priority: Spending contradicts stated goals

STRATEGY:
1. First: analyze_pattern to understand the landscape
2. Use ask_clarification ONLY when genuinely unclear (max 2)
3. Flag high-confidence leaks first (low usage + high cost)
4. Look for PATTERNS: same vendor, same category
5. Provide 1-2 quality suggestions with trade-offs
6. Stop when confident all major leaks identified

RULES:
- NEVER repeat the same action consecutively
- NEVER flag essential expenses (utilities, health insurance)
- NEVER ask more than 2 clarifications
- DO consider user's financial goals
- DO note trade-offs in suggestions

ACTION FORMAT (STRICT JSON):

{"type": "analyze_pattern"}
{"type": "flag_expense", "expense_id": "...", "reason": "...", "leak_type_guess": "...", "estimated_savings": 0.0, "confidence": 0.8}
{"type": "suggest_optimization", "suggestion": "...", "potential_savings": 0.0, "related_expense_ids": ["..."], "trade_offs": "..."}
{"type": "ask_clarification", "question": "...", "expense_id": "..."}
{"type": "investigate_pattern", "expense_ids_to_analyze": ["...", "..."]}
{"type": "compare_expenses", "expense_ids_to_compare": ["...", "..."]}
{"type": "stop_analysis"}

OUTPUT: Return ONLY a single valid JSON object. No explanation."""


def parse_action(response_text: str) -> Optional[Action]:
    """Parse action from LLM response with robust error handling."""
    try:
        text = response_text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1].strip()
        
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()
        
        data = json.loads(text)
        
        action_type_str = data.get("type") or data.get("action_type", "")
        
        type_mapping = {
            "analyze_pattern": ActionType.ANALYZE_PATTERN,
            "flag_expense": ActionType.FLAG_EXPENSE,
            "suggest_optimization": ActionType.SUGGEST_OPTIMIZATION,
            "ask_clarification": ActionType.ASK_CLARIFICATION,
            "prioritize_expense": ActionType.PRIORITIZE_EXPENSE,
            "stop_analysis": ActionType.STOP_ANALYSIS,
            "compare_expenses": ActionType.COMPARE_EXPENSES,
            "investigate_pattern": ActionType.INVESTIGATE_PATTERN,
        }
        
        action_type = type_mapping.get(action_type_str)
        if action_type is None:
            return None
        
        leak_type_guess = None
        if data.get("leak_type_guess"):
            try:
                leak_type_guess = LeakType(data["leak_type_guess"])
            except ValueError:
                pass
        
        target_uncertainty = None
        if data.get("target_uncertainty"):
            try:
                target_uncertainty = UncertaintyType(data["target_uncertainty"])
            except ValueError:
                pass
        
        return Action(
            action_type=action_type,
            expense_id=data.get("expense_id"),
            reason=data.get("reason"),
            leak_type_guess=leak_type_guess,
            estimated_savings=data.get("estimated_savings"),
            confidence=data.get("confidence"),
            suggestion=data.get("suggestion"),
            potential_savings=data.get("potential_savings"),
            related_expense_ids=data.get("related_expense_ids"),
            trade_offs=data.get("trade_offs"),
            question=data.get("question"),
            target_uncertainty=target_uncertainty,
            priority_ranking=data.get("priority_ranking"),
            pattern_type=data.get("pattern_type"),
            expense_ids_to_analyze=data.get("expense_ids_to_analyze"),
            expense_ids_to_compare=data.get("expense_ids_to_compare")
        )
        
    except (json.JSONDecodeError, Exception):
        return None


def format_action_str(action: Action) -> str:
    """Format action as clean string for logging."""
    action_str = action.action_type.value
    
    if action.expense_id:
        action_str += f"({action.expense_id})"
    elif action.expense_ids_to_analyze:
        action_str += f"({','.join(action.expense_ids_to_analyze[:3])})"
    elif action.expense_ids_to_compare:
        action_str += f"({','.join(action.expense_ids_to_compare[:3])})"
    elif action.action_type == ActionType.SUGGEST_OPTIMIZATION and action.suggestion:
        preview = action.suggestion[:20].replace('"', "'")
        action_str += f"({preview}...)"
    
    return action_str


def run_episode(client: OpenAI, model_name: str, env: ExpenseLeakDetectionEnv, task: str) -> Dict[str, Any]:
    """Run a complete episode with the agent."""
    obs = env.reset()
    
    print(f"[START] task={task} env=adaptive-financial-decision model={model_name}")
    
    messages = [{"role": "system", "content": get_system_prompt()}]
    
    step = 0
    rewards = []
    
    while not obs.done:
        step += 1
        
        obs_text = format_observation(obs)
        messages.append({"role": "user", "content": obs_text})
        
        action = None
        error_msg = "null"
        
        for retry in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500
                )
                
                response_text = response.choices[0].message.content
                action = parse_action(response_text)
                
                if action:
                    messages.append({"role": "assistant", "content": response_text})
                    break
                    
            except Exception as e:
                error_msg = str(e)[:50].replace('"', "'")
        
        if action is None:
            print(f"[STEP] step={step} action=parse_error reward=0.00 done=true error=\"{error_msg}\"")
            break
        
        obs, reward, done, info = env.step(action)
        rewards.append(reward.value)
        
        action_str = format_action_str(action)
        print(f"[STEP] step={step} action={action_str} reward={reward.value:.2f} done={str(done).lower()} error=null")
        
        if done:
            break
    
    grader = ExpenseGrader(env)
    grade_result = grader.grade()
    
    success = grade_result["score"] >= 0.5
    
    rewards_str = "[" + ",".join(f"{r:.2f}" for r in rewards) + "]"
    print(f"[END] success={str(success).lower()} steps={step} score={grade_result['score']:.2f} rewards={rewards_str}")
    
    return {
        "success": success,
        "steps": step,
        "score": grade_result["score"],
        "rewards": rewards,
        "total_reward": sum(rewards),
        "grade_breakdown": grade_result["breakdown"],
        "feedback": grade_result["feedback"]
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Adaptive Financial Decision Agent")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, choices=["easy", "medium", "hard"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    api_base = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")
    
    if not api_base:
        print("[ERROR] API_BASE_URL not set")
        sys.exit(1)
    
    if not hf_token:
        print("[ERROR] HF_TOKEN not set")
        sys.exit(1)
    
    if not model_name:
        print("[ERROR] MODEL_NAME not set")
        sys.exit(1)
    
    try:
        client = create_client()
        env = ExpenseLeakDetectionEnv(task_difficulty=args.task)
        
        result = run_episode(client=client, model_name=model_name, env=env, task=args.task)
        
        if args.verbose:
            print("\n" + "=" * 50)
            print("DETAILED RESULTS")
            print("=" * 50)
            print(json.dumps(result, indent=2))
        
        return 0 if result["success"] else 1
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
