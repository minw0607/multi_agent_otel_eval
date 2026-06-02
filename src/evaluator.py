"""
Hybrid evaluation: rule-based + LLM-as-judge.

HybridEvaluator      → scores agent output against a Mind2Web task
ToolCorrectnessEval  → precision / recall / F1 for tool selection
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage


@dataclass
class EvalResult:
    task_id: int
    passed: bool
    total_score: float
    rule_score: float
    llm_score: float
    llm_reason: str
    length_ok: bool
    has_numbers: bool
    goal_alignment: float
    structured_format: bool
    action_overlap: float
    safety_passed: bool
    safety_violations: List[str] = field(default_factory=list)


@dataclass
class ToolCorrectnessResult:
    precision: float
    recall: float
    f1: float
    exact_match: bool
    order_accuracy: float
    predicted_tools: List[str]
    reference_tools: List[str]
    correct_tools: List[str]
    missing_tools: List[str]
    extra_tools: List[str]


# ---------------------------------------------------------------------------
# Tool equivalence mapping
#
# Only genuinely interchangeable tools are treated as equivalent. Search tools
# (site_search / web_search) substitute for each other, but navigation,
# filtering, and transactions must be matched on their own merits — otherwise
# recall is trivially perfect and tool_f1 becomes uninformative.
# ---------------------------------------------------------------------------

TOOL_EQUIVALENTS: Dict[str, List[str]] = {
    "site_search":      ["site_search", "web_search"],
    "web_search":       ["web_search", "site_search"],
    "site_navigation":  ["site_navigation"],
    "filter_content":   ["filter_content"],
    "make_purchase":    ["make_purchase"],
    "book_reservation": ["book_reservation"],
}


def _tools_equivalent(t1: str, t2: str) -> bool:
    if t1 == t2:
        return True
    return t2 in TOOL_EQUIVALENTS.get(t1, []) or t1 in TOOL_EQUIVALENTS.get(t2, [])


def _action_to_tool(action: str) -> str:
    """
    Map a single Mind2Web action string → the tool that best represents it.

    Action format: '[element]  label -> ACTION: value'
    The mapping uses both the action verb and label/element keywords so that
    the reference tool sequence reflects real task diversity (search, filter,
    navigate, purchase) rather than collapsing everything to navigation.
    """
    a   = action.upper()
    low = action.lower()

    # Transaction intents
    if any(k in low for k in ("add to cart", "buy", "checkout", "place order", "purchase")):
        return "make_purchase"
    if any(k in low for k in ("book", "reserve", "reservation")):
        return "book_reservation"
    # Sorting / filtering (dropdown SELECT or filter/sort labels)
    if "SELECT" in a or any(k in low for k in ("sort", "filter", "refine")):
        return "filter_content"
    # Typing into a search field
    if "TYPE" in a and any(k in low for k in ("search", "find", "query", "keyword")):
        return "site_search"
    # Default: clicks, typing into form fields, scrolls
    return "site_navigation"


def _actions_to_tools(action_reprs: List[str]) -> List[str]:
    """Map a Mind2Web action sequence → expected tools (consecutive-deduplicated)."""
    tools, last = [], None
    for action in action_reprs:
        tool = _action_to_tool(action)
        if tool != last:
            tools.append(tool)
            last = tool
    return tools


def _parse_agent_tools(agent_output: str, trace_tool_calls: List[Dict] = None) -> List[str]:
    """Extract tool names from trace (accurate) or text (fallback)."""
    if trace_tool_calls:
        seen, tools = set(), []
        for tc in trace_tool_calls:
            name = tc.get("tool", "")
            if name and name not in seen:
                seen.add(name)
                tools.append(name)
        return tools

    all_tool_names = [
        "web_search", "site_search", "site_navigation", "filter_content",
        "get_page_info", "check_availability", "get_price_info",
        "book_reservation", "make_phone_call", "submit_form",
        "make_purchase", "budget_calculator",
    ]
    seen, tools = set(), []
    for name in all_tool_names:
        if re.search(rf"\b{name}\s*\(", agent_output, re.IGNORECASE) and name not in seen:
            seen.add(name)
            tools.append(name)
    return tools


class HybridEvaluator:
    """
    Scores agent output with a weighted combination of rule-based and LLM-as-judge scores.

    rule_score  → length, specificity, goal alignment, action verbs, action overlap
    llm_score   → holistic quality rated by judge LLM (0.0–1.0)
    total_score = rule_weight * rule_score + llm_weight * llm_score
    """

    ACTION_VERBS = ["CLICK", "TYPE", "SELECT", "SCROLL", "OPEN", "SUBMIT"]

    def __init__(self, judge_llm, pass_threshold: float = 0.7,
                 rule_weight: float = 0.4, llm_weight: float = 0.6):
        self.judge_llm      = judge_llm
        self.pass_threshold = pass_threshold
        self.rule_weight    = rule_weight
        self.llm_weight     = llm_weight

    def _rule_score(self, plan: str, task: str, reference_actions: List[str]) -> Dict:
        length_ok        = len(plan.split()) >= 40
        has_numbers      = bool(re.search(r"\d", plan))
        task_words       = set(re.findall(r"\w+", task.lower()))
        plan_words       = set(re.findall(r"\w+", plan.lower()))
        goal_alignment   = len(task_words & plan_words) / max(len(task_words), 1)
        structured_fmt   = any(v in plan.upper() for v in self.ACTION_VERBS)
        ref_text         = " ".join(reference_actions).upper()
        plan_verbs       = {v for v in self.ACTION_VERBS if v in plan.upper()}
        ref_verbs        = {v for v in self.ACTION_VERBS if v in ref_text}
        action_overlap   = len(plan_verbs & ref_verbs) / max(len(ref_verbs), 1) if ref_verbs else 0.5

        score = (0.2 * float(length_ok) + 0.1 * float(has_numbers) +
                 0.3 * goal_alignment + 0.2 * float(structured_fmt) + 0.2 * action_overlap)
        return {"rule_score": score, "length_ok": length_ok, "has_numbers": has_numbers,
                "goal_alignment": goal_alignment, "structured_format": structured_fmt,
                "action_overlap": action_overlap}

    def _llm_score(self, plan: str, task: str, reference: str) -> Tuple[float, str]:
        prompt = (
            f"Evaluate this action plan for a web navigation task.\n\n"
            f"Task: {task}\nReference: {reference}\nPlan: {plan}\n\n"
            f"Rate 0.0–1.0. Respond ONLY with JSON: "
            f'{{\"score\": <float>, \"reason\": \"<brief>\"}}'
        )
        try:
            resp = self.judge_llm.invoke([HumanMessage(content=prompt)])
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.content.strip(), flags=re.MULTILINE)
            data    = json.loads(cleaned)
            return max(0.0, min(1.0, float(data.get("score", 0.5)))), data.get("reason", "")
        except Exception as e:
            return 0.5, f"Evaluation error: {e}"

    def evaluate(self, agent_output: str, task, safety_result: Dict = None) -> EvalResult:
        plan_m = re.search(r"Final Plan:(.*?)(?:Final Answer:|$)", agent_output, re.DOTALL)
        plan   = plan_m.group(1).strip() if plan_m else agent_output[:500]

        rule   = self._rule_score(plan, task.confirmed_task, task.action_reprs)
        ref    = "\n".join(task.action_reprs[:5])
        llm_score, llm_reason = self._llm_score(plan, task.confirmed_task, ref)

        safe_passed   = True
        safe_violations = []
        if safety_result:
            safe_passed = (not safety_result["pii"]["pii_detected"]
                           and not safety_result["harmful"]["harmful_detected"]
                           and safety_result["injection"]["safe"])
            safe_violations = safety_result.get("injection", {}).get("violations", [])

        total = self.rule_weight * rule["rule_score"] + self.llm_weight * llm_score
        return EvalResult(
            task_id=task.idx, passed=(total >= self.pass_threshold and safe_passed),
            total_score=total, rule_score=rule["rule_score"], llm_score=llm_score,
            llm_reason=llm_reason, length_ok=rule["length_ok"], has_numbers=rule["has_numbers"],
            goal_alignment=rule["goal_alignment"], structured_format=rule["structured_format"],
            action_overlap=rule["action_overlap"], safety_passed=safe_passed,
            safety_violations=safe_violations,
        )


class ToolCorrectnessEval:
    """Evaluate tool selection quality against Mind2Web reference actions."""

    @staticmethod
    def evaluate(agent_output: str, reference_actions: List[str],
                 trace_tool_calls: List[Dict] = None) -> ToolCorrectnessResult:
        predicted = _parse_agent_tools(agent_output, trace_tool_calls)
        reference = _actions_to_tools(reference_actions)

        correct, missing, extra = [], [], list(predicted)
        for ref in reference:
            found = False
            for pred in predicted:
                if _tools_equivalent(ref, pred) and pred not in correct:
                    correct.append(pred)
                    if pred in extra:
                        extra.remove(pred)
                    found = True
                    break
            if not found:
                missing.append(ref)

        precision = len(correct) / len(predicted) if predicted else 0.0
        recall    = len(correct) / len(reference) if reference else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        exact     = not missing and not extra

        # Order accuracy via LCS
        def lcs(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if _tools_equivalent(a[i-1], b[j-1]):
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        order_acc = lcs(predicted, reference) / max(len(reference), 1)

        return ToolCorrectnessResult(
            precision=precision, recall=recall, f1=f1, exact_match=exact,
            order_accuracy=order_acc, predicted_tools=predicted, reference_tools=reference,
            correct_tools=correct, missing_tools=missing, extra_tools=extra,
        )
