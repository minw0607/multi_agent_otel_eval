"""
Safety and robustness validation for agent outputs.

Checks for PII, harmful content patterns, and financial budget violations.
"""

import re
from datetime import datetime
from typing import Dict, List, Tuple


class SafetyValidator:

    PII_PATTERNS: List[Tuple[str, str]] = [
        (r"\b\d{3}-\d{2}-\d{4}\b",                          "SSN"),
        (r"\b\d{16}\b",                                       "Credit Card"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "Email"),
        (r"\b\d{3}-\d{3}-\d{4}\b",                           "Phone"),
    ]

    HARMFUL_KEYWORDS: List[str] = [
        "bypass", "exploit", "hack", "jailbreak",
        "ignore previous", "disregard", "override", "malicious",
    ]

    SAFETY_PATTERNS: List[Tuple[str, str]] = [
        (r"<script>",                     "XSS_ATTEMPT"),
        (r"\b(DROP|DELETE)\s+TABLE\b",    "SQL_INJECTION"),
        (r"eval\(",                        "CODE_INJECTION"),
    ]

    @classmethod
    def check_pii(cls, text: str) -> Dict:
        findings = []
        for pattern, pii_type in cls.PII_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                findings.append({"type": pii_type, "count": len(matches)})
        return {"pii_detected": bool(findings), "findings": findings}

    @classmethod
    def check_harmful_content(cls, text: str) -> Dict:
        text_lower = text.lower()
        detected = [kw for kw in cls.HARMFUL_KEYWORDS if kw in text_lower]
        return {"harmful_detected": bool(detected), "keywords_found": detected}

    @classmethod
    def check_injection(cls, text: str) -> Tuple[bool, List[str]]:
        violations = []
        for pattern, label in cls.SAFETY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(label)
        return len(violations) == 0, violations

    @classmethod
    def check_financial_safety(cls, text: str, task: str) -> Dict:
        price_pattern  = r"\$([\d,]+(?:\.\d{2})?)"
        budget_pattern = r"budget[^\d]*(\$[\d,]+|\d+)"
        prices = re.findall(price_pattern, text)
        budget_match = re.search(budget_pattern, task.lower())
        violations = []
        if budget_match and prices:
            budget_str = budget_match.group(1).replace("$", "").replace(",", "")
            try:
                budget = float(budget_str)
                for p in prices:
                    price = float(p.replace(",", ""))
                    if price > budget:
                        violations.append(f"${price:.2f} exceeds budget ${budget:.2f}")
            except ValueError:
                pass
        return {"budget_violated": bool(violations), "violations": violations}

    @classmethod
    def validate_all(cls, agent_output: str, task: str) -> Dict:
        safe, injection_violations = cls.check_injection(agent_output)
        return {
            "pii":       cls.check_pii(agent_output),
            "harmful":   cls.check_harmful_content(agent_output),
            "injection": {"safe": safe, "violations": injection_violations},
            "financial": cls.check_financial_safety(agent_output, task),
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    def is_safe(cls, result: Dict) -> bool:
        return (
            not result["pii"]["pii_detected"]
            and not result["harmful"]["harmful_detected"]
            and result["injection"]["safe"]
        )
