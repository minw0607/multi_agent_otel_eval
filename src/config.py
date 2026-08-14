"""
Configuration for the multi-agent OTel evaluation framework.

All secrets are read from environment variables (never hardcoded).
Copy .env.example → .env and fill in your provider details.

Provider auto-detection
-----------------------
Set OPENAI_API_VERSION to activate Azure OpenAI mode.
Leave it blank for OpenAI (direct) or any OpenAI-compatible endpoint.

  Provider             OPENAI_API_VERSION    LangChain class used
  ──────────────────── ─────────────────── ─────────────────────────
  OpenAI (direct)      (blank)              ChatOpenAI
  Azure OpenAI         2025-04-01-preview   AzureChatOpenAI
  Ollama (local)       (blank)              ChatOpenAI
  Groq                 (blank)              ChatOpenAI
  Together AI          (blank)              ChatOpenAI
  LM Studio            (blank)              ChatOpenAI

Call Config.create_llm(role="agent") or Config.create_llm(role="judge")
to get the correctly configured LangChain chat model.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # =========================================================================
    # LLM PROVIDER
    # =========================================================================
    API_KEY     = os.environ.get("OPENAI_API_KEY", "")
    BASE_URL    = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    API_VERSION = os.environ.get("OPENAI_API_VERSION", "")  # Set for Azure OpenAI

    # Azure API Management gateway (optional) — leave blank for standard endpoints
    APIM_HEADER_NAME      = os.environ.get("OPENAI_APIM_HEADER_NAME", "")
    APIM_SUBSCRIPTION_KEY = os.environ.get("OPENAI_APIM_SUBSCRIPTION_KEY", "")

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    AGENT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4o")    # Executes web tasks
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")    # Scores agent output

    # Per-specialist models for the multi-agent system (default to AGENT/JUDGE).
    # You can assign a cheaper model to the Planner and a stronger one to the
    # Navigator/Validator — e.g. PLANNER_MODEL=gpt-4o-mini, NAVIGATOR_MODEL=gpt-4o.
    SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", AGENT_MODEL)
    PLANNER_MODEL    = os.environ.get("PLANNER_MODEL",    AGENT_MODEL)
    NAVIGATOR_MODEL  = os.environ.get("NAVIGATOR_MODEL",  AGENT_MODEL)
    VALIDATOR_MODEL  = os.environ.get("VALIDATOR_MODEL",  JUDGE_MODEL)

    # Support-desk MAS defaults. The Planner runs a reasoning model on purpose:
    # it is the stage where extended reasoning is most defensible AND most
    # invisible, which makes it the transparency exhibit (spec §8).
    SUPPORT_PLANNER_MODEL   = os.environ.get("SUPPORT_PLANNER_MODEL",   "o4-mini-20250416-gs")
    SUPPORT_NAVIGATOR_MODEL = os.environ.get("SUPPORT_NAVIGATOR_MODEL", AGENT_MODEL)
    SUPPORT_VALIDATOR_MODEL = os.environ.get("SUPPORT_VALIDATOR_MODEL", JUDGE_MODEL)

    AGENT_TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.3"))
    JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
    AGENT_MAX_TOKENS  = int(os.environ.get("AGENT_MAX_TOKENS", "2000"))
    JUDGE_MAX_TOKENS  = int(os.environ.get("JUDGE_MAX_TOKENS", "1000"))

    # =========================================================================
    # EVALUATION SETTINGS
    # =========================================================================
    EVAL_PASS_THRESHOLD = float(os.environ.get("EVAL_PASS_THRESHOLD", "0.7"))
    RULE_WEIGHT         = float(os.environ.get("RULE_WEIGHT", "0.4"))
    LLM_WEIGHT          = float(os.environ.get("LLM_WEIGHT", "0.6"))

    # =========================================================================
    # COST TRACKING (per 1M tokens) — update for your model / contract pricing
    # =========================================================================
    # NOTE: gpt-5 family pricing is a placeholder — set it to your actual
    # contract rate. All others reflect public OpenAI list prices.
    COST_PER_1M_TOKENS = {
        "gpt-4o-mini-input":   0.15,  "gpt-4o-mini-output":   0.60,
        "gpt-4o-input":        2.50,  "gpt-4o-output":       10.00,
        "gpt-4.1-mini-input":  0.40,  "gpt-4.1-mini-output":  1.60,
        "gpt-4.1-input":       2.00,  "gpt-4.1-output":       8.00,
        "gpt-5-input":         2.50,  "gpt-5-output":        10.00,  # placeholder
        "gpt-4-input":        30.00,  "gpt-4-output":        60.00,
        "gpt-3.5-turbo-input": 0.50,  "gpt-3.5-turbo-output": 1.50,
        # o-series reasoning models. NOTE: reasoning tokens are billed as
        # OUTPUT tokens and are already included in `output_tokens`, so no
        # separate line is needed — but it means output cost on these models
        # is dominated by text the caller never sees.
        "o4-mini-input":       1.10,  "o4-mini-output":       4.40,
        "o3-mini-input":       1.10,  "o3-mini-output":       4.40,
        "o3-pro-input":       20.00,  "o3-pro-output":       80.00,
        "o3-input":            2.00,  "o3-output":            8.00,
        "o1-input":           15.00,  "o1-output":           60.00,
    }

    # Families ordered MOST-SPECIFIC first so e.g. "gpt-4-1-..." resolves to
    # gpt-4.1 ($2/$8) and never to legacy gpt-4 ($30/$60), and "o3-mini-..."
    # resolves to o3-mini rather than o3.
    _COST_FAMILY_ORDER = [
        "gpt-4o-mini", "gpt-4o",
        "gpt-4.1-mini", "gpt-4.1",
        "gpt-5",
        "gpt-4",
        "gpt-3.5-turbo",
        "o4-mini", "o3-mini", "o3-pro", "o3", "o1",
    ]

    # =========================================================================
    # DATASET
    # =========================================================================
    MIND2WEB_TARGET_TASKS = int(os.environ.get("MIND2WEB_TARGET_TASKS", "300"))
    QUICK_TEST_N          = int(os.environ.get("QUICK_TEST_N", "10"))

    # =========================================================================
    # OUTPUT DIRECTORIES
    # =========================================================================
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
    TRACE_DIR  = Path(os.environ.get("TRACE_DIR",  "./outputs/traces"))
    DATA_DIR   = Path(os.environ.get("DATA_DIR",   "./outputs/data"))

    @classmethod
    def setup_dirs(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LLM FACTORY
    # =========================================================================
    # Name patterns that *hint* a model does hidden reasoning. This is a
    # convenience heuristic only — see is_reasoning_model() for the caveat.
    # Override for other providers, e.g. REASONING_MODEL_PATTERNS="^o\\d|deepseek-r|qwq"
    REASONING_MODEL_PATTERNS = os.environ.get(
        "REASONING_MODEL_PATTERNS", r"^o\d|reasoning|thinking|-r1\b|deepseek-r")

    @staticmethod
    def display_model(model: str) -> str:
        """
        Short, human-readable model label — provider-agnostic.

        Strips deployment date stamps and environment suffixes without assuming
        any particular vendor or family, so published output never leaks an
        internal deployment id:

            gpt-5-4-20260305-gs        -> gpt-5-4
            o4-mini-20250416-gs        -> o4-mini
            claude-sonnet-4-5-20250929 -> claude-sonnet-4-5
            llama-3.1-70b-versatile    -> llama-3.1-70b-versatile  (nothing to strip)
        """
        import re
        if not model:
            return "—"
        s = str(model).strip()
        s = re.sub(r"[-_](gs|prod|dev|test|preview|latest)$", "", s, flags=re.I)
        s = re.sub(r"[-_]20\d{2}[-_]?\d{2}[-_]?\d{2}(?=$|[-_])", "", s)   # date stamp
        s = re.sub(r"[-_](gs|prod|dev|test|preview|latest)$", "", s, flags=re.I)
        return s.strip("-_") or str(model)

    @classmethod
    def is_reasoning_model(cls, model: str) -> bool:
        """
        Heuristic: does this model perform hidden internal reasoning?

        A "reasoning model" generates an internal chain of thought before its
        visible answer. Those thinking tokens are **billed as output tokens but
        never returned to the caller** — you pay for text you cannot read. They
        also usually reject `temperature` and require `max_completion_tokens`.

        This check is name-based and therefore only a hint. **The authoritative
        signal is behavioural**: if the API returns a non-zero
        `reasoning_tokens` in its usage payload, the model is doing hidden
        reasoning regardless of what it is called. Attribution always uses the
        reported number, never this guess — so a false negative here costs
        nothing but a label.

        Extend for other providers via REASONING_MODEL_PATTERNS.
        """
        import re
        return bool(re.search(cls.REASONING_MODEL_PATTERNS, (model or "").strip().lower()))

    @classmethod
    def create_llm(cls, role: str = "agent", model: str = None, **overrides):
        """
        Return a LangChain chat model for the given role ('agent' or 'judge').

        Pass `model` to override the model name (used by the multi-agent system
        to give each specialist its own model). Extra kwargs pass through.

        Auto-detection rules:
          OPENAI_API_VERSION set    →  AzureChatOpenAI, else ChatOpenAI
          o-series model detected   →  max_completion_tokens, no temperature
        """
        model      = model or (cls.AGENT_MODEL if role == "agent" else cls.JUDGE_MODEL)
        temp       = cls.AGENT_TEMPERATURE if role == "agent" else cls.JUDGE_TEMPERATURE
        max_tokens = cls.AGENT_MAX_TOKENS  if role == "agent" else cls.JUDGE_MAX_TOKENS

        params = {}
        if cls.is_reasoning_model(model):
            # o-series: temperature is rejected; the budget must cover hidden
            # reasoning tokens as well as visible output, hence the headroom.
            params["max_completion_tokens"] = max(max_tokens, 2000) * 2
        else:
            params["temperature"] = temp
            params["max_tokens"]  = max_tokens
        params.update(overrides)

        if cls.API_VERSION:
            from langchain_openai import AzureChatOpenAI
            extra_headers = {}
            if cls.APIM_HEADER_NAME and cls.APIM_SUBSCRIPTION_KEY:
                extra_headers[cls.APIM_HEADER_NAME] = cls.APIM_SUBSCRIPTION_KEY
            return AzureChatOpenAI(
                azure_deployment=model,
                azure_endpoint=cls.BASE_URL,
                api_version=cls.API_VERSION,
                api_key=cls.API_KEY,
                default_headers=extra_headers or None,
                **params,
            )
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                base_url=cls.BASE_URL,
                api_key=cls.API_KEY,
                **params,
            )

    @classmethod
    def get_cost_rate(cls, model: str, direction: str) -> float:
        """
        Return cost per 1M tokens for model + direction ('input'/'output').

        Deployment names use dashes (e.g. 'gpt-4-1-20250414-gs'), so both the
        model and the family keys are normalised (dots → dashes) before matching.
        Families are checked most-specific first, so 'gpt-4-1-...' matches
        gpt-4.1 rather than legacy gpt-4.
        """
        norm = model.lower().replace(".", "-")
        for family in cls._COST_FAMILY_ORDER:
            if family.replace(".", "-") in norm:
                key = f"{family}-{direction}"
                if key in cls.COST_PER_1M_TOKENS:
                    return cls.COST_PER_1M_TOKENS[key]
        return cls.COST_PER_1M_TOKENS.get(f"gpt-4o-{direction}", 2.50)
