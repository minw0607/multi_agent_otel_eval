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
    }

    # Families ordered MOST-SPECIFIC first so e.g. "gpt-4-1-..." resolves to
    # gpt-4.1 ($2/$8) and never to legacy gpt-4 ($30/$60).
    _COST_FAMILY_ORDER = [
        "gpt-4o-mini", "gpt-4o",
        "gpt-4.1-mini", "gpt-4.1",
        "gpt-5",
        "gpt-4",
        "gpt-3.5-turbo",
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
    @classmethod
    def create_llm(cls, role: str = "agent", model: str = None):
        """
        Return a LangChain chat model for the given role ('agent' or 'judge').

        Pass `model` to override the model name (used by the multi-agent system
        to give each specialist its own model).

        Auto-detection rule:
          OPENAI_API_VERSION is set  →  AzureChatOpenAI
          OPENAI_API_VERSION is blank →  ChatOpenAI (OpenAI / Ollama / Groq / etc.)
        """
        model      = model or (cls.AGENT_MODEL if role == "agent" else cls.JUDGE_MODEL)
        temp       = cls.AGENT_TEMPERATURE if role == "agent" else cls.JUDGE_TEMPERATURE
        max_tokens = cls.AGENT_MAX_TOKENS  if role == "agent" else cls.JUDGE_MAX_TOKENS

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
                temperature=temp,
                max_tokens=max_tokens,
                default_headers=extra_headers or None,
            )
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                base_url=cls.BASE_URL,
                api_key=cls.API_KEY,
                temperature=temp,
                max_tokens=max_tokens,
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
