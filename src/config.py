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
    # COST TRACKING (per 1M tokens) — update for your model
    # =========================================================================
    COST_PER_1M_TOKENS = {
        "gpt-4o-input":   2.50,
        "gpt-4o-output":  10.00,
        "gpt-4o-mini-input":  0.15,
        "gpt-4o-mini-output": 0.60,
        "gpt-4-input":    30.00,
        "gpt-4-output":   60.00,
        "gpt-3.5-turbo-input":  0.50,
        "gpt-3.5-turbo-output": 1.50,
    }

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
    def create_llm(cls, role: str = "agent"):
        """
        Return a LangChain chat model for the given role ('agent' or 'judge').

        Auto-detection rule:
          OPENAI_API_VERSION is set  →  AzureChatOpenAI
          OPENAI_API_VERSION is blank →  ChatOpenAI (OpenAI / Ollama / Groq / etc.)
        """
        model      = cls.AGENT_MODEL      if role == "agent" else cls.JUDGE_MODEL
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
        """Return cost per 1M tokens for model + direction ('input'/'output')."""
        for prefix in ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]:
            if prefix in model:
                key = f"{prefix}-{direction}"
                if key in cls.COST_PER_1M_TOKENS:
                    return cls.COST_PER_1M_TOKENS[key]
        return cls.COST_PER_1M_TOKENS.get(f"gpt-4o-{direction}", 2.50)
