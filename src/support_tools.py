"""
Real tools for the customer-support desk, under an enforced sandbox contract.

Unlike `src/tools.py` (Mind2Web: WRITE operations are always mocked), these
tools perform **real I/O**:

  READ   search_kb / read_ticket / read_order / read_policy  → real file reads
         web_search                                          → real Tavily (optional)
  WRITE  draft_response / escalate                           → real file writes

The blast radius is contained by a contract enforced in code, not convention
(docs/transparency_spec.md §6):

  1. All writes confined to a `workspace/` root, with a path-traversal guard.
  2. No network writes. External access is read-only.
  3. Explicit per-run tool allowlist.
  4. Safety validation runs on INPUTS to write-tools, not only on outputs.
  5. `workspace/` is gitignored and reset per run.

Tool outputs are deliberately returned as substantial text, because tool output
becomes input context on the next LLM call — the quantity this framework is
trying to attribute.
"""

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import tool

from .support_dataset import SupportCorpus, load_support_corpus

# ---------------------------------------------------------------------------
# Sandbox root
# ---------------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("SUPPORT_WORKSPACE", "./workspace")).resolve()

_corpus: Optional[SupportCorpus] = None


def init_support_tools(corpus: SupportCorpus = None, workspace: Path = None,
                       reset: bool = True) -> Path:
    """
    Bind the corpus and prepare the sandbox. Must be called before using tools.
    Returns the resolved workspace root.
    """
    global _corpus, WORKSPACE
    _corpus = corpus or load_support_corpus()
    if workspace is not None:
        WORKSPACE = Path(workspace).resolve()
    if reset and WORKSPACE.exists():
        import shutil
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "drafts").mkdir(parents=True, exist_ok=True)
    return WORKSPACE


def _require_corpus() -> SupportCorpus:
    if _corpus is None:
        raise RuntimeError("Support tools not initialised — call init_support_tools() first.")
    return _corpus


def _safe_path(*parts: str) -> Path:
    """
    Resolve a path inside the workspace, refusing anything that escapes it.
    This is the sandbox guard (contract rule 1) — traversal via '..', absolute
    paths, or symlinks is rejected rather than sanitised silently.
    """
    candidate = (WORKSPACE.joinpath(*parts)).resolve()
    if not str(candidate).startswith(str(WORKSPACE) + os.sep) and candidate != WORKSPACE:
        raise PermissionError(
            f"Sandbox violation: refusing to write outside {WORKSPACE} (got {candidate})"
        )
    return candidate


def _slug(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return (s[:maxlen] or "unnamed")


def _fingerprint(value) -> str:
    """Stable 12-char fingerprint of normalised content (spec T2)."""
    norm = json.dumps(value, sort_keys=True, default=str) if not isinstance(value, str) else value.strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# READ tools — real, local
# ---------------------------------------------------------------------------

@tool
def search_kb(query: str) -> str:
    """Search the customer-facing knowledge base for relevant help articles.

    Args:
        query: What to look for, e.g. 'return window' or 'damaged on arrival'
    """
    c = _require_corpus()
    hits = c.search(query, top_k=3)
    if not hits:
        return f"No knowledge base articles matched '{query}'. Try different terms."
    out = [f"Knowledge base results for '{query}' ({len(hits)} articles):", ""]
    for a in hits:
        out += [f"--- [{a.id}] {a.title} ---", a.body, ""]
    return "\n".join(out)


@tool
def read_policy(topic: str) -> str:
    """Read the internal, authoritative policy for a topic. Use this to verify a draft.

    Args:
        topic: One of refunds, warranty, escalation, international, communication
    """
    c = _require_corpus()
    p = c.policy_for(topic)
    if not p:
        avail = ", ".join(sorted({x.topic for x in c.policies.values()}))
        return f"No policy found for '{topic}'. Available topics: {avail}"
    return f"--- POLICY [{p.id}] {p.title} ---\n{p.body}"


@tool
def read_order(order_id: str) -> str:
    """Look up an order in the system of record. Required before stating any order fact.

    Args:
        order_id: Order identifier, e.g. ORD-48210
    """
    c = _require_corpus()
    o = c.orders.get(order_id.strip().upper())
    if not o:
        return f"No order found with id '{order_id}'."
    items = "; ".join(f"{i['qty']}x {i['name']} (${i['price']:.2f})" for i in o.items)
    delivered = (f"{o.delivered_days_ago} days ago"
                 if o.delivered_days_ago is not None else "not yet delivered")
    return (f"--- ORDER {o.order_id} ---\n"
            f"Customer: {o.customer}\nItems: {items}\nSubtotal: ${o.subtotal:.2f}\n"
            f"Status: {o.status}\nDelivered: {delivered}\n"
            f"Destination country: {o.destination_country}\n"
            f"Payment method: {o.payment_method}\n"
            f"NorthPeak Care plan: {'yes' if o.care_plan else 'no'}")


@tool
def read_ticket(ticket_id: str) -> str:
    """Read the full text of a support ticket.

    Args:
        ticket_id: Ticket identifier, e.g. T-1006
    """
    c = _require_corpus()
    for t in c.tickets:
        if t.id.upper() == ticket_id.strip().upper():
            return t.as_prompt()
    return f"No ticket found with id '{ticket_id}'."


@tool
def web_search(query: str) -> str:
    """Search the public web. Read-only; used only when internal sources are insufficient.

    Args:
        query: Search query
    """
    if "TAVILY_API_KEY" in os.environ:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            results = TavilySearchResults(max_results=3).invoke({"query": query})
            return "\n\n".join(
                f"{r.get('title','')}\n{r.get('content','')[:250]}\n{r.get('url','')}"
                for r in results)
        except Exception as e:
            return f"Web search unavailable ({type(e).__name__}). Rely on internal sources."
    return ("Web search is not configured (no TAVILY_API_KEY). "
            "Answer using the knowledge base and policies instead.")


# ---------------------------------------------------------------------------
# WRITE tools — real, but confined to the workspace
# ---------------------------------------------------------------------------

@tool
def draft_response(ticket_id: str, body: str, cited_articles: str = "") -> str:
    """Save a draft reply to the customer. Writes a real file in the local workspace.

    Args:
        ticket_id: Ticket the reply answers, e.g. T-1006
        body: Full text of the reply to the customer
        cited_articles: Comma-separated KB article ids the reply relies on
    """
    from .safety import SafetyValidator
    # Contract rule 4: validate the INPUT to a write tool, not just model output.
    safety = SafetyValidator.validate_all(body, ticket_id)
    if not SafetyValidator.is_safe(safety):
        return (f"Draft REJECTED by safety validation for {ticket_id}: "
                f"{safety['injection']['violations'] or 'PII or harmful content detected'}. "
                "Rewrite without sensitive or unsafe content.")

    path = _safe_path("drafts", f"{_slug(ticket_id)}.md")
    cited = [c.strip() for c in cited_articles.split(",") if c.strip()]
    payload = (f"# Draft reply — {ticket_id}\n\n"
               f"_Generated {datetime.now():%Y-%m-%d %H:%M}_\n\n"
               f"**Cited articles:** {', '.join(cited) if cited else '(none)'}\n\n"
               f"---\n\n{body}\n")
    path.write_text(payload)
    return (f"Draft saved to {path.relative_to(WORKSPACE.parent)} "
            f"({len(body)} chars, {len(cited)} article(s) cited). "
            f"fingerprint={_fingerprint(body)}")


@tool
def escalate(ticket_id: str, reason: str) -> str:
    """Escalate a ticket to a supervisor. Appends a real record in the local workspace.

    Args:
        ticket_id: Ticket to escalate, e.g. T-1013
        reason: The policy trigger that requires escalation
    """
    path = _safe_path("escalations.jsonl")
    record = {"ticket_id": ticket_id, "reason": reason,
              "timestamp": datetime.now().isoformat()}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return (f"Ticket {ticket_id} escalated to a supervisor. Reason recorded: {reason}. "
            f"A supervisor responds within one business day.")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

READ_TOOLS  = [search_kb, read_policy, read_order, read_ticket, web_search]
WRITE_TOOLS = [draft_response, escalate]
ALL_SUPPORT_TOOLS = READ_TOOLS + WRITE_TOOLS

TOOL_EFFECT_CLASS = {
    "search_kb": "read/local", "read_policy": "read/local",
    "read_order": "read/local", "read_ticket": "read/local",
    "web_search": "read/external",
    "draft_response": "write/local", "escalate": "write/local",
}


def get_support_tools(allowlist: List[str] = None):
    """
    Return the tool objects permitted for this run (contract rule 3).
    Passing an allowlist is how a caller narrows the blast radius further.
    """
    if allowlist is None:
        return list(ALL_SUPPORT_TOOLS)
    names = set(allowlist)
    unknown = names - {t.name for t in ALL_SUPPORT_TOOLS}
    if unknown:
        raise ValueError(f"Unknown tools in allowlist: {sorted(unknown)}")
    return [t for t in ALL_SUPPORT_TOOLS if t.name in names]


def workspace_manifest() -> Dict:
    """Summarise what the agents actually wrote — evidence that effects were real."""
    drafts = sorted((WORKSPACE / "drafts").glob("*.md")) if (WORKSPACE / "drafts").exists() else []
    esc_path = WORKSPACE / "escalations.jsonl"
    escalations = []
    if esc_path.exists():
        escalations = [json.loads(l) for l in esc_path.read_text().splitlines() if l.strip()]
    return {
        "workspace": str(WORKSPACE),
        "drafts": [p.name for p in drafts],
        "draft_count": len(drafts),
        "escalations": escalations,
        "escalation_count": len(escalations),
    }
