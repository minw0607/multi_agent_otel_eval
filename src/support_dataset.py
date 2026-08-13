"""
Customer-support desk corpus — loader and ground truth.

The corpus (`data/support/`) is a **synthetic measuring instrument**, not a
convenience sample. Article boundaries, ticket phrasing, and order dates are
deliberately constructed so that attribution metrics have known correct answers:

  multi-hop            → tickets answerable only by combining 2+ articles
                         (validates context_growth / history accumulation)
  duplicate-bait       → one article relevant to two sub-questions
                         (validates duplicate_retrievals, known expected count)
  obvious-article-wrong→ surface match gives the wrong answer
                         (validates verification value, plan_execution_divergence)
  escalation-required  → a mandatory escalation trigger is present
                         (validates outcome scoring of the escalate decision)

Because ground truth exists, this corpus supports **outcome-based evaluation**
(did the reply cite the right article? was escalation correct?) in addition to
the trajectory-based scoring used for Mind2Web. See docs/transparency_spec.md §7.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "support"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class KBArticle:
    id: str
    title: str
    tags: List[str]
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}"


@dataclass
class Policy:
    id: str
    topic: str
    title: str
    body: str


@dataclass
class Order:
    order_id: str
    customer: str
    items: List[Dict[str, Any]]
    subtotal: float
    status: str
    delivered_days_ago: Optional[int]
    destination_country: str
    payment_method: str
    care_plan: bool


@dataclass
class SupportTicket:
    """One labeled ticket. `ground_truth` powers outcome evaluation."""
    id: str
    difficulty: str
    subject: str
    body: str
    order_id: Optional[str]
    ground_truth: Dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors over ground truth ---------------------------
    @property
    def expected_articles(self) -> List[str]:
        return self.ground_truth.get("kb_articles", [])

    @property
    def distractor_articles(self) -> List[str]:
        return self.ground_truth.get("distractor_articles", [])

    @property
    def should_escalate(self) -> bool:
        return bool(self.ground_truth.get("escalate", False))

    @property
    def trap(self) -> Optional[str]:
        return self.ground_truth.get("trap")

    @property
    def key_facts(self) -> List[str]:
        return self.ground_truth.get("key_facts", [])

    def as_prompt(self) -> str:
        oid = f"\nOrder ID: {self.order_id}" if self.order_id else ""
        return f"Ticket {self.id}\nSubject: {self.subject}{oid}\n\n{self.body}"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class SupportCorpus:
    """
    Loads the corpus from disk (real file I/O) and provides real keyword
    retrieval. No embeddings — retrieval is deterministic and reproducible,
    which matters because retrieval behaviour is itself under measurement.
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.articles: Dict[str, KBArticle] = {}
        self.policies: Dict[str, Policy] = {}
        self.orders: Dict[str, Order] = {}
        self.tickets: List[SupportTicket] = []
        self._load()

    def _read(self, name: str) -> Dict:
        path = self.data_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Support corpus file missing: {path}. Expected the repo's "
                f"data/support/ directory to be present."
            )
        with open(path) as f:
            return json.load(f)

    def _load(self):
        for a in self._read("kb.json")["articles"]:
            self.articles[a["id"]] = KBArticle(**a)
        for p in self._read("policies.json")["policies"]:
            self.policies[p["id"]] = Policy(**p)
        for o in self._read("orders.json")["orders"]:
            self.orders[o["order_id"]] = Order(**o)
        for t in self._read("tickets.json")["tickets"]:
            self.tickets.append(SupportTicket(
                id=t["id"], difficulty=t["difficulty"], subject=t["subject"],
                body=t["body"], order_id=t.get("order_id"),
                ground_truth=t.get("ground_truth", {}),
            ))

    # -- retrieval ---------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2]

    def search(self, query: str, top_k: int = 3) -> List[KBArticle]:
        """
        Deterministic keyword scoring: title and tag matches weigh more than
        body matches. Ties break on article id so results are stable across
        runs — a requirement when retrieval itself is the object of study.
        """
        q = set(self._tokenize(query))
        if not q:
            return []
        scored = []
        for art in self.articles.values():
            title_hits = len(q & set(self._tokenize(art.title)))
            tag_hits   = len(q & set(self._tokenize(" ".join(art.tags))))
            body_hits  = len(q & set(self._tokenize(art.body)))
            score = 3 * title_hits + 4 * tag_hits + body_hits
            if score > 0:
                scored.append((score, art.id, art))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [a for _, _, a in scored[:top_k]]

    def policy_for(self, topic: str) -> Optional[Policy]:
        topic = (topic or "").strip().lower()
        for p in self.policies.values():
            if p.topic == topic or topic in p.topic:
                return p
        return None

    # -- summary -----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        from collections import Counter
        diff = Counter(t.difficulty for t in self.tickets)
        traps = Counter(t.trap for t in self.tickets if t.trap)
        return {
            "kb_articles": len(self.articles),
            "policies": len(self.policies),
            "orders": len(self.orders),
            "tickets": len(self.tickets),
            "difficulty": dict(diff),
            "traps": dict(traps),
            "escalation_required": sum(1 for t in self.tickets if t.should_escalate),
        }


def load_support_corpus(data_dir: Path = None) -> SupportCorpus:
    """Load the support corpus. Raises if the data directory is missing."""
    return SupportCorpus(data_dir)
