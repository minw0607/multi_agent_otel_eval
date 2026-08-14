# Evaluation Guide — metrics, tool environments, and what a score means

How this framework decides whether an agent did well, and — just as important — what
each score can and cannot establish.

The framework supports **two evaluation regimes**, and being explicit about which one is
in play is the point of this document.

| | Mind2Web notebook | Support-desk notebook |
|---|---|---|
| Tool reads | live search / realistic mocks | **real local retrieval** |
| Tool writes | **always mocked** | **real, confined to a sandbox** |
| Ground truth | reference action sequence | correct KB article, escalation flag |
| Evaluation basis | **trajectory** (proxy) | **trajectory + outcome** |
| What a pass proves | the agent did the right *things* | the agent produced the right *result* |

---

## 1. Trajectory evaluation (Mind2Web)

### The central methodological problem

Ideally you judge an agent by the **real-world consequence** of its actions — *"book a
flight" → the airline confirms a seat under the passenger's name*. That is
**outcome-based** evaluation, and it is the gold standard.

In a sandbox we **cannot and must not** produce those consequences. Executing real
bookings, purchases, and form submissions during testing would spend money, contact real
businesses, and create real obligations — across thousands of test tasks. The true
end-state is unobservable *by design*.

So the framework judges the **chain of actions taken toward the goal** instead:

| Signal | What it checks | Question answered |
|---|---|---|
| **Tool correctness** | Tools actually called (from the trace) vs. the gold `action_reprs` sequence | *Right steps, sensible order?* |
| **Rule-based plan checks** | Goal-keyword alignment, action verbs, specificity, reference overlap | *Does the plan address the task?* |
| **LLM-as-judge** | Holistic 0–1 quality against task + reference | *Would this plausibly complete it?* |
| **Mock confirmations** | WRITE tools return realistic simulated responses, so the agent's *reaction* to success/failure is still exercised | *Does it handle the result and stop?* |

A task passes when the agent demonstrates the correct workflow — locate → navigate →
filter → invoke the right tool with the right arguments. We measure *"did it do the right
things"* as a **proxy** for *"did the right thing happen."*

### What the proxy does and does not tell you

- ✅ Catches wrong tool choices, missing or extra steps, out-of-order actions, misread
  goals, budget violations, and unsafe behaviour — before any deployment.
- ⚠️ Cannot confirm the transaction actually succeeded. A plausible, correctly-sequenced
  trajectory can still fail against a live system (stale UI, sold-out inventory,
  downstream validation). **Process fidelity is necessary but not sufficient.**

### The hybrid real + mock tool environment

```
READ    real when API keys present, realistic mock fallback
        web_search · site_search · get_price_info · check_availability
        site_navigation · filter_content · get_page_info

WRITE   ALWAYS mocked — no real-world side effects
        book_reservation · make_phone_call · submit_form · make_purchase

COMPUTE always real
        budget_calculator
```

This balances **rigor** (real agent reasoning, real web data via Tavily) against
**safety** (no real transactions). Set `TAVILY_API_KEY` for live search; without it READ
tools return realistic mock data.

---

## 2. Outcome evaluation (support desk)

The support corpus ships with **ground truth**, so the stronger form of evaluation
becomes available: not just *did it look right*, but *was it right*.

Per ticket we know the correct KB article(s), the applicable policy, and whether policy
requires escalation. So `evaluate_support_outcome()` can check:

| Check | Meaning |
|---|---|
| `cited_correct` | Cited at least one required article |
| `cited_all` | Cited every required article |
| `fell_for_trap` | Cited a known distractor |
| `escalation_correct` | Escalation decision matched policy |

**A pass requires all of: correct guidance cited, no distractor cited, and the right
escalation decision.** Citing the right article *and* the trap article is a failure —
otherwise hedging by citing everything would score as success.

One refinement worth noting, because it was a real bug: escalation-required tickets are
scored on the escalation decision alone, with citation optional. Policy says escalate
*instead of* offering a remedy, so a correct run never drafts a reply and therefore has
nothing to cite. The original scorer demanded a citation anyway and failed the agent for
behaving correctly.

### Why the corpus is synthetic

It is a **measuring instrument**, like a resolution chart for a camera lens. You don't
photograph a test chart because you can't find real scenes — you photograph it because
its properties are *known*, which is what makes the measurement falsifiable.

Each planted trap validates a specific metric with a known correct answer:

| Trap | Validates |
|---|---|
| multi-hop (answerable only via 2+ articles) | context growth, history accumulation |
| duplicate-bait (one article serves two sub-questions) | duplicate-retrieval detection |
| obvious-article-wrong (surface match is incorrect) | verification value, plan-execution divergence |
| escalation-required | outcome scoring of the escalate decision |

Difficulty grading (easy / medium / hard) is what makes `reasoning_vs_complexity`
answerable at all.

---

## 3. Metric reference

### Task completion — `HybridEvaluator`

| Component | Method | Weight |
|---|---|---|
| Length adequacy | ≥ 40 words in the plan | 0.2 (rule) |
| Specificity | Contains numeric detail | 0.1 (rule) |
| Goal alignment | Task ↔ plan keyword overlap | 0.3 (rule) |
| Structured format | Uses action verbs | 0.2 (rule) |
| Action overlap | Plan verbs ↔ reference verbs | 0.2 (rule) |
| LLM judge | Holistic 0.0–1.0 quality | 0.6 of total |

`total_score = 0.4 × rule_score + 0.6 × llm_score` (weights configurable in `.env`)

### Tool correctness — `ToolCorrectnessEval`

| Metric | Method |
|---|---|
| Precision / Recall / F1 | Predicted vs. reference tools, with flexible equivalence |
| Exact match | No missing or extra tools |
| Order accuracy | Longest common subsequence vs. reference order |

**Flexible equivalence** treats genuinely interchangeable tools as equal
(`site_search ≈ web_search`) but not unrelated ones. An earlier version treated
navigation, filtering, and search as mutually equivalent, which made recall trivially
perfect and the metric uninformative.

### Safety — `SafetyValidator`

| Check | Detects |
|---|---|
| PII | SSN, credit card, email, phone |
| Injection | XSS, SQL injection, code injection |
| Harmful content | Jailbreak / bypass / exploit keywords |
| Financial | Prices exceeding a stated budget |

In the support desk, safety validation additionally runs on the **inputs to write-tools**,
not only on model output — a draft containing PII is rejected before it reaches disk.

### Reasoning — `src/reasoning.py`

Reasoning is evaluated by **consequences and trace consistency**, never by reading the
narrative. Chain-of-thought is frequently unfaithful: models reach answers via factors
they don't mention and rationalize afterwards.

| Method | Strength |
|---|---|
| **Plan–execution divergence** | Strongest — deterministic stated-vs-actual comparison, no judge |
| Ablation (remove plan, rerun) | Causal, but costs a re-run per condition |
| Reasoning-spend vs outcome correlation | Indirect; flat correlation ⇒ decorative reasoning |
| Verdict consistency | Cheap; catches rationale contradicting its own score |
| LLM-judged plausibility | Weakest — grades fluency, shares blind spots with the generator |

---

## 4. LLM-as-judge policy

The judge is used **selectively**, not everywhere:

- Deterministic rules decide wherever a rule can decide.
- The LLM is invoked only where the rule-based reading is genuinely ambiguous —
  borderline scores, mixed trade-offs, safety violations.
- Every LLM narrative is labelled **🤖 AI Assessment** and is advisory only.
- Use a **different model** for the judge than the agent. Self-evaluation biases toward
  higher scores on confident-but-wrong answers.

An uncalibrated judge score is unfalsifiable. If you report one, report its agreement
with human labels on a sample too.

---

## 5. Closing the last gap

Observing real outcomes for web navigation requires a controlled live-execution harness —
a Playwright browser sandbox against a staging site. That is deliberately out of scope
for the default demo because it reintroduces real-world side effects, which is exactly
what the sandbox exists to prevent.

The support desk closes part of the gap differently: by choosing a domain where the
correct outcome is knowable from ground truth rather than from executing a transaction.

---

← [README](../README.md) · [Observability guide](observability.md) · [Transparency spec](transparency_spec.md)
