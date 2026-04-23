import re

from rapidfuzz import fuzz

from ctxrot.storage import CtxRotStore

# Strip DSPy structural markers before comparison
_DSPY_MARKER_RE = re.compile(r"\[\[\s*##\s*\w+\s*##\s*\]\]")

# Repetition threshold — above this, we flag onset
REPETITION_THRESHOLD = 0.4


def analyze_session(store: CtxRotStore, session_id: str) -> dict:
    """Analyze a session for context rot signals.

    Returns a dict with:
      - "has_content": whether store_content data was available
      - "repetition": per-iteration repetition scores (or None)
      - "efficiency": per-iteration efficiency ratios
      - "summary": human-readable summary dict
      - "session": session metadata (id, started_at, ended_at, model, mode,
                   terminal_state)
    """
    session = store.get_session(session_id)
    growth = store.get_growth_data(session_id)
    content = store.get_lm_call_content(session_id)

    efficiency = _compute_efficiency(growth)

    result: dict = {
        "has_content": bool(content),
        "repetition": None,
        "efficiency": efficiency,
        "summary": {},
        "session": session,
    }

    if content:
        result["repetition"] = _compute_repetition(content)

    result["summary"] = _build_summary(result)
    return result


# ── Repetition ────────────────────────────────────────────────────────────


def _clean_completion(text: str) -> str:
    """Strip DSPy markers and normalize whitespace."""
    text = _DSPY_MARKER_RE.sub("", text)
    return " ".join(text.split())


def _ngram_set(text: str, n: int = 3) -> frozenset[tuple[str, ...]]:
    """Build a frozenset of word n-gram tuples from text."""
    words = text.lower().split()
    if len(words) < n:
        return frozenset()
    return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two frozensets."""
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _compute_repetition(content: list[dict]) -> list[dict]:
    """Compute per-iteration repetition metrics from stored completions."""
    results = []
    prev_clean: str | None = None
    prev_ngrams: frozenset | None = None
    all_prior_ngrams: list[frozenset] = []

    for row in content:
        seq = row["seq"]
        completion = row.get("completion") or ""
        clean = _clean_completion(completion)
        ngrams = _ngram_set(clean)

        if prev_clean is None:
            # First iteration — baseline
            results.append(
                {
                    "seq": seq,
                    "ngram_jaccard": 0.0,
                    "sequence_similarity": 0.0,
                    "cumulative_max": 0.0,
                }
            )
        else:
            if prev_ngrams is None:
                prev_ngrams = frozenset()
            ngram_score = _jaccard(ngrams, prev_ngrams)
            raw_similarity = fuzz.ratio(clean, prev_clean)
            seq_score = raw_similarity / 100.0

            # Max similarity vs any prior completion
            cumulative = max(
                (_jaccard(ngrams, prior) for prior in all_prior_ngrams),
                default=0.0,
            )

            results.append(
                {
                    "seq": seq,
                    "ngram_jaccard": round(ngram_score, 3),
                    "sequence_similarity": round(seq_score, 3),
                    "cumulative_max": round(cumulative, 3),
                }
            )

        all_prior_ngrams.append(ngrams)
        prev_clean = clean
        prev_ngrams = ngrams

    return results


# ── Efficiency ────────────────────────────────────────────────────────────


def _compute_efficiency(growth: list[dict]) -> list[dict]:
    """Compute per-iteration efficiency ratio from growth data."""
    results = []
    for g in growth:
        prompt = g["prompt_tokens"]
        completion = g["completion_tokens"]
        ratio = completion / prompt if prompt > 0 else 0.0
        results.append(
            {
                "seq": g["seq"],
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "efficiency_ratio": round(ratio, 4),
            }
        )
    return results


# ── Summary ───────────────────────────────────────────────────────────────


def _build_summary(result: dict) -> dict:
    """Build a human-readable summary from analysis results."""
    summary: dict = {
        "repetition_detected": False,
        "onset_iteration": None,
        "max_repetition": 0.0,
    }

    session = result.get("session") or {}
    summary["terminal_state"] = session.get("terminal_state")

    repetition_rows = result.get("repetition")
    if repetition_rows:
        max_ngram = max(
            (row["ngram_jaccard"] for row in repetition_rows),
            default=0.0,
        )
        summary["max_repetition"] = max_ngram

        # Find onset — first iteration where ngram_jaccard exceeds threshold
        for row in repetition_rows:
            if row["ngram_jaccard"] > REPETITION_THRESHOLD:
                summary["repetition_detected"] = True
                summary["onset_iteration"] = row["seq"]
                break

    efficiency_rows = result.get("efficiency")
    if efficiency_rows and len(efficiency_rows) >= 2:
        first_iteration = efficiency_rows[0]
        last_iteration = efficiency_rows[-1]
        summary["initial_efficiency"] = first_iteration["efficiency_ratio"]
        summary["final_efficiency"] = last_iteration["efficiency_ratio"]

    return summary
