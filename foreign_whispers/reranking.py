"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


_PHRASE_CONTRACTIONS: list[tuple[str, str]] = [
    ("en este momento", "ahora"),
    ("con el fin de", "para"),
    ("a pesar de que", "aunque"),
    ("con respecto a", "sobre"),
    ("en relación con", "sobre"),
    ("por lo tanto", "así que"),
    ("sin embargo", "pero"),
    ("a través de", "por"),
    ("debido a que", "porque"),
    ("con el objetivo de", "para"),
    ("en la actualidad", "hoy"),
    ("de acuerdo con", "según"),
    ("por medio de", "por"),
    ("en el caso de", "si"),
    ("a lo largo de", "durante"),
    ("en primer lugar", "primero"),
    ("en segundo lugar", "segundo"),
    ("con la finalidad de", "para"),
    ("en la medida en que", "mientras"),
    ("por otra parte", "además"),
    ("en lo que respecta a", "sobre"),
    ("a causa de", "por"),
    ("es decir", "o sea"),
    ("al mismo tiempo", "a la vez"),
    ("de todas maneras", "igual"),
    ("por supuesto", "claro"),
    ("tiene que", "debe"),
    ("va a ser", "será"),
    ("vamos a", "vamos"),
]

_FILLER_WORDS = [
    "realmente", "básicamente", "simplemente", "actualmente",
    "literalmente", "absolutamente", "definitivamente", "obviamente",
    "esencialmente", "particularmente", "ciertamente",
]

CHARS_PER_SECOND = 15.0


def _rule_based_shorten(text: str) -> tuple[str, str]:
    """Apply phrase contractions and filler removal. Returns (shortened, rationale)."""
    import re
    result = text
    changes: list[str] = []

    for long, short in _PHRASE_CONTRACTIONS:
        pattern = re.compile(re.escape(long), re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(short, result)
            changes.append(f"'{long}'→'{short}'")

    for filler in _FILLER_WORDS:
        pattern = re.compile(r"\b" + re.escape(filler) + r"\b\s*", re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub("", result)
            changes.append(f"removed '{filler}'")

    result = re.sub(r"\s{2,}", " ", result).strip()
    rationale = "rule-based: " + ", ".join(changes) if changes else "no rules applied"
    return result, rationale


def _argos_retranslate(source_text: str) -> tuple[str, str]:
    """Simplify the English source, then re-translate with argostranslate."""
    import re
    simplified = source_text
    simplified = re.sub(
        r"\b(really|basically|actually|simply|literally|absolutely|"
        r"definitely|obviously|essentially|particularly|certainly|"
        r"just|very|quite|rather|somewhat|incredibly)\b\s*",
        "", simplified, flags=re.IGNORECASE,
    )
    simplified = re.sub(r"\s{2,}", " ", simplified).strip()

    if simplified == source_text or not simplified:
        return "", "no simplification possible"

    try:
        import argostranslate.translate
        retranslated = argostranslate.translate.translate(simplified, "en", "es")
        return retranslated, "argos re-translation from simplified source"
    except Exception as exc:
        logger.warning("argos re-translation failed: %s", exc)
        return "", f"argos failed: {exc}"


_marian_model = None
_marian_tokenizer = None


def _marian_translate(source_text: str) -> tuple[str, str]:
    """Translate English to Spanish using Helsinki-NLP/opus-mt-en-es (MarianMT).

    Model is loaded once and cached for subsequent calls.
    """
    global _marian_model, _marian_tokenizer
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        return "", "transformers not installed"

    model_name = "Helsinki-NLP/opus-mt-en-es"
    try:
        if _marian_tokenizer is None:
            logger.info("Loading MarianMT model %s (one-time download)...", model_name)
            _marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
            _marian_model = MarianMTModel.from_pretrained(model_name)
            logger.info("MarianMT model loaded.")

        tokens = _marian_tokenizer(source_text, return_tensors="pt", truncation=True)
        translated = _marian_model.generate(**tokens)
        result = _marian_tokenizer.decode(translated[0], skip_special_tokens=True)
        return result, "MarianMT (Helsinki-NLP/opus-mt-en-es) translation"
    except Exception as exc:
        logger.warning("MarianMT translation failed: %s", exc)
        return "", f"MarianMT failed: {exc}"


def _truncate_to_budget(text: str, max_chars: int) -> tuple[str, str]:
    """Truncate at the last sentence or clause boundary within budget."""
    if len(text) <= max_chars:
        return text, "already within budget"

    import re
    # Try sentence boundaries first
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result = ""
    for s in sentences:
        candidate = (result + " " + s).strip() if result else s
        if len(candidate) <= max_chars:
            result = candidate
        else:
            break

    if result and len(result) >= max_chars * 0.5:
        return result, "truncated at sentence boundary"

    # Fall back to clause boundaries
    clauses = re.split(r"[,;:]\s+", text)
    result = ""
    for c in clauses:
        candidate = (result + ", " + c).strip() if result else c
        if len(candidate) <= max_chars:
            result = candidate
        else:
            break

    if result and len(result) >= max_chars * 0.3:
        return result, "truncated at clause boundary"

    # Last resort: word boundary
    words = text.split()
    result = ""
    for w in words:
        candidate = (result + " " + w).strip() if result else w
        if len(candidate) <= max_chars:
            result = candidate
        else:
            break

    return result or text[:max_chars], "truncated at word boundary"


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    Uses four independent strategies and returns all viable candidates
    sorted shortest first:

    1. **Rule-based** — Spanish filler word removal and phrase contraction.
    2. **MarianMT** — independent translation via Helsinki-NLP/opus-mt-en-es.
    3. **Argos re-translation** — simplify the English source text, then
       re-translate via argostranslate for a naturally shorter output.
    4. **Smart truncation** — cut at sentence/clause/word boundaries to
       fit the character budget.

    Duration heuristic: ~15 characters/second for Romance-language TTS.
    """
    max_chars = int(target_duration_s * CHARS_PER_SECOND)

    if len(baseline_es) <= max_chars:
        return []

    candidates: list[TranslationCandidate] = []
    seen_texts: set[str] = set()

    # Strategy 1: rule-based shortening
    try:
        text, rationale = _rule_based_shorten(baseline_es)
        if text and text != baseline_es and text not in seen_texts:
            seen_texts.add(text)
            candidates.append(TranslationCandidate(
                text=text, char_count=len(text), brevity_rationale=rationale,
            ))
    except Exception as exc:
        logger.warning("rule-based shortening failed: %s", exc)

    # Strategy 2: MarianMT independent translation
    try:
        text, rationale = _marian_translate(source_text)
        if text and text != baseline_es and text not in seen_texts:
            seen_texts.add(text)
            candidates.append(TranslationCandidate(
                text=text, char_count=len(text), brevity_rationale=rationale,
            ))
    except Exception as exc:
        logger.warning("MarianMT translation failed: %s", exc)

    # Strategy 3: argos re-translation with simplified source
    try:
        text, rationale = _argos_retranslate(source_text)
        if text and text != baseline_es and text not in seen_texts:
            seen_texts.add(text)
            candidates.append(TranslationCandidate(
                text=text, char_count=len(text), brevity_rationale=rationale,
            ))
    except Exception as exc:
        logger.warning("argos re-translation failed: %s", exc)

    # Strategy 3: smart truncation (always produces a result within budget)
    try:
        best_input = min(
            [baseline_es] + [c.text for c in candidates],
            key=len,
        )
        text, rationale = _truncate_to_budget(best_input, max_chars)
        if text and text not in seen_texts:
            seen_texts.add(text)
            candidates.append(TranslationCandidate(
                text=text, char_count=len(text), brevity_rationale=rationale,
            ))
    except Exception as exc:
        logger.warning("truncation failed: %s", exc)

    candidates.sort(key=lambda c: c.char_count)
    return candidates


def pick_optimal_translation(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> TranslationCandidate:
    """Pick the best translation for *target_duration_s*.

    If the argostranslate baseline already fits the TTS budget, returns it
    directly.  Otherwise runs all four shortening strategies via
    ``get_shorter_translations`` and returns the shortest candidate that
    fits, or the shortest overall if none fit.
    """
    max_chars = int(target_duration_s * CHARS_PER_SECOND)

    if len(baseline_es) <= max_chars:
        return TranslationCandidate(
            text=baseline_es,
            char_count=len(baseline_es),
            brevity_rationale="argostranslate baseline (within budget)",
        )

    candidates = get_shorter_translations(
        source_text=source_text,
        baseline_es=baseline_es,
        target_duration_s=target_duration_s,
        context_prev=context_prev,
        context_next=context_next,
    )

    if not candidates:
        return TranslationCandidate(
            text=baseline_es,
            char_count=len(baseline_es),
            brevity_rationale="no shorter candidate found, using baseline",
        )

    # Prefer candidates that fit the budget; among those pick the longest
    # (closest to budget = least meaning lost). If none fit, pick shortest.
    fitting = [c for c in candidates if c.char_count <= max_chars]
    if fitting:
        return max(fitting, key=lambda c: c.char_count)
    return candidates[0]
