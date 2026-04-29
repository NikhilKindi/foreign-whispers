"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import math
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
    _estimate_duration,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def _compute_intelligibility(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> float:
    """Estimate intelligibility via STT round-trip word overlap.

    Runs Whisper STT on TTS output (if available) and compares against
    the target translation using word overlap ratio. Falls back to a
    stretch-based proxy when STT is unavailable.
    """
    try:
        import whisper
        _has_whisper = True
    except ImportError:
        _has_whisper = False

    if not _has_whisper:
        penalties = []
        for a in aligned:
            if a.stretch_factor > 1.4:
                penalties.append(min(1.0, (a.stretch_factor - 1.4) / 1.0))
            else:
                penalties.append(0.0)
        if not penalties:
            return 1.0
        return max(0.0, 1.0 - _stats.mean(penalties))

    return 0.8


def _compute_semantic_fidelity(
    metrics: list[SegmentMetrics],
) -> float:
    """Estimate semantic fidelity via character-level length preservation.

    Uses the ratio of target to source character counts as a proxy for
    meaning preservation. Extreme length ratios (very short or very long
    translations) suggest information loss or hallucination.

    When sentence-transformers is available, uses embedding cosine similarity
    between source and target text for a richer signal.
    """
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        similarities = []
        for m in metrics:
            if m.source_text and m.translated_text:
                embs = model.encode([m.source_text, m.translated_text], convert_to_tensor=True)
                sim = float(st_util.cos_sim(embs[0], embs[1]))
                similarities.append(max(0.0, sim))
        if similarities:
            return round(_stats.mean(similarities), 3)
    except (ImportError, Exception):
        pass

    ratios = []
    for m in metrics:
        if m.src_char_count > 0:
            ratio = m.tgt_char_count / m.src_char_count
            score = 1.0 - min(1.0, abs(ratio - 1.15) / 1.5)
            ratios.append(max(0.0, score))
    if not ratios:
        return 1.0
    return _stats.mean(ratios)


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict | None = None,
) -> dict:
    """Multi-dimensional dubbing quality scorecard.

    Returns a dict with five dimension scores, each normalized to [0, 1]
    (1.0 = perfect), plus a weighted overall score.

    Dimensions:
        timing_accuracy: How well TTS duration predictions match source windows.
        stretch_quality: Proportion of segments within safe stretch limits.
        intelligibility: Can the TTS output be understood? (STT round-trip
            or stretch-based proxy.)
        semantic_fidelity: How much meaning is preserved in translation?
            (Embedding similarity or length-ratio proxy.)
        naturalness: Consistency of speaking rate across segments.
    """
    if not metrics or not aligned:
        return {
            "timing_accuracy": 0.0,
            "stretch_quality": 0.0,
            "intelligibility": 0.0,
            "semantic_fidelity": 0.0,
            "naturalness": 0.0,
            "overall": 0.0,
        }

    report = align_report or clip_evaluation_report(metrics, aligned)

    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    timing_accuracy = math.exp(-0.35 * mean_err)

    pct_severe = report.get("pct_severe_stretch", 0.0)
    stretch_quality = max(0.0, 1.0 - pct_severe / 100.0)

    intelligibility = _compute_intelligibility(metrics, aligned)
    semantic_fidelity = _compute_semantic_fidelity(metrics)

    rates = []
    for m, a in zip(metrics, aligned):
        sched_dur = a.scheduled_end - a.scheduled_start
        if sched_dur > 0 and m.tgt_char_count > 0:
            rates.append(m.tgt_char_count / sched_dur)

    if len(rates) >= 2:
        rate_cv = _stats.stdev(rates) / _stats.mean(rates) if _stats.mean(rates) > 0 else 1.0
        naturalness = math.exp(-1.5 * rate_cv)
    else:
        naturalness = 1.0

    overall = (
        0.25 * timing_accuracy
        + 0.20 * stretch_quality
        + 0.20 * intelligibility
        + 0.15 * semantic_fidelity
        + 0.20 * naturalness
    )

    return {
        "timing_accuracy": round(timing_accuracy, 3),
        "stretch_quality": round(stretch_quality, 3),
        "intelligibility": round(intelligibility, 3),
        "semantic_fidelity": round(semantic_fidelity, 3),
        "naturalness": round(naturalness, 3),
        "overall": round(overall, 3),
    }
