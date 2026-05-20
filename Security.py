"""
tools/security.py — Security & Safety (Skill 4)

Responsibilities:
  1. Sanitize CV text before it reaches the LLM (blocks prompt injection
     hidden inside the CV, e.g. "Ignore previous instructions and …").
  2. PII handling helper (strips sensitive PII from logs).

Uses:
  - llm-guard  (prompt injection scanner + anonymizer)
  - Graceful degradation: if llm-guard is not installed, falls back to
    a simple regex-based injection blocker so the rest of the pipeline
    still works.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ── LLM Guard (optional but recommended) ─────────────────────────────────────
try:
    from llm_guard.input_scanners import PromptInjection, Anonymize
    from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF
    from llm_guard import scan_prompt
    LLM_GUARD_AVAILABLE = True
except ImportError:
    LLM_GUARD_AVAILABLE = False
    logger.warning(
        "llm-guard not installed. Falling back to regex injection blocker. "
        "Install with: pip install llm-guard"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simple regex fallback (always available)
# ─────────────────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    # Classic "ignore previous" attacks
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(everything|all)\s+(you('ve)?\s+)?(been\s+)?told",
    # Role-switching
    r"you\s+are\s+now\s+(a|an)\s+\w+",
    r"act\s+as\s+(a|an)\s+\w+",
    r"pretend\s+(you\s+are|to\s+be)",
    # Direct jailbreak signals
    r"DAN\s+mode",
    r"jailbreak",
    r"system\s+prompt",
    # Data exfiltration
    r"send\s+(me|the\s+user)\s+(your\s+)?(system|instructions|prompt)",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _INJECTION_PATTERNS]


def _regex_injection_scan(text: str) -> tuple[str, bool]:
    """
    Return (cleaned_text, was_flagged).
    Redacts matching segments with [REDACTED-INJECTION].
    """
    flagged = False
    for pattern in _COMPILED:
        if pattern.search(text):
            flagged = True
            text = pattern.sub("[REDACTED-INJECTION]", text)
    return text, flagged


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


class SecurityResult:
    def __init__(self, sanitized_text: str, injection_detected: bool, details: list[str]):
        self.sanitized_text = sanitized_text
        self.injection_detected = injection_detected
        self.details = details


def sanitize_cv_text(text: str) -> SecurityResult:
    """
    Sanitize raw CV text before passing to the LLM.

    Uses llm-guard if available, otherwise falls back to regex scanning.
    Always returns a SecurityResult — never raises.
    """
    details: list[str] = []
    injection_detected = False

    if LLM_GUARD_AVAILABLE:
        try:
            scanners = [PromptInjection()]
            sanitized, results_valid, results_score = scan_prompt(scanners, text)

            if not results_valid.get("PromptInjection", True):
                injection_detected = True
                score = results_score.get("PromptInjection", 0)
                details.append(
                    f"llm-guard detected prompt injection "
                    f"(score={score:.2f}). Text partially redacted."
                )
                # Also run regex as belt-and-suspenders
                sanitized, _ = _regex_injection_scan(sanitized)
            return SecurityResult(sanitized, injection_detected, details)

        except Exception as exc:
            logger.error("llm-guard scan failed, falling back to regex: %s", exc)

    # Fallback regex path
    sanitized, injection_detected = _regex_injection_scan(text)
    if injection_detected:
        details.append(
            "Regex scanner detected possible prompt-injection pattern(s). "
            "Segments redacted."
        )
    return SecurityResult(sanitized, injection_detected, details)


def redact_pii_for_logging(text: str) -> str:
    """
    Light PII redaction for safe logging.
    Redacts emails and phone numbers — keeps name/company for readability.
    """
    # Email
    text = re.sub(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
    # Phone (loose international pattern)
    text = re.sub(r"\+?[\d\s\-().]{7,15}\d", "[PHONE]", text)
    return text