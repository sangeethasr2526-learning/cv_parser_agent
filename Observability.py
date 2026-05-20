"""
tools/observability.py — Observability (Skill 6)

Wraps every tool call and LLM call with a Langfuse span so you can
watch the agent think live in the Langfuse dashboard.

Logged per span:
  - tool name
  - input (PII-redacted for logs)
  - output (truncated)
  - duration
  - whether OCR was used
  - any security flags raised

Graceful degradation: if Langfuse is not configured, all tracing calls
are no-ops — the pipeline keeps working.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

logger = logging.getLogger(__name__)

# ── Langfuse (optional) ───────────────────────────────────────────────────────
try:
    from langfuse import Langfuse

    _lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    LANGFUSE_AVAILABLE = bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )
    if not LANGFUSE_AVAILABLE:
        logger.info("Langfuse keys not set — tracing disabled.")
except ImportError:
    LANGFUSE_AVAILABLE = False
    _lf = None
    logger.warning("langfuse not installed. Tracing disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# Trace context (one trace per CV parse session)
# ─────────────────────────────────────────────────────────────────────────────


class TraceContext:
    """Holds the active Langfuse trace for one CV parse run."""

    def __init__(self):
        self._trace = None

    def start(self, session_id: str, candidate_name: str = "unknown") -> None:
        if LANGFUSE_AVAILABLE and _lf:
            self._trace = _lf.trace(
                name="cv_parse_session",
                session_id=session_id,
                metadata={"candidate": candidate_name},
            )

    def end(self) -> None:
        if self._trace:
            try:
                _lf.flush()
            except Exception:
                pass
            self._trace = None

    @property
    def trace(self):
        return self._trace


# ─────────────────────────────────────────────────────────────────────────────
# Span helper
# ─────────────────────────────────────────────────────────────────────────────


@contextmanager
def tool_span(
    ctx: TraceContext,
    tool_name: str,
    inputs: dict[str, Any],
    metadata: Optional[dict] = None,
) -> Generator[dict, None, None]:
    """
    Context manager that wraps a tool call in a Langfuse span.

    Usage:
        with tool_span(ctx, "extract_experience", {"section": text}) as span_data:
            result = do_extraction(text)
            span_data["output"] = str(result)[:500]
    """
    span_data: dict[str, Any] = {}
    span = None
    t0 = time.perf_counter()

    if LANGFUSE_AVAILABLE and ctx.trace:
        span = ctx.trace.span(
            name=tool_name,
            input=inputs,
            metadata=metadata or {},
        )

    try:
        yield span_data
    finally:
        elapsed = time.perf_counter() - t0
        span_data.setdefault("duration_ms", round(elapsed * 1000))

        if span:
            try:
                span.end(
                    output=span_data.get("output", ""),
                    metadata={
                        **(metadata or {}),
                        "duration_ms": span_data["duration_ms"],
                        **{k: v for k, v in span_data.items() if k not in ("output", "duration_ms")},
                    },
                )
            except Exception as exc:
                logger.debug("Langfuse span.end() failed: %s", exc)

        logger.debug(
            "[%s] completed in %.0fms | output_preview=%s",
            tool_name,
            elapsed * 1000,
            str(span_data.get("output", ""))[:120],
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM call logger
# ─────────────────────────────────────────────────────────────────────────────


def log_llm_call(
    ctx: TraceContext,
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    tool_name: str,
) -> None:
    """Log an LLM generation event to Langfuse."""
    if not (LANGFUSE_AVAILABLE and ctx.trace):
        return
    try:
        ctx.trace.generation(
            name=f"llm_{tool_name}",
            model=model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
    except Exception as exc:
        logger.debug("Langfuse generation log failed: %s", exc)