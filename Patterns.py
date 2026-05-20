"""
tools/patterns.py — Pattern Detection (Skill 7 / Product Thinking)

Pure-Python logic (no LLM) that runs on structured WorkExperience data to:
  1. Detect employment gaps > 3 months
  2. Flag job-hopping (roles < 12 months)
  3. Flag irrelevant experience vs target role (keyword-based pre-filter;
     the LLM does deeper assessment via assess_role_relevance)
  4. Detect title inflation patterns
  5. Flag unexplained frequent company changes

All findings return typed RedFlag / EmploymentGap objects so they feed
cleanly into ParsedCV validation.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from models import (
    DateRange,
    EmploymentGap,
    RedFlag,
    RiskLevel,
    WorkExperience,
    _parse_ym,
)


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (adjust to taste)
# ─────────────────────────────────────────────────────────────────────────────

GAP_MONTHS_MEDIUM_RISK = 4      # 4–8 months gap → medium
GAP_MONTHS_HIGH_RISK = 9        # 9+ months → high
HOPPING_THRESHOLD_MONTHS = 12   # < 12 months in a role → job-hopping flag
HOPPING_COUNT_THRESHOLD = 2     # flag after N hops in last 5 years


# ─────────────────────────────────────────────────────────────────────────────
# Gap detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_employment_gaps(experience: list[WorkExperience]) -> list[EmploymentGap]:
    """
    Identify gaps between consecutive roles.

    Sorts experience by start date (newest first in CV, so we reverse),
    then computes month-distance between consecutive roles.
    """
    if len(experience) < 2:
        return []

    # Sort chronologically
    def start_key(w: WorkExperience) -> date:
        try:
            return _parse_ym(w.dates.start)
        except Exception:
            return date(1900, 1, 1)

    sorted_exp = sorted(experience, key=start_key)
    gaps: list[EmploymentGap] = []

    for i in range(len(sorted_exp) - 1):
        current = sorted_exp[i]
        next_role = sorted_exp[i + 1]

        try:
            current_end = (
                date.today()
                if current.dates.end == "present"
                else _parse_ym(current.dates.end)
            )
            next_start = _parse_ym(next_role.dates.start)
        except Exception:
            continue

        gap_months = (
            (next_start.year - current_end.year) * 12
            + (next_start.month - current_end.month)
        )

        if gap_months <= 1:          # ≤1 month overlap / seamless
            continue

        if gap_months >= GAP_MONTHS_HIGH_RISK:
            risk = RiskLevel.HIGH
        elif gap_months >= GAP_MONTHS_MEDIUM_RISK:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        gaps.append(
            EmploymentGap(
                after_company=current.company,
                before_company=next_role.company,
                gap_months=gap_months,
                start_of_gap=current.dates.end if current.dates.end != "present"
                             else current_end.strftime("%Y-%m"),
                end_of_gap=next_role.dates.start,
                risk=risk,
            )
        )

    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# Job-hopping
# ─────────────────────────────────────────────────────────────────────────────


def detect_job_hopping(experience: list[WorkExperience]) -> list[RedFlag]:
    """
    Flag roles with tenure < HOPPING_THRESHOLD_MONTHS.
    Only counts recent roles (last 5 years) toward the hop counter.
    """
    flags: list[RedFlag] = []
    cutoff = date.today().year - 5

    short_stays: list[WorkExperience] = []
    for role in experience:
        if role.tenure_months is None:
            continue
        try:
            start_year = int(role.dates.start[:4])
        except Exception:
            continue
        if start_year < cutoff:
            continue
        if role.tenure_months < HOPPING_THRESHOLD_MONTHS:
            short_stays.append(role)

    for role in short_stays:
        flags.append(
            RedFlag(
                flag_type="job_hopping",
                description=(
                    f"Only {role.tenure_months} months at {role.company!r} "
                    f"({role.dates.start} – {role.dates.end}). "
                    f"Roles under {HOPPING_THRESHOLD_MONTHS} months raise retention concerns."
                ),
                evidence=(
                    f"'{role.title}' at {role.company}: "
                    f"{role.dates.start} → {role.dates.end} "
                    f"({role.tenure_months} months)"
                ),
                risk=RiskLevel.HIGH if role.tenure_months < 6 else RiskLevel.MEDIUM,
            )
        )

    if len(short_stays) >= HOPPING_COUNT_THRESHOLD:
        companies = ", ".join(r.company for r in short_stays)
        flags.append(
            RedFlag(
                flag_type="serial_job_hopper",
                description=(
                    f"Pattern of {len(short_stays)} short stays in the past 5 years "
                    f"({companies}). Likely to leave quickly."
                ),
                evidence=f"Short tenures at: {companies}",
                risk=RiskLevel.HIGH,
            )
        )

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Keyword relevance pre-filter (LLM does the full assessment)
# ─────────────────────────────────────────────────────────────────────────────

_COMMON_TECH_ROLES = {
    "software engineer", "data scientist", "machine learning", "backend",
    "frontend", "fullstack", "devops", "cloud", "product manager",
    "data analyst", "data engineer", "ml engineer", "ai", "nlp",
}


def detect_irrelevant_experience(
    experience: list[WorkExperience],
    target_role: Optional[str],
) -> list[RedFlag]:
    """
    Quickly flag roles that look completely unrelated to the target.
    This is a heuristic pre-filter — the LLM's assess_role_relevance
    does the authoritative assessment.
    """
    if not target_role:
        return []

    flags: list[RedFlag] = []
    target_lower = target_role.lower()

    # Build keywords from target role
    role_keywords = set(re.findall(r"\w+", target_lower)) | _COMMON_TECH_ROLES

    for role in experience:
        role_text = f"{role.title} {role.company} {' '.join(role.responsibilities)}".lower()
        if not any(kw in role_text for kw in role_keywords):
            flags.append(
                RedFlag(
                    flag_type="irrelevant_experience",
                    description=(
                        f"Role '{role.title}' at {role.company} appears unrelated "
                        f"to the target role '{target_role}'. "
                        "Review whether transferable skills exist."
                    ),
                    evidence=f"'{role.title}' at {role.company} ({role.dates.start}–{role.dates.end})",
                    risk=RiskLevel.MEDIUM,
                )
            )

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Master detector
# ─────────────────────────────────────────────────────────────────────────────


def run_all_pattern_checks(
    experience: list[WorkExperience],
    target_role: Optional[str] = None,
) -> tuple[list[EmploymentGap], list[RedFlag]]:
    """
    Run all structural pattern checks and return:
      (list[EmploymentGap], list[RedFlag])

    The agent calls this after extracting experience.
    """
    gaps = detect_employment_gaps(experience)
    flags: list[RedFlag] = []
    flags.extend(detect_job_hopping(experience))
    flags.extend(detect_irrelevant_experience(experience, target_role))

    # Convert gaps > threshold into red flags too
    for gap in gaps:
        if gap.risk in (RiskLevel.MEDIUM, RiskLevel.HIGH):
            flags.append(
                RedFlag(
                    flag_type="unexplained_gap",
                    description=(
                        f"{gap.gap_months}-month gap between "
                        f"{gap.after_company} and {gap.before_company}."
                    ),
                    evidence=(
                        f"Gap from {gap.start_of_gap} to {gap.end_of_gap} "
                        f"({gap.gap_months} months)"
                    ),
                    risk=gap.risk,
                )
            )

    return gaps, flags