"""
models.py — All Pydantic v2 schemas for the CV Parser Agent.

Covers:
  - Tool input / output contracts (Skill 1)
  - Validated CV output (Skill 3)
  - Red-flag & gap detection (Skill 7)
"""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceLevel(str, Enum):
    HIGH = "high"       # LLM is certain
    MEDIUM = "medium"   # Some ambiguity in source text
    LOW = "low"         # Inferred / OCR text was unclear


# ─────────────────────────────────────────────────────────────────────────────
# Primitive building blocks
# ─────────────────────────────────────────────────────────────────────────────


class DateRange(BaseModel):
    """
    Normalized date range for any role or education entry.
    Both dates are stored as YYYY-MM strings for easy gap arithmetic.
    Use 'present' for ongoing roles.
    """

    start: str = Field(
        ...,
        description="Start month as YYYY-MM, e.g. '2021-03'",
        examples=["2021-03"],
    )
    end: str = Field(
        ...,
        description="End month as YYYY-MM, or the literal string 'present'",
        examples=["2023-11", "present"],
    )

    @field_validator("start", "end")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if v.lower() == "present":
            return "present"
        if not re.fullmatch(r"\d{4}-\d{2}", v):
            raise ValueError(
                f"Date must be YYYY-MM or 'present', got: {v!r}"
            )
        return v


class ContactInfo(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Experience
# ─────────────────────────────────────────────────────────────────────────────


class WorkExperience(BaseModel):
    """
    Single job entry extracted from the CV.

    When NOT to populate:
      - Do NOT include volunteer / hobby projects unless clearly labeled as work.
      - Do NOT invent responsibilities not mentioned in the CV.
    """

    company: str
    title: str
    dates: DateRange
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Bullet-point responsibilities, verbatim or lightly cleaned.",
    )
    tenure_months: Optional[int] = Field(
        default=None,
        description="Auto-computed tenure in months. Leave None; will be set by validator.",
    )
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH

    @model_validator(mode="after")
    def compute_tenure(self) -> "WorkExperience":
        """Auto-compute tenure from DateRange."""
        try:
            start = _parse_ym(self.dates.start)
            end = date.today() if self.dates.end == "present" else _parse_ym(self.dates.end)
            delta = (end.year - start.year) * 12 + (end.month - start.month)
            self.tenure_months = max(delta, 0)
        except Exception:
            pass
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Education & Skills
# ─────────────────────────────────────────────────────────────────────────────


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    dates: Optional[DateRange] = None
    gpa: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH


class SkillCategory(BaseModel):
    """
    Skills grouped by category (e.g., 'Languages', 'Frameworks', 'Cloud').
    Do NOT invent skills not listed or strongly implied by the CV.
    """

    category: str
    skills: list[str]
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH


class Project(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    url: Optional[str] = None
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Red flags & gap detection
# ─────────────────────────────────────────────────────────────────────────────


class EmploymentGap(BaseModel):
    """A gap between two consecutive roles."""

    after_company: str
    before_company: str
    gap_months: int
    start_of_gap: str   # YYYY-MM
    end_of_gap: str     # YYYY-MM or 'present'
    risk: RiskLevel


class RedFlag(BaseModel):
    """
    A specific concern identified by the agent.

    Each flag MUST cite evidence — a quote or paraphrase from the CV.
    Do NOT raise a flag without evidence.
    """

    flag_type: str = Field(
        ...,
        description=(
            "Short label: e.g. 'job_hopping', 'skill_gap', "
            "'irrelevant_experience', 'unexplained_gap', 'title_inflation'"
        ),
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of the concern.",
    )
    evidence: str = Field(
        ...,
        description=(
            "Direct quote or paraphrase from the CV that supports this flag. "
            "Example: 'Left Company X after 4 months (Jan 2022 – Apr 2022).'"
        ),
    )
    risk: RiskLevel


class RoleRelevanceAssessment(BaseModel):
    """
    Assessment of how well the candidate matches the target role.
    Only populated when a target_role is provided by the user.
    """

    target_role: str
    relevance_score: int = Field(
        ..., ge=0, le=100,
        description="0 = completely irrelevant, 100 = perfect match."
    )
    relevant_experience: list[str] = Field(
        default_factory=list,
        description="Specific experiences / skills that support the role.",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Skills or experience areas required by the role that are absent.",
    )
    summary: str = Field(
        ...,
        description="2–3 sentence plain-English verdict for a recruiter.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Master output
# ─────────────────────────────────────────────────────────────────────────────


class ParsedCV(BaseModel):
    """
    Complete structured output of the CV parser.
    Every field carries a confidence signal.
    """

    contact: ContactInfo
    summary: Optional[str] = Field(
        default=None,
        description="Candidate's own summary/objective if present in CV.",
    )
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[SkillCategory] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

    # Analysis layer
    employment_gaps: list[EmploymentGap] = Field(default_factory=list)
    red_flags: list[RedFlag] = Field(default_factory=list)
    role_assessment: Optional[RoleRelevanceAssessment] = None

    # Meta
    ocr_used: bool = False
    parse_warnings: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Tool input schemas  (Skill 1 — Tool Contract Design)
# ─────────────────────────────────────────────────────────────────────────────


class ReadCVSectionInput(BaseModel):
    """Input for the read_cv_section tool."""

    section_name: str = Field(
        ...,
        description=(
            "Which section to read: 'header', 'summary', 'experience', "
            "'education', 'skills', 'projects', 'certifications', 'full'."
        ),
    )


class ExtractContactInput(BaseModel):
    """
    Input for extract_contact.
    Do NOT call this tool if there is no header section in the CV.
    """

    text: str = Field(..., description="Raw text of the CV header section.")


class ExtractExperienceInput(BaseModel):
    """
    Input for extract_experience.
    Do NOT call if the CV has no work-history section.
    """

    text: str = Field(..., description="Raw text of the experience section.")


class ExtractEducationInput(BaseModel):
    text: str = Field(..., description="Raw text of the education section.")


class ExtractSkillsInput(BaseModel):
    text: str = Field(..., description="Raw text of the skills section.")


class DetectPatternsInput(BaseModel):
    """
    Input for detect_patterns (gap & red-flag analysis).
    Call this AFTER extract_experience, not before.
    """

    experience: list[WorkExperience]
    target_role: Optional[str] = None


class AssessRoleRelevanceInput(BaseModel):
    """
    Input for assess_role_relevance.
    Do NOT call if the user did not provide a target_role.
    """

    parsed_cv: ParsedCV
    target_role: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_ym(ym: str) -> date:
    """Convert 'YYYY-MM' to a date object (day=1)."""
    year, month = map(int, ym.split("-"))
    return date(year, month, 1)