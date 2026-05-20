"""
eval/test_agent.py — Evaluation (Skill 5)

Uses DeepEval to measure whether the agent:
  1. Extracts all key fields from a strong CV
  2. Detects job-hopping correctly
  3. Flags employment gaps
  4. Produces evidence with every red flag
  5. Scores a strong candidate high on role relevance
  6. Scores a career-changer with appropriate caveats

Run with:
    cd cv_parser
    python -m pytest eval/test_agent.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make sure we can import from parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import CVParserAgent
from models import ParsedCV, RiskLevel

# ── DeepEval ─────────────────────────────────────────────────────────────────
try:
    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

# ── Fixtures ──────────────────────────────────────────────────────────────────

TEST_CV_DIR = Path(__file__).parent.parent / "tests" / "test_cvs"
TARGET_ROLE = "Senior Machine Learning Engineer"


@pytest.fixture(scope="module")
def agent():
    return CVParserAgent()


@pytest.fixture(scope="module")
def alice_result(agent) -> ParsedCV:
    return agent.parse(
        str(TEST_CV_DIR / "alice_strong_match.txt"),
        target_role=TARGET_ROLE,
    )


@pytest.fixture(scope="module")
def marcus_result(agent) -> ParsedCV:
    return agent.parse(
        str(TEST_CV_DIR / "marcus_job_hopper.txt"),
        target_role=TARGET_ROLE,
    )


@pytest.fixture(scope="module")
def priya_result(agent) -> ParsedCV:
    return agent.parse(
        str(TEST_CV_DIR / "priya_career_changer.txt"),
        target_role=TARGET_ROLE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test Group 1: Field extraction — Alice (strong match)
# ─────────────────────────────────────────────────────────────────────────────


class TestAliceExtraction:
    def test_contact_name_extracted(self, alice_result: ParsedCV):
        assert "alice" in alice_result.contact.name.lower(), (
            f"Expected 'alice' in contact name, got: {alice_result.contact.name}"
        )

    def test_contact_email_extracted(self, alice_result: ParsedCV):
        assert alice_result.contact.email is not None
        assert "@" in alice_result.contact.email

    def test_experience_count(self, alice_result: ParsedCV):
        assert len(alice_result.experience) >= 3, (
            f"Expected ≥3 roles, got {len(alice_result.experience)}"
        )

    def test_experience_dates_normalized(self, alice_result: ParsedCV):
        import re
        for role in alice_result.experience:
            assert re.match(r"\d{4}-\d{2}", role.dates.start), (
                f"Start date not normalized: {role.dates.start}"
            )

    def test_skills_extracted(self, alice_result: ParsedCV):
        all_skills = [s for cat in alice_result.skills for s in cat.skills]
        assert len(all_skills) >= 5, f"Expected ≥5 skills, got {len(all_skills)}"
        # PyTorch should be there
        joined = " ".join(all_skills).lower()
        assert "pytorch" in joined or "python" in joined

    def test_education_extracted(self, alice_result: ParsedCV):
        assert len(alice_result.education) >= 1

    def test_certifications_extracted(self, alice_result: ParsedCV):
        assert len(alice_result.certifications) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test Group 2: Role relevance — Alice
# ─────────────────────────────────────────────────────────────────────────────


class TestAliceRelevance:
    def test_role_assessment_present(self, alice_result: ParsedCV):
        assert alice_result.role_assessment is not None

    def test_strong_candidate_scores_high(self, alice_result: ParsedCV):
        score = alice_result.role_assessment.relevance_score
        assert score >= 70, (
            f"Expected score ≥70 for strong ML candidate, got {score}. "
            f"Summary: {alice_result.role_assessment.summary}"
        )

    def test_no_high_risk_red_flags(self, alice_result: ParsedCV):
        high_risk = [f for f in alice_result.red_flags if f.risk == RiskLevel.HIGH]
        assert len(high_risk) == 0, (
            f"Unexpected high-risk flags for Alice: {[f.flag_type for f in high_risk]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test Group 3: Job-hopping detection — Marcus
# ─────────────────────────────────────────────────────────────────────────────


class TestMarcusJobHopping:
    def test_job_hopping_flag_raised(self, marcus_result: ParsedCV):
        flag_types = [f.flag_type for f in marcus_result.red_flags]
        assert any("hopp" in ft for ft in flag_types), (
            f"Expected job_hopping flag, got: {flag_types}"
        )

    def test_all_flags_have_evidence(self, marcus_result: ParsedCV):
        for flag in marcus_result.red_flags:
            assert flag.evidence.strip(), (
                f"Flag '{flag.flag_type}' is missing evidence."
            )

    def test_weak_ml_match_scores_low(self, marcus_result: ParsedCV):
        if marcus_result.role_assessment:
            score = marcus_result.role_assessment.relevance_score
            assert score <= 40, (
                f"Expected low relevance score for Marcus vs ML role, got {score}"
            )

    def test_short_tenure_roles_detected(self, marcus_result: ParsedCV):
        short = [r for r in marcus_result.experience if (r.tenure_months or 99) < 12]
        assert len(short) >= 2, (
            f"Expected ≥2 short-tenure roles, detected {len(short)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test Group 4: Gap detection — Priya (gap 2020-11 → 2022-07)
# ─────────────────────────────────────────────────────────────────────────────


class TestPriyaGapDetection:
    def test_employment_gap_detected(self, priya_result: ParsedCV):
        assert len(priya_result.employment_gaps) >= 1, (
            "Expected at least one gap (Priya left HDFC Nov 2020, "
            "started MTech Jul 2022 — 20 months)"
        )

    def test_large_gap_is_high_or_medium_risk(self, priya_result: ParsedCV):
        risks = [g.risk for g in priya_result.employment_gaps]
        assert any(r in (RiskLevel.MEDIUM, RiskLevel.HIGH) for r in risks), (
            f"20-month gap should be medium/high risk. Got: {risks}"
        )

    def test_career_change_relevance_mid_range(self, priya_result: ParsedCV):
        if priya_result.role_assessment:
            score = priya_result.role_assessment.relevance_score
            # Priya has the degree + projects but no industry ML experience
            assert 30 <= score <= 75, (
                f"Career changer score should be 30-75, got {score}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test Group 5: DeepEval answer quality (if installed)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="deepeval not installed")
class TestDeepEvalQuality:
    """
    Uses DeepEval to check that the role assessment summary is relevant
    to the actual input (faithfulness + answer relevancy).
    """

    def test_alice_summary_relevance(self, alice_result: ParsedCV):
        if not alice_result.role_assessment:
            pytest.skip("No role assessment produced")

        exp_text = " | ".join(
            f"{r.title} at {r.company}" for r in alice_result.experience
        )
        test_case = LLMTestCase(
            input=f"Assess fit for: {TARGET_ROLE}. Experience: {exp_text}",
            actual_output=alice_result.role_assessment.summary,
            retrieval_context=[exp_text],
        )

        metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o-mini")
        assert_test(test_case, [metric])

    def test_marcus_red_flag_faithfulness(self, marcus_result: ParsedCV):
        if not marcus_result.red_flags:
            pytest.skip("No red flags produced")

        flag = marcus_result.red_flags[0]
        test_case = LLMTestCase(
            input="What are the red flags in this CV?",
            actual_output=flag.description,
            retrieval_context=[flag.evidence],
        )

        metric = FaithfulnessMetric(threshold=0.5, model="gpt-4o-mini")
        assert_test(test_case, [metric])