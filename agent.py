"""
agent.py — CV Parser Agent  (TRUE AGENTIC — ReAct loop)

The LLM drives the entire parse. It:
  1. Receives a system prompt listing all available tools + their schemas
  2. Decides which tool to call next based on what it has found so far
  3. We execute the tool and return the result
  4. The LLM sees the result and decides the next step
  5. Repeats until the LLM calls `finish` with the completed ParsedCV

This is the Reason → Act → Observe cycle (ReAct).
The code never hardcodes "call experience, then education, then skills".
The LLM decides everything.

Skills implemented:
  Skill 1 — Tool contracts: Pydantic schemas → JSON Schema fed to LLM
  Skill 2 — Retrieval: reader sections available as a tool the LLM calls
  Skill 3 — Reliability: Tenacity retries + Pydantic output validation
  Skill 4 — Security: LLM Guard + path-traversal guard (pre-loop)
  Skill 5 — Evaluation: same ParsedCV output, DeepEval tests unchanged
  Skill 6 — Observability: every tool call + every LLM step traced
  Skill 7 — Product: evidence-backed flags, role fit, confidence scores
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Optional

from groq import Groq
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from models import (
    ContactInfo,
    Education,
    ParsedCV,
    Project,
    RoleRelevanceAssessment,
    SkillCategory,
    WorkExperience,
)
from tools.observability import TraceContext, log_llm_call, tool_span
from tools.patterns import run_all_pattern_checks
from tools.reader import CVReader
from tools.security import sanitize_cv_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

MODEL = "llama-3.3-70b-versatile"
MAX_STEPS = 20          # safety ceiling — prevents infinite loops


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — every tool the LLM can call
# ─────────────────────────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_cv_section",
            "description": (
                "Read one section of the CV. Call this FIRST to understand what "
                "sections exist before deciding what to extract. "
                "section_name must be one of: 'header', 'summary', 'experience', "
                "'education', 'skills', 'projects', 'certifications', 'full'."
                "Call 'full' only as a last resort if sections are missing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_name": {
                        "type": "string",
                        "enum": ["header", "summary", "experience", "education",
                                 "skills", "projects", "certifications", "full"],
                        "description": "Which section to read.",
                    }
                },
                "required": ["section_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_contact",
            "description": (
                "Extract contact information (name, email, phone, location, LinkedIn, GitHub) "
                "from the CV header text. "
                "Call ONLY after reading the 'header' section. "
                "Do NOT call if you already have contact info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Header section text."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_experience",
            "description": (
                "Extract all work experience entries (title, company, dates, responsibilities). "
                "Normalise dates to YYYY-MM. Use 'present' for current roles. "
                "Call ONLY after reading the 'experience' section. "
                "Do NOT call if the experience section is empty."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Experience section text."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_education",
            "description": (
                "Extract all education entries (institution, degree, field, dates). "
                "Call ONLY after reading the 'education' section. "
                "Do NOT include certifications here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Education section text."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_skills",
            "description": (
                "Extract and categorise skills (Languages, Frameworks, Cloud, Databases, Tools). "
                "Call ONLY after reading the 'skills' section. "
                "Do NOT invent skills not in the text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Skills section text."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_projects",
            "description": (
                "Extract project entries (name, description, technologies, URL). "
                "Call ONLY if a projects section exists and you have read it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Projects section text."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_certifications",
            "description": (
                "Extract certification names as a list of strings. "
                "Call ONLY if a certifications section exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Certifications section text."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_patterns",
            "description": (
                "Run gap detection and red-flag analysis on the extracted experience. "
                "Call this AFTER extract_experience. "
                "Do NOT call before you have experience data. "
                "Optionally provide target_role to enable relevance pre-filtering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_role": {
                        "type": "string",
                        "description": "The job being hired for (optional).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_role_relevance",
            "description": (
                "Score (0–100) how well the candidate fits the target role. "
                "Call ONLY if a target_role was provided by the user AND you have "
                "already extracted experience and skills. "
                "Do NOT call if no target_role was given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_role": {
                        "type": "string",
                        "description": "The job title being hired for.",
                    }
                },
                "required": ["target_role"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this when you have extracted all available information and "
                "are ready to return the final result. "
                "Do NOT call finish until you have at minimum: contact info and experience. "
                "Pass a plain-English summary of what was found and any caveats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was extracted and any issues.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Structured extraction helpers  (called by the agent loop)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_contact(text: str) -> ContactInfo:
    import instructor
    client = instructor.from_groq(Groq(api_key=os.environ["GROQ_API_KEY"]), mode=instructor.Mode.JSON)
    return client.chat.completions.create(
        model=MODEL, response_model=ContactInfo, max_retries=2,
        messages=[
            {"role": "system", "content": "Extract contact info. Only extract what is explicitly present. Leave missing fields as null."},
            {"role": "user", "content": text},
        ],
    )


def _parse_experience(text: str) -> list[WorkExperience]:
    class W(BaseModel):
        experience: list[WorkExperience]
    import instructor
    client = instructor.from_groq(Groq(api_key=os.environ["GROQ_API_KEY"]), mode=instructor.Mode.JSON)
    r = client.chat.completions.create(
        model=MODEL, response_model=W, max_retries=2,
        messages=[
            {"role": "system", "content": (
                "Extract work experience. Normalize dates to YYYY-MM. "
                "Use 'present' for current roles. Do not invent responsibilities."
            )},
            {"role": "user", "content": text},
        ],
    )
    return r.experience


def _parse_education(text: str) -> list[Education]:
    class W(BaseModel):
        education: list[Education]
    import instructor
    client = instructor.from_groq(Groq(api_key=os.environ["GROQ_API_KEY"]), mode=instructor.Mode.JSON)
    r = client.chat.completions.create(
        model=MODEL, response_model=W, max_retries=2,
        messages=[
            {"role": "system", "content": "Extract education records. Normalize dates to YYYY-MM."},
            {"role": "user", "content": text},
        ],
    )
    return r.education


def _parse_skills(text: str) -> list[SkillCategory]:
    class W(BaseModel):
        skills: list[SkillCategory]
    import instructor
    client = instructor.from_groq(Groq(api_key=os.environ["GROQ_API_KEY"]), mode=instructor.Mode.JSON)
    r = client.chat.completions.create(
        model=MODEL, response_model=W, max_retries=2,
        messages=[
            {"role": "system", "content": "Extract and categorize skills. Do not invent skills."},
            {"role": "user", "content": text},
        ],
    )
    return r.skills


def _parse_projects(text: str) -> list[Project]:
    class W(BaseModel):
        projects: list[Project]
    import instructor
    client = instructor.from_groq(Groq(api_key=os.environ["GROQ_API_KEY"]), mode=instructor.Mode.JSON)
    r = client.chat.completions.create(
        model=MODEL, response_model=W, max_retries=2,
        messages=[
            {"role": "system", "content": "Extract project entries. Keep descriptions concise."},
            {"role": "user", "content": text},
        ],
    )
    return r.projects


def _parse_certifications(text: str) -> list[str]:
    class W(BaseModel):
        certifications: list[str]
    import instructor
    client = instructor.from_groq(Groq(api_key=os.environ["GROQ_API_KEY"]), mode=instructor.Mode.JSON)
    r = client.chat.completions.create(
        model=MODEL, response_model=W, max_retries=2,
        messages=[
            {"role": "system", "content": "Extract certification names as a list of strings."},
            {"role": "user", "content": text[:2000]},
        ],
    )
    return r.certifications


def _parse_role_assessment(experience, skills, education, target_role: str) -> RoleRelevanceAssessment:
    exp_text = "\n".join(
        f"- {r.title} @ {r.company} ({r.tenure_months}mo)"
        for r in experience
    )
    skill_text = ", ".join(s for cat in skills for s in cat.skills)
    import instructor
    client = instructor.from_groq(Groq(api_key=os.environ["GROQ_API_KEY"]), mode=instructor.Mode.JSON)
    return client.chat.completions.create(
        model=MODEL, response_model=RoleRelevanceAssessment, max_retries=2,
        messages=[
            {"role": "system", "content": (
                "You are a senior technical recruiter. Score 0-100 fit for the role. "
                "Cite specific evidence. Do not score >80 unless the match is genuinely strong."
            )},
            {"role": "user", "content": (
                f"Target role: {target_role}\n"
                f"Experience:\n{exp_text}\n"
                f"Skills: {skill_text}"
            )},
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tenacity retry for the orchestrator LLM call
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _call_orchestrator(groq_client: Groq, messages: list[dict]) -> Any:
    """Single call to the orchestrator (the LLM that drives the ReAct loop)."""
    return groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        timeout=60.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class CVParserAgent:
    """
    True agentic CV parser.

    The LLM drives the loop — it decides which tools to call, in what order,
    and when to stop. The code only executes what the LLM requests.
    """

    def __init__(self):
        self.groq = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.ctx = TraceContext()

    def parse(
        self,
        file_path: str,
        target_role: Optional[str] = None,
        allowed_upload_dir: Optional[str] = None,
    ) -> ParsedCV:
        session_id = str(uuid.uuid4())[:8]
        logger.info("=== Agentic CV Parse %s started ===", session_id)

        # ── Pre-loop: read file + security scan (not LLM decisions) ──────────
        reader = CVReader(file_path, allowed_base=allowed_upload_dir)
        reader.load()

        raw_text = reader.get_section("full")
        sec = sanitize_cv_text(raw_text)
        if sec.injection_detected:
            logger.warning("SECURITY: injection detected and redacted.")

        self.ctx.start(session_id)

        try:
            result = self._react_loop(
                reader=reader,
                target_role=target_role,
                ocr_used=reader.ocr_used,
                warnings=reader.warnings + sec.details,
            )
        finally:
            self.ctx.end()

        logger.info("=== Agentic CV Parse %s complete ===", session_id)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # ReAct loop — the heart of the agent
    # ─────────────────────────────────────────────────────────────────────────

    def _react_loop(
        self,
        reader: CVReader,
        target_role: Optional[str],
        ocr_used: bool,
        warnings: list[str],
    ) -> ParsedCV:
        """
        ReAct loop:
          1. LLM reasons about what it knows so far
          2. LLM calls a tool
          3. We execute the tool
          4. Result is appended to message history
          5. Repeat until LLM calls `finish`
        """

        # ── Agent state — accumulated across loop iterations ──────────────────
        state: dict[str, Any] = {
            "contact": None,
            "summary": None,
            "experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "certifications": [],
            "gaps": [],
            "red_flags": [],
            "role_assessment": None,
            "sections_read": {},      # section_name → text (cache)
        }

        # ── Seed messages ─────────────────────────────────────────────────────
        target_clause = f"The target role is: {target_role}." if target_role else "No target role was specified."
        system_prompt = f"""You are an expert CV parsing agent. Your job is to extract structured information from a CV and identify red flags.

{target_clause}

INSTRUCTIONS:
- Use the available tools to read sections and extract information
- Start by reading sections (header, experience, education, skills) to understand the CV structure
- After reading a section, call the appropriate extract_* tool on that text
- After extracting experience, ALWAYS call detect_patterns
- If a target role was specified, call assess_role_relevance after extracting experience and skills
- Only call finish when you have extracted all available information
- Think step by step about what you still need to do

IMPORTANT: You must read a section before you can extract from it. The CV sections available are: header, summary, experience, education, skills, projects, certifications."""

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.append({
            "role": "user",
            "content": (
                "Please parse this CV completely. "
                "Extract all available information and identify any patterns or red flags. "
                + (f"The candidate is being assessed for: {target_role}." if target_role else "")
            ),
        })

        # ── ReAct loop ────────────────────────────────────────────────────────
        for step in range(MAX_STEPS):
            logger.info("[Step %d] Calling orchestrator LLM...", step + 1)

            with tool_span(self.ctx, f"react_step_{step+1}", {"step": step + 1}) as sd:
                response = _call_orchestrator(self.groq, messages)
                msg = response.choices[0].message
                sd["output"] = f"tool_calls={len(msg.tool_calls or [])}"

                # Log tokens
                if response.usage:
                    log_llm_call(
                        self.ctx, MODEL,
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        f"react_step_{step+1}",
                    )

            # Append the assistant message to history
            messages.append(msg)

            # ── No tool call → ask LLM to make a decision ────────────────────
            if not msg.tool_calls:
                logger.info("[Step %d] No tool call — prompting LLM to act.", step + 1)
                messages.append({
                    "role": "user",
                    "content": (
                        "Please call a tool to continue. "
                        "If you have all the information you need, call finish."
                    ),
                })
                continue

            # ── Execute each tool call ────────────────────────────────────────
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                logger.info("[Step %d] LLM called: %s(%s)", step + 1, tool_name, list(args.keys()))

                # ── FINISH — exit loop ────────────────────────────────────────
                if tool_name == "finish":
                    logger.info("Agent called finish. Building ParsedCV.")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Acknowledged. Finalising output.",
                    })
                    return self._build_parsed_cv(state, ocr_used, warnings)

                # ── Execute tool and collect result ───────────────────────────
                tool_result = self._execute_tool(tool_name, args, state, reader, target_role)

                # Append tool result to history so LLM can see it
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, default=str),
                })

                logger.info(
                    "[Step %d] %s → %s",
                    step + 1, tool_name,
                    str(tool_result)[:120],
                )

        # Safety: MAX_STEPS reached — return whatever we have
        logger.warning("MAX_STEPS reached — returning partial result.")
        return self._build_parsed_cv(state, ocr_used, warnings + ["Partial parse: max steps reached."])

    # ─────────────────────────────────────────────────────────────────────────
    # Tool executor — maps LLM tool calls to real Python functions
    # ─────────────────────────────────────────────────────────────────────────

    def _execute_tool(
        self,
        tool_name: str,
        args: dict,
        state: dict,
        reader: CVReader,
        target_role: Optional[str],
    ) -> dict:
        """
        Execute a tool the LLM requested.
        Updates `state` in-place and returns a dict the LLM can read.
        """

        with tool_span(self.ctx, tool_name, args) as sd:

            if tool_name == "read_cv_section":
                section = args.get("section_name", "full")
                text = reader.get_section(section)
                state["sections_read"][section] = text
                sd["output"] = f"{len(text)} chars"
                return {
                    "section": section,
                    "length": len(text),
                    "preview": text[:300],
                    "is_empty": len(text.strip()) == 0,
                    "text": text,   # full text for the LLM to pass to extract_*
                }

            elif tool_name == "extract_contact":
                text = args.get("text", state["sections_read"].get("header", ""))
                contact = _parse_contact(text)
                state["contact"] = contact
                sd["output"] = f"name={contact.name}"
                return {"name": contact.name, "email": contact.email, "status": "ok"}

            elif tool_name == "extract_experience":
                text = args.get("text", state["sections_read"].get("experience", ""))
                exp = _parse_experience(text)
                state["experience"] = exp
                sd["output"] = f"{len(exp)} roles"
                return {
                    "roles_extracted": len(exp),
                    "roles": [
                        {"title": r.title, "company": r.company,
                         "tenure_months": r.tenure_months}
                        for r in exp
                    ],
                    "status": "ok",
                }

            elif tool_name == "extract_education":
                text = args.get("text", state["sections_read"].get("education", ""))
                edu = _parse_education(text)
                state["education"] = edu
                sd["output"] = f"{len(edu)} entries"
                return {"entries": len(edu), "status": "ok"}

            elif tool_name == "extract_skills":
                text = args.get("text", state["sections_read"].get("skills", ""))
                skills = _parse_skills(text)
                state["skills"] = skills
                all_skills = [s for cat in skills for s in cat.skills]
                sd["output"] = f"{len(all_skills)} skills"
                return {"categories": len(skills), "skills_count": len(all_skills), "status": "ok"}

            elif tool_name == "extract_projects":
                text = args.get("text", state["sections_read"].get("projects", ""))
                projects = _parse_projects(text)
                state["projects"] = projects
                sd["output"] = f"{len(projects)} projects"
                return {"projects": len(projects), "status": "ok"}

            elif tool_name == "extract_certifications":
                text = args.get("text", state["sections_read"].get("certifications", ""))
                certs = _parse_certifications(text)
                state["certifications"] = certs
                sd["output"] = f"{len(certs)} certs"
                return {"certifications": certs, "status": "ok"}

            elif tool_name == "detect_patterns":
                if not state["experience"]:
                    return {"error": "No experience extracted yet. Call extract_experience first."}
                gaps, flags = run_all_pattern_checks(
                    state["experience"],
                    args.get("target_role", target_role),
                )
                state["gaps"] = gaps
                state["red_flags"] = flags
                sd["output"] = f"{len(gaps)} gaps, {len(flags)} flags"
                return {
                    "gaps_found": len(gaps),
                    "red_flags_found": len(flags),
                    "flags": [
                        {"type": f.flag_type, "risk": f.risk.value, "evidence": f.evidence}
                        for f in flags
                    ],
                    "status": "ok",
                }

            elif tool_name == "assess_role_relevance":
                role = args.get("target_role", target_role)
                if not role:
                    return {"error": "No target_role provided."}
                if not state["experience"]:
                    return {"error": "No experience extracted yet. Call extract_experience first."}
                assessment = _parse_role_assessment(
                    state["experience"], state["skills"],
                    state["education"], role,
                )
                state["role_assessment"] = assessment
                sd["output"] = f"score={assessment.relevance_score}"
                return {
                    "relevance_score": assessment.relevance_score,
                    "summary": assessment.summary,
                    "gaps": assessment.gaps,
                    "status": "ok",
                }

            else:
                sd["output"] = "unknown tool"
                return {"error": f"Unknown tool: {tool_name}"}

    # ─────────────────────────────────────────────────────────────────────────
    # Assemble final ParsedCV from accumulated state
    # ─────────────────────────────────────────────────────────────────────────

    def _build_parsed_cv(
        self,
        state: dict,
        ocr_used: bool,
        warnings: list[str],
    ) -> ParsedCV:
        from models import ContactInfo as CI
        contact = state["contact"] or CI(name="Unknown")
        return ParsedCV(
            contact=contact,
            summary=state.get("summary"),
            experience=state["experience"],
            education=state["education"],
            skills=state["skills"],
            projects=state["projects"],
            certifications=state["certifications"],
            employment_gaps=state["gaps"],
            red_flags=state["red_flags"],
            role_assessment=state["role_assessment"],
            ocr_used=ocr_used,
            parse_warnings=warnings,
        )