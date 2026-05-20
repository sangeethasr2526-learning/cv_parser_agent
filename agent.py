"""
CV Parser Agent — Fully Agentic Version
-----------------------------------------
LLM decides which tools to use, in what order, and how many times.
Includes robust JSON cleaning to handle LLaMA quirks.
"""

import os
import re
import json
from openai import OpenAI

from tools import (
    detect_file_type,
    read_pdf,
    read_docx,
    ocr_image,
    detect_gaps,
    detect_patterns,
    validate_output,
)

# =============================================================================
# CLIENT SETUP
# =============================================================================

# Option A: Groq (current)
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", "paste-your-groq-key-here"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "llama-3.3-70b-versatile"

# Option B: Claude — uncomment to switch
# from anthropic import Anthropic
# client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
# MODEL = "claude-sonnet-4-6"

# Option C: GPT-4o — uncomment to switch
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# MODEL = "gpt-4o"

# =============================================================================
# JSON CLEANER — fixes common LLaMA JSON mistakes before parsing
# =============================================================================

def clean_json(raw: str) -> str:
    """
    Fixes common JSON generation mistakes by LLaMA:
    - "year": 2017-"2021"  →  "year": "2021"
    - Trailing commas      →  removed
    - Markdown code fences →  stripped
    """
    # Strip markdown fences
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    raw = raw.strip()

    # Fix: number-dash-string like 2017-"2021" → just take the last value "2021"
    raw = re.sub(r'\d{4}-"(\d{4})"', r'"\1"', raw)

    # Fix: bare number dash string like 2017-2021 in a value → "2021"
    raw = re.sub(r':\s*\d{4}-(\d{4})', r': "\1"', raw)

    # Fix: trailing commas before } or ]
    raw = re.sub(r',\s*([}\]])', r'\1', raw)

    return raw


def safe_parse_args(raw: str) -> dict:
    """Parse tool call arguments, cleaning JSON if needed."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(clean_json(raw))
        except json.JSONDecodeError as e:
            return {"_parse_error": str(e), "_raw": raw}


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_file_type",
            "description": (
                "Detects the type of a CV file (pdf, docx, image, text). "
                "Call this first when you have a file path and don't know the type yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf",
            "description": (
                "Extracts all text from a PDF file. "
                "Use when file type is pdf. "
                "If result has no text, the PDF is scanned — try ocr_image instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_docx",
            "description": "Extracts all text from a Word .docx file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ocr_image",
            "description": (
                "Reads text from a scanned image or photo using OCR. "
                "Use when file type is image, or as fallback when read_pdf returns no text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_gaps",
            "description": (
                "Finds employment gaps longer than 3 months from a list of jobs. "
                "Pass the experience list after extracting it. "
                "Use results to populate the gaps field and red_flags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experience": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "company": {"type": "string"},
                                "role":    {"type": "string"},
                                "start":   {"type": "string"},
                                "end":     {"type": "string"},
                            },
                        },
                    }
                },
                "required": ["experience"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_patterns",
            "description": (
                "Calculates average job tenure and detects job-hopping. "
                "Pass the experience list. Use results for patterns field and red_flags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experience": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "company": {"type": "string"},
                                "role":    {"type": "string"},
                                "start":   {"type": "string"},
                                "end":     {"type": "string"},
                            },
                        },
                    }
                },
                "required": ["experience"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_output",
            "description": (
                "Checks the final JSON has all required fields. "
                "Call when you think you are done. "
                "If it returns issues, fix them and call again. "
                "Only finish when valid is true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cv_json": {"type": "object"}
                },
                "required": ["cv_json"],
            },
        },
    },
]

# =============================================================================
# TOOL DISPATCHER
# =============================================================================

TOOL_MAP = {
    "detect_file_type": detect_file_type,
    "read_pdf":         read_pdf,
    "read_docx":        read_docx,
    "ocr_image":        ocr_image,
    "detect_gaps":      detect_gaps,
    "detect_patterns":  detect_patterns,
    "validate_output":  validate_output,
}

def run_tool(name: str, args: dict) -> str:
    if "_parse_error" in args:
        return json.dumps({"error": f"Could not parse tool arguments: {args['_parse_error']}"})
    fn = TOOL_MAP.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**args)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

# =============================================================================
# SYSTEM PROMPT — fully agentic
# =============================================================================

SYSTEM_PROMPT = """
You are an expert CV parsing agent for a recruitment team.

You have tools available. You decide which tools to use, in what order,
and how many times — based on what you find at each step.

Your goal: fully parse the CV file and return a complete JSON object with:
  - candidate: name, email, phone, location (strings)
  - skills: list of strings
  - experience: list of {company, role, start, end, description}
    start and end must be strings in YYYY-MM format or "present"
  - education: list of {institution, degree, year}
    year must be a single string like "2021" not "2017-2021"
  - projects: list of {name, description}
  - gaps: from detect_gaps tool
  - patterns: from detect_patterns tool
  - red_flags: list of concern strings you identify
  - summary: plain English paragraph assessment

IMPORTANT JSON rules:
  - All values must be valid JSON
  - year must be a single string e.g. "2021" never "2017-2021"
  - dates must be "YYYY-MM" strings e.g. "2021-08"
  - No trailing commas

Think before each tool call. If read_pdf returns empty text, try ocr_image.
Always call validate_output before finishing. Fix any issues it finds.
Return only valid JSON. No commentary outside the JSON.
""".strip()

# =============================================================================
# AGENT LOOP
# =============================================================================

def parse_cv(file_path: str, job_description: str = "") -> dict:
    user_message = f"Parse this CV file: {file_path}"
    if job_description:
        user_message += f"\n\nTarget job description:\n{job_description}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    print(f"\n{'='*50}")
    print(f"Parsing CV: {file_path}")
    print(f"Model:      {MODEL}")
    print(f"{'='*50}")

    max_iterations = 20
    for i in range(max_iterations):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            error_msg = str(e)
            print(f"  !! API error at iteration {i+1}: {error_msg[:120]}")
            # Tell the LLM what went wrong so it can recover
            messages.append({
                "role": "user",
                "content": f"Your last tool call failed with: {error_msg[:300]}. Please fix the JSON and try again. Remember: year must be a string like '2021', dates must be 'YYYY-MM'."
            })
            continue

        msg = response.choices[0].message
        messages.append(msg)
        finish_reason = response.choices[0].finish_reason

        # LLM finished
        if finish_reason == "stop":
            print(f"\n✓ Done after {i+1} iterations")
            content = msg.content or ""
            try:
                return json.loads(clean_json(content))
            except json.JSONDecodeError:
                return {"raw_response": msg.content, "error": "Could not parse final JSON"}

        # LLM wants to call tools
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                name     = tool_call.function.name
                raw_args = tool_call.function.arguments
                args     = safe_parse_args(raw_args)

                print(f"  -> [{i+1}] LLM chose: {name}({list(args.keys())})")
                result = run_tool(name, args)

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result,
                })

    return {"error": "Max iterations reached without completion"}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_cv> [job_description]")
        sys.exit(1)

    cv_path = sys.argv[1]
    jd      = sys.argv[2] if len(sys.argv) > 2 else ""

    result = parse_cv(cv_path, jd)
    print("\n-- RESULT ------------------------------------------")
    print(json.dumps(result, indent=2))