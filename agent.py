"""
CV Parser Agent — Fully Agentic Version
-----------------------------------------
LLM decides which tools to use, in what order, and how many times.
Includes robust JSON cleaning to handle LLaMA quirks.
Supports: Groq API (cloud) or Ollama local models (GPU/CPU).

Best local models for agentic tool calling (2026):
  1. qwen2.5:7b       — Best JSON + tool calling under 8GB VRAM
  2. mistral-nemo:12b — Strong tool calling, fits 8GB VRAM
  3. llama3.1:8b      — Reliable agentic pipelines
"""

import os
import re
import json
import subprocess
import sys
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
# LOCAL MODEL OPTIONS — best models for agentic tool calling (2026)
# All support real function/tool calling, not just text simulation.
# They will be auto-pulled if not already downloaded.
# =============================================================================

LOCAL_MODELS = {
    "1": {
        "name":      "qwen2.5:7b",
        "label":     "Qwen 2.5 7B       — Best JSON + tool calling, <8GB VRAM  (~5GB download)",
        "pull_name": "qwen2.5:7b",
    },
    "2": {
        "name":      "mistral-nemo:latest",
        "label":     "Mistral Nemo 12B  — Strong tool calling, fits 8GB VRAM   (~7GB download)",
        "pull_name": "mistral-nemo",
    },
    "3": {
        "name":      "llama3.1:8b",
        "label":     "Llama 3.1 8B      — Reliable agentic pipelines, 8GB VRAM (~5GB download)",
        "pull_name": "llama3.1:8b",
    },
}

# =============================================================================
# OLLAMA HELPERS
# =============================================================================

def check_ollama_gpu():
    """Check Ollama is running and report GPU status."""
    print("\n" + "=" * 50)
    print("Checking Ollama status...")

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("  ⚠  Ollama is not running. Start it with: ollama serve")
            sys.exit(1)
    except FileNotFoundError:
        print("  ✗  Ollama not installed. Get it from: https://ollama.com/download")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("  ⚠  Ollama not responding. Try: ollama serve")
        sys.exit(1)

    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if gpu.returncode == 0:
            print(f"  ✓  GPU detected: {gpu.stdout.strip()}")
            print("     Ollama will use this GPU automatically via CUDA.")
        else:
            print("  ⚠  GPU not detected — will run on CPU (slow).")
    except FileNotFoundError:
        print("  ⚠  nvidia-smi not found. GPU status unknown.")

    print("=" * 50)


def ensure_model_pulled(pull_name: str):
    """Pull the model if not already downloaded."""
    print(f"\nChecking if '{pull_name}' is available locally...")
    try:
        listed = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        # Match on base name (before colon) to handle :latest vs :7b tags
        base_name = pull_name.split(":")[0]
        if base_name in listed.stdout:
            print(f"  ✓  Model '{pull_name}' already downloaded.")
        else:
            print(f"  ↓  Pulling '{pull_name}' — this may take a few minutes...")
            subprocess.run(["ollama", "pull", pull_name], check=True)
            print(f"  ✓  Model '{pull_name}' downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"  ✗  Failed to pull model: {e}")
        sys.exit(1)


# =============================================================================
# STARTUP — choose backend + model
# =============================================================================

def select_backend() -> tuple:
    """
    Terminal prompt to choose between:
      [1] Groq API  (cloud, fast, requires GROQ_API_KEY)
      [2] Local Ollama model (runs on your GPU/CPU)

    Returns (client, model_name).
    """
    print("\n" + "=" * 50)
    print("  CV Parser Agent — Backend Selection")
    print("=" * 50)
    print("  [1] Groq API       — Cloud (fast, requires GROQ_API_KEY)")
    print("  [2] Local Ollama   — Runs on your machine (GPU/CPU)")
    print("=" * 50)

    while True:
        choice = input("  Enter 1 or 2 (default: 1): ").strip()

        if choice in ("", "1"):
            # ── Groq cloud ──────────────────────────────────────────────────
            api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if not api_key:
                api_key = input("  Enter your Groq API key: ").strip()
            if not api_key:
                print("  ✗  No API key provided. Exiting.")
                sys.exit(1)

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            model = "llama-3.3-70b-versatile"
            print(f"\n  ✓  Using Groq API  |  Model: {model}")
            print("=" * 50)
            return client, model

        elif choice == "2":
            # ── Local Ollama ─────────────────────────────────────────────────
            check_ollama_gpu()

            print("\n  Select local model (all support real tool calling):")
            print("  " + "-" * 56)
            for key, info in LOCAL_MODELS.items():
                print(f"  [{key}] {info['label']}")
            print("  " + "-" * 56)
            print("  NOTE: Selected model will be downloaded if not present.")
            print("  " + "-" * 56)

            while True:
                model_choice = input("  Enter 1, 2, or 3 (default: 1): ").strip()
                if model_choice in ("", "1"):
                    selected = LOCAL_MODELS["1"]
                    break
                elif model_choice in LOCAL_MODELS:
                    selected = LOCAL_MODELS[model_choice]
                    break
                else:
                    print("  Invalid choice. Enter 1, 2, or 3.")

            ensure_model_pulled(selected["pull_name"])
            os.environ.setdefault("OLLAMA_NUM_GPU", "1")

            client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1",
            )
            model = selected["name"]
            print(f"\n  ✓  Using Ollama local  |  Model: {model}")
            print("=" * 50)
            return client, model

        else:
            print("  Invalid choice. Enter 1 or 2.")


# =============================================================================
# JSON CLEANER — fixes common LLaMA / local model JSON mistakes
# =============================================================================

def clean_json(raw: str) -> str:
    """
    Fixes common JSON generation mistakes:
    - Qwen/DeepSeek <think>...</think> tags  →  stripped
    - "year": 2017-"2021"                    →  "year": "2021"
    - Trailing commas                        →  removed
    - Markdown code fences                   →  stripped
    """
    # Strip thinking tags (Qwen, DeepSeek)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Strip markdown fences
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    raw = raw.strip()

    # Fix: number-dash-string like 2017-"2021" → "2021"
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
# AGENT LOOP — fully agentic, unchanged
# =============================================================================

def parse_cv(client: OpenAI, model: str, file_path: str, job_description: str = "") -> dict:
    user_message = f"Parse this CV file: {file_path}"
    if job_description:
        user_message += f"\n\nTarget job description:\n{job_description}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    print(f"\n{'='*50}")
    print(f"Parsing CV: {file_path}")
    print(f"Model:      {model}")
    print(f"{'='*50}")

    max_iterations = 20
    for i in range(max_iterations):

        try:
            response = client.chat.completions.create(
                model=model,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            error_msg = str(e)
            # Hard stop on 404 — model name is wrong, retrying won't help
            if "404" in error_msg and "not found" in error_msg.lower():
                print(f"\n✗  Model not found: {model}")
                print("   Run 'ollama list' to see your installed models.")
                return {"error": f"Model '{model}' not found. Run 'ollama list' to verify."}
            print(f"  !! API error at iteration {i+1}: {error_msg[:120]}")
            messages.append({
                "role": "user",
                "content": (
                    f"Your last tool call failed with: {error_msg[:300]}. "
                    "Please fix the JSON and try again. "
                    "Remember: year must be a string like '2021', dates must be 'YYYY-MM'."
                )
            })
            continue

        msg = response.choices[0].message
        messages.append(msg)
        finish_reason = response.choices[0].finish_reason

        # LLM finished
        if finish_reason == "stop":
            print(f"\n✓ Done after {i+1} iterations")
            raw_content = msg.content or ""

            # Step 1: try parsing directly
            try:
                parsed = json.loads(clean_json(raw_content))
            except json.JSONDecodeError:
                parsed = None

            # Step 2: if parsed but is double-encoded {"raw_response": "{...}", "error": "..."}
            # this happens when the model returns our old error wrapper as its output
            if isinstance(parsed, dict) and "raw_response" in parsed:
                inner = parsed["raw_response"]
                try:
                    parsed = json.loads(clean_json(inner))
                except json.JSONDecodeError:
                    pass  # keep parsed as-is if inner is not valid JSON

            # Step 3: if still None, try the raw string one more time without cleaning
            if parsed is None:
                try:
                    parsed = json.loads(raw_content)
                except json.JSONDecodeError:
                    return {"raw_response": raw_content, "error": "Could not parse final JSON"}

            return parsed

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
    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_cv> [job_description]")
        sys.exit(1)

    cv_path = sys.argv[1]
    jd      = sys.argv[2] if len(sys.argv) > 2 else ""

    # ── Select backend + model interactively ──────────────────────────────
    client, MODEL = select_backend()

    # ── Run the fully agentic parser ──────────────────────────────────────
    result = parse_cv(client, MODEL, cv_path, jd)

    print("\n-- RESULT ------------------------------------------")
    print(json.dumps(result, indent=2))