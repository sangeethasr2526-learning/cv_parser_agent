# CV Parser Agent

A production-grade agentic CV parser built with **7 agent skills**: tool contract design, retrieval engineering, reliability, security, evaluation, observability, and product thinking.

## What it does

- Extracts structured data: contact, experience, education, skills, projects, certifications
- Detects **job-hopping patterns** (roles < 12 months, serial hops)
- Identifies **employment gaps** (>3 months) with risk scoring
- Flags **irrelevant experience** vs the target role
- Produces **evidence for every red flag** (never a flag without a quote)
- **Role relevance scoring** (0–100) with recruiter-grade summary
- **Confidence scores** per extracted field
- OCR fallback for scanned PDFs
- Prompt-injection protection (LLM Guard)
- Live tracing in Langfuse dashboard
- DeepEval test suite

---

## Project structure

```
cv_parser/
├── agent.py                  # Main orchestrator
├── models.py                 # All Pydantic schemas (Skill 1)
├── app.py                    # Streamlit UI (Skill 7)
├── requirements.txt
├── .env.example
│
├── tools/
│   ├── reader.py             # File reading + section splitting (Skill 2)
│   ├── security.py           # LLM Guard + path-traversal guard (Skill 4)
│   ├── observability.py      # Langfuse tracing (Skill 6)
│   └── patterns.py           # Gap/red-flag detection (Skill 7)
│
├── eval/
│   └── test_agent.py         # DeepEval test suite (Skill 5)
│
└── tests/
    └── test_cvs/
        ├── alice_strong_match.txt
        ├── marcus_job_hopper.txt
        └── priya_career_changer.txt
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <your-repo>
cd cv_parser
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tesseract** (OCR) requires a system install:
> - macOS: `brew install tesseract`
> - Ubuntu: `sudo apt install tesseract-ocr`
> - Windows: Download from https://github.com/tesseract-ocr/tesseract

### 3. Set environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
GROQ_API_KEY=gsk_...           # Get free at https://console.groq.com

# Optional — enables live Langfuse tracing dashboard
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Getting a Groq API key (free):**
1. Go to https://console.groq.com
2. Sign up → API Keys → Create key
3. Paste into `.env`

**Getting Langfuse keys (free, 50k traces/month):**
1. Go to https://cloud.langfuse.com
2. Create project → Settings → API Keys

---

## Running the Streamlit UI

```bash
streamlit run app.py
```

Open http://localhost:8501

1. Upload a PDF, DOCX, or TXT CV
2. (Optional) Enter the target role: e.g. `Senior ML Engineer`
3. Click **Parse CV**
4. Explore tabs: Contact · Experience · Education · Skills · Projects · Red Flags · Role Fit · Raw JSON

---

## Running via Python directly

```python
from agent import CVParserAgent

agent = CVParserAgent()
result = agent.parse(
    "path/to/cv.pdf",
    target_role="Senior Machine Learning Engineer",
)

print(result.contact.name)
print(result.role_assessment.relevance_score)
for flag in result.red_flags:
    print(f"[{flag.risk.value}] {flag.flag_type}: {flag.evidence}")
```

---

## Running evaluations

```bash
cd cv_parser
python -m pytest eval/test_agent.py -v
```

Expected output:
```
PASSED  TestAliceExtraction::test_contact_name_extracted
PASSED  TestAliceExtraction::test_experience_count
PASSED  TestAliceRelevance::test_strong_candidate_scores_high
PASSED  TestMarcusJobHopping::test_job_hopping_flag_raised
PASSED  TestMarcusJobHopping::test_all_flags_have_evidence
PASSED  TestPriyaGapDetection::test_employment_gap_detected
...
```

The DeepEval tests (answer relevancy + faithfulness) are skipped unless `deepeval` is installed and an OpenAI key is set (they use GPT-4o-mini as the eval model).

---

## The 7 Agent Skills — where each lives

| # | Skill | File(s) |
|---|-------|---------|
| 1 | Tool Contract Design | `models.py` — Pydantic v2 schemas auto-generate JSON Schema; `agent.py` — Instructor enforces them |
| 2 | Retrieval Engineering | `tools/reader.py` — PyMuPDF + Tesseract OCR + section splitter |
| 3 | Reliability | `agent.py` — Tenacity retries + Instructor auto-retry + Pydantic validation |
| 4 | Security | `tools/security.py` — LLM Guard injection scan; `tools/reader.py` — `os.path.realpath` path guard |
| 5 | Evaluation | `eval/test_agent.py` — DeepEval + pytest; 3 hand-crafted test CVs |
| 6 | Observability | `tools/observability.py` — Langfuse spans per tool call with input/output/duration |
| 7 | Product Thinking | `app.py` — Streamlit UI; evidence on every flag; confidence per field; relevance score bar |

---

## Key design decisions

**Why section splitting?**  
Long CVs (3+ pages) can exceed context windows. Splitting into sections (header, experience, education, skills) lets each tool call focus on exactly the right text, improving accuracy and reducing hallucination.

**Why is pattern detection (gaps, job-hopping) in pure Python, not the LLM?**  
Date arithmetic is deterministic. The LLM is unreliable at "left after 4 months" calculations. Pure Python on validated `YYYY-MM` dates is 100% accurate and adds zero latency.

**Why does every red flag require evidence?**  
A flag without a quote is useless to a recruiter. The Pydantic schema forces the LLM to produce a citation before the response validates.

**Why Instructor over raw JSON mode?**  
Instructor handles retries when the LLM produces malformed JSON, validates output against the Pydantic model, and surfaces the validation error back to the LLM for self-correction — all automatically.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY not set` | Add key to `.env` and restart |
| `PyMuPDF not found` | `pip install PyMuPDF` |
| OCR not working | Install Tesseract system package (see Setup step 2) |
| Langfuse not tracing | Check keys in `.env`; tracing silently degrades to no-op if keys are missing |
| `python-docx` error on DOCX | `pip install python-docx` |
| Tests failing on date extraction | The LLM occasionally misses ambiguous dates — re-run; Tenacity will retry |