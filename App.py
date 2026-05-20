"""
app.py — Streamlit Demo UI (Skill 7 — Product Thinking)

Features:
  - File uploader (PDF / DOCX / TXT)
  - Target role input
  - Live progress messages
  - Structured results: contact, experience, skills, education
  - Red flags with evidence badges
  - Employment gaps timeline
  - Role relevance score + summary
  - Confidence indicators per field
  - Raw JSON export
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Parser Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .flag-high   { background:#ff4b4b22; border-left:4px solid #ff4b4b; padding:10px; border-radius:4px; margin:6px 0; }
    .flag-medium { background:#ffa50022; border-left:4px solid #ffa500; padding:10px; border-radius:4px; margin:6px 0; }
    .flag-low    { background:#00c80022; border-left:4px solid #00c800; padding:10px; border-radius:4px; margin:6px 0; }
    .evidence    { font-size:0.82em; color:#aaa; font-style:italic; margin-top:4px; }
    .score-bar   { height:12px; border-radius:6px; background:linear-gradient(90deg,#ff4b4b,#ffa500,#00c800); }
    .conf-high   { color:#00c800; font-size:0.75em; }
    .conf-medium { color:#ffa500; font-size:0.75em; }
    .conf-low    { color:#ff4b4b; font-size:0.75em; }
    .gap-chip    { display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.8em; margin:2px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ CV Parser Agent")
    st.caption("Powered by Groq · Instructor · Pydantic")

    st.markdown("---")
    st.subheader("Upload CV")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        help="Scanned PDFs are supported via OCR.",
    )

    target_role = st.text_input(
        "Target Role (optional)",
        placeholder="e.g. Senior ML Engineer",
        help="Enables role relevance scoring and gap analysis.",
    )

    run_btn = st.button("🚀 Parse CV", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Skills built in:")
    for skill in [
        "1 · Tool Contracts (Pydantic + Instructor)",
        "2 · Retrieval (PyMuPDF + OCR)",
        "3 · Reliability (Tenacity + validation)",
        "4 · Security (LLM Guard)",
        "5 · Evaluation (DeepEval)",
        "6 · Observability (Langfuse)",
        "7 · Product Thinking (red flags + UX)",
    ]:
        st.caption(f"✅ {skill}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔍 CV Parser Agent")
st.markdown(
    "Upload a CV and get structured extraction, pattern analysis, "
    "red flags with evidence, and role fit scoring."
)

if not run_btn or not uploaded_file:
    st.info("Upload a CV and click **Parse CV** to begin.")
    st.stop()

# ── Check API key ─────────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not set. Add it to your .env file.")
    st.stop()

# ── Save upload to temp file ──────────────────────────────────────────────────
suffix = Path(uploaded_file.name).suffix
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

# ── Run agent ─────────────────────────────────────────────────────────────────
from agent import CVParserAgent  # noqa: E402 (imported here to avoid slow startup)
from models import RiskLevel     # noqa: E402

progress = st.progress(0, text="Starting…")
status   = st.empty()

def update(pct, msg):
    progress.progress(pct, text=msg)
    status.caption(msg)

try:
    update(10, "📄 Reading file…")
    agent = CVParserAgent()

    update(20, "🔒 Security scan…")
    update(35, "📇 Extracting contact info…")
    update(50, "💼 Extracting experience…")
    update(65, "🎓 Extracting education & skills…")
    update(80, "🔴 Detecting patterns & red flags…")
    update(90, "🎯 Assessing role fit…")

    result = agent.parse(tmp_path, target_role=target_role or None)
    update(100, "✅ Done!")

except Exception as exc:
    st.error(f"❌ Parse failed: {exc}")
    st.stop()

finally:
    import os as _os
    try:
        _os.unlink(tmp_path)
    except Exception:
        pass

progress.empty()
status.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

# ── OCR / warnings banner ────────────────────────────────────────────────────
if result.ocr_used:
    st.warning("🔬 OCR was used on one or more pages. Accuracy may vary.")
if result.parse_warnings:
    with st.expander("⚠️ Parse warnings"):
        for w in result.parse_warnings:
            st.caption(w)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "👤 Contact", "💼 Experience", "🎓 Education",
    "🛠 Skills", "📦 Projects", "🚨 Red Flags",
    "📊 Role Fit", "📋 Raw JSON",
])

# ── 1. Contact ───────────────────────────────────────────────────────────────
with tabs[0]:
    c = result.contact
    col1, col2 = st.columns(2)
    col1.metric("Name", c.name)
    col2.metric("Location", c.location or "—")
    col1.metric("Email", c.email or "—")
    col2.metric("Phone", c.phone or "—")
    if c.linkedin:
        st.markdown(f"🔗 [LinkedIn]({c.linkedin})")
    if c.github:
        st.markdown(f"🐙 [GitHub]({c.github})")
    st.caption(f"Confidence: {c.confidence.value}")

    if result.summary:
        st.markdown("---")
        st.subheader("Candidate Summary")
        st.write(result.summary)

# ── 2. Experience ────────────────────────────────────────────────────────────
with tabs[1]:
    if not result.experience:
        st.info("No experience extracted.")
    for role in result.experience:
        tenure = f"{role.tenure_months} months" if role.tenure_months else ""
        with st.expander(f"**{role.title}** @ {role.company}  ·  {role.dates.start} → {role.dates.end}  ·  {tenure}"):
            for resp in role.responsibilities:
                st.markdown(f"• {resp}")
            conf_class = f"conf-{role.confidence.value}"
            st.markdown(
                f'<span class="{conf_class}">Confidence: {role.confidence.value}</span>',
                unsafe_allow_html=True,
            )

# ── 3. Education ─────────────────────────────────────────────────────────────
with tabs[2]:
    if not result.education:
        st.info("No education extracted.")
    for edu in result.education:
        dates = ""
        if edu.dates:
            dates = f" · {edu.dates.start} → {edu.dates.end}"
        st.markdown(f"**{edu.degree}** — {edu.institution}{dates}")
        if edu.field_of_study:
            st.caption(f"Field: {edu.field_of_study}")
    if result.certifications:
        st.markdown("---")
        st.subheader("Certifications")
        for cert in result.certifications:
            st.markdown(f"🏅 {cert}")

# ── 4. Skills ────────────────────────────────────────────────────────────────
with tabs[3]:
    if not result.skills:
        st.info("No skills extracted.")
    for cat in result.skills:
        st.subheader(cat.category)
        st.markdown(" · ".join(f"`{s}`" for s in cat.skills))

# ── 5. Projects ──────────────────────────────────────────────────────────────
with tabs[4]:
    if not result.projects:
        st.info("No projects extracted.")
    for proj in result.projects:
        with st.expander(f"**{proj.name}**"):
            st.write(proj.description)
            if proj.technologies:
                st.markdown("Tech: " + ", ".join(f"`{t}`" for t in proj.technologies))
            if proj.url:
                st.markdown(f"🔗 [{proj.url}]({proj.url})")

# ── 6. Red Flags ─────────────────────────────────────────────────────────────
with tabs[5]:
    if not result.red_flags and not result.employment_gaps:
        st.success("✅ No significant red flags or gaps detected.")

    if result.employment_gaps:
        st.subheader("Employment Gaps")
        for gap in result.employment_gaps:
            color = {"high": "#ff4b4b", "medium": "#ffa500", "low": "#00c800"}[gap.risk.value]
            st.markdown(
                f'<span class="gap-chip" style="background:{color}33;border:1px solid {color}">'
                f'⏱ {gap.gap_months} months gap  ({gap.start_of_gap} → {gap.end_of_gap})  '
                f'between <b>{gap.after_company}</b> and <b>{gap.before_company}</b>'
                f'</span>',
                unsafe_allow_html=True,
            )
        st.markdown("")

    if result.red_flags:
        st.subheader("Red Flags")
        for flag in result.red_flags:
            css_class = f"flag-{flag.risk.value}"
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[flag.risk.value]
            st.markdown(
                f'<div class="{css_class}">'
                f'{icon} <b>{flag.flag_type.replace("_"," ").title()}</b>: {flag.description}'
                f'<div class="evidence">📎 Evidence: {flag.evidence}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── 7. Role Fit ───────────────────────────────────────────────────────────────
with tabs[6]:
    if not result.role_assessment:
        st.info("Provide a **Target Role** to enable role fit scoring.")
    else:
        ra = result.role_assessment
        score = ra.relevance_score

        col1, col2 = st.columns([1, 2])
        col1.metric("Relevance Score", f"{score}/100")
        col2.markdown(f"**Target Role:** {ra.target_role}")

        # Colour bar
        bar_color = "#ff4b4b" if score < 40 else "#ffa500" if score < 70 else "#00c800"
        st.markdown(
            f'<div style="background:#1e1e2e;border-radius:8px;padding:4px;">'
            f'<div style="width:{score}%;height:16px;border-radius:6px;background:{bar_color};transition:width 1s;"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("Recruiter Summary")
        st.write(ra.summary)

        col_r, col_g = st.columns(2)
        with col_r:
            st.subheader("✅ Relevant Experience")
            for item in ra.relevant_experience:
                st.markdown(f"• {item}")
        with col_g:
            st.subheader("❌ Gaps vs Role")
            for item in ra.gaps:
                st.markdown(f"• {item}")

# ── 8. Raw JSON ───────────────────────────────────────────────────────────────
with tabs[7]:
    json_str = result.model_dump_json(indent=2)
    st.download_button(
        "⬇️ Download JSON",
        data=json_str,
        file_name="parsed_cv.json",
        mime="application/json",
    )
    st.code(json_str, language="json")