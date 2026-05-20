"""
tools/reader.py — Retrieval Engineering (Skill 2)

Reads CV files (PDF / DOCX / TXT) and splits them into named sections
so the LLM receives focused, sized chunks — not one giant blob.

Features:
  - PyMuPDF for native PDF text extraction
  - Tesseract OCR fallback for scanned PDFs
  - python-docx for DOCX files
  - Heuristic section splitter (header / experience / education / skills / projects)
  - 30-second timeout guard on file reads
  - Path-traversal protection via os.path.realpath (Skill 4)
"""

from __future__ import annotations

import os
import re
import signal
import time
from pathlib import Path
from typing import Optional

# ── PDF ──────────────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ── OCR ──────────────────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── DOCX ─────────────────────────────────────────────────────────────────────
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

READ_TIMEOUT_SECONDS = 30

SECTION_PATTERNS: dict[str, list[str]] = {
    "summary": [
        r"\b(summary|profile|objective|about me|professional summary)\b",
    ],
    "experience": [
        r"\b(experience|work history|employment|career|professional background)\b",
    ],
    "education": [
        r"\b(education|academic|qualifications|degrees?|university|college)\b",
    ],
    "skills": [
        r"\b(skills?|technical skills?|competencies|technologies|expertise)\b",
    ],
    "projects": [
        r"\b(projects?|portfolio|personal projects?|side projects?)\b",
    ],
    "certifications": [
        r"\b(certifications?|certificates?|credentials?|courses?)\b",
    ],
}

VALID_SECTIONS = {
    "header", "summary", "experience", "education",
    "skills", "projects", "certifications", "full",
}


# ─────────────────────────────────────────────────────────────────────────────
# Timeout helper
# ─────────────────────────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("File read timed out after 30 seconds")


# ─────────────────────────────────────────────────────────────────────────────
# Core reader
# ─────────────────────────────────────────────────────────────────────────────

class CVReader:
    """
    Reads a CV file and exposes its text split into named sections.

    Usage:
        reader = CVReader("/uploads/alice_cv.pdf")
        reader.load()
        text = reader.get_section("experience")
        ocr_used = reader.ocr_used
    """

    def __init__(self, file_path: str, allowed_base: Optional[str] = None):
        """
        Parameters
        ----------
        file_path    : Absolute or relative path to the CV file.
        allowed_base : If set, raises ValueError if the resolved path is
                       outside this directory (path-traversal guard).
        """
        resolved = os.path.realpath(file_path)  # Skill 4 — path traversal
        if allowed_base:
            allowed_resolved = os.path.realpath(allowed_base)
            if not resolved.startswith(allowed_resolved):
                raise ValueError(
                    f"Path traversal blocked: {file_path!r} resolves outside "
                    f"allowed base {allowed_base!r}"
                )
        self.file_path = resolved
        self._raw_text: str = ""
        self._sections: dict[str, str] = {}
        self.ocr_used: bool = False
        self.warnings: list[str] = []

    # ── Public ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Extract raw text from file then split into sections."""
        ext = Path(self.file_path).suffix.lower()

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(READ_TIMEOUT_SECONDS)
        try:
            if ext == ".pdf":
                self._raw_text = self._read_pdf()
            elif ext in (".docx", ".doc"):
                self._raw_text = self._read_docx()
            elif ext in (".txt", ".md"):
                self._raw_text = Path(self.file_path).read_text(encoding="utf-8", errors="replace")
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        finally:
            signal.alarm(0)  # cancel alarm

        self._sections = self._split_sections(self._raw_text)

    def get_section(self, section_name: str) -> str:
        """
        Return the text for a named section.
        'full' returns the entire document.
        'header' returns the first ~400 chars (usually contact info).
        """
        if section_name not in VALID_SECTIONS:
            raise ValueError(
                f"Unknown section {section_name!r}. "
                f"Choose from: {sorted(VALID_SECTIONS)}"
            )
        if section_name == "full":
            return self._raw_text
        if section_name == "header":
            return self._raw_text[:600].strip()
        return self._sections.get(section_name, "").strip()

    def all_sections(self) -> dict[str, str]:
        return {k: v for k, v in self._sections.items() if v.strip()}

    # ── PDF reading ──────────────────────────────────────────────────────────

    def _read_pdf(self) -> str:
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF")

        doc = fitz.open(self.file_path)
        pages_text: list[str] = []

        for page in doc:
            text = page.get_text("text").strip()
            if text:
                pages_text.append(text)
            elif OCR_AVAILABLE:
                # Scanned page — rasterize and OCR
                self.ocr_used = True
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img).strip()
                if ocr_text:
                    pages_text.append(ocr_text)
                else:
                    self.warnings.append(
                        f"Page {page.number + 1}: OCR returned no text."
                    )
            else:
                self.warnings.append(
                    f"Page {page.number + 1}: no text extracted and OCR unavailable."
                )

        doc.close()
        return "\n\n".join(pages_text)

    # ── DOCX reading ─────────────────────────────────────────────────────────

    def _read_docx(self) -> str:
        if not DOCX_AVAILABLE:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")
        doc = Document(self.file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # ── Section splitter ─────────────────────────────────────────────────────

    def _split_sections(self, text: str) -> dict[str, str]:
        """
        Heuristically split raw text into labelled sections.

        Strategy:
          1. Walk lines looking for heading-like lines that match a section keyword.
          2. Everything between two headings belongs to the first heading.
          3. Unmatched content at the top goes to 'header'.
        """
        lines = text.splitlines()
        sections: dict[str, list[str]] = {s: [] for s in SECTION_PATTERNS}
        current_section: Optional[str] = None
        header_lines: list[str] = []
        in_header = True

        for line in lines:
            stripped = line.strip()
            matched_section = self._match_section_heading(stripped)

            if matched_section:
                in_header = False
                current_section = matched_section
            elif in_header:
                header_lines.append(line)
            elif current_section:
                sections[current_section].append(line)

        result: dict[str, str] = {}
        result["header"] = "\n".join(header_lines)
        for name, collected in sections.items():
            result[name] = "\n".join(collected)
        return result

    def _match_section_heading(self, line: str) -> Optional[str]:
        """Return section name if line looks like a heading for that section."""
        # Headings are usually short (< 60 chars) and all-caps or title-case
        if not line or len(line) > 80:
            return None
        lowered = line.lower()
        for section, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, lowered):
                    return section
        return None