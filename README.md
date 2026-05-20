# CV Parser Agent

An AI-powered CV parsing system that extracts structured candidate information from resumes using LLM-based processing and intelligent parsing workflows.

---

# Overview

The CV Parser Agent is designed to automate resume analysis by extracting important candidate details such as:

* Personal Information
* Skills
* Work Experience
* Education
* Projects
* Certifications
* Technical Stack

The system can serve as a foundational module for:

* Agentic Hiring Systems
* Candidate Screening Platforms
* Resume Matching Engines
* Talent Intelligence Systems
* HR Automation Pipelines

---

# Features

* AI-powered resume parsing
* Structured candidate information extraction
* Modular agent architecture
* Lightweight and extensible design
* Easy integration into hiring workflows
* Python-based implementation

---

# Project Structure

```bash
cv_parser/
│
├── agent.py             # Main CV parsing agent
├── tools.py             # Helper tools and utilities
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── venv/                # Virtual environment (ignored in git)
```

---

# Tech Stack

* Python
* LLM-based parsing workflows
* Virtual Environment (venv)
* Git & GitHub

---

# Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/sangeethasr2526-learning/cv_parser_agent.git
cd cv_parser_agent
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate Virtual Environment

### Windows

```bash
.\venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---
# CV Parser Agent 🤖

A fully agentic CV/Resume parser powered by Groq (LLaMA 3.3) that extracts structured data from PDF, DOCX, and image-based CVs.

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/sangeethasr2526-learning/cv_parser_agent.git
cd cv_parser_agent
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
- Get your free Groq API key from 👉 https://console.groq.com
- Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run
```bash
python agent.py "path/to/cv.pdf"
```

## Notes
- Never share or commit your `.env` file
- `.env` is already in `.gitignore` so it won't be pushed to GitHub


---

# Future Enhancements

* PDF and DOCX resume ingestion
* Vector embedding generation
* Skill matching engine
* Candidate ranking system
* Multi-agent orchestration
* API integration
* Frontend dashboard
* Behavioral analysis integration

---

# Use Cases

* Automated resume screening
* Intelligent candidate parsing
* Recruitment workflow automation
* AI-based hiring systems
* Talent analytics pipelines

---

# Contribution

Contributions, improvements, and feature suggestions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request



# Author

Sangeetha Sundar

GitHub: [https://github.com/sangeethasr2526-learning](https://github.com/sangeethasr2526-learning)
