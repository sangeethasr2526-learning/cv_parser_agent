Agentic Hiring System
An end-to-end AI-powered recruitment pipeline combining LLM Agents, ML Models, and Deep Learning techniques.

📌 Overview
The Agentic Hiring System is a multi-layered artificial intelligence architecture designed to automate candidate screening, conduct intelligent interviews, and predict hiring outcomes. The system integrates Large Language Model (LLM) agents, Machine Learning (ML) models, Deep Learning (DL) techniques, and orchestration logic to create an end-to-end recruitment pipeline. The architecture is modular, allowing each layer to handle a specific task such as parsing resumes, generating interview questions, evaluating responses, and making final hiring decisions. This layered approach improves scalability, transparency, and decision accuracy.

🗂️ Project Structure
agentic-hiring-system/
│
├── phase1-screening/                  # CV & JD Screening + Matching
│   ├── cv_parser/                     # LLM-based CV extraction
│   ├── cv_feature_builder/            # ML feature engineering
│   ├── jd_analyzer/                   # LLM-based JD extraction
│   ├── jd_feature_builder/            # ML weighted feature vectors
│   └── matching_engine/               # CV-JD similarity matching
│
├── phase2-interview/                  # Interview Orchestration + Evaluation
│   ├── hiring_manager_agent/          # Question generation (LLM)
│   ├── interview_orchestrator/        # Interview flow control
│   ├── answer_evaluator/              # Response evaluation (LLM + ML)
│   ├── behavior_analysis/             # Audio/Video analysis (DL)
│   └── candidate_state_store/         # Feature store for all scores
│
├── phase3-prediction/                 # Predictive Analytics + Decision
│   ├── cv_outcome_predictor/          # Pre-interview prediction (ML)
│   ├── predictive_engine/             # Final scoring engine
│   ├── decision_layer/                # Hire / Maybe / Reject output
│   └── model_training/                # Feedback loop & retraining
│
├── README.md
├── requirements.txt
└── .gitignore
🧱 Architecture Phases
🔵 Phase 1 — CV & JD Screening + Matching
Layer	Component	Type
Layer 0	CV / Resume	Data/Input
Layer 0	Job Description	Data/Input
Layer 1	CV Parser Agent	LLM Agent
Layer 1	CV Feature Builder	ML/DL Model
Layer 1	JD Analyzer Agent	LLM Agent
Layer 1	JD Feature Builder	ML/DL Model
Layer 2	CV-JD Matching Engine	ML/DL Model
🟢 Phase 2 — Interview Orchestration + Response Evaluation
Layer	Component	Type
Layer 3	Hiring Manager Agent	LLM Agent
Layer 3	Interview Orchestrator	Function/Orchestration
Layer 3	Interview Responses (Text/Audio/Video)	Data/Input
Layer 3	Interview Responses (Audio/Video)	Function/Orchestration
Layer 4	Answer Evaluator Agent	LLM Agent
Layer 4	Behavior Analysis Engine	ML/DL Model
Layer 5	Candidate State Store	Memory/State Store
🟡 Phase 3 — Predictive Analytics + Decision + Feedback
Layer	Component	Type
Layer 6	Predictive Decision Engine	Prediction/Decision
Layer 7	Decision Layer	Prediction/Decision
Layer 8	CV Outcome Predictor	ML/DL Model
Layer 9	Model Training Pipeline	ML/DL Model
👥 Team & Branch Structure
Member	Branch	Artifact
Sangeetha	cv-parser	CV Parser Agent
Varshita	cv-parser	CV Parser Agent
Anuragine	cv-feature-builder	CV Feature Builder
Yashwanth	jd-analyzer	JD Analyzer Agent
Nivedhitha	jd-feature-builder	JD Feature Builder
Meghana	matching-engine	CV-JD Matching Engine
🔧 Tech Stack
Component Type	Technologies
LLM Agents	GPT-4 / Claude / LangChain
ML Models	Scikit-learn / XGBoost
DL Models	PyTorch / TensorFlow
Feature Engineering	HuggingFace / BERT Embeddings
Orchestration	FastAPI / LangGraph
State Store	Redis / PostgreSQL
🚀 Getting Started
# 1. Clone the repository
git clone https://github.com/Sangeetha-Sundar1608/agentic-hiring-system.git

# 2. Navigate to project
cd agentic-hiring-system

# 3. Switch to your branch
git checkout your-branch-name

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start working!
📌 Git Workflow
# Pull latest updates
git pull origin main

# After making changes
git add .
git commit -m "Your clear commit message"
git push origin your-branch-name-