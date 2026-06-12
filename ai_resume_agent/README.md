# AI Resume Screening & Interview Preparation Agent

This project implements an intelligent, multi-agent recruitment and coaching pipeline orchestrated with **LangGraph** and **Streamlit**. It automatically parses candidate resumes, analyzes skill gaps against a job description, calculates job compatibility matching, and generates simulated interview prep materials.

## 🏗️ Multi-Agent Architecture

```
User Uploads Resume & Job Description
                  |
                  v
+------------------------------------+
| 🤖 Node 1: Resume Parser Agent     |
| Extracts Skills, Edu, Exp, Projs   |
+------------------------------------+
                  |
                  v
+------------------------------------+
| 📊 Node 2: Skill Gap Analyzer      |
| Maps Matches & Missing Skills      |
+------------------------------------+
                  |
                  v
+------------------------------------+
| 💼 Node 3: Job Match Agent         |
| Match Score, Fit Verdict & Roles   |
+------------------------------------+
                  |
                  v
+------------------------------------+
| 🎤 Node 4: Interview Coach Agent   |
| Technical & HR Q&A Simulator       |
+------------------------------------+
                  |
                  v
       Final Dashboard Reports
```

1. **Resume Parser Agent**: Standardizes raw text from PDF/DOCX/TXT into a structured profile schema including contact details, experience history, projects, and tech stack.
2. **Skill Gap Analyzer Agent**: Compares candidate's skills against job requirements, identifying what qualifications align and what key skills are missing, generating recommendations to bridge gaps.
3. **Job Match Agent**: Calculates a compatibility percentage (0-100), provides a colored alignment verdict, summarizes major hiring strengths & weaknesses, and recommends alternative matching roles.
4. **Interview Coach Agent**: Simulates technical and HR interview scenarios by creating tailored questions (with best-approach strategies and detailed model answers).

---

## 🛠️ Setup & Local Execution

### 1. Requirements

Verify that your system has Python 3.9+ and pip installed. The project relies on the following libraries (included in `requirements.txt`):
* `streamlit`
* `langchain`
* `langgraph`
* `langchain-google-genai`
* `langchain-openai`
* `pypdf`
* `python-docx`
* `python-dotenv`

### 2. Installation

Install all dependencies using pip:
```bash
pip install -r requirements.txt
```

### 3. Launching the App

To start the Streamlit local server, navigate to the project folder and run:
```bash
streamlit run app.py
```

Streamlit will print the local server URLs:
* Local URL: `http://localhost:8501`
* Network URL: `http://192.168.x.x:8501`

---

## 💡 Portability & API Setup
You can either define your API Keys in a local `.env` file (e.g., `GEMINI_API_KEY` or `OPENAI_API_KEY`) or enter them directly inside the settings panel in the Streamlit Sidebar. 
An option is included to **Load Mock Resume & Job Desc** for testing the visual layouts and pipeline flows.
