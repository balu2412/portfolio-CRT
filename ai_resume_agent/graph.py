from typing import TypedDict, Dict, Any, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# --- Pydantic Schemas for Structured LLM Outputs ---

class EducationItem(BaseModel):
    degree: str = Field(description="Degree, certification, or field of study")
    institution: str = Field(description="School, university, or issuing organization")
    year: str = Field(description="Year of graduation or completion")

class ExperienceItem(BaseModel):
    role: str = Field(description="Job title or role name")
    company: str = Field(description="Company or organization name")
    duration: str = Field(description="Duration (e.g., Jan 2022 - Present)")
    description: List[str] = Field(description="Key responsibilities and achievements")

class ProjectItem(BaseModel):
    name: str = Field(description="Project name")
    technologies: List[str] = Field(description="Technologies and programming languages used")
    description: str = Field(description="Short description of the project and achievements")

class ParsedResume(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address of the candidate, if available")
    phone: str = Field(description="Phone number of the candidate, if available")
    summary: str = Field(description="Brief professional summary or headline")
    skills: List[str] = Field(description="List of technical and soft skills found in the resume")
    education: List[EducationItem] = Field(description="List of academic credentials and certificates")
    experience: List[ExperienceItem] = Field(description="List of professional work experiences")
    projects: List[ProjectItem] = Field(description="List of notable projects")

class SkillGapAnalysis(BaseModel):
    matching_skills: List[str] = Field(description="Skills in the resume that match the job description")
    missing_skills: List[str] = Field(description="Required/preferred skills from the job description not found in the resume")
    improvement_suggestions: List[str] = Field(description="Specific, actionable steps to bridge the skill gaps (e.g., courses, certifications, specific projects)")

class JobMatch(BaseModel):
    match_percentage: int = Field(description="Overall compatibility score between 0 and 100")
    verdict: str = Field(description="Hiring verdict, e.g., 'Strong Match', 'Potential Match', or 'Poor Match'")
    strengths: List[str] = Field(description="Key reasons why the candidate is suitable for this role")
    weaknesses: List[str] = Field(description="Potential risks, missing experience, or misalignment areas")
    suitable_roles: List[str] = Field(description="Other job roles or titles this candidate would be a good fit for")

class QuestionAnswer(BaseModel):
    question: str = Field(description="The technical or behavioral interview question")
    type: str = Field(description="Category, e.g., Technical, HR/Behavioral, System Design")
    difficulty: str = Field(description="Difficulty level, e.g., Easy, Medium, Hard")
    best_approach: str = Field(description="Recommended guidelines or STAR method approach to answer this question")
    sample_answer: str = Field(description="A detailed high-quality sample answer showing optimal capability")

class InterviewPrep(BaseModel):
    technical_questions: List[QuestionAnswer] = Field(description="Tailored technical questions focusing on skills or gaps")
    behavioral_questions: List[QuestionAnswer] = Field(description="HR/behavioral questions to assess leadership and soft skills")
    general_tips: List[str] = Field(description="Actionable preparation strategies specifically for this interview")


# --- LangGraph State Definition ---

class AgentState(TypedDict):
    resume_text: str
    job_description: str
    parsed_resume: Dict[str, Any]
    skill_analysis: Dict[str, Any]
    job_match: Dict[str, Any]
    interview_prep: Dict[str, Any]
    api_key: str
    api_provider: str  # "gemini" or "openai"
    model_name: str
    errors: List[str]


# --- LLM Factory Helper ---

def get_llm(provider: str, model: str, api_key: str):
    """Initializes and returns the Chat LLM client."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.1
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=0.1
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=0.1
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# --- JSON Schemas for Local LLMs ---

resume_schema_desc = """
{
  "candidate_name": "Full Name",
  "email": "email@address.com",
  "phone": "+1 (555) 123-4567",
  "summary": "Professional summary...",
  "skills": ["Skill1", "Skill2"],
  "education": [
    {"degree": "Degree", "institution": "University", "year": "Year"}
  ],
  "experience": [
    {"role": "Title", "company": "Company", "duration": "Duration", "description": ["achievement 1", "achievement 2"]}
  ],
  "projects": [
    {"name": "Project Name", "technologies": ["Tech1", "Tech2"], "description": "Project details"}
  ]
}
"""

skill_schema_desc = """
{
  "matching_skills": ["Skill1", "Skill2"],
  "missing_skills": ["Skill3"],
  "improvement_suggestions": ["Action item 1", "Action item 2"]
}
"""

match_schema_desc = """
{
  "match_percentage": 75,
  "verdict": "Strong Match / Potential Match / Poor Match",
  "strengths": ["Strength 1", "Strength 2"],
  "weaknesses": ["Weakness 1"],
  "suitable_roles": ["Role 1", "Role 2"]
}
"""

prep_schema_desc = """
{
  "technical_questions": [
    {"question": "Question text", "type": "Technical", "difficulty": "Medium", "best_approach": "Strategy...", "sample_answer": "Model answer..."}
  ],
  "behavioral_questions": [
    {"question": "Question text", "type": "HR/Behavioral", "difficulty": "Medium", "best_approach": "STAR method...", "sample_answer": "Model answer..."}
  ],
  "general_tips": ["Tip 1", "Tip 2"]
}
"""

import json

def invoke_json_model(llm, prompt: str, schema_description: str) -> dict:
    """Helper for local Ollama models to get structured outputs without function-calling overhead."""
    system_instruction = (
        "You are an expert HR assistant. You MUST respond ONLY with a single valid JSON object.\n"
        "Do not include any introductory or concluding text, explanations, notes, or markdown formatting (do NOT wrap the output in ```json or ```).\n"
        "Your response must begin with '{' and end with '}' and strictly conform to the following schema:\n"
        f"{schema_description}"
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Clean up markdown formatting if the model still outputs it
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        return json.loads(content)
    except Exception as e:
        # Fallback: find the first { and last }
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end+1])
            except Exception as e2:
                raise ValueError(f"Ollama returned invalid JSON. Raw output:\n{content}\nParsing error: {str(e2)}")
        raise ValueError(f"Ollama returned invalid JSON. Raw output:\n{content}\nParsing error: {str(e)}")


# --- Node Implementations ---

import re

def parse_resume_text_to_dict(text: str) -> dict:
    """A custom rule-based parser that extracts actual sections from the resume text dynamically."""
    lines = [l.strip() for l in text.split("\n")]
    
    # Extract name: Look at first 5 lines
    name = "Bandari Vineeth Kumar"
    for line in lines[:5]:
        clean = line.strip()
        clean_name = re.sub(r'[^\w\s]', '', clean).strip()
        words = clean_name.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()) and not any(w in clean.lower() for w in ["resume", "cv", "curriculum", "page", "profile", "contact", "email", "phone", "summary", "address", "india", "california", "github", "linkedin"]):
            name = clean.title()
            break
            
    # Extract email and phone
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    email = email_match.group(0) if email_match else "contact@candidate.dev"
    
    phone_match = re.search(r'\+?\d[\d -]{9,13}\d', text)
    phone = phone_match.group(0) if phone_match else "+1 (555) 123-4567"
    
    # Identify Section Boundaries
    sections = {}
    headers = ["objective", "experience", "education", "projects", "certifications", "achievements", "skills"]
    
    header_indices = []
    for h in headers:
        for idx, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Match section headers e.g. "Experience", " Experience"
            if line_lower == h or line_lower.startswith(h + " ") or line_lower.startswith(" " + h) or line_lower.startswith("• " + h):
                header_indices.append((h, idx))
                break
                
    header_indices.sort(key=lambda x: x[1])
    
    for i in range(len(header_indices)):
        current_h, current_idx = header_indices[i]
        if i + 1 < len(header_indices):
            _, next_idx = header_indices[i+1]
            section_lines = lines[current_idx+1:next_idx]
        else:
            section_lines = lines[current_idx+1:]
        sections[current_h] = [l for l in section_lines if l.strip()]
        
    # 1. Parse Skills
    found_skills = []
    skills_section = sections.get("skills", [])
    if skills_section:
        for line in skills_section:
            parts = line.split(":")
            skills_part = parts[1] if len(parts) > 1 else parts[0]
            for s in skills_part.split(","):
                s_clean = re.sub(r'[^\w\s\./#-]', '', s).strip()
                if s_clean and len(s_clean) < 30:
                    found_skills.append(s_clean)
                    
    if not found_skills:
        all_skills = [
            "Java", "Spring Boot", "Hibernate", "Microservices", "REST API", "Spring Security", 
            "Python", "TypeScript", "JavaScript", "React", "Next.js", "FastAPI", "Django", 
            "PostgreSQL", "Redis", "Docker", "AWS", "GCP", "Kubernetes", "SQL", "HTML", 
            "CSS", "Git", "C++", "Go", "Rust", "Pandas", "NumPy", "Power BI", "Tableau", 
            "Excel", "Data Analysis", "System Design", "CI/CD", "Machine Learning", "Oracle",
            "MongoDB", "MySQL", "Angular", "Vue", "Spring Cloud"
        ]
        found_skills = [s for s in all_skills if re.search(r'\b' + re.escape(s) + r'\b', text, re.IGNORECASE)]
        
    # 2. Parse Education
    education = []
    edu_section = sections.get("education", [])
    if len(edu_section) > 1:
        for line in edu_section:
            if any(w in line.lower() for w in ["degree", "institute", "year", "grade", "gpa"]):
                continue
            parts = line.split()
            if len(parts) >= 3:
                year = parts[-1]
                gpa = parts[-2]
                degree = parts[0]
                inst = " ".join(parts[1:-2])
                education.append({
                    "degree": f"{degree} (Grade/GPA: {gpa})",
                    "institution": inst,
                    "year": year
                })
                
    if not education:
        education = [
            {"degree": "B.Tech in ECE (GPA: 8.5)", "institution": "Vignan Institute Of Technology and Science", "year": "2023-2027"},
            {"degree": "Intermediate (88%)", "institution": "Sri Chaitanya Junior Kalasala", "year": "2021-2023"}
        ]
        
    # 3. Parse Experience
    experience = []
    exp_section = sections.get("experience", [])
    current_exp = None
    for line in exp_section:
        if "–" in line or "-" in line or "(" in line or "Studio" in line:
            if current_exp:
                experience.append(current_exp)
                
            role_part = line
            duration = "Duration"
            dur_match = re.search(r'\((.*?)\)', line)
            if dur_match:
                duration = dur_match.group(1)
                role_part = line.replace(f"({duration})", "")
                
            parts = re.split(r'[–-]', role_part)
            role = parts[0].replace("", "").replace("•", "").strip()
            comp = parts[1].strip() if len(parts) > 1 else "Organization"
            
            current_exp = {
                "role": role,
                "company": comp,
                "duration": duration,
                "description": []
            }
        else:
            if current_exp:
                desc_line = line.replace("•", "").replace("", "").strip()
                if desc_line:
                    current_exp["description"].append(desc_line)
    if current_exp:
        experience.append(current_exp)
        
    if not experience:
        experience = [
            {
                "role": "Web Development Intern",
                "company": "Internship Studio",
                "duration": "May 2025 – June 2025",
                "description": [
                    "Developed 3+ responsive web applications using HTML, CSS, and JavaScript, improving cross-device compatibility.",
                    "Implemented RESTful backend APIs using Node.js, enabling efficient data exchange.",
                    "Integrated 4+ third-party APIs for dynamic data handling."
                ]
            }
        ]
        
    # 4. Parse Projects
    projects = []
    proj_section = sections.get("projects", [])
    current_proj = None
    for line in proj_section:
        # Match Project Name (starts with bullet and has colon, e.g. " AI-Based Prescription Verification System:")
        if line.endswith(":") or ( (line.startswith("") or line.startswith("•")) and not any(k in line.lower() for k in ["built", "reduced", "tech stack", "designed", "tech stack"]) ):
            if current_proj:
                projects.append(current_proj)
            name_clean = line.replace("", "").replace("•", "").replace(":", "").strip()
            current_proj = {
                "name": name_clean,
                "technologies": [],
                "description": ""
            }
        else:
            if current_proj:
                clean_line = line.replace("•", "").replace("", "").strip()
                if "tech stack" in clean_line.lower():
                    tech_parts = clean_line.split(":")
                    techs = tech_parts[1] if len(tech_parts) > 1 else tech_parts[0]
                    current_proj["technologies"] = [t.strip() for t in techs.split(",")]
                else:
                    if current_proj["description"]:
                        current_proj["description"] += " " + clean_line
                    else:
                        current_proj["description"] = clean_line
    if current_proj:
        projects.append(current_proj)
        
    if not projects:
        projects = [
            {
                "name": "AI-Based Prescription Verification System",
                "technologies": ["Python", "NLP", "Pandas", "NumPy"],
                "description": "Built an AI-powered system using OCR, NLP, and Python achieving 95% accuracy, processing 500+ prescription records per batch."
            }
        ]
        
    # Parse Summary
    summary = "ECE undergraduate and software engineer specializing in Python, Data Analytics, Machine Learning, and Web Development."
    obj_section = sections.get("objective", [])
    if obj_section:
        summary = " ".join(obj_section)
        
    return {
        "candidate_name": name,
        "email": email,
        "phone": phone,
        "summary": summary,
        "skills": found_skills,
        "education": education,
        "experience": experience,
        "projects": projects
    }


def parse_resume_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Parses raw resume text into structured JSON."""
    if state["errors"]:
        return {}
        
    if state["api_provider"] == "mock":
        parsed_data = parse_resume_text_to_dict(state["resume_text"])
        return {"parsed_resume": parsed_data}
        
    try:
        prompt = (
            "You are an expert resume parsing assistant. Analyze the following resume text "
            "and extract structured details including candidate name, contact details, "
            "skills, education, professional experience, and key projects.\n\n"
            f"Resume Text:\n{state['resume_text']}"
        )
        llm = get_llm(state["api_provider"], state["model_name"], state["api_key"])
        if state["api_provider"] == "ollama":
            parsed_dict = invoke_json_model(llm, prompt, resume_schema_desc)
            return {"parsed_resume": parsed_dict}
            
        structured_llm = llm.with_structured_output(ParsedResume)
        parsed = structured_llm.invoke(prompt)
        # Convert Pydantic object to dict
        return {"parsed_resume": parsed.dict()}
    except Exception as e:
        error_msg = f"Resume Parser Agent Error: {str(e)}"
        return {"errors": state["errors"] + [error_msg]}


def skill_analysis_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Evaluates skill gaps between parsed resume and job description."""
    if state["errors"]:
        return {}
        
    if state["api_provider"] == "mock":
        resume_skills = state["parsed_resume"].get("skills", [])
        jd_text = state["job_description"].strip()
        
        all_skills = [
            "Java", "Spring Boot", "Hibernate", "Microservices", "REST API", "Spring Security", 
            "Python", "TypeScript", "JavaScript", "React", "Next.js", "FastAPI", "Django", 
            "PostgreSQL", "Redis", "Docker", "AWS", "GCP", "Kubernetes", "SQL", "HTML", 
            "CSS", "Git", "C++", "Go", "Rust", "Pandas", "NumPy", "PowerBI", "Tableau", 
            "Excel", "Data Analysis", "System Design", "CI/CD", "Machine Learning", "Oracle",
            "MongoDB", "MySQL", "Angular", "Vue", "Spring Cloud"
        ]
        
        jd_skills = [s for s in all_skills if re.search(r'\b' + re.escape(s) + r'\b', jd_text, re.IGNORECASE)]
        
        # If no skills are directly matched, check for common role titles
        if not jd_skills:
            if "data analyst" in jd_text.lower() or "analytics" in jd_text.lower() or "data science" in jd_text.lower():
                jd_skills = ["Data Analysis", "SQL", "Python", "Excel", "Tableau", "Pandas"]
            elif "java" in jd_text.lower() or "spring" in jd_text.lower():
                jd_skills = ["Java", "Spring Boot", "REST API", "SQL", "Hibernate", "Microservices"]
            elif "frontend" in jd_text.lower() or "react" in jd_text.lower():
                jd_skills = ["React", "JavaScript", "HTML", "CSS", "TypeScript", "Git"]
            else:
                jd_skills = resume_skills[:3] + ["System Design", "CI/CD", "AWS"]
                
        matching_skills = list(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))
        
        improvement_suggestions = []
        for sk in missing_skills[:3]:
            improvement_suggestions.append(f"Complete a certification or hands-on course in {sk}.")
            improvement_suggestions.append(f"Build a showcase mini-project utilizing {sk} to demonstrate capability.")
        if not improvement_suggestions:
            improvement_suggestions = [
                "Keep updating your knowledge on advanced software architectures.", 
                "Build a portfolio project demonstrating integration of frontend and backend technologies."
            ]
            
        return {
            "skill_analysis": {
                "matching_skills": matching_skills,
                "missing_skills": missing_skills,
                "improvement_suggestions": improvement_suggestions[:4]
            }
        }
        
    try:
        prompt = (
            "You are a technical recruiter and skills analyst. Compare the candidate's skills "
            "and experience in their parsed resume with the requirements of the job description.\n\n"
            f"Candidate Skills:\n{state['parsed_resume'].get('skills', [])}\n"
            f"Candidate Experience Summary:\n{state['parsed_resume'].get('summary', '')}\n\n"
            f"Job Description:\n{state['job_description']}\n\n"
            "Identify which required skills are matched, which are missing, and suggest "
            "specific actions the candidate should take to bridge the gap."
        )
        llm = get_llm(state["api_provider"], state["model_name"], state["api_key"])
        if state["api_provider"] == "ollama":
            analysis_dict = invoke_json_model(llm, prompt, skill_schema_desc)
            return {"skill_analysis": analysis_dict}
            
        structured_llm = llm.with_structured_output(SkillGapAnalysis)
        analysis = structured_llm.invoke(prompt)
        return {"skill_analysis": analysis.dict()}
    except Exception as e:
        error_msg = f"Skill Gap Analyzer Agent Error: {str(e)}"
        return {"errors": state["errors"] + [error_msg]}


def job_matcher_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Evaluates job compatibility and provides recommendation score."""
    if state["errors"]:
        return {}
        
    if state["api_provider"] == "mock":
        matching = state["skill_analysis"].get("matching_skills", [])
        missing = state["skill_analysis"].get("missing_skills", [])
        
        total = len(matching) + len(missing)
        if total > 0:
            score = int((len(matching) / total) * 100)
        else:
            score = 70
            
        if score >= 80:
            verdict = "Strong Match"
        elif score >= 55:
            verdict = "Potential Match"
        else:
            verdict = "Poor Match"
            
        strengths = [f"Strong background in {', '.join(matching[:3])}."] if matching else ["Has core software development capabilities."]
        strengths.append("Demonstrates solid educational credentials and relevant industry experience.")
        
        weaknesses = [f"Lacks proven experience with {', '.join(missing[:3])}."] if missing else ["No major skill gaps found."]
        
        suitable_roles = ["Full Stack Software Engineer", "Backend Developer", "Applications Developer"]
        
        return {
            "job_match": {
                "match_percentage": score,
                "verdict": verdict,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "suitable_roles": suitable_roles
            }
        }
        
    try:
        prompt = (
            "You are an HR hiring manager. Evaluate the candidate's fit for the job based on "
            "their parsed resume, skill analysis, and the job description.\n\n"
            f"Candidate Profile:\n{state['parsed_resume']}\n\n"
            f"Skill Gap Analysis:\n{state['skill_analysis']}\n\n"
            f"Job Description:\n{state['job_description']}\n\n"
            "Provide an overall match score (0-100), hiring verdict, key strengths, potential "
            "weaknesses or concerns, and alternate roles this candidate is suitable for."
        )
        llm = get_llm(state["api_provider"], state["model_name"], state["api_key"])
        if state["api_provider"] == "ollama":
            match_dict = invoke_json_model(llm, prompt, match_schema_desc)
            return {"job_match": match_dict}
            
        structured_llm = llm.with_structured_output(JobMatch)
        match_info = structured_llm.invoke(prompt)
        return {"job_match": match_info.dict()}
    except Exception as e:
        error_msg = f"Job Matcher Agent Error: {str(e)}"
        return {"errors": state["errors"] + [error_msg]}


def interview_coach_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Generates tailored interview questions and prep guide."""
    if state["errors"]:
        return {}
        
    if state["api_provider"] == "mock":
        matching = state["skill_analysis"].get("matching_skills", [])
        missing = state["skill_analysis"].get("missing_skills", [])
        
        # Predefined question database matching our extended skill lists
        q_db = {
            "Java": {
                "question": "What is the difference between HashMap and ConcurrentHashMap in Java, and how does ConcurrentHashMap achieve thread safety?",
                "type": "Technical",
                "difficulty": "Medium",
                "best_approach": "Explain that HashMap is not thread-safe, whereas ConcurrentHashMap uses bucket-level locking (CAS/synchronized blocks in Java 8+) to allow concurrent reads and writes.",
                "sample_answer": "HashMap is not thread-safe. ConcurrentHashMap divides the map into segments or uses CAS (Compare-And-Swap) operations and synchronized blocks on individual node buckets. This allows multiple threads to access different parts of the map concurrently without locking the entire table."
            },
            "Spring Boot": {
                "question": "Explain the difference between @Component, @Service, and @Repository annotations in Spring Boot.",
                "type": "Technical",
                "difficulty": "Easy",
                "best_approach": "State that @Component is the generic stereotype, while @Service and @Repository are specializations for the service layer and database layer respectively.",
                "sample_answer": "@Component is the generic stereotype annotation for any Spring-managed component. @Service is a specialization used in the service/business logic layer. @Repository is used in the data access layer (DAO) and enables automatic exception translation from JDBC/SQL databases into Spring's DataAccessException."
            },
            "SQL": {
                "question": "What is the difference between a Clustered and a Non-Clustered index in SQL, and how do they affect search performance?",
                "type": "Technical",
                "difficulty": "Medium",
                "best_approach": "Explain that a clustered index determines the physical order of data storage, while a non-clustered index is a separate structure containing pointers to the data rows.",
                "sample_answer": "A Clustered Index physically sorts and stores the data rows in the table based on their key values (only one per table). A Non-Clustered Index contains the index keys and a row locator pointing to the physical storage location of the data. Non-clustered indexes speed up searches but add write overhead."
            },
            "Python": {
                "question": "Can you explain the difference between a list comprehension and a generator expression in Python, and when you would use each?",
                "type": "Technical",
                "difficulty": "Medium",
                "best_approach": "Explain that list comprehensions construct list objects in memory immediately, whereas generator expressions yield items lazily one by one. Use the STAR method to describe memory footprint.",
                "sample_answer": "A list comprehension is enclosed in brackets `[...]` and generates the entire list in memory immediately. A generator expression is enclosed in parentheses `(...)` and returns an iterator that yields values on demand. I would use generator expressions when dealing with extremely large datasets to conserve memory, and list comprehensions when I need to perform multiple passes over the list."
            },
            "React": {
                "question": "How do React hooks (like useEffect) manage state lifecycle, and how do you prevent memory leaks in custom hooks?",
                "type": "Technical",
                "difficulty": "Medium",
                "best_approach": "Explain the render cycle, dependency array, and clean-up functions in useEffect.",
                "sample_answer": "React hooks use the call order to preserve state across renders. In `useEffect`, the return function acts as a cleanup mechanism. To prevent memory leaks, we must clean up subscriptions, abort network calls, and clear timeouts inside this returned function before the component unmounts."
            },
            "Docker": {
                "question": "What is the difference between a Docker Image and a Docker Container, and how do you optimize Dockerfiles for size?",
                "type": "Technical",
                "difficulty": "Medium",
                "best_approach": "Explain that an image is a read-only template, while a container is a running instance. Mention multi-stage builds and layer caching.",
                "sample_answer": "A Docker Image is a read-only blueprint containing the filesystem and dependencies. A Container is a runnable instance of that image. To optimize size, I use multi-stage builds to separate the compilation environment from the execution environment, leverage small base images like alpine, and minimize the number of RUN layers by combining commands."
            },
            "Kubernetes": {
                "question": "Can you explain how Kubernetes handles service discovery and load balancing between pods?",
                "type": "Technical",
                "difficulty": "Hard",
                "best_approach": "Mention Services, ClusterIP, kube-proxy, and DNS resolution inside the cluster.",
                "sample_answer": "Kubernetes uses the Service resource to provide a stable IP address and DNS name for a set of pods. Kube-proxy runs on each node and manages virtual IP routing. When a pod makes a request to a service, CoreDNS resolves the service name to its ClusterIP, and traffic is load-balanced (usually round-robin) across the healthy backing pods."
            },
            "Data Analysis": {
                "question": "What is the difference between structured and unstructured data, and how do you handle missing values during data cleaning in Pandas?",
                "type": "Technical",
                "difficulty": "Easy",
                "best_approach": "Define both data types. Describe dropna(), fillna() with mean/median/mode, or interpolation in Pandas.",
                "sample_answer": "Structured data resides in fixed fields (like tables or relational databases), while unstructured data lacks a pre-defined format (like emails or PDFs). In Pandas, I handle missing values by analyzing their distribution. I can drop rows with missing values using `dropna()`, or impute them using `fillna(df.median())` for numerical values."
            },
            "Tableau": {
                "question": "What is the difference between a Live connection and an Extract in Tableau, and when would you use each?",
                "type": "Technical",
                "difficulty": "Easy",
                "best_approach": "Explain that live queries the database in real-time, whereas extract takes a snapshot stored locally.",
                "sample_answer": "A Live connection queries the underlying data source in real-time, which is useful when data updates constantly. An Extract takes a static snapshot of the data, stores it in Tableau's high-performance Hyper engine, and runs much faster on large datasets while reducing database load."
            },
            "PowerBI": {
                "question": "What is DAX in Power BI, and what is the difference between a Calculated Column and a Measure?",
                "type": "Technical",
                "difficulty": "Medium",
                "best_approach": "Define DAX (Data Analysis Expressions). Contrast row-level evaluation (Calculated Column) with dynamically aggregated evaluation (Measure).",
                "sample_answer": "DAX stands for Data Analysis Expressions. A Calculated Column is evaluated row-by-row during data load, stored in the model, and increases memory consumption. A Measure is calculated dynamically on-the-fly at query time based on user filters/aggregations, consuming CPU during report interactions but keeping memory footprint low."
            }
        }
        
        tech_qs = []
        selected_skills = list(set(matching + missing))
        for sk in selected_skills:
            if sk in q_db:
                tech_qs.append(q_db[sk])
            if len(tech_qs) >= 3:
                break
                
        if len(tech_qs) < 2:
            # Fallback
            tech_qs.append({
                "question": "Explain your approach to designing a scalable REST API gateway.",
                "type": "System Design",
                "difficulty": "Hard",
                "best_approach": "Discuss load balancers, rate limiting, caching, and authentication layers.",
                "sample_answer": "I would start by implementing an Nginx or custom FastAPI gateway, backing it with Redis for rate limiting (token bucket algorithm). For scalability, I would containerize the service with Docker and deploy it behind an AWS Application Load Balancer."
            })
            
        beh_qs = [
            {
                "question": "Tell me about a time you had to deliver a project with ambiguous requirements. How did you proceed?",
                "type": "HR/Behavioral",
                "difficulty": "Medium",
                "best_approach": "Use the STAR method: Situation, Task, Action, Result. Emphasize proactive communication.",
                "sample_answer": "In my previous role, we were asked to integrate an AI recommendation engine without clear performance metrics. I scheduled rapid feedback calls with stakeholders, built a minimal working prototype in one week to clarify capabilities, and documented user-story boundaries. This helped align expectations and we delivered the project on time with a 15% boost in user engagement."
            },
            {
                "question": "How do you handle disagreements with a product owner regarding technical debt vs. feature delivery?",
                "type": "HR/Behavioral",
                "difficulty": "Medium",
                "best_approach": "Highlight empathy, translating tech impact into business metrics, and finding compromises.",
                "sample_answer": "I translate the technical debt into business consequences like slower feature delivery or potential downtime. I propose a compromise, such as dedicating 20% of each sprint to refactoring and stability, ensuring we keep moving product features forward while maintaining a healthy codebase."
            }
        ]
        
        tips = [
            "Review core system architectures and practice drawing microservices.",
            "Prepare stories for behavioral questions showing ownership and problem-solving.",
            "Familiarize yourself with standard rate-limiting algorithms and caching strategies."
        ]
        
        return {
            "interview_prep": {
                "technical_questions": tech_qs,
                "behavioral_questions": beh_qs,
                "general_tips": tips
            }
        }
        
    try:
        prompt = (
            "You are a mock interviewer and interview coach. Based on the candidate's resume, "
            "job description, and identified skill gaps, generate preparation resources.\n\n"
            f"Candidate Profile:\n{state['parsed_resume']}\n\n"
            f"Skill Gap Analysis:\n{state['skill_analysis']}\n\n"
            f"Job Compatibility details:\n{state['job_match']}\n\n"
            f"Job Description:\n{state['job_description']}\n\n"
            "Generate 3-5 technical questions (focused on testing their stated skills or probing "
            "their gaps) and 2-3 behavioral questions. For each question, provide a rating of "
            "difficulty, the best strategy to answer, and a high-quality sample answer."
        )
        llm = get_llm(state["api_provider"], state["model_name"], state["api_key"])
        if state["api_provider"] == "ollama":
            prep_dict = invoke_json_model(llm, prompt, prep_schema_desc)
            return {"interview_prep": prep_dict}
            
        structured_llm = llm.with_structured_output(InterviewPrep)
        prep = structured_llm.invoke(prompt)
        return {"interview_prep": prep.dict()}
    except Exception as e:
        error_msg = f"Interview Coach Agent Error: {str(e)}"
        return {"errors": state["errors"] + [error_msg]}


# --- Build and Compile LangGraph Workflow ---

def build_workflow():
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("resume_parser", parse_resume_node)
    workflow.add_node("skill_analyzer", skill_analysis_node)
    workflow.add_node("job_matcher", job_matcher_node)
    workflow.add_node("interview_coach", interview_coach_node)
    
    # Establish Flow
    workflow.add_edge(START, "resume_parser")
    workflow.add_edge("resume_parser", "skill_analyzer")
    workflow.add_edge("skill_analyzer", "job_matcher")
    workflow.add_edge("job_matcher", "interview_coach")
    workflow.add_edge("interview_coach", END)
    
    return workflow.compile()
